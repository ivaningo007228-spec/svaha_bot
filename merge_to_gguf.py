"""
Мост после train.py: LoRA -> слитая модель Float16 -> GGUF Q5_K_M -> ollama create.

Одна команда из папки проекта:
    py merge_to_gguf.py

llama-cpp-python только запускает уже готовый GGUF и не умеет квантовать Q5_K_M.
Скрипт сам скачивает исходники llama.cpp и llama-quantize одной и той же сборки,
конвертирует Hugging Face в F16 и жмёт Q5_K_M системной командой.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
TOOLS_DIR = BASE_DIR / ".tools"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
USER_AGENT = "svaha-bot-merge"


def _project_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70, flush=True)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Скачиваю {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total}%", end="", flush=True)
    if total:
        print()
    tmp.replace(dest)


def _read_url(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def latest_llama_tag() -> str:
    """Тег сборки, где лежат и исходники конвертера, и llama-quantize.exe."""
    payload = json.loads(_read_url("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"))
    tag = str(payload.get("tag_name") or "")
    if len(tag) > 1 and tag.startswith("b") and tag[1:].isdigit():
        return tag
    for asset in payload.get("assets") or []:
        if asset.get("name") == "nightly-tag.txt" and asset.get("browser_download_url"):
            nightly = _read_url(asset["browser_download_url"]).strip()
            if nightly:
                return nightly
    raise SystemExit("Не удалось узнать сборку llama.cpp с бинарником llama-quantize")


def latest_adapter(adapter_dir: Path) -> Path:
    """Последний checkpoint-N, иначе финальный адаптер в корне папки."""
    if not adapter_dir.is_dir():
        raise SystemExit(f"Папка LoRA не найдена: {adapter_dir}")

    checkpoints: list[tuple[int, Path]] = []
    for path in adapter_dir.glob("checkpoint-*"):
        if not (path / "adapter_config.json").is_file():
            continue
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if checkpoints:
        checkpoints.sort()
        chosen = checkpoints[-1][1]
        print(f"Последний чекпоинт: {chosen}")
        return chosen

    if (adapter_dir / "adapter_config.json").is_file():
        print(f"Чекпоинтов нет, беру финальный адаптер: {adapter_dir}")
        return adapter_dir

    raise SystemExit(f"В {adapter_dir} нет adapter_config.json")


def merge_lora(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    """База Qwen в Float16 на CPU, плюс LoRA, затем merge_and_unload."""
    _banner("1. Слияние LoRA с Qwen в Float16")
    print(f"База:    {base_model}")
    print(f"LoRA:    {adapter_dir}")
    print(f"Куда:    {merged_dir}")
    print("Гружу на CPU: Float16 7B не влезает в 8 ГБ видеопамяти.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    load_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            **load_kwargs,
        )

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    with torch.inference_mode():
        merged = model.merge_and_unload()

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Слитая модель сохранена: {merged_dir}", flush=True)

    del merged
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_llama_cpp() -> tuple[Path, Path]:
    """Исходник convert_hf_to_gguf.py и llama-quantize одной сборки."""
    override_src = os.getenv("LLAMA_CPP_DIR", "").strip()
    override_bin = os.getenv("LLAMA_QUANTIZE", "").strip()
    if override_src and override_bin:
        return Path(override_src), Path(override_bin)

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tag_file = TOOLS_DIR / "llama.tag"
    src_dir = TOOLS_DIR / "llama.cpp"
    bin_dir = TOOLS_DIR / "llama-bin"
    convert_py = src_dir / "convert_hf_to_gguf.py"

    quantize = next(bin_dir.rglob("llama-quantize.exe"), None) if bin_dir.exists() else None
    if quantize is None and bin_dir.exists():
        quantize = next(bin_dir.rglob("llama-quantize"), None)

    cached_tag = tag_file.read_text(encoding="utf-8").strip() if tag_file.is_file() else ""
    if convert_py.is_file() and quantize is not None and cached_tag:
        _install_convert_deps(src_dir)
        return convert_py, quantize

    tag = latest_llama_tag()
    _banner(f"Качаю llama.cpp {tag}")

    if src_dir.exists():
        shutil.rmtree(src_dir)
    archive = TOOLS_DIR / f"llama.cpp-{tag}.zip"
    present = {path.name for path in TOOLS_DIR.iterdir()}
    _download(f"{LLAMA_REPO}/archive/refs/tags/{tag}.zip", archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(TOOLS_DIR)
    extracted_dirs = [
        path
        for path in TOOLS_DIR.iterdir()
        if path.is_dir() and path.name not in present and path.name != src_dir.name
    ]
    if len(extracted_dirs) != 1:
        raise SystemExit(f"Архив llama.cpp распаковался неожиданно: {extracted_dirs}")
    extracted_dirs[0].rename(src_dir)
    archive.unlink(missing_ok=True)

    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    quant_zip = TOOLS_DIR / f"llama-{tag}-bin-win-cpu-x64.zip"
    _download(
        f"{LLAMA_REPO}/releases/download/{tag}/llama-{tag}-bin-win-cpu-x64.zip",
        quant_zip,
    )
    with zipfile.ZipFile(quant_zip) as bundle:
        bundle.extractall(bin_dir)
    quant_zip.unlink(missing_ok=True)

    convert_py = src_dir / "convert_hf_to_gguf.py"
    quantize = next(bin_dir.rglob("llama-quantize.exe"), None)
    if quantize is None:
        quantize = next(bin_dir.rglob("llama-quantize"), None)
    if not convert_py.is_file() or quantize is None:
        raise SystemExit("В скачанной сборке llama.cpp нет convert_hf_to_gguf.py или llama-quantize")

    tag_file.write_text(tag, encoding="utf-8")
    _install_convert_deps(src_dir)
    return convert_py, quantize


def _install_convert_deps(src_dir: Path) -> None:
    try:
        import gguf  # noqa: F401
        return
    except ImportError:
        pass

    requirements = src_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if not requirements.is_file():
        requirements = src_dir / "requirements.txt"
    if requirements.is_file():
        _run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
        return
    _run([sys.executable, "-m", "pip", "install", "gguf", "protobuf", "sentencepiece", "numpy"])


def convert_gguf(merged_dir: Path, gguf_path: Path, keep_f16: bool) -> None:
    _banner("2. Конвертация в GGUF Q5_K_M")
    convert_py, quantize = _ensure_llama_cpp()
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_path.with_name(gguf_path.stem + ".f16.gguf")

    _run(
        [
            sys.executable,
            str(convert_py),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )
    if not f16_path.is_file():
        raise SystemExit(f"Конвертер не создал {f16_path}")
    _run([str(quantize), str(f16_path), str(gguf_path), "Q5_K_M"], cwd=quantize.parent)

    if not keep_f16 and f16_path.exists():
        f16_path.unlink()
        print(f"Удалил промежуточный F16: {f16_path}")
    print(f"GGUF готов: {gguf_path}", flush=True)


def _from_line(gguf_path: Path, modelfile: Path) -> str:
    try:
        relative = gguf_path.resolve().relative_to(modelfile.parent.resolve())
        return "./" + relative.as_posix()
    except ValueError:
        return gguf_path.resolve().as_posix()


def update_modelfile(modelfile: Path, gguf_path: Path) -> str:
    if not modelfile.is_file():
        raise SystemExit(f"Modelfile не найден: {modelfile}")
    target = _from_line(gguf_path, modelfile)
    lines = modelfile.read_text(encoding="utf-8").splitlines()
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith("FROM "):
            updated.append(f"FROM {target}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.insert(0, f"FROM {target}")
    modelfile.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"Modelfile: FROM {target}")
    return target


def create_ollama_model(model_name: str, modelfile: Path) -> None:
    _banner("3. Обновление модели в Ollama")
    ollama = shutil.which("ollama")
    if not ollama:
        raise SystemExit("Команда ollama не найдена в PATH")
    _run([ollama, "create", model_name, "-f", str(modelfile)], cwd=modelfile.parent)
    print(f"Ollama-модель {model_name} пересобрана.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Слить LoRA в Qwen Float16, сжать в Q5_K_M и обновить vanya_q5 в Ollama"
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--adapter",
        default=os.getenv("ADAPTER_PATH", str(BASE_DIR / "vanya_lora_weights")),
    )
    parser.add_argument(
        "--merged-dir",
        default=os.getenv("MERGED_HF_PATH", str(BASE_DIR / "vanya_merged_hf")),
    )
    parser.add_argument(
        "--gguf",
        default=os.getenv("GGUF_PATH", str(BASE_DIR / "models" / "vanya_q5_trained.gguf")),
    )
    parser.add_argument(
        "--modelfile",
        default=str(BASE_DIR / "Modelfile"),
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "vanya_q5"),
    )
    parser.add_argument("--skip-merge", action="store_true", help="Не сливать заново, взять vanya_merged_hf")
    parser.add_argument("--skip-gguf", action="store_true", help="Не конвертировать, взять уже готовый GGUF")
    parser.add_argument("--skip-ollama", action="store_true", help="Не запускать ollama create")
    parser.add_argument("--keep-f16", action="store_true", help="Оставить промежуточный F16 GGUF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_root = _project_path(args.adapter)
    merged_dir = _project_path(args.merged_dir)
    gguf_path = _project_path(args.gguf)
    modelfile = _project_path(args.modelfile)

    if not args.skip_merge:
        merge_lora(args.base_model, latest_adapter(adapter_root), merged_dir)
    elif not (merged_dir / "config.json").is_file():
        raise SystemExit(f"Слитая модель не найдена: {merged_dir}")

    if not args.skip_gguf:
        convert_gguf(merged_dir, gguf_path, keep_f16=args.keep_f16)
    elif not gguf_path.is_file():
        raise SystemExit(f"GGUF не найден: {gguf_path}")

    update_modelfile(modelfile, gguf_path)
    if not args.skip_ollama:
        create_ollama_model(args.ollama_model, modelfile)

    _banner("Готово")
    print(f"GGUF:    {gguf_path}")
    print(f"Ollama:  {args.ollama_model}")
    print("Юзербот уже смотрит на эту модель через OLLAMA_MODEL.", flush=True)


if __name__ == "__main__":
    main()
