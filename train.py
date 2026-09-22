"""
Скрипт дообучения (Fine-Tuning) модели Qwen2.5-7B-Instruct методом LoRA (PEFT)
Чистый, нативный цикл обучения PyTorch без использования SFTTrainer и SFTConfig.
Оптимизирован для Windows 11 и видеокарты NVIDIA RTX 4060 (8 GB VRAM).
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb

# Загрузка переменных окружения из .env
load_dotenv()

# Поддержка UTF-8 в консоли Windows
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qwen_trainer")

BASE_DIR = Path(__file__).parent.resolve()


def check_gpu_environment():
    """Проверка доступности CUDA и видеопамяти."""
    if not torch.cuda.is_available():
        logger.warning(
            "⚠️ CUDA не обнаружена! Обучение на CPU для модели 7B практически невозможно.\n"
            "Убедитесь, что установлены драйверы NVIDIA и версия PyTorch с поддержкой CUDA."
        )
    else:
        device_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("=" * 70)
        print("🖥️  АППАРАТНОЕ ОКРУЖЕНИЕ (WINDOWS 11 / NATIVE PYTORCH)")
        print("=" * 70)
        print(f"Видеокарта:      {device_name}")
        print(f"Объем VRAM:      {total_vram:.2f} GB")
        print(f"Версия PyTorch:  {torch.__version__}")
        print("Режим:           Нативный PyTorch + QLoRA 4-bit (BitsAndBytes)")
        print("-" * 70)


class ChatDataset(Dataset):
    """Датасет диалогов из JSONL с форматированием через apply_chat_template."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 256):
        self.samples = []
        logger.info("Загрузка датасета из %s...", jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data and isinstance(data["messages"], list):
                        self.samples.append(data["messages"])
                except Exception:
                    continue

        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info("Успешно загружено диалогов: %d", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_messages = self.samples[idx]

        # Гарантируем, что самый первый элемент ВСЕГДА имеет вид {"role": "system", "content": ...}
        if not raw_messages or raw_messages[0].get("role") != "system":
            sys_prompt = os.getenv("SYSTEM_PROMPT") or "Ты 17-летний Иван из Бийска."
            messages = [{"role": "system", "content": sys_prompt}] + [
                m for m in raw_messages if m.get("role") in ("user", "assistant")
            ]
        else:
            messages = raw_messages

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Маскирование паддингов: заменяем все pad_token_id и незаполненные позиции на -100,
        # чтобы модель не училась генерировать пустые токены
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Коллатор для batch_size=1 без лишнего паддинга."""
    item = batch[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "labels": item["labels"].unsqueeze(0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Нативное PyTorch дообучение Qwen2.5-7B-Instruct (QLoRA 4-bit + PEFT)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Базовая модель HuggingFace (по умолчанию: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("DATASET_PATH", str(BASE_DIR / "my_training_dataset.jsonl")),
        help="Путь к файлу датасета JSONL (по умолчанию: my_training_dataset.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("ADAPTER_PATH", str(BASE_DIR / "vanya_lora_weights")),
        help="Папка для сохранения обученных LoRA-весов (по умолчанию: ./vanya_lora_weights)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Количество шагов обучения (по умолчанию: 1000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Скорость обучения / Learning rate (по умолчанию: 5e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Размер батча (СТРОГО 1 для 8GB VRAM)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Шаги накопления градиента / Gradient accumulation (по умолчанию: 8)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Интервал сохранения промежуточных чекпоинтов (по умолчанию: 100)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения (по умолчанию автоматически проверяется checkpoint-400)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Максимальная длина последовательности токенов (по умолчанию: 256)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="Ранг LoRA матрицы r (по умолчанию: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="Коэффициент масштабирования lora_alpha (по умолчанию: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (по умолчанию: 0.05)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Частота вывода логов в консоль (по умолчанию: 10)",
    )

    args = parser.parse_args()

    # Проверка GPU
    check_gpu_environment()

    dataset_file = Path(args.dataset_path).resolve()
    if not dataset_file.exists():
        logger.error("Файл датасета не найден: %s", dataset_file)
        print("Сначала запустите скрипт сборки датасета: python dataset_creator.py")
        sys.exit(1)

    print("=" * 70)
    print("🚀 ПАРАМЕТРЫ ТРЕНИРОВКИ")
    print("=" * 70)
    print(f"Базовая модель:        {args.model_id}")
    print(f"Файл датасета:         {dataset_file}")
    print(f"Папка сохранения:      {args.output_dir}")
    print(f"Квантование:           4-bit NF4 (Double Quant=True, Compute=Float16)")
    print(f"Оптимизатор:           bnb.optim.AdamW8bit (lr={args.learning_rate})")
    print(f"Параметры LoRA:        r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Целевые слои:          q_proj, v_proj, k_proj, o_proj")
    print(f"Размер батча:          {args.batch_size} (накопление x{args.grad_accum} -> эффективный: {args.batch_size * args.grad_accum})")
    print(f"Шагов обучения:        {args.max_steps}")
    print(f"Чекпоинты каждые:      {args.save_steps} шагов")
    print(f"Макс. длина контекста: {args.max_seq_length} токенов")
    print("-" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ЗАГРУЗКА ТОКЕНИЗАТОРА
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Загрузка токенизатора для %s...", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ЗАГРУЗКА МОДЕЛИ В 4-BIT QLoRA (BitsAndBytes)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Настройка BitsAndBytesConfig (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Загрузка базовой модели в 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # Сразу после загрузки модели (ДО наложения LoRA) оборачиваем в k-bit подготовку
    logger.info("Подготовка модели для k-bit обучения...")
    model = prepare_model_for_kbit_training(model)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. НАКАТЫВАНИЕ LORA АДАПТЕРА (ИЛИ ПОДГРУЗКА ИЗ ЧЕКПОИНТА)
    # ─────────────────────────────────────────────────────────────────────────
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    ckpt_400 = output_path / "checkpoint-400"
    resume_path = None

    if args.resume_from_checkpoint:
        cand = Path(args.resume_from_checkpoint).resolve()
        if cand.exists():
            resume_path = cand
        else:
            logger.warning("Указанный чекпоинт '%s' не найден!", cand)

    if resume_path is None and ckpt_400.exists():
        resume_path = ckpt_400

    if resume_path is not None:
        try:
            start_step = int(resume_path.name.split("-")[-1])
        except (ValueError, IndexError):
            start_step = 400

        logger.info(
            "Обнаружен чекпоинт %s! Загрузка обученных весов через PeftModel.from_pretrained(is_trainable=True)...",
            resume_path
        )
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
        print(f"\n🔄 ВОЗОБНОВЛЕНИЕ ОБУЧЕНИЯ: загружен чекпоинт {resume_path.name}")
        print(f"Шаги: с {start_step} до {args.max_steps} (осталось пройти: {max(0, args.max_steps - start_step)} шагов)")
    else:
        start_step = 0
        logger.info("Чекпоинт не найден. Инициализация нового адаптера LoRA...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        print("🌱 Инициализирован новый LoRA-адаптер с нуля.")

    model.print_trainable_parameters()

    # Определение рабочего устройства (CUDA)
    device = next(model.parameters()).device
    logger.info("Модель размещена на устройстве: %s", device)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ПОДГОТОВКА DATALOADER
    # ─────────────────────────────────────────────────────────────────────────
    dataset = ChatDataset(
        jsonl_path=dataset_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ОПТИМИЗАТОР (8-битный AdamW из bitsandbytes)
    # ─────────────────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. КЛАССИЧЕСКИЙ ЦИКЛ ОБУЧЕНИЯ (NATIVE PYTORCH LOOP)
    # ─────────────────────────────────────────────────────────────────────────
    # Обязательно активируем расчет градиентов для входных эмбеддингов
    # перед циклом обучения замороженной базовой модели с LoRA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()
    optimizer.zero_grad()

    print("\n" + "=" * 70)
    print("🔥 ЗАПУСК ОБУЧЕНИЯ (NATIVE PYTORCH)...")
    print("=" * 70)
    if start_step > 0:
        print(f"Шаги: {start_step} -> {args.max_steps} | Накопление градиентов: каждые {args.grad_accum} шагов")
    else:
        print(f"Шагов: {args.max_steps} | Накопление градиентов: каждые {args.grad_accum} шагов")
    print(f"Чекпоинты: каждые {args.save_steps} шагов -> {output_path}/checkpoint-N")
    print("Для безопасной остановки нажмите Ctrl+C — текущие веса сохранятся автоматически!")
    print("Следите за loss в терминале.")
    print("-" * 70 + "\n", flush=True)

    step = start_step
    running_loss = 0.0
    stop_training = False
    interrupted = False

    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if step >= args.max_steps:
        print(f"⚠️ Обучение уже завершено (текущий шаг {step} >= max_steps {args.max_steps}).")
        stop_training = True

    try:
        for epoch in range(100):
            for batch in dataloader:
                step += 1

                # Ручной перенос каждого тензора на GPU
                input_ids = batch["input_ids"].to(target_device)
                attention_mask = batch["attention_mask"].to(target_device)
                labels = batch["labels"].to(target_device)

                # Прямой проход
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Масштабирование для накопления градиентов
                loss_scaled = loss / args.grad_accum
                loss_scaled.backward()

                running_loss += loss.item()

                # Шаг оптимизатора каждые grad_accum шагов или на последнем шаге
                if step % args.grad_accum == 0 or step == args.max_steps:
                    optimizer.step()
                    optimizer.zero_grad()

                # Вывод метрик в консоль
                if step % args.logging_steps == 0 or step == start_step + 1 or step == args.max_steps:
                    div = args.logging_steps if step > (start_step + 1) else 1
                    avg_loss = running_loss / div
                    print(f"Step: {step}/{args.max_steps} | Loss: {avg_loss:.4f}", flush=True)
                    running_loss = 0.0

                # Промежуточное сохранение чекпоинта каждые save_steps шагов
                if args.save_steps > 0 and step % args.save_steps == 0 and step > start_step and step < args.max_steps:
                    ckpt_dir = output_path / f"checkpoint-{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n💾 [Чекпоинт {step}] Сохранение промежуточных весов в {ckpt_dir}...", flush=True)
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

                if step >= args.max_steps:
                    stop_training = True
                    break

            if stop_training:
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\n" + "!" * 70)
        print(f"⚠️ Обучение прервано пользователем (Ctrl+C) на шаге {step}!")
        print("Сохранение текущих весов LoRA в основную папку...")
        print("!" * 70 + "\n", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Сохранение адаптера LoRA в %s...", output_path)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("\n" + "=" * 70)
    if interrupted:
        print(f"💾 ВЕСА УСПЕШНО СОХРАНЕНЫ ПОСЛЕ ПРЕРЫВАНИЯ (Шаг {step}/{args.max_steps})!")
    else:
        print("🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"📁 Обученные LoRA-веса сохранены в: {output_path}")
    print("\nИнструкция для подключения в Ollama:")
    print("1. Создайте файл Modelfile:")
    print("   FROM qwen2.5:7b-instruct-q4_K_M")
    print(f"   ADAPTER {output_path.as_posix()}")
    print("2. Соберите модель:")
    print("   ollama create vanya_qwen -f Modelfile")
    print("3. Пропишите в .env:")
    print("   OLLAMA_MODEL=vanya_qwen")
    print("=" * 70)


if __name__ == "__main__":
    main()
