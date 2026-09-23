"""
Интерактивный скрипт тестирования обученной LoRA-маски речи (Ваня)
Локальный запуск на RTX 4060 (8GB VRAM) в режиме 4-bit QLoRA.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import sys
import time
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
logger = logging.getLogger("inference")

# ─────────────────────────────────────────────────────────────────────────────
# ПУТИ И НАСТРОЙКИ ПО УМОЛЧАНИЮ (легко изменить здесь)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Основной путь к адаптеру (можно указать checkpoint-100, checkpoint-200 или ./vanya_lora_weights)
DEFAULT_ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./vanya_lora_weights")


DEFAULT_SYSTEM_PROMPT = (
    "Ты 17-летний Иван из Бийска. Твоя задача — вести живой, естественный диалог с девушкой, "
    "общаться легко, непринужденно, с юмором, в вежливом пацанском стиле, без канцеляризмов и длинных занудных монологов."
)


def load_model_and_tokenizer(base_model_id: str, adapter_path: str):
    """
    Загрузка базовой модели Qwen2.5-7B в 4-битном режиме (NF4)
    и наложение обученного LoRA-адаптера.
    """
    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.exists():
        fallback_dir = BASE_DIR / "vanya_lora_weights"
        if fallback_dir.exists():
            logger.warning(
                "Указанный путь к адаптеру '%s' не найден. Используем: %s",
                adapter_dir, fallback_dir
            )
            adapter_dir = fallback_dir
        else:
            raise FileNotFoundError(
                f"Папка с LoRA-весами не найдена ни по пути {adapter_dir}, ни по {fallback_dir}."
            )

    logger.info("Загрузка токенизатора из %s...", adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model_id)
    tokenizer_source = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Настройка 4-битного квантования BitsAndBytes (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Загрузка базовой модели '%s'...", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Подгрузка обученного LoRA-адаптера из: %s...", adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    return model, tokenizer, str(adapter_dir)


def get_time_schedule_prompt() -> str:
    """Динамическое определение текущего времени на ПК и расписания Вани."""
    current_time = datetime.now().strftime("%H:%M")
    return (
        f'Текущее время на часах: {current_time}. Учитывай это, когда отвечаешь на вопросы о том, '
        'что ты сейчас делаешь или где находишься(с 8:00 до 14:00 в будние дни школа, '
        '15:30 - 16:30 в понедельник - репетитор по физике, 15:00 - 17:00 вторник - репетитор по математике, '
        '14:00-15:00 среда - репетитор по русскому, 16:00 - 17:00 четверг - репетитор по физике, '
        '18:00 - 19:00 пятница - репетитор по русскому; ночью отвечай что нибудь типа "да ничего не делаю" или "валяюсь", '
        'в остальное время - "программирую", "кушаю", "с другом разговариваю" и т.п.).'
    )


def prepare_messages_for_chat_template(
    dialog_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Формирует список сообщений для tokenizer.apply_chat_template.
    Динамически добавляет текущее время на ПК и расписание в самое начало системного промпта.
    Гарантирует, что самый первый элемент ВСЕГДА имеет ровно такой вид:
    {"role": "system", "content": ...}.
    """
    base_system_prompt = os.getenv("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    time_prefix = get_time_schedule_prompt()
    system_prompt = f"{time_prefix}\n\n{base_system_prompt}"

    # 1. Первый элемент в списке ВСЕГДА строго role: system с динамическим временем
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    # 2. Добавляем диалоговые реплики (user и assistant), исключая любые дублирующиеся system
    for msg in dialog_history:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    return messages


def generate_reply(
    model,
    tokenizer,
    history: list[dict[str, str]],
    temperature: float = 0.4,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.08,
    max_new_tokens: int = 64,
) -> str:
    """Генерация ответа Ивана с использованием Chat Template модели Qwen."""
    # Формируем корректный список сообщений с обязательным системным промптом на 1 месте
    messages = prepare_messages_for_chat_template(history)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply


def main():
    parser = argparse.ArgumentParser(
        description="Интерактивный чат с дообученной LoRA-моделью Вани (4-bit inference)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"Путь к чекпоинту адаптера (по умолчанию: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Базовая модель HuggingFace (по умолчанию: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Температура генерации (по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p сэмплирование (по умолчанию: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k фильтрация (по умолчанию: 50)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
        help="Штраф за повторы / repetition_penalty (по умолчанию: 1.08)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Максимальное количество токенов ответа (по умолчанию: 64)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Системный промпт Ивана (по умолчанию берётся из переменной SYSTEM_PROMPT в .env)",
    )

    args = parser.parse_args()

    # Если передан флаг --system-prompt, сохраняем его в окружение
    if args.system_prompt:
        os.environ["SYSTEM_PROMPT"] = args.system_prompt
    elif not os.getenv("SYSTEM_PROMPT"):
        os.environ["SYSTEM_PROMPT"] = DEFAULT_SYSTEM_PROMPT

    active_system_prompt = os.getenv("SYSTEM_PROMPT")

    print("=" * 70)
    print("🤖 ЗАПУСК ЛОКАЛЬНОГО ТЕСТИРОВАНИЯ (QWEN 4-BIT + LORA)")
    print("=" * 70)
    print(f"Базовая модель:        {args.base_model}")
    print(f"Путь к адаптеру:       {args.adapter_path}")
    print(f"Параметры генерации:   temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, repetition_penalty={args.repetition_penalty}, max_new_tokens={args.max_new_tokens}")
    print(f"Системный промпт:      {active_system_prompt[:80]}...")
    print("-" * 70)

    try:
        model, tokenizer, loaded_adapter_path = load_model_and_tokenizer(
            base_model_id=args.base_model,
            adapter_path=args.adapter_path,
        )
    except Exception as e:
        logger.error("Ошибка при загрузке модели: %s", e)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✨ МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА В VRAM!")
    print(f"Используемый адаптер: {loaded_adapter_path}")
    print("=" * 70)
    print("Инструкции:")
    print(" • Пишите сообщения от лица девушки из Дайвинчика.")
    print(" • Для очистки истории переписки введите: /reset или reset")
    print(" • Для выхода введите: /exit или exit (или нажмите Ctrl+C)")
    print("-" * 70 + "\n")

    # Хранилище реплик диалога (system добавляется на 1 место в prepare_messages_for_chat_template)
    dialog_history: list[dict[str, str]] = []

    while True:
        try:
            user_text = input("Девушка: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nВыход из чата. До связи!")
            break

        if not user_text:
            continue

        if user_text.lower() in ("/exit", "exit", "quit", "/quit", "q"):
            print("Выход из чата. До связи!")
            break

        if user_text.lower() in ("/reset", "reset", "clear", "/clear"):
            dialog_history.clear()
            print("🧹 [История диалога очищена. Начинаем сначала!]\n")
            continue

        # Добавляем реплику собеседницы
        dialog_history.append({"role": "user", "content": user_text})

        # Ограничиваем историю диалога последними 12 репликами (скользящее окно)
        if len(dialog_history) > 12:
            dialog_history = dialog_history[-12:]

        # Генерация ответа
        start_t = time.time()
        try:
            reply = generate_reply(
                model=model,
                tokenizer=tokenizer,
                history=dialog_history,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - start_t
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}\n")
            continue

        # Очищаем ответ от лишних переносов строк
        reply = reply.replace("\n", " ").strip()

        # Добавляем реплику Ивана в историю диалога строго с ролью assistant
        dialog_history.append({"role": "assistant", "content": reply})

        print(f"Ваня:    {reply}  ({elapsed:.2f}с)\n")


if __name__ == "__main__":
    main()
