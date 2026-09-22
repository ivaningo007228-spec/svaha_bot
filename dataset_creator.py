"""
Dataset Creator для дообучения (Fine-Tuning) Qwen / LLM
Парсит экспорты чатов Telegram (папки ChatExport_...) из директории raw_chats/,
применяет строжайшую фильтрацию приватности, цензуру 18+/пошлости, удаление имён и старого возраста,
и формирует чистый обучающий датасет в формате JSONL (ChatML / OpenAI format).
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import unicodedata
import os
from dotenv import load_dotenv

# Поддержка UTF-8 вывода в консоли Windows
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
logger = logging.getLogger("dataset_creator")

BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

# Системный промпт по умолчанию (личность 17-летнего Ивана из Бийска)
DEFAULT_SYSTEM_PROMPT = (
    "Ты 17-летний Иван из Бийска. Твоя задача — вести живой, естественный диалог с девушкой, "
    "общаться легко, непринужденно, с юмором, в вежливом пацанском стиле, без канцеляризмов и длинных занудных монологов."
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ЧЕРНЫЙ СПИСОК 18+ И ИЗЛИШНЕЙ ПОШЛОСТИ (ЦЕНЗУРА ДАТАСЕТА)
# Если реплика (девушки или Ивана) содержит любое из этих совпадений — ВСЯ пара удаляется.
# ─────────────────────────────────────────────────────────────────────────────
VULGAR_STOP_PATTERNS = [
    # Секс, порно и интим
    r"\bсекс\w*",
    r"\bпорн\w*",
    r"\bинтим\w*",
    r"\bнюдс\w*",
    r"\bнюдес\w*",
    # Действия сексуального характера
    r"\b(?:по|пере|за|вы|от|на)?трах\w*",
    r"\b(?:по|на|от|вы)?дроч\w*",
    r"\bпереспа\w*",
    r"\bперепихон\w*",
    r"\b(?:от|по)?сос(?:ать|ешь|ет|ут|ал|ала|али|и|у)\b",
    r"\bотсос\w*",
    r"\bкуни\b|\bкуни[лс]?\w*",
    r"\bв\s+рот\b|\bв\s+ротик\b",
    r"\bкончи(?:л|ла|ть|шь|те)\s+(?:на|в|тебе|мне)\b",
    r"\bконча(?:ть|ю|ешь|ет|ют)\b",
    r"\bсперм\w*",
    # Анатомические интимные подробности
    r"\b(?!член\s+(?:семьи|совета|партии|клуба|кружка)\b)член\w*",
    r"\bвагин\w*",
    r"\bпенис\w*",
    r"\bклитор\w*",
    r"\bанал\w*",
    r"\bсиськ\w*|\bсисе[чк]\w*",
    r"\bтитьк\w*",
    r"\bсоск(?:и|ов|ами|ах)\b",
    r"\bстояк\b|\bэрекц\w*",
    r"\bдилдо\b|\bдилдак\w*|\bвибратор\w*",
    # Обнажение и нижнее белье в контексте интима
    r"\bголая\b|\bголый\b|\bголые\b|\bголышом\b|\bголяком\b",
    r"\bраздет(?:ая|ый|ые|а|о|ы)\b",
    r"\bбез\s+(?:одежды|лифчика|трусов|трусиков)\b",
    r"\bтрусики\b",
    r"\bлифчик\w*",
    # Возбуждение и грязь
    r"\bвозбужд\w*",
    r"\bшлюх\w*|\bшалав\w*|\bпроститут\w*|\bэскорт\w*",
]
VULGAR_REGEX = re.compile("|".join(VULGAR_STOP_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# 2. АНОНИМИЗАЦИЯ ИМЕН (ПРЕДВАРИТЕЛЬНО СКОМПИЛИРОВАННЫЕ РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ)
# ─────────────────────────────────────────────────────────────────────────────
# Звательные формы и короткие обращения ("Привет, Маш", "забей, Кать", "Даш, привет")
VOCATIVE_SHORT_NAMES = [
    "маш", "кать", "даш", "ань", "насть", "поль", "сонь", "лер", "алин",
    "вик", "юль", "лиз", "ксюш", "инн", "диан", "оль", "тань", "лен",
    "наташ", "рит", "ев", "крист", "ян", "влад", "ирин", "надюх", "любаш", "жень",
]

# Полные и уменьшительные женские имена
COMMON_FEMALE_NAMES = [
    "мария", "маша", "машка", "машенька", "машуля",
    "дарья", "даша", "дашка", "дашенька", "дашуля",
    "екатерина", "катя", "катька", "катюша",
    "анна", "аня", "анька", "анюта", "анечка", "аннушка",
    "анастасия", "настя", "настька", "настенька",
    "полина", "полинка", "полиночка",
    "софия", "софья", "соня", "сонька", "сонечка",
    "валерия", "лера", "лерка", "лерочка",
    "алина", "алинка", "алиночка",
    "виктория", "вика", "викуль", "викуля",
    "юлия", "юля", "юлька", "юлечка",
    "елизавета", "лиза", "лизка", "лизочка",
    "ксения", "ксюша", "ксю", "ксюшенька",
    "инна", "иннуля", "инночка",
    "кристина", "кристя", "кристинка",
    "диана", "дианка", "вероника", "никуля",
    "владислава", "влада", "владочка",
    "яна", "янка",
    "ольга", "оля", "олька", "олечка",
    "татьяна", "таня", "танька", "танечка",
    "светлана", "света", "светка",
    "марина", "маринка",
    "елена", "лена", "ленка",
    "наталья", "наталия", "наташа", "наташка",
    "маргарита", "рита", "ритка",
    "мирослава", "мира",
    "ева", "евочка",
    "арина", "аринка",
    "ульяна", "ульянка",
    "кира", "кирка",
    "александра", "саша", "сашка",
    "ирина", "ира", "ирка",
    "надежда", "надя",
    "любовь", "люба", "любочка",
    "евгения", "женя",
    "милана", "мила", "ангелина", "лина",
    "камилла", "камила", "алиса", "варвара", "варя",
    "злата", "алёна", "алена", "карина", "виолетта", "элина", "снежана", "ярослава",
]

STOP_STEMS = {"мир", "сон", "мат", "час", "вид", "род", "дом", "друг"}

# Предкомпиляция быстрых регулярных выражений для имен
_voc_pat = "|".join(sorted(set(VOCATIVE_SHORT_NAMES), key=len, reverse=True))
_fem_pat = "|".join(sorted(set(COMMON_FEMALE_NAMES), key=len, reverse=True))
_fem_cap_pat = "|".join(sorted(set(n.capitalize() for n in COMMON_FEMALE_NAMES), key=len, reverse=True))

RE_VOC_END = re.compile(rf"(?i),\s*(?:{_voc_pat})(?=[!?.:;,\s]|$)")
RE_VOC_START = re.compile(rf"(?i)(?:^|[.!?]\s*)\b(?:{_voc_pat})\s*(?:,\s*|[!?:;]+\s*)")
RE_VOC_STANDALONE = re.compile(rf"(?i)\b(?:{_voc_pat})\b")

RE_FEM_COMMA = re.compile(rf"(?i),\s*(?:{_fem_pat})\s*,")
RE_FEM_END = re.compile(rf"(?i),\s*(?:{_fem_pat})(?=[!?.:;,\s]|$)")
RE_FEM_START = re.compile(rf"(?i)(?:^|[.!?]\s*)\b(?:{_fem_pat})\s*(?:,\s*|[!?:;]+\s*)")
RE_FEM_CAP = re.compile(rf"\b(?:{_fem_cap_pat})\b")

# Предкомпиляция выражений для санитаризации возраста
RE_EXACT_AGE = re.compile(r"^(1[456]|пятнадцать|шестнадцать)$", re.IGNORECASE)
AGE_REPLACE_RULES = [
    (re.compile(r"(?i)\b,\s*(?:1[456]|пятнадцать|шестнадцать)\s*,\s*Бийск\b"), ", 17, Бийск"),
    (re.compile(r"(?i)\bмне\s+(?:было\s+|будет\s+)?(1[456])\s*лет\b"), "мне 17 лет"),
    (re.compile(r"(?i)\bмне\s+(?:было\s+|будет\s+)?(1[456])\b"), "мне 17"),
    (re.compile(r"(?i)\bмне\s+(?:было\s+|будет\s+)?(пятнадцать|шестнадцать)\s*лет\b"), "мне семнадцать лет"),
    (re.compile(r"(?i)\bмне\s+(?:было\s+|будет\s+)?(пятнадцать|шестнадцать)\b"), "мне семнадцать"),
    (re.compile(r"(?i)\b(1[456])\s*лет\b"), "17 лет"),
    (re.compile(r"(?i)\b(пятнадцать|шестнадцать)\s*лет\b"), "семнадцать лет"),
    (re.compile(r"(?i)\bв\s+свои\s+(?:1[3456]|13-14|14-15|15-16)\s*лет\b"), "в свои 17 лет"),
    (re.compile(r"(?i)\bучусь\s+в\s+[891]0?\s+классе\b"), "учусь в 11 классе"),
    (re.compile(r"(?i)\bв\s+[891]0?\s+класс\b"), "в 11 класс"),
]


def extract_chat_name_variants(chat_name: str) -> list[str]:
    """Генерирует возможные формы имени девушки на основе названия чата."""
    norm = unicodedata.normalize("NFKD", chat_name)
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", norm)
    variants = set()

    for w in words:
        if len(w) < 3:
            continue
        variants.add(w)
        w_lower = w.lower()

        # Суффиксы -ка / -ку
        if w_lower.endswith(("ка", "ку", "ке", "кой", "ки")):
            stem = w[:-2]
            for end in ("ка", "ку", "ке", "кой", "ки", "а", "у", "е", "ей", "ой", ""):
                if len(stem + end) >= 3 and stem.lower() not in STOP_STEMS:
                    variants.add(stem + end)
        # Суффиксы -ина / -ова
        elif w_lower.endswith(("ина", "иной", "ину", "ине", "ова", "овой", "ову", "ове")):
            stem = w[:-3]
            for end in ("ина", "иной", "ину", "ине", "ова", "овой", "ову", "ове"):
                variants.add(stem + end)
        # Окончания -а / -я
        elif w_lower.endswith(("а", "я")):
            stem = w[:-1]
            if len(stem) >= 3 and stem.lower() not in STOP_STEMS:
                for end in ("а", "е", "у", "ой", "ей", "ы", "и", ""):
                    variants.add(stem + end)

    return sorted(variants, key=len, reverse=True)


def remove_names_from_assistant(text: str, chat_names: list[str] = None) -> tuple[str, bool]:
    """
    Автоматически вырезает женские имена из ответов ассистента:
    'Привет, Маш' -> 'Привет'
    'забей, Кать' -> 'забей'
    'Даш, привет' -> 'Привет'
    'Слушай, Кать, а пошли' -> 'Слушай, а пошли'
    """
    res = text
    orig = text

    # Защищаем сеть магазинов 'Мария-Ра' / 'Мария Ра' от вырезания имени Мария
    res = re.sub(r"(?i)\bмария[\s-]ра\b", "__MARIA_RA__", res)

    # 1. Звательные формы (быстрая замена через предкомпилированные regex)
    res = RE_VOC_END.sub("", res)
    res = RE_VOC_START.sub("", res)
    res = RE_VOC_STANDALONE.sub("", res)

    # 2. Общие женские имена
    res = RE_FEM_COMMA.sub(",", res)
    res = RE_FEM_END.sub("", res)
    res = RE_FEM_START.sub("", res)
    res = RE_FEM_CAP.sub("", res)

    # 3. Дополнительные имена из названия конкретного чата (если заданы)
    if chat_names:
        for cn in chat_names:
            pat = re.escape(cn)
            res = re.sub(rf"(?i),\s*{pat}(?=[!?.:;,\s]|$)", "", res)
            res = re.sub(rf"(?i)(?:^|[.!?]\s*)\b{pat}\s*(?:,\s*|[!?:;]+\s*)", "", res)
            res = re.sub(rf"(?i)\b{pat}\b", "", res)

    # Восстанавливаем Мария-Ра
    res = res.replace("__MARIA_RA__", "Мария-Ра")

    modified = (res != orig)

    if modified:
        # Аккуратная подчистка знаков препинания и пробелов
        res = re.sub(r"[ \t]+", " ", res)
        res = re.sub(r"^[,\s\-:]+", "", res)
        res = re.sub(r"\s+([,!?\.])", r"\1", res)
        res = re.sub(r",\s*,+", ",", res)
        res = res.strip()
        if res and res[0].islower():
            res = res[0].upper() + res[1:]

    return res, modified


# ─────────────────────────────────────────────────────────────────────────────
# 3. САНИТИЗАЦИЯ СТАРОГО ВОЗРАСТА (ИЗМЕНЕНИЕ 14/15/16 ЛЕТ НА 17)
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_age_in_assistant(text: str) -> tuple[str, bool]:
    """
    Заменяет упоминания старого возраста (15 лет, 16, пятнадцать) на 17 лет,
    а также 9/10 класс на 11 класс, чтобы модель точно знала, что ей 17 лет.
    """
    res = text
    orig = text

    # Точный ответ одним числом или словом: '15' -> '17'
    if RE_EXACT_AGE.match(res.strip()):
        return "17", True

    for pattern, repl in AGE_REPLACE_RULES:
        if pattern.search(res):
            res = pattern.sub(repl, res)

    return res, (res != orig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ИЗВЛЕЧЕНИЕ И ОЧИСТКА ТЕКСТА СООБЩЕНИЙ
# ─────────────────────────────────────────────────────────────────────────────
def clean_message_text(text_field) -> str:
    """
    Извлекает чистый текст сообщения из JSON экспорта Telegram,
    удаляет ссылки, медиа-плейсхолдеры и нормализует пробелы.
    """
    if isinstance(text_field, str):
        raw = text_field
    elif isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("link", "text_link", "url"):
                    continue
                parts.append(item.get("text", ""))
        raw = "".join(parts)
    else:
        return ""

    # 1. Удаление ссылок (http, https, www, t.me, tg://)
    raw = re.sub(r"https?://\S+|www\.\S+|t\.me/\S+|tg://\S+", "", raw)

    # 2. Удаление медиа-строк экспорта Telegram Desktop
    if "(File not included" in raw:
        return ""

    # 3. Нормализация пробелов и переносов строк
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
    cleaned = "\n".join(l for l in lines if l)

    # 4. Игнорируем пустые или состоящие только из знаков препинания сообщения
    if not cleaned or not re.search(r"[A-Za-zА-Яа-яЁё0-9]", cleaned):
        return ""

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 5. ОБРАБОТКА ОДНОГО ФАЙЛА RESULT.JSON
# ─────────────────────────────────────────────────────────────────────────────
def process_chat_export_file(
    file_path: Path,
    system_prompt: str,
    session_gap_hours: float = 3.0,
) -> tuple[list[dict], dict]:
    """
    Обрабатывает один файл result.json:
    - фильтрует медиа, системные звонки и служебные сообщения
    - склеивает реплики
    - отсеивает пошлость и 18+ (удаляя всю пару реплик)
    - анонимизирует имена и старый возраст в ответах ассистента
    """
    stats = {
        "chat_id": None,
        "chat_name": "",
        "total_messages": 0,
        "valid_text_messages": 0,
        "vulgar_pairs_filtered": 0,
        "names_cleaned": 0,
        "age_sanitized": 0,
        "clean_pairs_generated": 0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Не удалось прочитать файл %s: %s", file_path, exc)
        return [], stats

    chat_name = str(data.get("name") or "Собеседница").strip()
    chat_id = data.get("id")
    stats["chat_id"] = chat_id
    stats["chat_name"] = chat_name

    messages = data.get("messages", [])
    stats["total_messages"] = len(messages)

    # Игнорируем ботов или пустые системные чаты
    if not messages or chat_name.lower() in ("leomatchbot", "telegram", "saved messages"):
        return [], stats

    chat_name_variants = extract_chat_name_variants(chat_name)

    raw_dialog_entries: list[tuple[str, str, int]] = []

    for msg in messages:
        # Игнорируем системные сообщения, звонки и сервисные действия
        if msg.get("type") != "message":
            continue
        if msg.get("action"):
            continue

        text = clean_message_text(msg.get("text"))
        if not text:
            continue

        # Игнорируем технические команды бота
        if text.lower() in ("!старт", "!стоп", "бот офф", "bot off", "бот онн", "bot on") or text.startswith("!"):
            continue

        from_id = str(msg.get("from_id") or "")
        from_name = str(msg.get("from") or "")

        # Определение роли:
        # Если отправитель равен chat_id или имени чата — это девушка (user)
        # Иначе — это аккаунт автора (assistant)
        if (chat_id is not None and from_id == f"user{chat_id}") or from_name == chat_name:
            role = "user"
        else:
            role = "assistant"

        ts = int(msg.get("date_unixtime") or 0)
        raw_dialog_entries.append((role, text, ts))

    stats["valid_text_messages"] = len(raw_dialog_entries)
    if not raw_dialog_entries:
        return [], stats

    # 1. Объединение подряд идущих сообщений от одного отправителя
    session_gap_seconds = int(session_gap_hours * 3600)
    merged_turns: list[tuple[str, str, int]] = []

    for role, text, ts in raw_dialog_entries:
        if merged_turns and merged_turns[-1][0] == role:
            prev_role, prev_text, prev_ts = merged_turns[-1]
            if ts - prev_ts < session_gap_seconds:
                merged_turns[-1] = (role, prev_text + "\n" + text, ts)
            else:
                merged_turns.append((role, text, ts))
        else:
            merged_turns.append((role, text, ts))

    dataset_samples: list[dict] = []

    # 2. Формирование пар (user -> assistant) с цензурой и анонимизацией
    for i in range(len(merged_turns) - 1):
        curr_role, curr_text, curr_ts = merged_turns[i]
        next_role, next_text, next_ts = merged_turns[i + 1]

        if curr_role == "user" and next_role == "assistant":
            # Ответ должен быть дан в пределах разумного времени беседы (до 48 часов)
            if next_ts - curr_ts > 48 * 3600:
                continue

            # ── ФИЛЬТРАЦИЯ 18+ И ИЗЛИШНЕЙ ПОШЛОСТИ ──
            # Если любая из реплик содержит пошлость — удаляем ВСЮ пару целиком!
            if VULGAR_REGEX.search(curr_text) or VULGAR_REGEX.search(next_text):
                stats["vulgar_pairs_filtered"] += 1
                continue

            # ── АНОНИМИЗАЦИЯ ИМЕН В ОТВЕТЕ ASSISTANT ──
            cleaned_asst, name_modified = remove_names_from_assistant(next_text, chat_name_variants)
            if name_modified:
                stats["names_cleaned"] += 1

            # ── САНИТИЗАЦИЯ ВОЗРАСТА В ОТВЕТЕ ASSISTANT ──
            cleaned_asst, age_modified = sanitize_age_in_assistant(cleaned_asst)
            if age_modified:
                stats["age_sanitized"] += 1

            # Если после очистки ответ стал пустым — пропускаем
            if not cleaned_asst.strip():
                continue

            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": curr_text},
                    {"role": "assistant", "content": cleaned_asst},
                ]
            }
            dataset_samples.append(sample)

    stats["clean_pairs_generated"] = len(dataset_samples)
    return dataset_samples, stats


# ─────────────────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Создание чистого JSONL-датасета для Fine-Tuning Qwen из Telegram-экспортов."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=str(BASE_DIR / "raw_chats"),
        help="Путь к директории с экспортами чатов (по умолчанию: raw_chats/ в корне проекта)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=str(BASE_DIR / "my_training_dataset.jsonl"),
        help="Путь к выходному файлу JSONL (по умолчанию: my_training_dataset.jsonl в корне)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Системный промпт (по умолчанию: 'Ты 17-летний Иван из Бийска...')",
    )
    parser.add_argument(
        "--gap-hours",
        type=float,
        default=3.0,
        help="Интервал в часах между репликами для разделения диалоговых сессий (по умолчанию: 3.0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_file).resolve()
    system_prompt = args.system_prompt.strip()

    print("=" * 70, flush=True)
    print("🚀 Генератор датасета Qwen: Telegram Chats -> my_training_dataset.jsonl", flush=True)
    print("=" * 70, flush=True)
    print(f"📁 Директория сырых чатов: {input_path}", flush=True)
    print(f"💾 Итоговый файл датасета: {output_path}", flush=True)
    print(f"📝 Системный промпт: {system_prompt[:75]}...", flush=True)
    print("-" * 70, flush=True)

    if not input_path.exists():
        logger.error("Папка '%s' не найдена!", input_path)
        sys.exit(1)

    export_files = sorted(input_path.glob("**/result.json"))
    if not export_files:
        logger.error("В папке '%s' не найдено ни одного файла result.json!", input_path)
        print("\nПожалуйста, поместите папки экспорта ChatExport_... в директорию raw_chats/.", flush=True)
        sys.exit(1)

    print(f"Найдено файлов экспорта result.json: {len(export_files)}\n", flush=True)

    all_samples: list[dict] = []
    total_raw_msgs = 0
    total_valid_msgs = 0
    total_vulgar_filtered = 0
    total_names_cleaned = 0
    total_age_sanitized = 0
    seen_chat_ids: set[int] = set()

    for idx, fpath in enumerate(export_files, 1):
        samples, stats = process_chat_export_file(
            file_path=fpath,
            system_prompt=system_prompt,
            session_gap_hours=args.gap_hours,
        )

        cid = stats.get("chat_id")
        if cid and cid in seen_chat_ids:
            continue
        if cid:
            seen_chat_ids.add(cid)

        all_samples.extend(samples)
        total_raw_msgs += stats["total_messages"]
        total_valid_msgs += stats["valid_text_messages"]
        total_vulgar_filtered += stats["vulgar_pairs_filtered"]
        total_names_cleaned += stats["names_cleaned"]
        total_age_sanitized += stats["age_sanitized"]

        folder_name = fpath.parent.name
        print(
            f"[{idx:02d}/{len(export_files):02d}] Чат '{stats['chat_name']}' ({folder_name}): "
            f"{stats['total_messages']} сообщ. -> {stats['clean_pairs_generated']} чистых пар "
            f"| 18+ удалено: {stats['vulgar_pairs_filtered']} "
            f"| Имен очищено: {stats['names_cleaned']}",
            flush=True,
        )

    if not all_samples:
        logger.warning("Не удалось сформировать ни одного обучающего примера!")
        sys.exit(0)

    # Построчная запись в итоговый JSONL файл
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for item in all_samples:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70, flush=True)
    print("🎉 ДАТАСЕТ УСПЕШНО СОБРАН И ОЧИЩЕН!", flush=True)
    print("=" * 70, flush=True)
    print(f"📊 Всего уникальных чатов обработано: {len(seen_chat_ids)}", flush=True)
    print(f"📨 Всего сырых сообщений Telegram просканировано: {total_raw_msgs:,}", flush=True)
    print(f"💬 Извлечено текстовых реплик: {total_valid_msgs:,}", flush=True)
    print(f"🚫 ОТФИЛЬТРОВАНО И УДАЛЕНО ПОШЛЫХ/18+ ПАР РЕПЛИК: {total_vulgar_filtered:,}", flush=True)
    print(f"🛡️  Вырезано упоминаний женских имен из ответов: {total_names_cleaned:,}", flush=True)
    print(f"🎂 Исправлено упоминаний старого возраста на 17 лет: {total_age_sanitized:,}", flush=True)
    print(f"🎯 ИТОГО ЧИСТЫХ ОБУЧАЮЩИХ ПРИМЕРОВ (строк JSONL): {len(all_samples):,}", flush=True)
    print(f"💾 Итоговый файл датасета: {output_path} ({file_size_mb:.2f} МБ)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
