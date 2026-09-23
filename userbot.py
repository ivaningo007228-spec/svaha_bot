"""
Telegram Userbot — Мультиаккаунтная CRM-автоматизация поддержки клиентов
Pyrogram + локальная Ollama (vanya_q5) + openpyxl + Умный автолайкер Дайвинчика

Зависимости: pyrogram, tgcrypto, openpyxl, ollama, python-dotenv
"""

import asyncio
from datetime import datetime, time as dtime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid
from ollama import AsyncClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# Поддержка UTF-8 вывода в консоли Windows
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import openpyxl
from openpyxl import load_workbook
from dotenv import load_dotenv
from pyrogram import Client, enums, filters, idle, raw
from pyrogram.types import Message

# ─────────────────────────────────────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("userbot")
logger = log

# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация и директории
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR: Path = Path(__file__).parent.resolve()
SESSIONS_DIR: Path = BASE_DIR / "sessions"
DATABASES_DIR: Path = BASE_DIR / "databases"
LOGS_CHATS_DIR: Path = BASE_DIR / "logs_chats"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
DATABASES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_CHATS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Список аккаунтов (уникальные имена/номера)
_raw_accounts = os.getenv("TELEGRAM_ACCOUNTS") or os.getenv("ACCOUNTS") or "acc_main,acc_second"
ACCOUNTS: list[str] = [acc.strip() for acc in _raw_accounts.split(",") if acc.strip()]
if not ACCOUNTS:
    ACCOUNTS = ["acc_main", "acc_second"]

# Чтение API_ID и API_HASH строго из .env
_raw_api_id = (os.getenv("API_ID") or "").strip().strip('"').strip("'")
_raw_api_hash = (os.getenv("API_HASH") or "").strip().strip('"').strip("'")

if not _raw_api_id or not _raw_api_hash:
    raise ValueError(
        "Критическая ошибка: переменные API_ID и API_HASH не найдены или пусты в файле .env!\n"
        "Пожалуйста, укажите в вашем .env:\n"
        "API_ID=ваш_api_id\n"
        "API_HASH=ваш_api_hash"
    )

try:
    API_ID: int = int(_raw_api_id)
except ValueError:
    raise ValueError(f"Критическая ошибка: API_ID в файле .env должен быть целым числом, получено: '{_raw_api_id}'")

API_HASH: str = _raw_api_hash

DEVICE_MODEL = "Desktop"
SYSTEM_VERSION = "Windows 10"
APP_VERSION = "5.10.0"  # Актуальная версия tdesktop
LANG_CODE = "ru"
SYSTEM_LANG_CODE = "ru-RU"

# Патч Client.__init__ и InitConnection, чтобы Pyrogram принимал system_lang_code="ru-RU"
# и передавал точные метаданные Telegram Desktop для Windows в MTProto
_orig_client_init = Client.__init__


def _patched_client_init(self, *args, system_lang_code: str = SYSTEM_LANG_CODE, **kwargs):
    _orig_client_init(self, *args, **kwargs)
    self.system_lang_code = system_lang_code


Client.__init__ = _patched_client_init

_orig_init_connection = raw.functions.InitConnection.__init__


def _patched_init_connection(self, *args, **kwargs):
    if kwargs.get("system_lang_code") in ("ru", None, ""):
        kwargs["system_lang_code"] = SYSTEM_LANG_CODE
    _orig_init_connection(self, *args, **kwargs)


raw.functions.InitConnection.__init__ = _patched_init_connection

# Патч Client.handle_updates для отслеживания и перехвата raw.types.UpdatesTooLong
_orig_handle_updates = Client.handle_updates

# Набор известных ID лид-бота Дайвинчика (@leomatchbot)
_raw_leobot_ids = os.getenv("KNOWN_LEOMATCHBOT_IDS", "1234060895,602220152,5029377485")
_KNOWN_LEOMATCHBOT_IDS: set[int] = {
    int(x.strip()) for x in _raw_leobot_ids.split(",") if x.strip().lstrip("-").isdigit()
}

# ── Динамический белый список чатов (в памяти и на диске) ─────────────────────
ENV_FILE: Path = BASE_DIR / ".env"
WHITELIST_JSON_FILE: Path = DATABASES_DIR / "whitelist.json"
GLOBAL_WHITELIST_IDS: set[int] = set()


def _sync_update_env_whitelist(whitelist_ids: set[int]) -> None:
    """Обновить переменную WHITELIST_CHAT_IDS в файле .env для сохранения между перезапусками."""
    if not ENV_FILE.exists():
        return
    try:
        content = ENV_FILE.read_text(encoding="utf-8")
        id_str = ",".join(str(i) for i in sorted(whitelist_ids) if i > 0)
        pattern = r"^(WHITELIST_CHAT_IDS\s*=).*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            new_content = re.sub(pattern, f"WHITELIST_CHAT_IDS={id_str}", content, flags=re.MULTILINE)
        else:
            new_content = content.rstrip() + f"\n\n# Белый список чатов\nWHITELIST_CHAT_IDS={id_str}\n"
        ENV_FILE.write_text(new_content, encoding="utf-8")
        log.debug("[WHITELIST] Переменная WHITELIST_CHAT_IDS в .env успешно обновлена (%d ID)", len(whitelist_ids))
    except Exception as exc:
        log.error("[WHITELIST] Ошибка обновления .env файла: %s", exc)


def _sync_save_whitelist_json(filepath: Path, chat_ids: set[int], metadata: dict[str, dict]) -> None:
    """Сохранить белый список в JSON-файл конфигурации."""
    try:
        data = {
            "chat_ids": sorted(list(chat_ids)),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chats": metadata,
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.debug("[WHITELIST] Белый список сохранен в %s", filepath.name)
    except Exception as exc:
        log.error("[WHITELIST] Ошибка сохранения JSON %s: %s", filepath, exc)


def _sync_load_whitelist_json(filepath: Path) -> tuple[set[int], dict[str, dict]]:
    """Загрузить белый список из JSON-файла конфигурации."""
    if not filepath.exists():
        return set(), {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chat_ids = set()
        for x in data.get("chat_ids", []):
            try:
                chat_ids.add(int(x))
            except (ValueError, TypeError):
                pass
        chats = data.get("chats", {})
        return chat_ids, chats
    except Exception as exc:
        log.error("[WHITELIST] Ошибка загрузки JSON %s: %s", filepath, exc)
        return set(), {}


def _is_allowed_dialog(client: Client, user_id: int | None, username: str | None = None) -> bool:
    """
    Проверка, относится ли ID/username к разрешенным личным диалогам:
    1. Официальный бот Дайвинчика (@leomatchbot) — всегда валиден.
    2. Собственный аккаунт (Saved Messages / исходящие команды) — всегда валиден.
    3. Клиенты/девушки из актуального белого списка (GLOBAL_WHITELIST_IDS и active_chats).
    4. Входящие личные сообщения (Private Chats с пользователями user_id > 0).
    Любые ID суперканалов (начинающиеся на -100) и групп (< 0) немедленно отсекаются.
    """
    if user_id is None and not username:
        return False

    # Любые ID суперканалов (-100...) и групп (< 0) категорически запрещены
    if user_id is not None:
        if user_id < 0 or str(user_id).startswith("-100"):
            return False

    # 1. Проверка на официального бота Дайвинчика (@leomatchbot) — всегда валиден!
    if username and username.lower().lstrip("@") == "leomatchbot":
        if user_id:
            _KNOWN_LEOMATCHBOT_IDS.add(user_id)
            GLOBAL_WHITELIST_IDS.add(user_id)
        return True

    if user_id is not None and user_id in _KNOWN_LEOMATCHBOT_IDS:
        return True

    # 2. Проверка на свой собственный аккаунт
    me = getattr(client, "me", None)
    if me and getattr(me, "id", None) == user_id:
        return True

    account_bot = getattr(client, "_account_bot", None)
    if account_bot:
        if account_bot.my_id and user_id == account_bot.my_id:
            return True

        if user_id is not None:
            if user_id in account_bot.active_chats or str(user_id) in account_bot.active_chats:
                return True

        if username:
            clean_uname = username.lower().lstrip("@")
            for key in account_bot.active_chats:
                if isinstance(key, str) and key.lower().lstrip("@") == clean_uname:
                    return True

    # 3. Проверка в актуальном глобальном set белого списка в памяти
    if user_id is not None and user_id in GLOBAL_WHITELIST_IDS:
        return True

    # 4. Любые личные сообщения между пользователями (положительный user_id > 0)
    # разрешаются в диспетчер, чтобы новые люди после мэтча могли написать и быть добавлены в белый список
    if user_id is not None and user_id > 0:
        return True

    return False


def _is_channel_or_group_update(u) -> bool:
    """
    Проверяет, является ли обновление системным событием канала, супергруппы или группового чата.
    """
    type_name = type(u).__name__

    # 1. Любые обновления с Channel или Chat в имени (UpdateNewChannelMessage, UpdateChat... и т.д.)
    if "Channel" in type_name or "Chat" in type_name:
        return True

    # 2. Прямые атрибуты channel_id или chat_id
    if getattr(u, "channel_id", None) is not None:
        return True
    if getattr(u, "chat_id", None) is not None:
        return True

    # 3. Проверка сообщения внутри обновления
    msg = getattr(u, "message", None)
    if msg is not None:
        peer_id = getattr(msg, "peer_id", None)
        if isinstance(peer_id, (raw.types.PeerChannel, raw.types.PeerChat)):
            return True
        if getattr(peer_id, "channel_id", None) is not None or getattr(peer_id, "chat_id", None) is not None:
            return True

        from_id = getattr(msg, "from_id", None)
        if isinstance(from_id, (raw.types.PeerChannel, raw.types.PeerChat)):
            return True
        if getattr(from_id, "channel_id", None) is not None or getattr(from_id, "chat_id", None) is not None:
            return True

    # 4. Проверка атрибута peer (например, UpdateReadHistoryInbox)
    peer = getattr(u, "peer", None)
    if peer is not None:
        if isinstance(peer, (raw.types.PeerChannel, raw.types.PeerChat)):
            return True
        if getattr(peer, "channel_id", None) is not None or getattr(peer, "chat_id", None) is not None:
            return True

    return False


def _extract_user_ids_from_update(u) -> list[int]:
    """Извлекает ID пользователей из сырого обновления личного диалога."""
    user_ids = []

    uid = getattr(u, "user_id", None)
    if isinstance(uid, int):
        user_ids.append(uid)

    peer = getattr(u, "peer", None)
    if isinstance(peer, raw.types.PeerUser):
        user_ids.append(peer.user_id)

    msg = getattr(u, "message", None)
    if msg is not None:
        p = getattr(msg, "peer_id", None)
        if isinstance(p, raw.types.PeerUser):
            user_ids.append(p.user_id)
        f = getattr(msg, "from_id", None)
        if isinstance(f, raw.types.PeerUser):
            user_ids.append(f.user_id)

    return user_ids


def _is_update_from_allowed_dialog(client: Client, u) -> bool:
    """Проверяет, относится ли обновление к разрешенному личному диалогу."""
    user_ids = _extract_user_ids_from_update(u)
    if not user_ids:
        # Безличные технические обновления без peers (например, UpdateDeleteMessages, UpdateConfig)
        return True
    return any(_is_allowed_dialog(client, uid) for uid in user_ids)


async def _patched_handle_updates(self, updates):
    """
    Патч Client.handle_updates с жесткой фильтрацией:
    1. Игнорирует любые каналы, супергруппы (-100...) и групповые чаты.
    2. Пропускает СТРОГО личные сообщения от @leomatchbot, белого списка чатов и владельца.
    3. При возникновении ValueError (например, Peer id invalid) или KeyError тихо делает return.
    """
    try:
        # 1. UpdatesTooLong — уведомление о переполнении очереди
        if isinstance(updates, raw.types.UpdatesTooLong):
            log.warning(
                "[DISPATCHER] ВНИМАНИЕ: Получено событие raw.types.UpdatesTooLong! "
                "Telegram сообщил о переполнении очереди обновлений."
            )
            return

        # 2. Групповые чаты (UpdateShortChatMessage) — отсекаем мгновенно
        if isinstance(updates, raw.types.UpdateShortChatMessage):
            return

        # 3. Личное короткое сообщение (UpdateShortMessage)
        if isinstance(updates, raw.types.UpdateShortMessage):
            if not _is_allowed_dialog(self, getattr(updates, "user_id", None)):
                return
            await _orig_handle_updates(self, updates)
            return

        # 4. Короткое обновление (UpdateShort)
        if isinstance(updates, raw.types.UpdateShort):
            u = getattr(updates, "update", None)
            if u is None:
                return
            if _is_channel_or_group_update(u):
                return
            if not _is_update_from_allowed_dialog(self, u):
                return
            await _orig_handle_updates(self, updates)
            return

        # 5. Пакет обновлений (Updates или UpdatesCombined)
        if isinstance(updates, (raw.types.Updates, raw.types.UpdatesCombined)):
            # Регистрируем leomatchbot и сопоставляем пользователей из updates.users с белым списком
            users_list = getattr(updates, "users", []) or []
            for u in users_list:
                uid = getattr(u, "id", None)
                uname = getattr(u, "username", None)
                if uname and uname.lower() == "leomatchbot" and uid:
                    _KNOWN_LEOMATCHBOT_IDS.add(uid)
                    GLOBAL_WHITELIST_IDS.add(uid)
                if uid and uname:
                    account_bot = getattr(self, "_account_bot", None)
                    if account_bot:
                        clean_u = uname.lower().lstrip("@")
                        for k in list(account_bot.active_chats.keys()):
                            if isinstance(k, str) and k.lower().lstrip("@") == clean_u:
                                account_bot.active_chats[uid] = account_bot.active_chats[k]
                                account_bot.active_chats[str(uid)] = account_bot.active_chats[k]
                                GLOBAL_WHITELIST_IDS.add(uid)

            # Фильтруем updates.updates: убираем любые каналы, супергруппы и чужие диалоги
            raw_updates = getattr(updates, "updates", []) or []
            valid_updates = []
            for u in raw_updates:
                if _is_channel_or_group_update(u):
                    continue
                if _is_update_from_allowed_dialog(self, u):
                    valid_updates.append(u)

            # Если после фильтрации не осталось целевых обновлений — выходим
            if not valid_updates:
                return

            updates.updates = valid_updates
            # Очищаем список chats, чтобы Pyrogram не делал fetch_peers/resolve_peer для каналов
            updates.chats = []
            await _orig_handle_updates(self, updates)
            return

        # 6. Для остальных неизвестных типов событий вызываем базовый обработчик
        await _orig_handle_updates(self, updates)

    except (ValueError, KeyError):
        return
    except Exception as exc:
        log.debug("[DISPATCHER] Игнорируем ошибку при обработке обновлений: %s", exc)
        return


Client.handle_updates = _patched_handle_updates

# ── Настройки ИИ (локальная Ollama, без загрузки весов в процесс бота) ──────
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "vanya_q5")
# Штраф выше ~1.1 заставляет Qwen бросать кириллицу и срываться в иероглифы.
OLLAMA_TEMPERATURE: float = 0.45
OLLAMA_REPEAT_PENALTY: float = 1.08


def _ollama_message_text(response: object) -> str:
    """Достаёт текст ответа Ollama и схлопывает переносы, как раньше делал декодер."""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content") or ""
    else:
        content = getattr(message, "content", "") or ""
    return str(content).replace("\n", " ").strip()

# Склейка мыслей собеседника: общие словари модуля
pending_messages: dict[int, list[str]] = {}
debouncer_tasks: dict[int, asyncio.Task] = {}

# ── Векторная память диалогов (Qdrant + MiniLM на CPU) ─────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
QDRANT_COLLECTION: str = "vanya_memories"
QDRANT_VECTOR_SIZE: int = 384

qdrant_client: AsyncQdrantClient | None = None
embed_model: SentenceTransformer | None = None
_memory_ready: bool = False
_memory_lock: asyncio.Lock | None = None


def _coerce_chat_id(chat_id: int | str) -> int:
    """Приводит chat_id к int, чтобы фильтр Qdrant совпадал с payload."""
    text = str(chat_id).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    raise ValueError(f"chat_id не является числом: {chat_id}")


def _embed_text_sync(text: str) -> list[float]:
    """Синхронный эмбеддинг на CPU. Вызывается через asyncio.to_thread."""
    if embed_model is None:
        raise RuntimeError("Модель эмбеддингов не загружена")
    vector = embed_model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()


async def init_vector_memory() -> None:
    """
    Один раз при старте юзербота:
    - поднимает AsyncQdrantClient к localhost:6333;
    - создаёт коллекцию vanya_memories (384, Cosine), если её ещё нет;
    - грузит SentenceTransformer all-MiniLM-L6-v2 строго на CPU.
    """
    global qdrant_client, embed_model, _memory_ready, _memory_lock

    if _memory_lock is None:
        _memory_lock = asyncio.Lock()

    async with _memory_lock:
        if _memory_ready:
            return

        log.info(
            "[QDRANT] Подключение к %s:%s, коллекция '%s'...",
            QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
        )
        qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        if not await qdrant_client.collection_exists(QDRANT_COLLECTION):
            await qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=QDRANT_VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            log.info("[QDRANT] Коллекция '%s' создана (size=384, distance=Cosine)", QDRANT_COLLECTION)
        else:
            log.info("[QDRANT] Коллекция '%s' уже существует", QDRANT_COLLECTION)

        try:
            await qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="chat_id",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
        except Exception as index_exc:
            log.debug("[QDRANT] Индекс chat_id уже есть или не создан: %s", index_exc)

        log.info("[QDRANT] Загрузка эмбеддера %s на CPU...", EMBEDDING_MODEL_NAME)
        embed_model = await asyncio.to_thread(
            SentenceTransformer, EMBEDDING_MODEL_NAME, device="cpu"
        )
        _memory_ready = True
        log.info("[QDRANT] Векторная память готова (эмбеддер на CPU, VRAM не занята)")


async def close_vector_memory() -> None:
    """Закрыть HTTP-клиент Qdrant при остановке юзербота."""
    global qdrant_client, _memory_ready
    if qdrant_client is not None:
        try:
            await qdrant_client.close()
        except Exception as exc:
            log.debug("[QDRANT] Ошибка закрытия клиента: %s", exc)
        qdrant_client = None
    _memory_ready = False


async def recall_dialog_memories(chat_id: int | str, user_text: str, limit: int = 3) -> str:
    """
    RAG: векторизует новое сообщение девушки и достаёт 2–3 похожие прошлые пары
    строго из этого chat_id.
    """
    cleaned = (user_text or "").strip()
    if not _memory_ready or qdrant_client is None or not cleaned:
        return ""

    try:
        numeric_chat_id = _coerce_chat_id(chat_id)
    except ValueError:
        return ""

    try:
        vector = await asyncio.to_thread(_embed_text_sync, cleaned)
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="chat_id",
                    match=qmodels.MatchValue(value=numeric_chat_id),
                )
            ]
        )
        # qdrant-client 1.19 убрал search(); query_points — тот же nearest-поиск по вектору.
        response = await qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        log.warning("[QDRANT] Поиск памяти для чата %s не удался: %s", chat_id, exc)
        return ""

    lines: list[str] = []
    for hit in hits or []:
        payload = getattr(hit, "payload", None) or {}
        user_message = str(payload.get("user_message") or "").strip()
        assistant_reply = str(payload.get("assistant_reply") or "").strip()
        if not user_message or not assistant_reply:
            continue
        lines.append(f"[User: {user_message} / Assistant: {assistant_reply}]")

    if not lines:
        return ""

    block = "Контекст прошлых бесед с этим собеседником:\n" + "\n".join(lines)
    log.info("[QDRANT] Для чата %s подмешано %d воспоминаний в системный промпт", chat_id, len(lines))
    return block


_QDRANT_PROFILE_KEYS = {
    "имя": "Имя",
    "name": "Имя",
    "возраст": "Возраст",
    "age": "Возраст",
    "город": "Город",
    "city": "Город",
    "хобби": "Увлечения/Хобби",
    "увлечение": "Увлечения/Хобби",
    "увлечения": "Увлечения/Хобби",
    "hobby": "Увлечения/Хобби",
    "hobbies": "Увлечения/Хобби",
}


def _merge_profile_payload(target: dict[str, str], payload: dict) -> None:
    """Забирает из payload Qdrant уже известные слоты профиля, пустые ячейки не затирает."""
    for key, value in payload.items():
        column = _QDRANT_PROFILE_KEYS.get(str(key).strip().lower())
        if column is None or _profile_cell_empty(value):
            continue
        if _profile_cell_empty(target.get(column), name_field=(column == "Имя")):
            target[column] = str(value).strip()

    nested = payload.get("profile")
    if isinstance(nested, dict):
        _merge_profile_payload(target, nested)

    blob = f"{payload.get('assistant_reply') or ''}\n{payload.get('user_message') or ''}"
    marker = re.search(r"ОБНОВИТЬ_ДАННЫЕ:\s*([^\n]+)", blob, re.IGNORECASE)
    if not marker:
        return
    for pattern, field in _PROFILE_MARKER_FIELDS:
        found = re.search(pattern, marker.group(1), re.IGNORECASE)
        if found and _profile_cell_empty(target.get(field), name_field=(field == "Имя")):
            target[field] = found.group(1).strip()


async def collect_profile_from_qdrant(chat_id: int | str) -> dict[str, str]:
    """Читает накопленные точки этого chat_id и вытаскивает известные слоты профиля."""
    found: dict[str, str] = {}
    if not _memory_ready or qdrant_client is None:
        return found
    try:
        numeric_chat_id = _coerce_chat_id(chat_id)
    except ValueError:
        return found

    query_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="chat_id",
                match=qmodels.MatchValue(value=numeric_chat_id),
            )
        ]
    )
    offset = None
    seen = 0
    try:
        while seen < 200:
            points, offset = await qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=query_filter,
                limit=64,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", None) or {}
                if isinstance(payload, dict):
                    _merge_profile_payload(found, payload)
                seen += 1
            if offset is None:
                break
    except Exception as exc:
        log.warning("[SLOTS] Qdrant не отдал профиль чата %s: %s", chat_id, exc)
    return found


async def save_dialog_memory(
    chat_id: int | str,
    user_message: str,
    assistant_reply: str,
    *,
    embed_text: str | None = None,
    initiative: bool = False,
) -> None:
    """Сохраняет успешную пару «сообщение девушки → ответ Вани» в Qdrant."""
    user_message = (user_message or "").strip()
    assistant_reply = (assistant_reply or "").strip()
    if not _memory_ready or qdrant_client is None or not user_message or not assistant_reply:
        return

    try:
        numeric_chat_id = _coerce_chat_id(chat_id)
        vector_source = (embed_text or user_message).strip()
        vector = await asyncio.to_thread(_embed_text_sync, vector_source)
        payload: dict[str, object] = {
            "chat_id": numeric_chat_id,
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if initiative:
            payload["initiative"] = True
        point = qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        )
        await qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=[point])
        log.info(
            "[QDRANT] Память сохранена для чата %s: '%s' → '%s'",
            numeric_chat_id,
            user_message[:40].replace("\n", " "),
            assistant_reply[:40].replace("\n", " "),
        )
    except Exception as exc:
        log.warning("[QDRANT] Не удалось сохранить память чата %s: %s", chat_id, exc)

# ── Настройки Автолайкера Дайвинчика (@leomatchbot) ──
AUTOCLICKER_ENABLED: bool = False
_raw_autolike = (os.getenv("AUTOLIKE_ENABLED") or "true").strip().lower()
AUTOLIKE_ENABLED: bool = _raw_autolike in ("true", "1", "yes", "y", "on")
try:
    AUTOLIKE_DAILY_LIMIT: int = int(os.getenv("AUTOLIKE_DAILY_LIMIT", "120"))
except ValueError:
    AUTOLIKE_DAILY_LIMIT = 120

EXCEL_COLUMNS = [
    "Имя",
    "Username/ID",
    "Дата приглашения",
    "Возраст",
    "Место учебы",
    "Увлечения/Хобби",
    "Доп. инфо",
    "Город",
]

# ── Синхронные функции работы с файлами для вызова через asyncio.to_thread ──

def _sync_append_chat_log(user_id: int | str, role: str, text: str) -> None:
    """Дозапись одного сообщения в файл logs_chats/{user_id}.txt."""
    filepath = LOGS_CHATS_DIR / f"{user_id}.txt"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_text = text.replace("\r\n", " ").replace("\n", " ").strip()
    if not clean_text:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"[{now_str}] {role}: {clean_text}\n")


def _sync_read_chat_log_window(user_id: int | str, limit: int = 15) -> list[dict[str, str]]:
    """Чтение скользящего окна (последние limit сообщений) из logs_chats/{user_id}.txt."""
    filepath = LOGS_CHATS_DIR / f"{user_id}.txt"
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        recent_lines = lines[-limit:]
        parsed_messages: list[dict[str, str]] = []
        line_pattern = re.compile(
            r"^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*([A-Za-zА-Яа-яЁё]+):\s*(.*)$"
        )
        for line in recent_lines:
            line = line.strip()
            if not line:
                continue
            m = line_pattern.match(line)
            if m:
                raw_role = m.group(1).lower()
                role = "assistant" if raw_role in ("assistant", "ассистент", "me", "bot") else "user"
                content = m.group(2).strip()
                parsed_messages.append({"role": role, "content": content})
            else:
                if ":" in line:
                    parts = line.split(":", 1)
                    raw_role = parts[0].strip().lower()
                    role = "assistant" if raw_role in ("assistant", "ассистент", "me", "bot") else "user"
                    parsed_messages.append({"role": role, "content": parts[1].strip()})
                else:
                    parsed_messages.append({"role": "user", "content": line})
        return parsed_messages
    except Exception as exc:
        log.error("Ошибка чтения файла лога чата %s: %s", filepath, exc)
        return []


def _excel_headers(ws) -> list:
    headers: list = []
    column = 1
    while column <= max(ws.max_column or 1, 1):
        value = ws.cell(row=1, column=column).value
        if value in (None, ""):
            break
        headers.append(value)
        column += 1
    return headers


def _ensure_excel_headers(ws) -> bool:
    """Дописывает в шапку недостающие колонки, старые строки не сдвигает."""
    headers = _excel_headers(ws)
    changed = False
    for column_name in EXCEL_COLUMNS:
        if column_name not in headers:
            ws.cell(row=1, column=len(headers) + 1, value=column_name)
            headers.append(column_name)
            changed = True
    return changed


def _sync_ensure_seen_users_excel(filepath: Path) -> None:
    """Проверка и создание файла базы данных просмотренных анкет seen_users."""
    if not filepath.exists():
        wb = openpyxl.Workbook()
        try:
            ws = wb.active
            ws.title = "SeenUsers"
            ws.append(["User ID", "Дата просмотра", "Действие (Лайк/Дизлайк)", "Имя/Инфо"])
            wb.save(filepath)
        finally:
            wb.close()


def _sync_load_seen_users(filepath: Path) -> set[str]:
    """Загрузка уже просмотренных ID анкет из Excel."""
    if not filepath.exists():
        return set()
    seen: set[str] = set()
    try:
        wb = load_workbook(filepath, read_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row and row[0] is not None:
                val = str(row[0]).strip()
                if val:
                    seen.add(val)
        wb.close()
    except Exception as exc:
        log.error("Ошибка загрузки seen_users из %s: %s", filepath, exc)
    return seen


def _sync_append_seen_user(filepath: Path, user_id: str, action: str, info: str = "") -> None:
    """Добавление записи о просмотренной анкете в Excel."""
    wb = load_workbook(filepath)
    try:
        ws = wb.active
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append([str(user_id), now_str, action, info])
        wb.save(filepath)
    finally:
        wb.close()

SLEEP_AFTER_MY_MSG_MINUTES = 15
PING_MIN_HOURS = 1.0
PING_MAX_HOURS = 3.0
MAX_HISTORY_MESSAGES = 20

# Готовые фразы пинга. Без точек на конце и без эмодзи; одна ")" допустима.
PING_PHRASES_UNIVERSAL: list[str] = [
    "привет) как дела?",
    "приветик, чем занимаешься?",
    "ку, что делаешь?",
    "привет, как день проходит?",
]
PING_PHRASES_EVENING: list[str] = [
    "привет, как вечер?",
    "ку, отдыхаешь уже?",
    "приветик, какие планы на вечер?",
]
PING_PHRASES_DAY: list[str] = [
    "привет, как бодрячком?",
    "приветик, хорошего дня) чем занята?",
]


def _collapse_trailing_user_burst(dialog_history: list[dict[str, str]], joined_text: str) -> None:
    """Убирает хвост отдельных реплик, если они уже склеены в один входящий промпт."""
    joined = (joined_text or "").strip()
    if not joined or not dialog_history:
        return
    idx = len(dialog_history)
    collected: list[str] = []
    while idx > 0 and dialog_history[idx - 1].get("role") == "user":
        collected.append(str(dialog_history[idx - 1].get("content") or "").strip())
        idx -= 1
        ordered = list(reversed(collected))
        if "\n".join(ordered) == joined or " ".join(ordered) == joined:
            del dialog_history[idx:]
            return
        if len(collected) > 12:
            return


def _word_core(word: str) -> str:
    return word.lower().strip(".,!?;:…\"'«»()").strip()


_REPLY_CONJUNCTIONS = {"и", "а", "но", "или", "либо", "что", "чтобы", "если", "когда", "хотя", "потому"}


def _drop_trailing_sentence_marks(text: str) -> str:
    """Убирает точки и восклицательные знаки только в конце строки. Вопросы не трогает."""
    trimmed = (text or "").strip().rstrip(".!")
    return trimmed.strip()


def split_reply_chunks(text: str) -> list[str]:
    """
    Короткий ответ остаётся одним сообщением.
    Длинный (больше 15 слов или 100 символов) режется по предложениям, запятым и союзам
    на порции примерно по 7–12 слов.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    words = cleaned.split(" ")
    if len(words) <= 15 and len(cleaned) <= 100:
        short = _drop_trailing_sentence_marks(cleaned)
        return [short] if short else []

    def boundary_before(index: int) -> bool:
        if index <= 0 or index >= len(words):
            return False
        prev = words[index - 1]
        if re.search(r"[.!?…]$", prev) or prev.endswith(","):
            return True
        return _word_core(words[index]) in _REPLY_CONJUNCTIONS

    chunks: list[str] = []
    start = 0
    total = len(words)
    while start < total:
        remaining = total - start
        if remaining <= 12:
            tail = _drop_trailing_sentence_marks(" ".join(words[start:]).strip(" ,"))
            if tail:
                chunks.append(tail)
            break
        cut = None
        for count in range(12, 6, -1):
            index = start + count
            if boundary_before(index):
                cut = index
                break
        if cut is None:
            for count in range(13, min(18, remaining)):
                index = start + count
                if boundary_before(index):
                    cut = index
                    break
        if cut is None:
            cut = start + 12
        piece = _drop_trailing_sentence_marks(" ".join(words[start:cut]).strip(" ,"))
        if piece:
            chunks.append(piece)
        start = cut
    if chunks:
        return chunks
    fallback = _drop_trailing_sentence_marks(cleaned)
    return [fallback] if fallback else []


def choose_ping_phrase(now: datetime | None = None) -> tuple[str, str]:
    """Случайная фраза из пула текущего времени суток. Qwen здесь не участвует."""
    if now is None:
        now = datetime.now()
    hour = now.hour
    if hour < 12:
        period = "утро"
        pool = PING_PHRASES_UNIVERSAL + PING_PHRASES_DAY
    elif hour < 17:
        period = "день"
        pool = PING_PHRASES_UNIVERSAL + PING_PHRASES_DAY
    else:
        period = "вечер"
        pool = PING_PHRASES_UNIVERSAL + PING_PHRASES_EVENING
    return random.choice(pool), period

# ── Текстовые маркеры и фразы (читаются из .env) ──
MARKER_INVITATION: str = (os.getenv("MARKER_INVITATION") or "МАРКЕР_ПРИГЛАШЕНИЕ").strip()
INVITE_MARKER = MARKER_INVITATION  # Алиас для совместимости

REPLY_ON_INVITATION: str = (
    (
        os.getenv("REPLY_ON_INVITATION")
        or "не уверен что получится в этот день )) надо будет глянуть по планам"
    )
    .strip()
    .strip('"')
    .strip("'")
)

_raw_greetings = (
    (os.getenv("GREETING_FRASES") or "Привет! Я с Дайвинчика ✨,приветик")
    .strip()
    .strip('"')
    .strip("'")
)
GREETING_FRASES: list[str] = [
    g.strip().strip('"').strip("'") for g in _raw_greetings.split(",") if g.strip()
]
if not GREETING_FRASES:
    GREETING_FRASES = ["Привет! Я с Дайвинчика ✨", "приветик"]

DEFAULT_MEDIA_RESPONSES: list[str] = [
    "блин ща интернет тупит, не грузит чето( позже гляну",
    "я ща на улице, тут связь ппц плохая, не открывается. че там?",
    "у меня телега чет лагает жестко, картинки не прогружаются) расскажи на словах че там",
    "ща не могу посмотреть, позже чекну)",
]

_raw_media_resp = (os.getenv("MEDIA_RESPONSES") or "").strip().strip('"').strip("'")
if _raw_media_resp:
    if "|" in _raw_media_resp:
        MEDIA_RESPONSES = [m.strip().strip('"').strip("'") for m in _raw_media_resp.split("|") if m.strip()]
    elif "\n" in _raw_media_resp:
        MEDIA_RESPONSES = [m.strip().strip('"').strip("'") for m in _raw_media_resp.splitlines() if m.strip()]
    else:
        MEDIA_RESPONSES = [m.strip().strip('"').strip("'") for m in _raw_media_resp.split(";") if m.strip()]
    if not MEDIA_RESPONSES:
        MEDIA_RESPONSES = list(DEFAULT_MEDIA_RESPONSES)
else:
    MEDIA_RESPONSES = list(DEFAULT_MEDIA_RESPONSES)

UPDATE_MARKER_PREFIX: str = (os.getenv("UPDATE_MARKER_PREFIX") or "ОБНОВИТЬ_ДАННЫЕ:").strip()

RUSSIAN_LANGUAGE_LOCK = (
    "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ КИТАЙСКИЕ ИЕРОГЛИФЫ ИЛИ АНГЛИЙСКИЙ ЯЗЫК. "
    "ТЫ ОБЩАЕШЬСЯ СТРОГО НА РУССКОМ ЯЗЫКЕ. "
    "ВСЕ СЛОВА, ВЫРАЖЕНИЯ И МЫСЛИ ДОЛЖНЫ БЫТЬ НАПИСАНЫ ИСКЛЮЧИТЕЛЬНО КИРИЛЛИЦЕЙ, "
    "КАК У ОБЫЧНОГО РУССКОГО ПАРНЯ. "
    "ЕСЛИ ТЫ ПОПЫТАЕШЬСЯ СГЕНЕРИРОВАТЬ ИЕРОГЛИФ, ДИАЛОГ БУДЕТ УНИЧТОЖЕН"
)

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_han(text: str) -> bool:
    """Есть ли в строке хотя бы один китайский иероглиф."""
    return bool(_HAN_RE.search(text or ""))


def _strip_han(text: str) -> str:
    """Убирает иероглифы и оставляет остальной текст."""
    cleaned = _HAN_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" ,;:-")


def _seal_russian_prompt(prompt: str) -> str:
    """Ставит языковой замок в самое начало и в самый конец системного промпта."""
    lock = RUSSIAN_LANGUAGE_LOCK
    body = (prompt or "").strip()
    while body.startswith(lock):
        body = body[len(lock):].strip()
    while body.endswith(lock):
        body = body[: -len(lock)].strip()
    if not body:
        return f"{lock}\n\n{lock}"
    return f"{lock}\n\n{body}\n\n{lock}"


DEFAULT_SYSTEM_PROMPT = f"""\
{RUSSIAN_LANGUAGE_LOCK}

Ты — 17-летний парень по имени Иван (Ваня). Твоя цель — вести живой, естественный и позитивный диалог с девушкой, с которой у тебя совпал лайк в Дайвинчике, общаться легко, непринужденно и плавно подводить к встрече.

ТВОЯ ЛИЧНОСТЬ (ИНФОРМАЦИЯ О СЕБЕ):
- Тебе 17 лет, твой рост 175 см. У тебя отличное настроение, ты открыт к общению и уверен в себе.
- Учишься в 11 классе в 11-й гимназии в городе Бийск.
- Увлекаешься программированием и любишь читать книги.
- На вопросы о будущем отвечай, что пока выбираешь, куда поступать.
- ПРИНЦИП ПОРЦИОННОСТИ: СТРОГО ЗАПРЕЩЕНО вываливать факты о себе без прямого вопроса от девушки. Раскрывай информацию о себе только тогда, когда она сама прямо спросит, и строго мелкими порциями. Не перегружай диалог рассказами о себе.

ПРАВИЛА ОБЩЕНИЯ И ЖЕСТКИЕ ОГРАНИЧЕНИЯ:
1. СТИЛЬ И ДЛИНА РЕПЛИК (1-3 ПРЕДЛОЖЕНИЯ, 7-20 СЛОВ): Ты общаешься свободно, развернуто и естественно, как живой 17-летний парень. Твои реплики должны быть полноценными предложениями (в среднем 1-3 предложения, от 7 до 20 слов), чтобы диалог выглядел вовлеченным. Если уместно по контексту, органично добавляй в конец короткий встречный вопрос, чтобы девушке было за что зацепиться (например, не просто 'Программирую', а 'Да вот, код сижу ковыряю, проект один делаю. А ты чем маешься?'). Без точек в конце строки, без графических эмодзи и скобочек.
2. ВСТРЕЧНЫЕ ВОПРОСЫ И ИНИЦИАТИВА: Не будь допрашивающим роботом. Встречный вопрос задавай только тогда, когда это уместно и развивает тему:
   – Если тема разговора тупиковая или девушка ответила односложно (нужно перехватить инициативу).
   – Если диалог только начался или сменилась тема.
   Если идет активное обсуждение конкретного факта, достаточно просто развернуто и живо поделиться мыслью по делу без лишнего допроса.
3. ЗАПРЕТ НА ЭХО И ПОВТОРЕНИЯ (ТЫ НЕ ПОПУГАЙ): ЗАПРЕЩЕНО повторять или копировать слова и фразы из последнего сообщения девушки (например, если она написала 'Новый год и лето', запрещено отвечать 'Новый год и лето звучат круто'). Ты не попугай. Твоя задача — вести диалог вперед своими словами, а не комментировать её слова её же фразами.
4. БОРЬБА СО СПАМОМ ПРИВЕТСТВИЙ: Запрещено здороваться, если диалог уже начат. Слово 'привет' или 'приветик' допускается ИСКЛЮЧИТЕЛЬНО в самом первом сообщении чата или если девушка сама только что поздоровалась. Если тебе передана история переписки, где вы уже общались, сразу отвечай на её реплику без приветствий.
5. ГЕНДЕРНЫЙ КОНТРОЛЬ: Твой собеседник — ВСЕГДА ДЕВУШКА. Ты обязан обращаться к ней СТРОГО в женском роде (например: ты сказала, ты видела, ты подумала).
6. ПОЛНЫЙ ЗАПРЕТ НА СКОБОЧКИ И СМАЙЛИКИ: НИКАКИХ графических эмодзи и скобочек ) или )) по умолчанию. Забудь про них. Пиши абсолютно без них. Использовать ОДНУ скобочку ')' разрешено только в 1 случае из 10, если ты реально пошутил или подколол.
7. БЕЗ ТОЧЕК: Никогда не ставь точку в самом конце сообщения.
8. ОБРАЗ И ТОН: Живой, непринужденный, вежливый пацанский стиль без канцеляризмов и пошлости.

ТЕХНИЧЕСКИЕ ИНСТРУКЦИИ (МАРКЕРЫ):

А. ОБНОВЛЕНИЕ ДАННЫХ (Excel):
Если в процессе диалога девушка сообщила какой-то новый факт о себе (свой точный возраст, где учится, чем увлекается или как её зовут, если имя не было известно), ты ОБЯЗАН в самом конце своего текстового ответа дописать скрытый сервисный маркер строго в таком формате:
ОБНОВИТЬ_ДАННЫЕ: имя=[значение], возраст=[значение], город=[значение], учеба=[значение], хобби=[значение]
(Пиши только те поля, которые она реально назвала. Если ничего нового не сказала — маркер не пишется).

Б. ПЕРЕХВАТ ПРИГЛАШЕНИЙ (Свидание):
Тебе ЖЕСТКО ЗАПРЕЩЕНО самому звать девушку гулять или на свидание. Ты должен общаться так, чтобы инициатива пойти гулять исходила от неё.

Если девушка напрямую или косвенно зовет тебя встретиться, погулять или увидеться (например: 'пошли гулять', 'может встретимся?', 'какие планы на выходные?'):
1. Сгенерируй мягкий, живой и немного интригующий ответ в своем обычном пацанском стиле без точек в конце. Ты не должен говорить четкое 'да', но и не должен отказывать (говорить 'нет'). Твой ответ должен быть уклончивым, например: 
   - 'ну в целом можно) надо глянуть что там по планам на днях, отпишусь тебе'
   - 'слушай, идея хорошая) у меня там были дела, но если получится раскидать их, то погнали'
   - 'можно как-нибудь выбраться) я прикину по времени и скажу точно'
   Придумывай похожие живые варианты, не пиши всегда одно и то же.
2. В самый конец этого сообщения (после текста ответа) принудительно допиши скрытый маркер: МАРКЕР_ПРИГЛАШЕНИЕ

{RUSSIAN_LANGUAGE_LOCK}
"""

def _load_system_prompt() -> str:
    """Загрузка системного промпта из .env с защитой от ошибок парсинга многострочных строк."""
    env_sp = os.getenv("SYSTEM_PROMPT", "").strip()
    if env_sp:
        if "\\n" in env_sp and "\n" not in env_sp:
            env_sp = env_sp.replace("\\n", "\n")
        return _seal_russian_prompt(env_sp.strip('"').strip("'"))

    # Резервный поиск напрямую в файле .env, если dotenv споткнулся о внутренние кавычки
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            content = env_path.read_text(encoding="utf-8")
            m = re.search(r'SYSTEM_PROMPT\s*=\s*["\'](.*)', content, re.DOTALL)
            if m:
                raw_text = m.group(1)
                # Отсекаем следующие переменные или конец строки/файла
                split_next = re.split(r'\n[A-Z0-9_]+\s*=', raw_text, maxsplit=1)
                val = split_next[0].strip().rstrip('"').rstrip("'").strip()
                if val:
                    if "\\n" in val and "\n" not in val:
                        val = val.replace("\\n", "\n")
                    return _seal_russian_prompt(val)
        except Exception:
            pass
    return _seal_russian_prompt(DEFAULT_SYSTEM_PROMPT)


SYSTEM_PROMPT: str = _load_system_prompt()

# ─────────────────────────────────────────────────────────────────────────────
# Валидация человеческих имен
# ─────────────────────────────────────────────────────────────────────────────

STOP_NAME_WORDS = {
    "inst", "tg", "channel", "bot", "vip", "http", "www",
    "admin", "client", "клиент", "неизвестно", "unknown",
    "канал", "чат", "инст", "тг", "бот", "id", "com"
}


def is_valid_name(name_str: str | None) -> bool:
    """
    Проверяет, является ли переданная строка реальным человеческим именем:
    - Длина от 2 до 15 символов;
    - Только русские или английские буквы (допускается дефис или пробел в двойных именах);
    - Внутри нет цифр, символов @, подчёркиваний, смайликов, ссылок и стоп-слов.
    """
    if not name_str or not isinstance(name_str, str):
        return False
    s = name_str.strip()
    if not (2 <= len(s) <= 15):
        return False
    lower = s.lower()
    for stop in STOP_NAME_WORDS:
        if stop in lower:
            return False
    if not re.fullmatch(r"[А-Яа-яЁёA-Za-z]+(?:[- ][А-Яа-яЁёA-Za-z]+)?", s):
        return False
    return True


# Алиас для обратной совместимости
Is_valid_name = is_valid_name


# Порядок важен: бот выведывает ровно первый пустой слот.
PROFILE_SLOTS: tuple[tuple[str, str], ...] = (
    ("Имя", "Имя"),
    ("Возраст", "Возраст"),
    ("Город", "Город"),
    ("Увлечение", "Увлечения/Хобби"),
)

_EMPTY_PROFILE_TOKENS = {
    "",
    "none",
    "null",
    "неизвестно",
    "unknown",
    "клиент",
    "-",
    "—",
}

_PROFILE_MARKER_FIELDS = (
    (r"имя=([^,\n]+)", "Имя"),
    (r"возраст=([^,\n]+)", "Возраст"),
    (r"город=([^,\n]+)", "Город"),
    (r"учеба=([^,\n]+)", "Место учебы"),
    (r"хобби=([^,\n]+)", "Увлечения/Хобби"),
    (r"увлечение=([^,\n]+)", "Увлечения/Хобби"),
)


def _profile_cell_empty(value: object, *, name_field: bool = False) -> bool:
    """Пусто, если значения нет, это пробелы, заглушка или (для имени) не человеческое имя."""
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.lower() in _EMPTY_PROFILE_TOKENS:
        return True
    if name_field and not is_valid_name(text):
        return True
    return False


def missing_slots_from_profile(row: dict | None) -> list[str]:
    """Возвращает подписи только тех слотов, которые в строке профиля сейчас пустые."""
    source = row or {}
    missing: list[str] = []
    for label, column in PROFILE_SLOTS:
        if _profile_cell_empty(source.get(column), name_field=(column == "Имя")):
            missing.append(label)
    return missing


def build_slot_goal(slot: str) -> str:
    """Скрытая микро-инструкция: один факт за раз, без допроса."""
    return (
        f"ТЕКУЩАЯ ТАКТИЧЕСКАЯ ЦЕЛЬ: Ты до сих пор не знаешь {slot} девушки. "
        "Твоя задача — максимально естественно, коротко и ненавязчиво вплести в свой ответ встречный вопрос, "
        "чтобы узнать этот факт. Не спрашивай про остальные вещи. Если все данные уже заполнены, общайся на свободные темы. "
        "Спрашивай строго про один факт за раз и только тогда, когда это уместно по контексту переписки. "
        "Не будь допрашивающим роботом."
    )


UNKNOWN_NAME_INSTRUCTION = (
    "ВНИМАНИЕ: Имя девушки тебе НЕИЗВЕСТНО. Тебе СТРОГО ЗАПРЕЩЕНО выдумывать ей имя или использовать её никнейм. "
    "Ты ОБЯЗАН в этом сообщении лаконично и ненавязчиво, в своем пацанском стиле, спросить как её зовут. "
    "Как только она назовет имя, в следующих сообщениях общайся нормально."
)


def _get_time_schedule_prompt() -> str:
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


def _with_city_marker(prompt: str) -> str:
    """В уже загруженном промпте маркер обновления тоже умеет принимать город."""
    if re.search(r"город\s*=", prompt, re.IGNORECASE):
        return prompt
    if "хобби=[значение]" in prompt:
        return prompt.replace("хобби=[значение]", "город=[значение], хобби=[значение]", 1)
    return prompt


def _build_system_prompt(client_row: dict | None, name: str | None = None) -> str:
    """Сформировать системный промпт с динамическим временем на ПК, блоком контекста из Excel и инструкцией по имени."""
    effective_name = name
    if effective_name is None and client_row:
        effective_name = client_row.get("Имя")

    context_parts: list[str] = []
    if effective_name and is_valid_name(effective_name):
        context_parts.append(f"имя: {effective_name}")

        if client_row:
            if not _profile_cell_empty(client_row.get("Возраст")):
                context_parts.append(f"возраст: {client_row['Возраст']}")
            if not _profile_cell_empty(client_row.get("Город")):
                context_parts.append(f"город: {client_row['Город']}")
            if not _profile_cell_empty(client_row.get("Место учебы")):
                context_parts.append(f"место учёбы: {client_row['Место учебы']}")
            if not _profile_cell_empty(client_row.get("Увлечения/Хобби")):
                context_parts.append(f"хобби: {client_row['Увлечения/Хобби']}")
            if not _profile_cell_empty(client_row.get("Доп. инфо")):
                context_parts.append(f"доп. инфо: {client_row['Доп. инфо']}")

    prompt_body = _with_city_marker(SYSTEM_PROMPT)
    if effective_name == "Неизвестно" or (effective_name is not None and not is_valid_name(effective_name)):
        prompt_body = f"{UNKNOWN_NAME_INSTRUCTION}\n\n{SYSTEM_PROMPT}"

    time_prefix = _get_time_schedule_prompt()

    if context_parts:
        context_block = "Контекст собеседника: " + "; ".join(context_parts) + "."
        return f"{time_prefix}\n\n{context_block}\n\n{prompt_body}"
    return f"{time_prefix}\n\n{prompt_body}"


def prepare_messages_for_chat_template(
    dialog_history: list[dict[str, str]],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """
    Формирует список сообщений для Ollama chat.
    Гарантирует, что самый первый элемент ВСЕГДА имеет ровно такой вид:
    {"role": "system", "content": ...}.
    """
    if system_prompt is None:
        base_system_prompt = os.getenv("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
        time_prefix = _get_time_schedule_prompt()
        system_prompt = f"{time_prefix}\n\n{base_system_prompt}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(system_prompt)}
    ]

    for msg in dialog_history:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    return messages


async def _click_leobot_button(client: Client, message: Message, action: str) -> bool:
    """
    Нажатие на кнопку лайка (❤️) или дизлайка (👎) в сообщении Дайвинчика.
    Поддерживает inline_keyboard, reply_keyboard и отправку текстового эмодзи.
    """
    try:
        # 1. Проверяем inline_keyboard
        if message.reply_markup and getattr(message.reply_markup, "inline_keyboard", None):
            for r_idx, row in enumerate(message.reply_markup.inline_keyboard):
                for c_idx, btn in enumerate(row):
                    t = (getattr(btn, "text", "") or "").strip()
                    if action == "like":
                        if any(s in t for s in ["❤️", "❤", "👍", "😍", "1", "1️⃣", "Лайк"]):
                            await message.click(r_idx, c_idx)
                            return True
                    else:
                        if any(s in t for s in ["👎", "2", "2️⃣", "3", "3️⃣", "Дизлайк"]):
                            await message.click(r_idx, c_idx)
                            return True

        # 2. Проверяем reply_keyboard (обычные кнопки внизу)
        if message.reply_markup and getattr(message.reply_markup, "keyboard", None):
            for row in message.reply_markup.keyboard:
                for btn in row:
                    t = (getattr(btn, "text", str(btn)) or "").strip()
                    if action == "like":
                        if any(s in t for s in ["❤️", "❤", "👍", "😍", "1", "Лайк"]):
                            await client.send_message(message.chat.id, t)
                            return True
                    else:
                        if any(s in t for s in ["👎", "2", "3", "Дизлайк"]):
                            await client.send_message(message.chat.id, t)
                            return True

        # 3. Фолбек: отправка текстового символа в чат Дайвинчика
        fallback_char = "❤️" if action == "like" else "👎"
        await client.send_message(message.chat.id, fallback_char)
        return True
    except Exception as exc:
        log.error("Ошибка при клике кнопки в сообщении %d: %s", message.id, exc)
        return False


# Глобальный лок для предотвращения конфликтов в интерактивной консоли при авторизации
_AUTH_LOCK = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Класс аккаунта с полной изоляцией логики и состояния
# ─────────────────────────────────────────────────────────────────────────────


class AccountBot:
    """
    Изолированный контекст и клиент Telegram-аккаунта.
    Каждый аккаунт имеет:
    - Свой файл сессии: sessions/{name}.session
    - Свою изолированную базу данных: databases/dates_{name}.xlsx
    - Свою базу просмотренных анкет: databases/seen_users_{name}.xlsx
    - Свой лок базы данных, активные чаты, таймер сна и флаг активности
    - Свою фоновую задачу реанимации диалогов и воркер автолайкера Дайвинчика
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.session_file: Path = SESSIONS_DIR / f"{name}.session"
        self.excel_file: Path = DATABASES_DIR / f"dates_{name}.xlsx"
        self.excel_lock: asyncio.Lock = asyncio.Lock()

        # База данных просмотренных анкет Дайвинчика
        self.seen_users_file: Path = DATABASES_DIR / f"seen_users_{name}.xlsx"
        self.seen_users_lock: asyncio.Lock = asyncio.Lock()
        self.seen_users: set[str] = set()

        # Изолированное состояние аккаунта
        self.bot_active: bool = True
        self.sleep_until: datetime | None = None
        self.my_id: int | None = None
        self.active_chats: dict[int | str, dict] = {
            "leomatchbot": {"name": "Дайвинчик", "username": "leomatchbot"},
            "@leomatchbot": {"name": "Дайвинчик", "username": "leomatchbot"},
        }

        # Автолайкер Дайвинчика
        self.autoliker_task: asyncio.Task | None = None
        self.leobot_queue: asyncio.Queue[Message] = asyncio.Queue()
        self.likes_today: int = 0
        self.likes_reset_time: datetime = datetime.now() + timedelta(hours=24)
        self.last_handled_leobot_msg_id: int | None = None

        # Pyrogram Client с метаданными Telegram Desktop для Windows и пулом воркеров
        self.client: Client = Client(
            name=self.name,
            workdir=str(SESSIONS_DIR),
            api_id=API_ID,
            api_hash=API_HASH,
            workers=20,
            device_model="Desktop",
            system_version="Windows 10",
            app_version="5.10.0",
            lang_code="ru",
            system_lang_code="ru-RU",
        )
        self.client._account_bot = self

        self._stop_event: asyncio.Event = asyncio.Event()
        self.resurrector_task: asyncio.Task | None = None
        self.day_worker_task: asyncio.Task | None = None
        self.night_mode_active: bool = False
        self.day_mode_event: asyncio.Event = asyncio.Event()

        # Чаты этого аккаунта, чьи таски лежат в модульных pending_messages / debouncer_tasks
        self._debounce_chat_ids: set[int] = set()
        self.typing_activity: dict[int, float] = {}
        # Слушатель: онлайн без статуса «печатает»
        self._listen_tasks: dict[int, asyncio.Task] = {}
        self._listen_stop: dict[int, asyncio.Event] = {}
        self._listen_ready: dict[int, asyncio.Event] = {}
        # Писатель: цикл TYPING и отменяемая генерация
        self._typing_tasks: dict[int, asyncio.Task] = {}
        self._typing_stop: dict[int, asyncio.Event] = {}
        self._generate_tasks: dict[int, asyncio.Task] = {}
        self._chunk_tasks: dict[int, asyncio.Task] = {}

        self._register_handlers()

    # ── Регистрация обработчиков сообщений и сырых обновлений ───────────────

    def _register_handlers(self) -> None:
        @self.client.on_raw_update(group=-1)
        async def _on_raw_update(client: Client, update: raw.base.Update, users: dict, chats: dict):
            await self.handle_raw_update(update, users, chats)

        @self.client.on_message(filters.me & filters.text, group=-2)
        async def _on_clicker_cmd(client: Client, message: Message):
            await self.handle_clicker_command(message)

        @self.client.on_message(filters.incoming | filters.outgoing, group=0)
        async def _on_toggle(client: Client, message: Message):
            await self.toggle_bot(message)

        @self.client.on_message(filters.outgoing, group=1)
        async def _on_manual_control(client: Client, message: Message):
            await self.manual_chat_control(message)

        @self.client.on_message(filters.outgoing, group=2)
        async def _on_track(client: Client, message: Message):
            await self.track_my_messages(message)

        @self.client.on_message(filters.private & filters.user("leomatchbot"), group=3)
        async def _on_leobot(client: Client, message: Message):
            await self.on_leobot_message(message)

        @self.client.on_message(filters.private & ~filters.bot & ~filters.service, group=4)
        async def _on_girl(client: Client, message: Message):
            await self.handle_girl_message(message)

    async def handle_raw_update(self, update: raw.base.Update, users: dict, chats: dict) -> None:
        """Диспетчер сырых обновлений MTProto. Отслеживает UpdatesTooLong и статус печати."""
        update_name = type(update).__name__
        if isinstance(update, (raw.types.UpdatesTooLong, raw.types.UpdateChannelTooLong)):
            log.warning(
                "[%s][RAW DISPATCHER] ВНИМАНИЕ: Получено событие %s! Telegram сообщил о пропущенных обновлениях.",
                self.name,
                update_name,
            )
            return

        if isinstance(update, raw.types.UpdateUserTyping):
            user_id = int(update.user_id)
            if isinstance(update.action, raw.types.SendMessageTypingAction):
                self.typing_activity[user_id] = time.monotonic()
                log.debug("[%s][TYPING] Пользователь %s печатает", self.name, user_id)
            elif isinstance(update.action, raw.types.SendMessageCancelAction):
                self.typing_activity.pop(user_id, None)
                log.debug("[%s][TYPING] Пользователь %s перестал печатать", self.name, user_id)

    # ── Проверки активности и режима сна ───────────────────────────────────

    def is_bot_asleep(self) -> bool:
        """Проверить, активен ли режим «сна» ИИ для этого аккаунта."""
        return self.sleep_until is not None and datetime.now() < self.sleep_until

    def is_ai_frozen(self) -> bool:
        """Проверить, заморожен ли ИИ (выключен вручную или активен режим сна)."""
        return (not self.bot_active) or self.is_bot_asleep()

    def calculate_mode_delta(self, now: datetime | None = None) -> tuple[bool, float, datetime]:
        """
        Вычисляет статус ночного режима и точное время сна (дельту) до ближайшей смены фазы:
        - Ночной диапазон: с 21:00:00 вечера до 09:00:00 утра (с переходом через полночь 00:00).
        - Дневной диапазон: с 09:00:00 утра до 21:00:00 вечера.

        Возвращает кортеж: (is_night, seconds_to_sleep, target_datetime).
        """
        if now is None:
            now = datetime.now()

        today = now.date()
        t_09_today = datetime.combine(today, dtime(9, 0, 0))
        t_21_today = datetime.combine(today, dtime(21, 0, 0))

        if now < t_09_today:
            # 1. Ночь: текущее время между 00:00:00 и 08:59:59
            # Целевая точка переключения на ДЕНЬ — 09:00 сегодня
            is_night = True
            target_dt = t_09_today
        elif now < t_21_today:
            # 2. День: текущее время между 09:00:00 и 20:59:59
            # Целевая точка переключения на НОЧЬ — 21:00 сегодня
            is_night = False
            target_dt = t_21_today
        else:
            # 3. Ночь: текущее время между 21:00:00 и 23:59:59
            # Переход через полночь: целевая точка переключения на ДЕНЬ — 09:00 завтра
            is_night = True
            target_dt = datetime.combine(today + timedelta(days=1), dtime(9, 0, 0))

        seconds_to_sleep = (target_dt - now).total_seconds()
        if seconds_to_sleep <= 0:
            seconds_to_sleep = 1.0

        return is_night, seconds_to_sleep, target_dt

    def is_night_time(self) -> bool:
        """
        Проверка системного времени: попадает ли оно в ночной диапазон с 21:00 вечера до 09:00 утра.
        В этот промежуток юзерботу категорически запрещено писать первым и инициировать диалог.
        """
        is_night, _, _ = self.calculate_mode_delta()
        return is_night

    def activate_sleep_mode(self) -> None:
        """Заморозить ИИ для этого аккаунта на SLEEP_AFTER_MY_MSG_MINUTES минут."""
        self.sleep_until = datetime.now() + timedelta(minutes=SLEEP_AFTER_MY_MSG_MINUTES)
        log.info("[%s] Режим сна активирован до %s", self.name, self.sleep_until.strftime("%H:%M:%S"))

    # ── Работа с Excel (изолированная БД аккаунта, полностью асинхронно через to_thread) ─

    def _sync_ensure_excel(self) -> None:
        if not self.excel_file.exists():
            wb = openpyxl.Workbook()
            try:
                ws = wb.active
                ws.title = "Clients"
                ws.append(EXCEL_COLUMNS)
                wb.save(self.excel_file)
                log.info("[%s] Создан файл БД: %s", self.name, self.excel_file)
            finally:
                wb.close()
            return
        wb = load_workbook(self.excel_file)
        try:
            ws = wb.active
            if _ensure_excel_headers(ws):
                wb.save(self.excel_file)
                log.info("[%s] В %s добавлена колонка профиля «Город»", self.name, self.excel_file.name)
            else:
                log.info("[%s] Файл БД %s уже существует, используем его.", self.name, self.excel_file)
        finally:
            wb.close()

    async def ensure_excel(self) -> None:
        """Убедиться, что файл dates_{name}.xlsx существует со всеми колонками (в отдельном потоке)."""
        await asyncio.to_thread(self._sync_ensure_excel)

    async def ensure_seen_users_excel(self) -> None:
        """Убедиться, что файл базы данных просмотренных анкет seen_users_{name}.xlsx существует."""
        async with self.seen_users_lock:
            await asyncio.to_thread(_sync_ensure_seen_users_excel, self.seen_users_file)
            try:
                await asyncio.to_thread(_sync_ensure_seen_users_excel, DATABASES_DIR / "seen_users.xlsx")
            except Exception:
                pass

    async def append_chat_log(self, user_id: int | str, role: str, text: str) -> None:
        """Асинхронная дозапись сообщения в текстовый файл logs_chats/{user_id}.txt."""
        await asyncio.to_thread(_sync_append_chat_log, user_id, role, text)

    async def read_chat_log_window(self, user_id: int | str, limit: int = 15) -> list[dict[str, str]]:
        """Асинхронное чтение скользящего окна (последние 10-15 сообщений) из logs_chats/{user_id}.txt."""
        return await asyncio.to_thread(_sync_read_chat_log_window, user_id, limit)

    def _sync_load_active_chats(self) -> list[tuple[str, str]]:
        loaded: list[tuple[str, str]] = []
        if not self.excel_file.exists():
            return loaded
        try:
            wb = load_workbook(self.excel_file)
            try:
                ws = wb.active
                headers = [cell.value for cell in ws[1]]
                if "Username/ID" not in headers:
                    return loaded
                id_idx = headers.index("Username/ID")
                name_idx = headers.index("Имя") if "Имя" in headers else None

                for row in ws.iter_rows(min_row=2, values_only=True):
                    raw_id = row[id_idx]
                    if not raw_id:
                        continue
                    name_val = row[name_idx] if name_idx is not None and row[name_idx] else "Неизвестно"
                    name_str = str(name_val).strip()
                    if not is_valid_name(name_str):
                        name_str = "Неизвестно"

                    str_id = str(raw_id).strip()
                    loaded.append((name_str, str_id))
            finally:
                wb.close()
        except Exception as exc:
            log.error("[%s] Ошибка загрузки активных чатов из Excel: %s", self.name, exc)
        return loaded

    async def load_whitelist(self) -> None:
        """
        Загрузить ранее сохраненные чаты из dates_{name}.xlsx, JSON-конфигов и .env
        в память (active_chats и GLOBAL_WHITELIST_IDS).
        """
        now_time = asyncio.get_event_loop().time()
        count = 0

        # 1. Загрузка из Excel (dates_{name}.xlsx)
        chat_rows = await asyncio.to_thread(self._sync_load_active_chats)
        for name_str, str_id in chat_rows:
            chat_info = {
                "name": name_str,
                "username": str_id,
                "history": [],
                "pinged": False,
                "last_msg_time": now_time,
                "last_msg_is_me": False,
            }
            if str_id.lstrip("-").isdigit():
                uid = int(str_id)
                self.active_chats[uid] = chat_info
                GLOBAL_WHITELIST_IDS.add(uid)
            self.active_chats[str_id] = chat_info
            self.active_chats[str_id.lstrip("@")] = chat_info
            count += 1

        # 2. Загрузка из JSON-конфигураций (whitelist_{name}.json и общий whitelist.json)
        acc_json = DATABASES_DIR / f"whitelist_{self.name}.json"
        for jf in [WHITELIST_JSON_FILE, acc_json]:
            j_ids, j_chats = await asyncio.to_thread(_sync_load_whitelist_json, jf)
            for uid in j_ids:
                GLOBAL_WHITELIST_IDS.add(uid)
                if uid not in self.active_chats:
                    chat_meta = j_chats.get(str(uid), {})
                    name = chat_meta.get("name", "Неизвестно")
                    uname = chat_meta.get("username", str(uid))
                    chat_info = {
                        "name": name,
                        "username": uname,
                        "history": [],
                        "pinged": False,
                        "last_msg_time": now_time,
                        "last_msg_is_me": False,
                    }
                    self.active_chats[uid] = chat_info
                    self.active_chats[str(uid)] = chat_info
                    if uname:
                        self.active_chats[uname] = chat_info
                        self.active_chats[uname.lstrip("@")] = chat_info
                    count += 1

        # 3. Загрузка из переменной окружения WHITELIST_CHAT_IDS (.env)
        env_raw = os.getenv("WHITELIST_CHAT_IDS", "").strip()
        if env_raw:
            for part in env_raw.split(","):
                part = part.strip()
                if part.lstrip("-").isdigit():
                    uid = int(part)
                    GLOBAL_WHITELIST_IDS.add(uid)
                    if uid not in self.active_chats:
                        chat_info = {
                            "name": "Неизвестно",
                            "username": str(uid),
                            "history": [],
                            "pinged": False,
                            "last_msg_time": now_time,
                            "last_msg_is_me": False,
                        }
                        self.active_chats[uid] = chat_info
                        self.active_chats[str(uid)] = chat_info
                        count += 1

        log.info(
            "[%s][WHITELIST] Загружен белый список: %d записей в active_chats, %d уникальных ID в GLOBAL_WHITELIST_IDS",
            self.name, len(self.active_chats), len(GLOBAL_WHITELIST_IDS)
        )

    async def load_active_chats_from_excel(self) -> None:
        """Совместимость со старым вызовом: перенаправляет на load_whitelist."""
        await self.load_whitelist()

    async def add_to_whitelist(
        self,
        user_id: int | str,
        username: str | None = None,
        name: str = "Неизвестно",
        extra: str = "",
    ) -> None:
        """
        Динамическое добавление нового chat_id в белый список:
        1. Немедленно добавляет в active_chats и GLOBAL_WHITELIST_IDS в оперативной памяти.
        2. Синхронизирует с файлом Excel (dates_{name}.xlsx).
        3. Сохраняет в JSON-конфиг whitelist_{name}.json и общий whitelist.json.
        4. Обновляет переменную WHITELIST_CHAT_IDS в .env для сохранения между перезапусками.
        """
        try:
            str_id = str(user_id).strip()
            if not str_id or str_id.startswith("-100"):
                return
            int_id = int(str_id) if str_id.lstrip("-").isdigit() else None
            if int_id is not None and int_id <= 0:
                return

            if int_id in _KNOWN_LEOMATCHBOT_IDS or (username and username.lower().lstrip("@") == "leomatchbot"):
                if int_id:
                    _KNOWN_LEOMATCHBOT_IDS.add(int_id)
                    GLOBAL_WHITELIST_IDS.add(int_id)
                return

            if not is_valid_name(name):
                name = "Неизвестно"

            now_time = asyncio.get_event_loop().time()
            chat_info = None
            if int_id:
                chat_info = self.active_chats.get(int_id)
            if not chat_info and username:
                chat_info = self.active_chats.get(username)

            if chat_info:
                if is_valid_name(name) and chat_info.get("name") in ("Неизвестно", "Клиент", "", None):
                    chat_info["name"] = name
                if username and not chat_info.get("username"):
                    chat_info["username"] = username
            else:
                chat_info = {
                    "name": name,
                    "username": username or str_id,
                    "history": [],
                    "pinged": False,
                    "last_msg_time": now_time,
                    "last_msg_is_me": False,
                }

            if int_id:
                self.active_chats[int_id] = chat_info
                self.active_chats[str_id] = chat_info
                GLOBAL_WHITELIST_IDS.add(int_id)
            if username:
                self.active_chats[username] = chat_info
                self.active_chats[username.lstrip("@")] = chat_info

            # 1. Синхронизация с Excel
            client_row = await self.excel_read_client(str_id)
            if not client_row and username:
                client_row = await self.excel_read_client(username)

            if not client_row:
                await self.excel_append_row(
                    name=name,
                    user_id=str_id,
                    date_str="",
                    extra=extra or (f"Добавлен автоматически (@{username})" if username else "Добавлен автоматически"),
                )
            elif is_valid_name(name) and client_row.get("Имя") in ("Неизвестно", "Клиент", "", None):
                await self.excel_update_field(str_id, "Имя", name)

            # 2. Синхронизация с JSON и .env
            metadata_map = {}
            for k, v in self.active_chats.items():
                if isinstance(k, int) and k > 0 and k not in _KNOWN_LEOMATCHBOT_IDS and k != self.my_id:
                    metadata_map[str(k)] = {
                        "name": v.get("name", "Неизвестно"),
                        "username": v.get("username", str(k)),
                    }

            acc_json = DATABASES_DIR / f"whitelist_{self.name}.json"
            valid_ids = {i for i in GLOBAL_WHITELIST_IDS if i > 0 and i not in _KNOWN_LEOMATCHBOT_IDS and i != self.my_id}
            await asyncio.to_thread(_sync_save_whitelist_json, acc_json, valid_ids, metadata_map)
            await asyncio.to_thread(_sync_save_whitelist_json, WHITELIST_JSON_FILE, valid_ids, metadata_map)
            await asyncio.to_thread(_sync_update_env_whitelist, valid_ids)

            log.info(
                "[%s][WHITELIST] Чат %s (ID: %s, @%s) добавлен в белый список (память + JSON + .env + Excel)",
                self.name, name, str_id, username or ""
            )
        except Exception as exc:
            log.error("[%s][WHITELIST] Ошибка добавления в белый список %s: %s", self.name, user_id, exc, exc_info=True)

    async def remove_from_whitelist(self, user_id: int | str, username: str | None = None) -> None:
        """Удалить пользователя из белого списка (память + диск + .env + Excel)."""
        try:
            str_id = str(user_id).strip()
            int_id = int(str_id) if str_id.lstrip("-").isdigit() else None

            if int_id:
                self.active_chats.pop(int_id, None)
                GLOBAL_WHITELIST_IDS.discard(int_id)
            self.active_chats.pop(str_id, None)
            if username:
                self.active_chats.pop(username, None)
                self.active_chats.pop(username.lstrip("@"), None)

            # Удаление из Excel
            identifiers = [str_id]
            if username:
                identifiers.append(username)
            await self.excel_delete_client(identifiers)

            # Обновление JSON и .env
            metadata_map = {}
            for k, v in self.active_chats.items():
                if isinstance(k, int) and k > 0 and k not in _KNOWN_LEOMATCHBOT_IDS and k != self.my_id:
                    metadata_map[str(k)] = {
                        "name": v.get("name", "Неизвестно"),
                        "username": v.get("username", str(k)),
                    }

            acc_json = DATABASES_DIR / f"whitelist_{self.name}.json"
            valid_ids = {i for i in GLOBAL_WHITELIST_IDS if i > 0 and i not in _KNOWN_LEOMATCHBOT_IDS and i != self.my_id}
            await asyncio.to_thread(_sync_save_whitelist_json, acc_json, valid_ids, metadata_map)
            await asyncio.to_thread(_sync_save_whitelist_json, WHITELIST_JSON_FILE, valid_ids, metadata_map)
            await asyncio.to_thread(_sync_update_env_whitelist, valid_ids)
            log.info("[%s][WHITELIST] Чат %s удален из белого списка", self.name, user_id)
        except Exception as exc:
            log.error("[%s][WHITELIST] Ошибка удаления из белого списка %s: %s", self.name, user_id, exc)

    async def excel_read_client(self, user_id: int | str) -> dict | None:
        """Прочитать строку клиента из Excel по Username/ID (в отдельном потоке)."""
        log.debug("[%s][EXCEL_READ] Ожидание блокировки excel_lock для %s...", self.name, user_id)
        async with self.excel_lock:
            log.debug("[%s][EXCEL_READ] Блокировка получена. Чтение записи %s из Excel в потоке...", self.name, user_id)

            def _sync_read() -> dict | None:
                if not self.excel_file.exists():
                    return None
                try:
                    wb = load_workbook(self.excel_file, data_only=True)
                    try:
                        ws = wb.active
                        headers = [cell.value for cell in ws[1]]
                        if "Username/ID" not in headers:
                            return None
                        id_idx = headers.index("Username/ID")
                        target = str(user_id).strip().lower()

                        for row in ws.iter_rows(min_row=2, values_only=True):
                            val = str(row[id_idx] or "").strip()
                            if val == str(user_id) or val.lstrip("@").lower() == target.lstrip("@"):
                                return dict(zip(headers, row))
                        return None
                    finally:
                        wb.close()
                except Exception as exc:
                    log.error("[%s] Ошибка чтения из Excel: %s", self.name, exc)
                    return None

            result = await asyncio.to_thread(_sync_read)
            log.debug("[%s][EXCEL_READ] Чтение записи %s завершено (найдено: %s)", self.name, user_id, bool(result))
            return result

    async def excel_append_row(
        self,
        name: str,
        user_id: str | int,
        date_str: str,
        age: str = "",
        study: str = "",
        hobby: str = "",
        extra: str = "",
        city: str = "",
    ) -> None:
        """Добавить новую строку в Excel под блокировкой (в отдельном потоке)."""
        log.debug("[%s][EXCEL_APPEND] Ожидание блокировки excel_lock для %s (%s)...", self.name, name, user_id)
        async with self.excel_lock:
            log.debug("[%s][EXCEL_APPEND] Блокировка получена. Запись в Excel в отдельном потоке...", self.name)

            def _sync_append() -> None:
                try:
                    wb = load_workbook(self.excel_file)
                    try:
                        ws = wb.active
                        _ensure_excel_headers(ws)
                        headers = _excel_headers(ws)
                        values = {
                            "Имя": name,
                            "Username/ID": str(user_id),
                            "Дата приглашения": date_str,
                            "Возраст": age,
                            "Место учебы": study,
                            "Увлечения/Хобби": hobby,
                            "Доп. инфо": extra,
                            "Город": city,
                        }
                        ws.append([values.get(header, "") for header in headers])
                        wb.save(self.excel_file)
                        log.info("[%s][EXCEL_APPEND] Запись сохранена: %s (%s)", self.name, name, user_id)
                    finally:
                        wb.close()
                except Exception as exc:
                    log.error("[%s] Ошибка записи в Excel: %s", self.name, exc)

            await asyncio.to_thread(_sync_append)
            log.debug("[%s][EXCEL_APPEND] Операция добавления завершена, блокировка освобождена", self.name)

    async def excel_update_field(self, user_id: str | int, field: str, value: str) -> None:
        """Обновить конкретное поле клиента в Excel под блокировкой (в отдельном потоке)."""
        if field not in EXCEL_COLUMNS:
            return
        log.debug("[%s][EXCEL_UPDATE] Ожидание блокировки excel_lock для %s (поле '%s')...", self.name, user_id, field)
        async with self.excel_lock:
            log.debug("[%s][EXCEL_UPDATE] Блокировка получена. Обновление поля '%s' для %s в отдельном потоке...", self.name, field, user_id)

            def _sync_update() -> None:
                try:
                    wb = load_workbook(self.excel_file)
                    try:
                        ws = wb.active
                        _ensure_excel_headers(ws)
                        headers = _excel_headers(ws)
                        if field not in headers or "Username/ID" not in headers:
                            return
                        col_idx = headers.index(field) + 1
                        id_idx = headers.index("Username/ID") + 1
                        target = str(user_id).strip().lower()

                        for row in range(2, ws.max_row + 1):
                            val = str(ws.cell(row=row, column=id_idx).value or "").strip()
                            if val == str(user_id) or val.lstrip("@").lower() == target.lstrip("@"):
                                ws.cell(row=row, column=col_idx, value=value)
                                wb.save(self.excel_file)
                                log.info("[%s][EXCEL_UPDATE] Поле '%s' обновлено -> '%s' для %s", self.name, field, value, user_id)
                                return
                    finally:
                        wb.close()
                except Exception as exc:
                    log.error("[%s] Ошибка обновления поля в Excel: %s", self.name, exc)

            await asyncio.to_thread(_sync_update)
            log.debug("[%s][EXCEL_UPDATE] Операция обновления поля '%s' завершена", self.name, field)

    async def excel_delete_client(self, identifiers: int | str | list[int | str]) -> None:
        """Физически удалить строку клиента из Excel по ID или username (в отдельном потоке)."""
        if not isinstance(identifiers, list):
            identifiers = [identifiers]
        targets = {str(i).strip().lstrip("@").lower() for i in identifiers if i}

        log.debug("[%s][EXCEL_DELETE] Ожидание блокировки excel_lock для удаления %s...", self.name, identifiers)
        async with self.excel_lock:
            log.debug("[%s][EXCEL_DELETE] Блокировка получена. Удаление строк из Excel в отдельном потоке...", self.name)

            def _sync_delete() -> None:
                try:
                    wb = load_workbook(self.excel_file)
                    try:
                        ws = wb.active
                        headers = [cell.value for cell in ws[1]]
                        if "Username/ID" not in headers:
                            return
                        id_col = headers.index("Username/ID") + 1

                        for row_idx in range(ws.max_row, 1, -1):
                            cell_val = str(ws.cell(row=row_idx, column=id_col).value or "").strip()
                            if cell_val in targets or cell_val.lstrip("@").lower() in targets:
                                ws.delete_rows(row_idx)
                                log.info("[%s][EXCEL_DELETE] Удалена строка %d (значение '%s')", self.name, row_idx, cell_val)
                        wb.save(self.excel_file)
                    finally:
                        wb.close()
                except Exception as exc:
                    log.error("[%s] Ошибка удаления строки из Excel: %s", self.name, exc)

            await asyncio.to_thread(_sync_delete)
            log.debug("[%s][EXCEL_DELETE] Операция удаления завершена", self.name)

    # ── Вспомогательные функции истории и состояния чата ────────────────────

    def _append_to_history(self, user_id: int | str, role: str, content: str) -> None:
        """Добавить сообщение в историю чата и дописать в logs_chats/{user_id}.txt."""
        chat_data = self.active_chats.get(user_id) or self.active_chats.get(str(user_id))
        cleaned = content.strip()
        if chat_data:
            if "history" not in chat_data:
                chat_data["history"] = []
            if not chat_data["history"] or chat_data["history"][-1].get("content") != cleaned:
                chat_data["history"].append({"role": role, "content": cleaned})
                if len(chat_data["history"]) > MAX_HISTORY_MESSAGES * 2:
                    chat_data["history"] = chat_data["history"][-MAX_HISTORY_MESSAGES:]

        if cleaned:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.append_chat_log(user_id, role, cleaned))
            except RuntimeError:
                _sync_append_chat_log(user_id, role, cleaned)

    async def _fetch_tg_chat_history(self, user_id: int | str, limit: int = 15) -> list[dict[str, str]]:
        """
        Выгружает из Telegram последние 10-15 сообщений чата и упаковывает в хронологический
        список словарей [{'role': 'user' | 'assistant', 'content': 'текст'}].
        """
        chat_data = self.active_chats.get(user_id) or self.active_chats.get(str(user_id), {})
        existing = chat_data.get("history", [])

        # Приводим user_id к корректному типу для Pyrogram
        str_id = str(user_id).strip()
        peer: int | str = int(str_id) if str_id.lstrip("-").isdigit() else str_id
        if isinstance(peer, str) and peer.startswith("@"):
            peer = peer.lstrip("@")

        if hasattr(self, "client") and self.client and getattr(self.client, "is_connected", False):
            try:
                tg_messages: list[dict[str, str]] = []
                # get_chat_history возвращает сообщения от самых новых к старым
                async for m in self.client.get_chat_history(peer, limit=limit):
                    text = (m.text or m.caption or "").strip()
                    if not text:
                        if m.photo:
                            text = "[Фото]"
                        elif m.video:
                            text = "[Видео]"
                        elif m.animation:
                            text = "[GIF]"
                        elif m.voice:
                            text = "[Голосовое сообщение]"
                        elif m.video_note:
                            text = "[Видеосообщение]"
                        elif m.sticker:
                            text = f"[Стикер {m.sticker.emoji or ''}]".strip()
                        elif m.audio:
                            text = "[Аудио]"
                        elif m.document:
                            text = "[Документ]"
                        else:
                            continue
                    elif m.photo or m.video or m.animation or m.voice or m.document:
                        tag = "[Фото] " if m.photo else "[Видео] " if m.video else "[GIF] " if m.animation else "[Голосовое] " if m.voice else ""
                        text = f"{tag}{text}"

                    if text.lower() in ("!старт", "!стоп", "бот офф", "bot off", "бот онн", "bot on"):
                        continue
                    if m.from_user and m.from_user.username == "leomatchbot":
                        continue

                    # Если outgoing=True или отправитель равен me — это assistant, иначе user
                    is_me = bool(
                        getattr(m, "outgoing", False)
                        or (m.from_user and (getattr(m.from_user, "is_self", False) or m.from_user.id == self.my_id))
                    )
                    role = "assistant" if is_me else "user"
                    tg_messages.append({"role": role, "content": text})

                if tg_messages:
                    tg_messages.reverse()  # Хронологический порядок: от старых к самым новым
                    if chat_data:
                        chat_data["history"] = tg_messages
                    log.info(
                        "[%s][HISTORY] Успешно выгружено %d сообщений из истории Telegram для %s",
                        self.name, len(tg_messages), user_id
                    )
                    return tg_messages
            except Exception as exc:
                log.warning(
                    "[%s][HISTORY] Не удалось выгрузить историю из Telegram для %s: %s. Используем локальную память",
                    self.name, user_id, exc
                )

        return existing[-limit:]

    _sync_chat_history_from_tg = _fetch_tg_chat_history  # Алиас для обратной совместимости

    def _update_chat_state(self, user_id: int | str, *, is_me: bool) -> None:
        chat_data = self.active_chats.get(user_id) or self.active_chats.get(str(user_id))
        if chat_data:
            chat_data["last_msg_time"] = asyncio.get_event_loop().time()
            chat_data["last_msg_is_me"] = is_me

    async def _handle_update_marker(self, user_id: int | str, data_str: str) -> None:
        """Обновить поля Excel по маркеру ОБНОВИТЬ_ДАННЫЕ."""
        log.info("[%s][MARKER] Обработка маркера данных: '%s' для user_id=%s", self.name, data_str, user_id)
        chat_data = self.active_chats.get(user_id) or self.active_chats.get(str(user_id), {})
        username = chat_data.get("username", str(user_id))
        for pattern, field in _PROFILE_MARKER_FIELDS:
            m = re.search(pattern, data_str, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if field == "Имя":
                    if is_valid_name(val):
                        log.info("[%s][MARKER] Обновление имени клиента на '%s'...", self.name, val)
                        await self.excel_update_field(str(user_id), field, val)
                        if username and username != str(user_id):
                            await self.excel_update_field(username, field, val)
                        if user_id in self.active_chats:
                            self.active_chats[user_id]["name"] = val
                        if str(user_id) in self.active_chats:
                            self.active_chats[str(user_id)]["name"] = val
                else:
                    log.info("[%s][MARKER] Обновление поля '%s' -> '%s'...", self.name, field, val)
                    await self.excel_update_field(username, field, val)
                    if str(user_id) != username:
                        await self.excel_update_field(str(user_id), field, val)
        log.info("[%s][MARKER] Обработка маркера данных для %s завершена", self.name, user_id)

    async def get_missing_profile_slots(
        self,
        chat_id: int | str,
        client_row: dict | None = None,
    ) -> list[str]:
        """
        Слоты профиля, которых сейчас нет ни в Excel, ни в памяти Qdrant этого чата.
        Порядок: имя, возраст, город, увлечение. Пустые — None, пустая строка или отсутствие поля.
        """
        chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
        username = chat_data.get("username")
        row = client_row
        if row is None:
            row = await self.excel_read_client(chat_id)
            if not row and username:
                row = await self.excel_read_client(username)

        merged: dict = dict(row or {})
        live_name = chat_data.get("name")
        if _profile_cell_empty(merged.get("Имя"), name_field=True) and is_valid_name(live_name):
            merged["Имя"] = live_name

        if missing_slots_from_profile(merged):
            qdrant_facts = await collect_profile_from_qdrant(chat_id)
            for column, value in qdrant_facts.items():
                if _profile_cell_empty(merged.get(column), name_field=(column == "Имя")) and not _profile_cell_empty(
                    value, name_field=(column == "Имя")
                ):
                    merged[column] = value

        missing = missing_slots_from_profile(merged)
        log.info(
            "[%s][SLOTS] Чат %s: пустые слоты — %s",
            self.name,
            chat_id,
            ", ".join(missing) if missing else "нет",
        )
        return missing

    # ── Генерация ответа ИИ (локальная Ollama, без блокировки event loop) ────

    async def ask_ai(
        self,
        user_id: int | str,
        user_text: str,
        message: Message | None = None,
        temperature: float = OLLAMA_TEMPERATURE,
    ) -> str:
        """
        Генерация ответа через локальную Ollama (модель vanya_q5).
        Контекст собеседника берётся из Excel, Qdrant и скользящего окна переписки.
        Отмена задачи не превращается в пустую строку: Debouncer ловит CancelledError.
        """
        log.info("[%s][AI] Запрос в Ollama для user_id=%s. Текст: '%s'", self.name, user_id, user_text[:60].replace("\n", " "))
        chat_data = self.active_chats.get(user_id) or self.active_chats.get(str(user_id), {})

        username = chat_data.get("username")
        name = chat_data.get("name")

        log.debug("[%s][AI] Загрузка контекста клиента %s из Excel...", self.name, user_id)
        client_row = await self.excel_read_client(user_id)
        if not client_row and username:
            client_row = await self.excel_read_client(username)
        if not name and client_row:
            name = client_row.get("Имя")
        if not name or not is_valid_name(name):
            name = "Неизвестно"
        log.debug("[%s][AI] Контекст: имя='%s', найдено в Excel=%s", self.name, name, bool(client_row))

        # 1. Системный промпт с динамическим временем и блоком "Контекст собеседника" из Excel
        system_prompt = _build_system_prompt(client_row, name=name)

        # 1.1. RAG: похожие прошлые пары этой девушки из Qdrant — в начало системного контекста
        memory_block = await recall_dialog_memories(user_id, str(user_text))
        if memory_block:
            system_prompt = f"{memory_block}\n\n{system_prompt}"

        # 1.2. Одна тактическая цель: выведать ровно один пустой слот профиля
        missing_slots = await self.get_missing_profile_slots(user_id, client_row=client_row)
        if missing_slots:
            slot = missing_slots[0]
            system_prompt = f"{system_prompt}\n\n{build_slot_goal(slot)}"
            log.info("[%s][SLOTS] Чат %s: тактическая цель — %s", self.name, user_id, slot)

        # Языковой замок остаётся первой и последней фразой, даже после RAG и цели слота.
        system_prompt = _seal_russian_prompt(system_prompt)

        # 2. Скользящее окно контекста: считываем ТОЛЬКО последние 10-15 сообщений из logs_chats/{user_id}.txt
        history = await self.read_chat_log_window(user_id, limit=15)
        if not history:
            # Фолбек: если файл лога еще пуст, выгружаем историю из Telegram и сохраняем в файл лога
            history = await self._fetch_tg_chat_history(user_id, limit=15)
            for h in history:
                await self.append_chat_log(user_id, h.get("role", "user"), h.get("content", ""))

        cleaned_user_text = str(user_text).strip()
        effective_user_text = cleaned_user_text

        # 2.1. Обработка Reply: если входящее сообщение является ответом на другое сообщение
        if message:
            replied_msg = getattr(message, "reply_to_message", None)
            replied_id = getattr(message, "reply_to_message_id", None)
            if not replied_msg and replied_id:
                try:
                    str_id = str(user_id).strip()
                    peer: int | str = int(str_id) if str_id.lstrip("-").isdigit() else str_id
                    replied_msg = await self.client.get_messages(peer, replied_id)
                except Exception as fetch_exc:
                    log.debug("[%s][AI] Не удалось получить сообщение по reply_to_message_id=%s: %s", self.name, replied_id, fetch_exc)

            if replied_msg:
                replied_text = (replied_msg.text or replied_msg.caption or "").strip()
                if not replied_text:
                    if replied_msg.photo:
                        replied_text = "[Фото]"
                    elif replied_msg.voice:
                        replied_text = "[Голосовое сообщение]"
                    elif replied_msg.video or replied_msg.video_note:
                        replied_text = "[Видео]"
                    elif replied_msg.sticker:
                        replied_text = f"[Стикер {replied_msg.sticker.emoji or ''}]".strip()
                    else:
                        replied_text = "[Медиа]"

                is_reply_to_vanya = (
                    getattr(replied_msg, "outgoing", False)
                    or (replied_msg.from_user and replied_msg.from_user.id == self.my_id)
                )
                sender_label = "сообщение Вани" if is_reply_to_vanya else "сообщение собеседника"
                effective_user_text = f'[Ответ на {sender_label}: "{replied_text}"] Входящее сообщение девушки: "{cleaned_user_text}"'
                log.info(
                    "[%s][AI] Обнаружен Reply от девушки на %s! Сформирован контекст для Ollama: '%s'",
                    self.name, sender_label, effective_user_text[:120]
                )

        # Добавляем сообщения из скользящего окна текстового файла
        dialog_history: list[dict[str, str]] = []
        for msg in history:
            msg_role = str(msg.get("role", "user"))
            msg_content = msg.get("content")
            if msg_content and isinstance(msg_content, str) and msg_content.strip():
                dialog_history.append({"role": msg_role, "content": msg_content.strip()})

        # Склеенная мысль уже лежит в логе отдельными репликами — убираем хвост, чтобы не дублировать её
        _collapse_trailing_user_burst(dialog_history, cleaned_user_text)

        # Убеждаемся, что текущее сообщение девушки с контекстом ответа является САМЫМ ПОСЛЕДНИМ в массиве
        if dialog_history and dialog_history[-1].get("role") == "user" and dialog_history[-1].get("content") == cleaned_user_text:
            dialog_history[-1]["content"] = effective_user_text
        elif not dialog_history or dialog_history[-1].get("role") != "user" or dialog_history[-1].get("content") != effective_user_text:
            dialog_history.append({"role": "user", "content": effective_user_text})

        # 3. Собираем messages через prepare_messages_for_chat_template(history)
        messages = prepare_messages_for_chat_template(dialog_history, system_prompt=system_prompt)

        log.info(
            "[%s][AI] Сформирован контекст для генерации: всего %d сообщений (диалоговых реплик: %d). Последнее: [%s] '%s'",
            self.name, len(messages), len(messages) - 1, messages[-1]["role"], messages[-1]["content"][:60]
        )

        # 4. Асинхронный запрос в Ollama. Event loop Pyrogram в это время свободен.
        client = AsyncClient(host=os.getenv("OLLAMA_HOST", OLLAMA_HOST))
        try:
            log.info("[%s][AI] Запрос в Ollama, модель %s", self.name, os.getenv("OLLAMA_MODEL", OLLAMA_MODEL))
            response = await client.chat(
                model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
                messages=messages,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repeat_penalty": OLLAMA_REPEAT_PENALTY,
                    "num_predict": 96,
                },
            )
            ai_text = _ollama_message_text(response)
            log.info(
                "[%s][AI] Успешный ответ от Ollama для %s (длина: %d симв.): '%s'",
                self.name, user_id, len(ai_text), ai_text[:60]
            )
            return ai_text
        except asyncio.CancelledError:
            log.info("[%s][AI] Генерация для %s отменена: собеседница дописала мысль", self.name, user_id)
            raise
        except Exception as e:
            log.error("[%s] Критическая ошибка запроса в Ollama: %s", self.name, e, exc_info=True)
            print(f"[{self.name}][DEBUG ИИ] Критическая ошибка генерации: {e}")
            return ""
        finally:
            try:
                await client.close()
            except Exception:
                log.debug("[%s][AI] Клиент Ollama уже закрыт", self.name)

    async def generate_text(
        self,
        user_id: int | str,
        user_text: str,
        message: Message | None = None,
        temperature: float = OLLAMA_TEMPERATURE,
    ) -> str:
        """Алиас для ask_ai: генерация текста через локальную Ollama."""
        return await self.ask_ai(user_id, user_text, message=message, temperature=temperature)

    async def _guard_russian_reply(
        self,
        chat_id: int | str,
        reply: str,
        message: Message | None,
        user_text: str | None,
    ) -> str:
        """
        Перед отправкой в Telegram: если в ответе есть иероглиф, один раз
        перегенерировать его с температурой на 0.1 ниже. Повторный сбой —
        вырезать иероглифы и оставить русскую часть.
        """
        if not _contains_han(reply):
            return reply

        reroll_temp = max(0.1, OLLAMA_TEMPERATURE - 0.1)
        log.warning(
            "[%s][AI] В ответе для %s найдены иероглифы, повтор генерации с температурой %.2f",
            self.name, chat_id, reroll_temp,
        )
        source = (user_text or "").strip()
        if not source and message is not None:
            source = (message.text or message.caption or "").strip()

        rerolled = ""
        if source:
            rerolled = await self.ask_ai(chat_id, source, message=message, temperature=reroll_temp)
            rerolled = re.sub(r"<think>[\s\S]*?</think>", "", rerolled or "").strip()
            rerolled = rerolled.replace(MARKER_INVITATION, "").replace(INVITE_MARKER, "").strip()
            update_match = re.search(r"ОБНОВИТЬ_ДАННЫЕ:\s*([^\n]+)", rerolled)
            if update_match:
                await self._handle_update_marker(chat_id, update_match.group(1).strip())
                rerolled = rerolled.replace(update_match.group(0), "").strip()

        if rerolled and not _contains_han(rerolled):
            log.info("[%s][AI] Повторная генерация для %s пришла на русском", self.name, chat_id)
            return rerolled

        cleaned = _strip_han(rerolled) or _strip_han(reply)
        log.warning(
            "[%s][AI] Повторный сбой языка для %s, иероглифы вырезаны: '%s'",
            self.name, chat_id, cleaned[:60],
        )
        return cleaned

    async def _remember_successful_reply(
        self,
        chat_id: int | str,
        message: Message | None,
        assistant_reply: str,
        user_text: str | None = None,
    ) -> None:
        """После успешной отправки ответа векторизует реплику девушки и пишет пару в Qdrant."""
        user_message = (user_text or "").strip()
        if not user_message and message is not None:
            user_message = (message.text or message.caption or "").strip()
        if not user_message:
            return
        await save_dialog_memory(chat_id, user_message, assistant_reply)

    async def _hold_typing(self, chat_id: int | str, seconds: float) -> None:
        """Держит статус печати заданное время, обновляя его каждые 4 секунды."""
        elapsed = 0.0
        while elapsed < seconds:
            try:
                await self.client.send_chat_action(chat_id, enums.ChatAction.TYPING)
            except Exception as exc:
                log.debug("[%s][PROCESS_REPLY] Не удалось показать TYPING в %s: %s", self.name, chat_id, exc)
            step = min(4.0, seconds - elapsed)
            await asyncio.sleep(step)
            elapsed += step

    async def _send_reply_chunks(
        self,
        chat_id: int | str,
        chunks: list[str],
        reply_to_msg_id: int | None,
    ) -> None:
        """
        Шлёт длинный ответ порциями. Между кусками печать гаснет на 1.5–3 секунды.
        Новое сообщение девушки отменяет этот цикл через _chunk_tasks.
        """
        task_key = chat_id
        self._chunk_tasks[task_key] = asyncio.current_task()
        try:
            for index, raw_chunk in enumerate(chunks):
                chunk = _drop_trailing_sentence_marks(raw_chunk)
                if not chunk:
                    log.info("[%s][PROCESS_REPLY] Чат %s: кусок %d пуст после очистки, пропуск", self.name, chat_id, index + 1)
                    continue
                typing_for = min(5.0, max(2.0, len(chunk) * 0.05))
                log.info(
                    "[%s][PROCESS_REPLY] Чат %s: печать %.1f с, кусок %d/%d",
                    self.name, chat_id, typing_for, index + 1, len(chunks),
                )
                await self._hold_typing(chat_id, typing_for)
                try:
                    if index == 0 and reply_to_msg_id:
                        await self.client.send_message(chat_id, chunk, reply_to_message_id=reply_to_msg_id)
                    else:
                        await self.client.send_message(chat_id, chunk)
                except Exception as send_err:
                    log.warning(
                        "[%s][PROCESS_REPLY] Ошибка отправки куска %d (%s), повтор без reply: %s",
                        self.name, index + 1, reply_to_msg_id, send_err,
                    )
                    await self.client.send_message(chat_id, chunk)
                self._append_to_history(chat_id, "assistant", chunk)
                self._update_chat_state(chat_id, is_me=True)
                if index < len(chunks) - 1:
                    try:
                        await self.client.send_chat_action(chat_id, enums.ChatAction.CANCEL)
                    except Exception as exc:
                        log.debug("[%s][PROCESS_REPLY] Не удалось снять TYPING в %s: %s", self.name, chat_id, exc)
                    breath = random.uniform(1.5, 3.0)
                    log.info("[%s][PROCESS_REPLY] Чат %s: пауза %.1f с перед следующей мыслью", self.name, chat_id, breath)
                    await asyncio.sleep(breath)
        except asyncio.CancelledError:
            try:
                await self.client.send_chat_action(chat_id, enums.ChatAction.CANCEL)
            except Exception:
                pass
            raise
        finally:
            if self._chunk_tasks.get(task_key) is asyncio.current_task():
                self._chunk_tasks.pop(task_key, None)

    async def process_ai_reply(
        self,
        chat_id: int | str,
        reply: str,
        message: Message | None = None,
        user_text: str | None = None,
    ) -> None:
        """Обработка ответа ИИ: парсинг маркеров, отправка и обновление Excel."""
        log.debug("[%s][PROCESS_REPLY] Начало обработки ответа ИИ для чата %s...", self.name, chat_id)
        reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()

        # Защита от спама скобочек: удаляем множественные скобочки в конце (например, )) или )))
        reply = re.sub(r"\s*\){2,}$", "", reply).strip()

        # Защита от повторного приветствия посреди уже идущего диалога
        chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
        history = chat_data.get("history", [])
        if len(history) > 2:
            last_user_msg = ""
            for h in reversed(history):
                if h.get("role") == "user":
                    last_user_msg = h.get("content", "").lower()
                    break
            greetings = ("привет", "приветик", "хай", "салют", "здарова", "здравствуй", "добрый день", "доброе утро", "добрый вечер")
            girl_greeted = any(g in last_user_msg for g in greetings)
            if not girl_greeted:
                clean_reply = re.sub(r"^(Привет|Приветик|Здарова|Хай)[!,\.]\s*", "", reply, flags=re.IGNORECASE).strip()
                if clean_reply:
                    reply = clean_reply[0].upper() + clean_reply[1:]

        has_invite = (MARKER_INVITATION in reply) or (INVITE_MARKER in reply)
        reply = reply.replace(MARKER_INVITATION, "").replace(INVITE_MARKER, "").strip()

        update_match = re.search(r"ОБНОВИТЬ_ДАННЫЕ:\s*([^\n]+)", reply)
        if update_match:
            log.info("[%s][PROCESS_REPLY] Найден маркер обновления данных для чата %s", self.name, chat_id)
            await self._handle_update_marker(chat_id, update_match.group(1).strip())
            reply = reply.replace(update_match.group(0), "").strip()

        # Определяем, нужно ли визуально линковать ответ через reply_to_message_id.
        # Бот отвечает обычным текстом, если девушка написала простое сообщение.
        # Но если входящее сообщение от девушки было ответом на старое сообщение бота (или контекст требует привязки),
        # бот отправляет свое сообщение с параметром reply_to_message_id=message.id
        should_reply_link = False
        reply_to_msg_id: int | None = None
        if message and getattr(message, "id", None):
            replied_to = getattr(message, "reply_to_message", None)
            replied_to_id = getattr(message, "reply_to_message_id", None)
            if replied_to or replied_to_id:
                should_reply_link = True
                reply_to_msg_id = message.id
                log.info(
                    "[%s][PROCESS_REPLY] Входящее сообщение %d является Reply — ответ будет привязан к message.id=%d",
                    self.name, message.id, reply_to_msg_id
                )
            else:
                log.debug("[%s][PROCESS_REPLY] Простое сообщение без Reply — отправка обычным текстом без линкования", self.name)

        if has_invite:
            invite_text = REPLY_ON_INVITATION
            log.info("[%s][PROCESS_REPLY] Обнаружено приглашение от клиента %s! Отправка нейтрального ответа...", self.name, chat_id)
            try:
                if should_reply_link and reply_to_msg_id:
                    await self.client.send_message(chat_id, invite_text, reply_to_message_id=reply_to_msg_id)
                else:
                    await self.client.send_message(chat_id, invite_text)
            except Exception as send_err:
                log.warning("[%s][PROCESS_REPLY] Ошибка отправки с reply_to_message_id (%s), пробуем обычный send_message: %s", self.name, reply_to_msg_id, send_err)
                await self.client.send_message(chat_id, invite_text)
            self._append_to_history(chat_id, "assistant", invite_text)
            self._update_chat_state(chat_id, is_me=True)

            chat_data = self.active_chats.get(chat_id, {})
            name = chat_data.get("name", "Неизвестно")
            if not is_valid_name(name):
                name = "Неизвестно"
            username = chat_data.get("username", str(chat_id))
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log.info("[%s][PROCESS_REPLY] Добавление записи о приглашении в Excel (%s, %s)...", self.name, name, username)
            await self.excel_append_row(name, username, date_str)
            log.info("[%s][PROCESS_REPLY] Отправка уведомления о приглашении в Избранное...", self.name)
            await self.client.send_message(
                "me",
                f"🔔 [{self.name}] Запись внесена в Excel. Клиент: {name} ({username})",
            )
            log.info("[%s][PROCESS_REPLY] Приглашение успешно зафиксировано для %s (%s)", self.name, name, username)
            await self._remember_successful_reply(chat_id, message, invite_text, user_text=user_text)
            return

        reply = await self._guard_russian_reply(chat_id, reply, message, user_text)

        if reply:
            chunks = split_reply_chunks(reply)
            log.debug(
                "[%s][PROCESS_REPLY] Отправка клиенту %s: %d частей (reply_link=%s)",
                self.name, chat_id, len(chunks), should_reply_link,
            )
            try:
                if len(chunks) > 1:
                    await self._send_reply_chunks(chat_id, chunks, reply_to_msg_id if should_reply_link else None)
                else:
                    text = _drop_trailing_sentence_marks(chunks[0] if chunks else reply)
                    if not text:
                        log.info("[%s][PROCESS_REPLY] Чат %s: после очистки финала отправлять нечего", self.name, chat_id)
                        return
                    try:
                        if should_reply_link and reply_to_msg_id:
                            await self.client.send_message(chat_id, text, reply_to_message_id=reply_to_msg_id)
                        else:
                            await self.client.send_message(chat_id, text)
                    except Exception as send_err:
                        log.warning("[%s][PROCESS_REPLY] Ошибка отправки с reply_to_message_id (%s), пробуем обычный send_message: %s", self.name, reply_to_msg_id, send_err)
                        await self.client.send_message(chat_id, text)
                    self._append_to_history(chat_id, "assistant", text)
                    self._update_chat_state(chat_id, is_me=True)
            except asyncio.CancelledError:
                log.info("[%s][PROCESS_REPLY] Отправка порций клиенту %s прервана новым сообщением", self.name, chat_id)
                raise
            log.info(
                "[%s][PROCESS_REPLY] Ответ успешно отправлен клиенту %s (%d сообщ.): '%s'",
                self.name, chat_id, len(chunks), reply[:60].replace("\n", " ")
            )
            await self._remember_successful_reply(chat_id, message, reply, user_text=user_text)

    # ── Логика задержки, прочтения истории и набора текста ───────────────────

    async def _delayed_reply(self, message: Message, chat_id: int, text: str) -> None:
        """
        Умная динамическая задержка и реалистичные статусы присутствия (Стелс-режим генерации):
        1. При получении сообщения бот НЕ отмечает его прочитанным и НЕ заходит в сеть (не шлет статус онлайн/typing).
        2. Сразу после входящего события запускается асинхронная пауза await asyncio.sleep(random.randint(1, 5))
           (имитация того, что человек не смотрит в экран неотрывно).
        3. Параллельно со стелс-паузой запускается тяжелая генерация текста моделью Qwen (ask_ai / generate_text),
           чтобы время паузы не пропадало впустую.
        4. Только ПОСЛЕ того как прошел этот начальный sleep (и пока модель параллельно думает/генерирует),
           бот отмечает сообщение прочитанным (read_chat_history), заходит в сеть и запускает фоновую задачу
           (asyncio.create_task) циклического обновления статуса enums.ChatAction.TYPING каждые 4 секунды.
        5. Итоговая отправка сообщения происходит по завершении генерации, с учетом общей динамической
           задержки (7-15 сек). Если генерация заняла меньше времени, бот досыпает остаток timing-паузы в статусе TYPING.
        """
        chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
        if chat_data.get("last_msg_is_me"):
            log.info("[%s][REPLY_WORKER] Последнее сообщение от меня — пропускаем ИИ-ответ для %d", self.name, chat_id)
            return

        if self.is_ai_frozen():
            log.info("[%s][REPLY_WORKER] ИИ заморожен (сна/выкл) — отмена ответа для %d", self.name, chat_id)
            return

        # Фиксируем целевую "человеческую задержку" (7-15 с) и стартовое время
        target_delay = random.uniform(7.0, 15.0)
        start_time = time.time()

        # Запускаем тяжелую генерацию текста моделью Qwen параллельно, чтобы время стелс-паузы не тратилось впустую
        generate_task = asyncio.create_task(self.ask_ai(chat_id, text, message=message))
        typing_task: asyncio.Task | None = None
        typing_stop_event = asyncio.Event()

        try:
            # Шаг 1. Стелс-пауза (1-5 с): бот оффлайн, сообщение НЕ прочитано, send_chat_action НЕ вызывается
            stealth_delay = random.randint(1, 5)
            log.info(
                "[%s][REPLY_WORKER] Чат %d: Стелс-пауза %d с (оффлайн, не прочитано). "
                "Параллельно запущена генерация текста Qwen (целевая общая задержка: %.1f с)...",
                self.name, chat_id, stealth_delay, target_delay
            )
            await asyncio.sleep(stealth_delay)

            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                log.info("[%s][REPLY_WORKER] Состояние чата %d изменилось за время стелс-паузы — отмена ответа", self.name, chat_id)
                generate_task.cancel()
                return

            # Шаг 2. Только ПОСЛЕ стелс-паузы входим в сеть: читаем сообщение и шлем первичный TYPING
            log.debug("[%s][REPLY_WORKER] Чат %d: Стелс-пауза истекла. Вход в сеть: прочтение сообщения и статус TYPING...", self.name, chat_id)
            try:
                await self.client.read_chat_history(chat_id)
            except Exception as exc:
                log.debug("[%s][REPLY_WORKER] Ошибка read_chat_history для %d: %s", self.name, chat_id, exc)

            try:
                await self.client.send_chat_action(chat_id, enums.ChatAction.TYPING)
            except Exception as exc:
                log.debug("[%s][REPLY_WORKER] Ошибка send_chat_action TYPING для %d: %s", self.name, chat_id, exc)

            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                generate_task.cancel()
                return

            # Шаг 3. Запускаем фоновую задачу циклического обновления статуса TYPING каждые 4 секунды
            async def _keep_typing():
                while not typing_stop_event.is_set():
                    try:
                        await asyncio.wait_for(typing_stop_event.wait(), timeout=4.0)
                    except asyncio.TimeoutError:
                        pass
                    if not typing_stop_event.is_set():
                        try:
                            await self.client.send_chat_action(chat_id, enums.ChatAction.TYPING)
                        except Exception as exc:
                            log.debug("[%s][REPLY_WORKER] Ошибка send_chat_action TYPING в %d: %s", self.name, chat_id, exc)

            typing_task = asyncio.create_task(_keep_typing())

            # Шаг 4. Дожидаемся окончания генерации текста моделью Qwen
            raw_reply = await generate_task
            if not raw_reply:
                log.warning("[%s][REPLY_WORKER] Чат %d: ИИ вернул пустой ответ", self.name, chat_id)
                return

            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                log.info("[%s][REPLY_WORKER] Во время генерации ИИ изменилось состояние — отмена для %d", self.name, chat_id)
                return

            # Шаг 5. Динамическая задержка: если ИИ сгенерировал быстрее целевой задержки (7-15 с),
            # досыпаем остаток времени с активным статусом TYPING
            elapsed = time.time() - start_time
            remaining_delay = target_delay - elapsed
            if remaining_delay > 0:
                log.info(
                    "[%s][REPLY_WORKER] Чат %d: ИИ готов за %.2f с. Досыпаем остаток паузы %.2f с (TYPING активен)...",
                    self.name, chat_id, elapsed, remaining_delay
                )
                await asyncio.sleep(remaining_delay)
            else:
                log.info(
                    "[%s][REPLY_WORKER] Чат %d: ИИ генерировал %.2f с (дольше целевых %.1f с). Отправка МГНОВЕННО!",
                    self.name, chat_id, elapsed, target_delay
                )

            # Проверяем еще раз после сна
            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                log.info("[%s][REPLY_WORKER] Во время остатка паузы изменилось состояние — отмена для %d", self.name, chat_id)
                return

            # Останавливаем статус 'печатает' перед отправкой сообщения
            typing_stop_event.set()
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except (asyncio.CancelledError, Exception):
                    pass
                typing_task = None

            # Шаг 6. Отправка ответа клиенту
            log.info("[%s][REPLY_WORKER] Чат %d: Отправка ответа клиенту через process_ai_reply...", self.name, chat_id)
            await self.process_ai_reply(chat_id, raw_reply, message=message, user_text=text)
            log.info("[%s][REPLY_WORKER] Чат %d: Цикл ответа полностью завершен", self.name, chat_id)

        except asyncio.CancelledError:
            log.debug("[%s][REPLY_WORKER] Чат %d: Задача ответа отменена (получено более новое сообщение)", self.name, chat_id)
            raise
        except Exception as exc:
            log.error("[%s][REPLY_WORKER] Ошибка в _delayed_reply для чата %d: %s", self.name, chat_id, exc, exc_info=True)
        finally:
            typing_stop_event.set()
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except (asyncio.CancelledError, Exception):
                    pass
            if generate_task and not generate_task.done():
                generate_task.cancel()

    async def _delayed_media_reply(self, message: Message, chat_id: int) -> None:
        """
        Обработка входящих медиафайлов (фото, видео, анимации, голосовые сообщения, стикеры):
        1. Стелс-пауза 1–5 секунд перед появлением в сети.
        2. Затем заходим в сеть, отмечаем прочитанным и включаем статус TYPING.
        3. Имитирует набор текста со случайной задержкой 4–8 секунд и активным статусом ChatAction.TYPING.
        4. Отправляет одну из случайных фраз о плохом интернете / занятости.
        """
        try:
            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me"):
                log.info("[%s][MEDIA_REPLY] Последнее сообщение от меня — отмена ответа на медиа для %d", self.name, chat_id)
                return

            if self.is_ai_frozen():
                log.info("[%s][MEDIA_REPLY] ИИ заморожен (сна/выкл) — отмена ответа на медиа для %d", self.name, chat_id)
                return

            # Шаг 1. Стелс-пауза 1–5 секунд перед входом в сеть
            init_delay = random.randint(1, 5)
            await asyncio.sleep(init_delay)

            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                return

            # Вход в сеть, отметка медиа прочитанным и первичный статус набора текста (TYPING)
            try:
                await self.client.read_chat_history(chat_id)
                log.debug("[%s][MEDIA_REPLY] Чат %d: медиа отмечено прочитанным", self.name, chat_id)
            except Exception as read_exc:
                log.debug("[%s][MEDIA_REPLY] Не удалось отметить медиа прочитанным в чате %d: %s", self.name, chat_id, read_exc)

            try:
                await self.client.send_chat_action(chat_id, enums.ChatAction.TYPING)
            except Exception as exc:
                log.debug("[%s][MEDIA_REPLY] Ошибка send_chat_action TYPING для %d: %s", self.name, chat_id, exc)

            chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
            if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                return

            # Выбираем одну случайную фразу
            reply_text = random.choice(MEDIA_RESPONSES) if MEDIA_RESPONSES else "блин ща интернет тупит, не грузит чето( позже гляну"

            # Случайная пауза набора текста 4–8 секунд
            total_delay = random.uniform(4.0, 8.0)
            log.info(
                "[%s][MEDIA_REPLY] Чат %d: пауза %.1f с на набор ответа на медиа, включение TYPING...",
                self.name, chat_id, total_delay
            )

            typing_stop_event = asyncio.Event()

            async def _keep_media_typing():
                while not typing_stop_event.is_set():
                    try:
                        await asyncio.wait_for(typing_stop_event.wait(), timeout=4.0)
                    except asyncio.TimeoutError:
                        pass
                    if not typing_stop_event.is_set():
                        try:
                            await self.client.send_chat_action(chat_id, enums.ChatAction.TYPING)
                        except Exception as exc:
                            log.debug("[%s][MEDIA_REPLY] Ошибка send_chat_action TYPING: %s", self.name, exc)

            typing_task = asyncio.create_task(_keep_media_typing())
            try:
                await asyncio.sleep(total_delay)
                chat_data = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id), {})
                if chat_data.get("last_msg_is_me") or self.is_ai_frozen():
                    return
            finally:
                typing_stop_event.set()
                typing_task.cancel()
                try:
                    await typing_task
                except (asyncio.CancelledError, Exception):
                    pass

            # Отправка ответа клиенту
            log.info("[%s][MEDIA_REPLY] Чат %d: отправка ответа на медиа: '%s'", self.name, chat_id, reply_text)
            is_media_reply = bool(
                message
                and getattr(message, "id", None)
                and (getattr(message, "reply_to_message", None) or getattr(message, "reply_to_message_id", None))
            )
            try:
                if is_media_reply:
                    await self.client.send_message(chat_id, reply_text, reply_to_message_id=message.id)
                else:
                    await self.client.send_message(chat_id, reply_text)
            except Exception as send_exc:
                log.warning("[%s][MEDIA_REPLY] Ошибка отправки (%s), пробуем send_message...", self.name, send_exc)
                await self.client.send_message(chat_id, reply_text)

            self._append_to_history(chat_id, "assistant", reply_text)
            self._update_chat_state(chat_id, is_me=True)
            log.info("[%s][MEDIA_REPLY] Чат %d: ответ на медиа успешно отправлен", self.name, chat_id)
        except asyncio.CancelledError:
            log.debug("[%s][MEDIA_REPLY] Чат %d: задача ответа на медиа отменена новым сообщением", self.name, chat_id)
            raise
        except Exception as exc:
            log.error("[%s][MEDIA_REPLY] Ошибка в _delayed_media_reply для чата %d: %s", self.name, chat_id, exc, exc_info=True)

    # ── Обработчики Pyrogram (Handlers) ─────────────────────────────────────

    async def handle_clicker_command(self, message: Message) -> None:
        """Управление автокликером Дайвинчика через команды (.clicker on / .clicker off)."""
        global AUTOCLICKER_ENABLED
        text = (message.text or message.caption or "").strip()
        cmd = text.lower()

        if cmd == ".clicker off":
            AUTOCLICKER_ENABLED = False
            log.info("[%s][CLICKER] Автокликер Дайвинчика ОТКЛЮЧЕН по команде пользователя", self.name)
            print("[INFO] Автокликер Дайвинчика успешно ОТКЛЮЧЕН")
            try:
                await message.reply_text("⏸️ Автокликер Дайвинчика успешно ОТКЛЮЧЕН")
            except Exception as exc:
                log.warning("[%s][CLICKER] Ошибка reply_text: %s, отправляем через send_message", self.name, exc)
                await self.client.send_message(message.chat.id, "⏸️ Автокликер Дайвинчика успешно ОТКЛЮЧЕН")
            return
        elif cmd == ".clicker on":
            AUTOCLICKER_ENABLED = True
            log.info("[%s][CLICKER] Автокликер Дайвинчика ВКЛЮЧЕН по команде пользователя", self.name)
            print("[INFO] Автокликер Дайвинчика успешно ВКЛЮЧЕН")
            try:
                await message.reply_text("▶️ Автокликер Дайвинчика успешно ВКЛЮЧЕН")
            except Exception as exc:
                log.warning("[%s][CLICKER] Ошибка reply_text: %s, отправляем через send_message", self.name, exc)
                await self.client.send_message(message.chat.id, "▶️ Автокликер Дайвинчика успешно ВКЛЮЧЕН")
            return

        message.continue_propagation()

    async def toggle_bot(self, message: Message) -> None:
        """Ручной переключатель режима (ON/OFF через Избранное)."""
        if message.chat.type != enums.ChatType.PRIVATE:
            return

        if message.chat.id != self.my_id:
            message.continue_propagation()
            return

        text = (message.text or message.caption or "").strip().lower()

        if text in ("бот офф", "bot off"):
            log.info("[%s][TOGGLE] Получена команда выключения бота", self.name)
            self.bot_active = False
            log.debug("[%s][TOGGLE] Отправка подтверждения в Избранное...", self.name)
            await self.client.send_message(
                "me", f"🤖 [{self.name}] Бот успешно выключен. Режим ручного управления"
            )
            log.info("[%s][TOGGLE] Бот выключен через Избранное", self.name)
            return
        elif text in ("бот онн", "bot on"):
            log.info("[%s][TOGGLE] Получена команда включения бота", self.name)
            self.bot_active = True
            log.debug("[%s][TOGGLE] Отправка подтверждения в Избранное...", self.name)
            await self.client.send_message(
                "me", f"🤖 [{self.name}] Бот включен. Система на связи 24/7"
            )
            log.info("[%s][TOGGLE] Бот включен через Избранное", self.name)
            return

        message.continue_propagation()

    async def manual_chat_control(self, message: Message) -> None:
        """Ручное управление чатами (!старт / !стоп)."""
        if message.chat.id == self.my_id:
            return

        text_val = (message.text or message.caption or "").strip()
        if text_val.lower() in ["!старт", "!стоп"]:
            cmd = text_val.lower()
            log.info("[%s][MANUAL_CTRL] Начало обработки команды '%s' в чате %d", self.name, cmd, message.chat.id)

            if cmd == "!старт":
                log.debug("[%s][MANUAL_CTRL] Шаг 1: Удаление сообщения команды '!старт'...", self.name)
                try:
                    await message.delete()
                    log.debug("[%s][MANUAL_CTRL] Шаг 1: Сообщение '!старт' удалено", self.name)
                except Exception as exc:
                    log.warning("[%s][MANUAL_CTRL] Не удалось удалить сообщение '!старт': %s", self.name, exc)

                chat_id = message.chat.id
                chat_id_str = str(chat_id)
                username = message.chat.username or ""
                name = "Неизвестно"

                log.debug("[%s][MANUAL_CTRL] Шаг 2: Получение информации о пользователе %d (get_users)...", self.name, chat_id)
                try:
                    peer_user = await self.client.get_users(chat_id)
                    raw_name = peer_user.first_name or ""
                    if not username and peer_user.username:
                        username = peer_user.username
                    if is_valid_name(raw_name):
                        name = raw_name
                    else:
                        name = "Неизвестно"
                    log.debug("[%s][MANUAL_CTRL] Шаг 2: Получены данные: имя='%s', username='%s'", self.name, name, username)
                except Exception as exc:
                    log.warning("[%s][MANUAL_CTRL] Ошибка получения данных пользователя %d: %s", self.name, chat_id, exc)

                if not is_valid_name(name):
                    name = "Неизвестно"

                await self.add_to_whitelist(
                    user_id=chat_id,
                    username=username,
                    name=name,
                    extra=f"Добавлен вручную через !старт (@{username})" if username else "Добавлен вручную через !старт",
                )
                self._update_chat_state(chat_id, is_me=True)

                # Сразу фоново подгружаем предыдущую историю диалога из Telegram, если общались ранее
                asyncio.create_task(self._sync_chat_history_from_tg(chat_id))

                log.debug("[%s][MANUAL_CTRL] Шаг 4: Отправка подтверждения в Избранное...", self.name)
                await self.client.send_message("me", f"✅ [{self.name}] Чат с {name} добавлен в белый список!")
                log.info("[%s][MANUAL_CTRL] !старт — чат %s (%d) успешно добавлен в белый список", self.name, name, chat_id)
                return

            elif cmd == "!стоп":
                log.debug("[%s][MANUAL_CTRL] Шаг 1: Удаление сообщения команды '!стоп'...", self.name)
                try:
                    await message.delete()
                    log.debug("[%s][MANUAL_CTRL] Шаг 1: Сообщение '!стоп' удалено", self.name)
                except Exception as exc:
                    log.warning("[%s][MANUAL_CTRL] Не удалось удалить сообщение '!стоп': %s", self.name, exc)

                chat_id = message.chat.id
                username = message.chat.username or ""
                name = "Клиент"

                log.debug("[%s][MANUAL_CTRL] Шаг 2: Получение данных пользователя %d...", self.name, chat_id)
                try:
                    peer_user = await self.client.get_users(chat_id)
                    name = peer_user.first_name or "Клиент"
                    if not username and peer_user.username:
                        username = peer_user.username
                    log.debug("[%s][MANUAL_CTRL] Шаг 2: Пользователь: %s (@%s)", self.name, name, username)
                except Exception as exc:
                    log.warning("[%s][MANUAL_CTRL] Ошибка получения данных пользователя %d: %s", self.name, chat_id, exc)

                await self.remove_from_whitelist(chat_id, username=username)

                log.debug("[%s][MANUAL_CTRL] Шаг 4: Отправка подтверждения в Избранное...", self.name)
                await self.client.send_message("me", f"🛑 [{self.name}] Чат с {name} удален из белого списка!")
                log.info("[%s][MANUAL_CTRL] !стоп — чат %s (%d) удален из белого списка и Excel", self.name, name, chat_id)
                return

        message.continue_propagation()

    async def track_my_messages(self, message: Message) -> None:
        """Отслеживание моих сообщений (заморозка ИИ на 15 минут)."""
        text = (message.text or message.caption or "").strip()
        if text.lower() in ("бот офф", "bot off", "бот онн", "bot on", "!старт", "!стоп", ".clicker off", ".clicker on"):
            return

        if message.chat.id == self.my_id:
            return

        peer_id = message.chat.id
        # Если я вручную написал новому пользователю в ЛС — автоматически добавляем в белый список
        if message.chat.type == enums.ChatType.PRIVATE and peer_id > 0 and peer_id not in _KNOWN_LEOMATCHBOT_IDS:
            if peer_id not in self.active_chats and str(peer_id) not in self.active_chats:
                raw_name = message.chat.first_name or ""
                name = raw_name if is_valid_name(raw_name) else "Неизвестно"
                log.info("[%s][TRACK] Мое исходящее сообщение новому пользователю %d. Добавление в белый список...", self.name, peer_id)
                await self.add_to_whitelist(
                    user_id=peer_id,
                    username=message.chat.username,
                    name=name,
                    extra="Исходящий диалог",
                )

        log.debug("[%s][TRACK] Мое исходящее сообщение в чат %d. Заморозка ИИ на 15 мин...", self.name, message.chat.id)
        self.activate_sleep_mode()
        if peer_id in self.active_chats:
            self._update_chat_state(peer_id, is_me=True)
            if text:
                self._append_to_history(peer_id, "assistant", text)

    async def on_leobot_message(self, message: Message) -> None:
        """Перехват сообщений от лид-бота Дайвинчика (leomatchbot)."""
        # Регистрируем ID бота Дайвинчика
        if message.from_user and message.from_user.id:
            _KNOWN_LEOMATCHBOT_IDS.add(message.from_user.id)
            GLOBAL_WHITELIST_IDS.add(message.from_user.id)
        if message.chat and message.chat.id:
            _KNOWN_LEOMATCHBOT_IDS.add(message.chat.id)
            GLOBAL_WHITELIST_IDS.add(message.chat.id)

        text = (message.text or message.caption or "").strip()
        text_lower = text.lower()

        # 1. Проверяем, является ли сообщение мэтчем / анкетой с контактом
        match_keywords = [
            "начинай общаться",
            "начинайте общаться",
            "взаимная симпатия",
            "понравилась твоя анкета",
            "кому-то понравилась",
            "есть взаимная",
            "общаться:",
            "общаться с",
        ]
        is_match = any(kw in text_lower for kw in match_keywords)

        # Проверка наличия ссылки на контакт пользователя в entities или кнопках
        has_user_link = False
        entities = (message.entities or []) + (getattr(message, "caption_entities", None) or [])
        for ent in entities:
            url = getattr(ent, "url", None) or ""
            if "tg://user?id=" in url or ("t.me/" in url and "leomatchbot" not in url.lower()):
                has_user_link = True
                break
            if ent.type == enums.MessageEntityType.MENTION:
                m_txt = text[ent.offset:ent.offset + ent.length].lstrip("@")
                if m_txt.lower() != "leomatchbot":
                    has_user_link = True
                    break

        if not has_user_link and message.reply_markup and getattr(message.reply_markup, "inline_keyboard", None):
            for row in message.reply_markup.inline_keyboard:
                for btn in row:
                    url = getattr(btn, "url", None) or ""
                    if "tg://user?id=" in url or ("t.me/" in url and "leomatchbot" not in url.lower()):
                        has_user_link = True
                        break
                if has_user_link:
                    break

        if is_match or has_user_link:
            log.info("[%s][LEOBOT] Перехвачено сообщение с мэтчем / контактом от leomatchbot", self.name)
            name = "Неизвестно"
            for pattern in [
                r"пользователю\s+([А-Яа-яЁёA-Za-z]+)",
                r"пользователь\s+([А-Яа-яЁёA-Za-z]+)",
                r"с\s+([А-Яа-яЁёA-Za-z]+),?\s+\d+",
                r"с\s+([А-Яа-яЁёA-Za-z]+)",
                r"общаться[:\s]+([А-Яа-яЁёA-Za-z]+)",
                r"^([А-Яа-яЁёA-Za-z]+),?\s+\d+",
            ]:
                m = re.search(pattern, text)
                if m:
                    cand = m.group(1).strip()
                    if is_valid_name(cand):
                        name = cand
                        break

            target_user_id: int | None = None
            target_username: str | None = None

            # Поиск в entities
            for ent in entities:
                url = getattr(ent, "url", None)
                if url:
                    uid_match = re.search(r"tg://user\?id=(\d+)", url)
                    uname_match = re.search(r"t\.me/([A-Za-z0-9_]+)", url)
                    if uid_match:
                        target_user_id = int(uid_match.group(1))
                        break
                    elif uname_match and uname_match.group(1).lower() != "leomatchbot":
                        target_username = uname_match.group(1)
                elif ent.type == enums.MessageEntityType.MENTION:
                    m_uname = text[ent.offset:ent.offset + ent.length].lstrip("@")
                    if m_uname.lower() != "leomatchbot":
                        target_username = m_uname

            # Поиск в кнопках inline_keyboard
            if not target_user_id and message.reply_markup and getattr(message.reply_markup, "inline_keyboard", None):
                for row in message.reply_markup.inline_keyboard:
                    for btn in row:
                        url = getattr(btn, "url", None) or ""
                        uid_match = re.search(r"tg://user\?id=(\d+)", url)
                        uname_match = re.search(r"t\.me/([A-Za-z0-9_]+)", url)
                        if uid_match:
                            target_user_id = int(uid_match.group(1))
                            break
                        elif uname_match and uname_match.group(1).lower() != "leomatchbot":
                            target_username = uname_match.group(1)
                            break
                    if target_user_id:
                        break

            # Поиск по тексту напрямую через regex
            if not target_user_id and not target_username:
                uid_match = re.search(r"tg://user\?id=(\d+)", text)
                if uid_match:
                    target_user_id = int(uid_match.group(1))
                else:
                    uname_match = re.search(r"t\.me/([A-Za-z0-9_]+)", text)
                    if uname_match and uname_match.group(1).lower() != "leomatchbot":
                        target_username = uname_match.group(1)
                    else:
                        m_at = re.search(r"@([A-Za-z0-9_]{4,32})", text)
                        if m_at and m_at.group(1).lower() != "leomatchbot":
                            target_username = m_at.group(1)

            # Получение ID через client.get_users если известен username
            if not target_user_id and target_username:
                try:
                    user_obj = await self.client.get_users(target_username)
                    target_user_id = user_obj.id
                    if user_obj.first_name and not is_valid_name(name):
                        name = user_obj.first_name
                except Exception as exc:
                    log.error("[%s][LEOBOT] Ошибка get_users(@%s): %s", self.name, target_username, exc)

            if target_user_id:
                if not is_valid_name(name):
                    try:
                        user_obj = await self.client.get_users(target_user_id)
                        if user_obj.first_name and is_valid_name(user_obj.first_name):
                            name = user_obj.first_name
                        if user_obj.username and not target_username:
                            target_username = user_obj.username
                    except Exception:
                        name = "Неизвестно"

                display_id = target_username or str(target_user_id)
                log.info(
                    "[%s][LEOBOT] Распознан мэтч: имя='%s', target_user_id=%d, target_username=%s",
                    self.name, name, target_user_id, target_username
                )

                # Автоматически добавляем нового пользователя в белый список (память + диск + .env + Excel)
                await self.add_to_whitelist(
                    user_id=target_user_id,
                    username=target_username,
                    name=name,
                    extra=f"Мэтч из Дайвинчика (@{target_username})" if target_username else "Мэтч из Дайвинчика",
                )

                # Проверяем ночной режим перед планированием приветствия
                if self.is_night_time() or self.night_mode_active:
                    log.info(
                        "[%s][LEOBOT] Мэтч %s (%d) зафиксирован в базе, но сейчас ночное время (21:00–09:00). "
                        "Приветствие не отправляется.",
                        self.name, name, target_user_id
                    )
                else:
                    # Запускаем отложенное приветствие в фоновом режиме (через 90 секунд)
                    log.info("[%s][LEOBOT] Запуск фонового таймера приветствия (90 с) для %s (%d)...", self.name, name, target_user_id)
                    asyncio.create_task(self._delayed_greeting(target_user_id, name, display_id))
                return
            else:
                log.warning("[%s][LEOBOT] Не удалось определить ID клиента из сообщения leomatchbot: '%s'", self.name, text[:80])

        # 2. Если это не мэтч, а анкета для оценки в режиме автолайкера:
        global AUTOCLICKER_ENABLED
        if AUTOLIKE_ENABLED and AUTOCLICKER_ENABLED:
            await self.leobot_queue.put(message)
        else:
            log.debug("[%s][LEOBOT] Автокликер отключен (AUTOCLICKER_ENABLED=%s). Анкета пропущена.", self.name, AUTOCLICKER_ENABLED)

    async def _delayed_greeting(self, target_user_id: int, name: str, display_id: str) -> None:
        """Отложенная отправка приветствия через 90 секунд в фоновом режиме."""
        try:
            log.info("[%s][GREETING] Старт ожидания 90 сек перед приветствием для %s (%s)...", self.name, name, display_id)
            await asyncio.sleep(90)

            # Проверка ночного режима: если наступило ночное время (21:00–09:00), запрещено писать первым
            if self.is_night_time() or self.night_mode_active:
                log.info(
                    "[%s][GREETING] Активен ночной режим (время: %s, диапазон: 21:00–09:00). "
                    "Отмена отправки приветствия клиенту %d, чтобы не будить человека ночью.",
                    self.name, datetime.now().time().strftime("%H:%M:%S"), target_user_id
                )
                return

            # Проверяем: если девушка уже написала первой за эти 90 секунд, не шлем приветствие повторно
            chat_data = self.active_chats.get(target_user_id) or self.active_chats.get(str(target_user_id), {})
            if chat_data.get("history") and len(chat_data.get("history")) > 0:
                log.info("[%s][GREETING] В чате %d уже есть переписка — отмена повторного приветствия", self.name, target_user_id)
                return

            greeting = random.choice(GREETING_FRASES) if GREETING_FRASES else "Привет!"
            log.info("[%s][GREETING] Таймер 90 сек истек. Отправка приветствия клиенту %d...", self.name, target_user_id)
            await self.client.send_message(target_user_id, greeting)
            self._append_to_history(target_user_id, "assistant", greeting)
            self._update_chat_state(target_user_id, is_me=True)
            log.info("[%s][GREETING] Приветствие ('%s') успешно отправлено клиенту %s (%d)", self.name, greeting, name, target_user_id)
        except Exception as exc:
            log.error("[%s][GREETING] Ошибка отправки приветствия клиенту %d: %s", self.name, target_user_id, exc)

    def _peer_is_typing(self, chat_id: int) -> bool:
        """True, если за последние 6 секунд приходило ChatAction.TYPING от этого пользователя."""
        last_seen = self.typing_activity.get(chat_id)
        if last_seen is None:
            return False
        return (time.monotonic() - last_seen) < 6.0

    def _writer_active(self, chat_id: int) -> bool:
        typing_task = self._typing_tasks.get(chat_id)
        generate_task = self._generate_tasks.get(chat_id)
        chunk_task = self._chunk_tasks.get(chat_id)
        typing_alive = typing_task is not None and not typing_task.done()
        generate_alive = generate_task is not None and not generate_task.done()
        chunk_alive = chunk_task is not None and not chunk_task.done()
        return typing_alive or generate_alive or chunk_alive

    def _start_listener(self, chat_id: int) -> None:
        """Первое сообщение: пауза 1–3 сек и вход в сеть без статуса печати."""
        existing = self._listen_tasks.get(chat_id)
        if existing is not None and not existing.done():
            return
        self._listen_stop[chat_id] = asyncio.Event()
        self._listen_ready[chat_id] = asyncio.Event()
        self._listen_tasks[chat_id] = asyncio.create_task(
            self._listen_online(chat_id),
            name=f"listen_{self.name}_{chat_id}",
        )

    async def _appear_online(self, chat_id: int, *, read: bool) -> None:
        """Онлайн и взгляд в чат. Статус TYPING отсюда не отправляется."""
        try:
            await self.client.invoke(raw.functions.account.UpdateStatus(offline=False))
        except Exception as exc:
            log.debug("[%s][DEBOUNCE] Не удалось стать онлайн для чата %d: %s", self.name, chat_id, exc)
        if not read:
            return
        try:
            await self.client.read_chat_history(chat_id)
        except Exception as exc:
            log.debug("[%s][DEBOUNCE] Не удалось открыть чат %d: %s", self.name, chat_id, exc)

    async def _listen_online(self, chat_id: int) -> None:
        ready = self._listen_ready.get(chat_id)
        stop = self._listen_stop.get(chat_id)
        try:
            pause = random.randint(1, 3)
            log.info(
                "[%s][DEBOUNCE] Чат %d: пауза %d сек, затем вход в сеть без печати",
                self.name, chat_id, pause,
            )
            await asyncio.sleep(pause)
            if stop is not None and stop.is_set():
                return
            log.info("[%s][DEBOUNCE] Чат %d: Ваня в сети и смотрит чат, статус печати выключен", self.name, chat_id)
            await self._appear_online(chat_id, read=True)
            if ready is not None and not ready.is_set():
                ready.set()
            while stop is not None and not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=20.0)
                except asyncio.TimeoutError:
                    await self._appear_online(chat_id, read=False)
        except asyncio.CancelledError:
            raise
        finally:
            if ready is not None and not ready.is_set():
                ready.set()

    def _stop_listener(self, chat_id: int) -> None:
        stop = self._listen_stop.pop(chat_id, None)
        if stop is not None:
            stop.set()
        ready = self._listen_ready.pop(chat_id, None)
        if ready is not None and not ready.is_set():
            ready.set()
        task = self._listen_tasks.pop(chat_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _start_typing(self, chat_id: int) -> None:
        existing = self._typing_tasks.get(chat_id)
        if existing is not None and not existing.done():
            return
        self._typing_stop[chat_id] = asyncio.Event()
        self._typing_tasks[chat_id] = asyncio.create_task(
            self._typing_loop(chat_id),
            name=f"typing_{self.name}_{chat_id}",
        )

    async def _typing_loop(self, chat_id: int) -> None:
        stop = self._typing_stop.get(chat_id)
        while stop is not None and not stop.is_set():
            try:
                await self.client.send_chat_action(chat_id, enums.ChatAction.TYPING)
            except Exception as exc:
                log.debug("[%s][DEBOUNCE] Не удалось показать TYPING в чате %d: %s", self.name, chat_id, exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass

    def _stop_typing(self, chat_id: int) -> None:
        stop = self._typing_stop.pop(chat_id, None)
        if stop is not None:
            stop.set()
        task = self._typing_tasks.pop(chat_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _silence_writer(self, chat_id: int) -> None:
        """Мгновенно убирает «печатает» и отменяет генерацию. Очередь текста не трогает."""
        self._stop_typing(chat_id)
        generate_task = self._generate_tasks.get(chat_id)
        if generate_task is not None and not generate_task.done():
            generate_task.cancel()
        chunk_task = self._chunk_tasks.get(chat_id)
        if chunk_task is not None and not chunk_task.done():
            chunk_task.cancel()
        asyncio.create_task(
            self._send_typing_cancel(chat_id),
            name=f"typing_cancel_{self.name}_{chat_id}",
        )

    async def _send_typing_cancel(self, chat_id: int) -> None:
        try:
            await self.client.send_chat_action(chat_id, enums.ChatAction.CANCEL)
        except Exception as exc:
            log.debug("[%s][DEBOUNCE] Не удалось снять TYPING в чате %d: %s", self.name, chat_id, exc)

    def _cancel_thought_debounce(self, chat_id: int, *, drop_pending: bool) -> None:
        task = debouncer_tasks.get(chat_id)
        if drop_pending:
            pending_messages.pop(chat_id, None)
            debouncer_tasks.pop(chat_id, None)
            self._debounce_chat_ids.discard(chat_id)
        if task and not task.done():
            task.cancel()
        self._silence_writer(chat_id)
        if drop_pending:
            self._stop_listener(chat_id)

    def _queue_debounced_thought(self, chat_id: int, message: Message, msg_text: str) -> None:
        """Слушатель копит текст. Если Ваня уже печатает, он замолкает и снова ждёт."""
        old_task = debouncer_tasks.get(chat_id)
        listen = self._listen_tasks.get(chat_id)
        listen_alive = listen is not None and not listen.done()

        if self._writer_active(chat_id):
            log.info("[%s][DEBOUNCE] Чат %d: девушка перебила набор, Ваня замолкает", self.name, chat_id)
            self._silence_writer(chat_id)

        if not listen_alive:
            self._start_listener(chat_id)

        if old_task is not None and not old_task.done():
            old_task.cancel()

        pending_messages.setdefault(chat_id, []).append(msg_text)
        self._debounce_chat_ids.add(chat_id)
        task = asyncio.create_task(
            self.wait_for_finishing_thought(chat_id, message),
            name=f"debounce_{self.name}_{chat_id}",
        )
        debouncer_tasks[chat_id] = task
        log.info(
            "[%s][DEBOUNCE] Чат %d: в очереди %d частей, режим слушателя, таймер 1.5–4 сек",
            self.name, chat_id, len(pending_messages[chat_id]),
        )

    async def wait_for_finishing_thought(self, chat_id: int, original_message: Message) -> None:
        """
        Случайная тишина 1.5–4 секунды без статуса печати. Потом TYPING и ask_ai.
        Отмена не удаляет накопленный текст и не роняет юзербота.
        """
        current = asyncio.current_task()
        try:
            ready = self._listen_ready.get(chat_id)
            if ready is not None and not ready.is_set():
                await ready.wait()
            if debouncer_tasks.get(chat_id) is not current or self._stop_event.is_set():
                return

            silence = random.uniform(1.5, 4.0)
            log.info(
                "[%s][DEBOUNCE] Чат %d: ждём тишину %.1f сек",
                self.name, chat_id, silence,
            )
            await asyncio.sleep(silence)
            if debouncer_tasks.get(chat_id) is not current or self._stop_event.is_set():
                return
            if self.is_ai_frozen():
                log.info("[%s][DEBOUNCE] Чат %d: ИИ заморожен, мысль не отправляем", self.name, chat_id)
                pending_messages.pop(chat_id, None)
                debouncer_tasks.pop(chat_id, None)
                self._debounce_chat_ids.discard(chat_id)
                self._silence_writer(chat_id)
                self._stop_listener(chat_id)
                return

            parts = [part.strip() for part in pending_messages.get(chat_id, []) if part and part.strip()]
            combined = "\n".join(parts).strip()
            if not combined:
                pending_messages.pop(chat_id, None)
                debouncer_tasks.pop(chat_id, None)
                self._debounce_chat_ids.discard(chat_id)
                self._stop_listener(chat_id)
                return

            log.info(
                "[%s][DEBOUNCE] Чат %d: тишина выдержана, включаем печать и ask_ai (%d частей)",
                self.name, chat_id, len(parts),
            )
            self._start_typing(chat_id)
            target_delay = random.uniform(7.0, 15.0)
            started = time.time()
            generate_task = asyncio.create_task(
                self.ask_ai(chat_id, combined, message=original_message),
                name=f"ask_ai_{self.name}_{chat_id}",
            )
            self._generate_tasks[chat_id] = generate_task
            try:
                raw_reply = await generate_task
            finally:
                if self._generate_tasks.get(chat_id) is generate_task:
                    self._generate_tasks.pop(chat_id, None)

            if debouncer_tasks.get(chat_id) is not current or self._stop_event.is_set():
                return
            if not raw_reply:
                log.warning("[%s][DEBOUNCE] Чат %d: модель вернула пустой ответ", self.name, chat_id)
                pending_messages.pop(chat_id, None)
                debouncer_tasks.pop(chat_id, None)
                self._debounce_chat_ids.discard(chat_id)
                self._silence_writer(chat_id)
                self._stop_listener(chat_id)
                return

            elapsed = time.time() - started
            remaining = target_delay - elapsed
            if remaining > 0:
                log.info(
                    "[%s][DEBOUNCE] Чат %d: ответ готов за %.1f с, досыпаем %.1f с в статусе печати",
                    self.name, chat_id, elapsed, remaining,
                )
                await asyncio.sleep(remaining)
            if debouncer_tasks.get(chat_id) is not current or self._stop_event.is_set():
                return

            pending_messages.pop(chat_id, None)
            debouncer_tasks.pop(chat_id, None)
            self._debounce_chat_ids.discard(chat_id)
            self._stop_typing(chat_id)
            self._stop_listener(chat_id)
            await self.process_ai_reply(chat_id, raw_reply, message=original_message, user_text=combined)
            log.info("[%s][DEBOUNCE] Чат %d: ответ на склеенную мысль отправлен", self.name, chat_id)
        except asyncio.CancelledError:
            log.debug(
                "[%s][DEBOUNCE] Чат %d: ожидание или генерация отменены, очередь текста сохранена",
                self.name, chat_id,
            )
            raise

    async def handle_girl_message(self, message: Message) -> None:
        """Перехват входящих сообщений (текстовых и медиа) от девушек / клиентов."""
        chat_id = message.chat.id
        msg_preview = (message.text or message.caption or "")[:40].replace("\n", " ")
        if not msg_preview:
            if message.photo:
                msg_preview = "[Фото]"
            elif message.video:
                msg_preview = "[Видео]"
            elif message.animation:
                msg_preview = "[GIF]"
            elif message.voice:
                msg_preview = "[Голосовое]"
            elif message.video_note:
                msg_preview = "[Кружок]"
            elif message.sticker:
                msg_preview = f"[Стикер {message.sticker.emoji or ''}]"
            elif message.audio:
                msg_preview = "[Аудио]"
            elif message.document:
                msg_preview = "[Документ]"
            else:
                msg_preview = "[Медиа]"

        log.debug("[%s][GIRL_MSG] Входящее сообщение в чат %d (msg_id=%s): '%s'", self.name, chat_id, message.id, msg_preview)

        if chat_id == self.my_id:
            message.continue_propagation()
            return

        if getattr(message, "outgoing", False) or (message.from_user and message.from_user.id == self.my_id):
            message.continue_propagation()
            return

        if message.from_user and message.from_user.username == "leomatchbot":
            message.continue_propagation()
            return

        username = message.chat.username or (message.from_user.username if message.from_user else "") or ""

        # Автоматическое добавление нового диалога в белый список при переходе инициативы в новый чат
        if (chat_id not in self.active_chats) and (str(chat_id) not in self.active_chats) and (username not in self.active_chats):
            log.info(
                "[%s][GIRL_MSG] Новое входящее сообщение от пользователя %d (@%s)! "
                "Автоматическое добавление в белый список и базу данных...",
                self.name, chat_id, username
            )
            raw_name = ""
            if message.from_user and message.from_user.first_name:
                raw_name = message.from_user.first_name
            elif message.chat and message.chat.first_name:
                raw_name = message.chat.first_name

            girl_name = raw_name if is_valid_name(raw_name) else "Неизвестно"
            await self.add_to_whitelist(
                user_id=chat_id,
                username=username,
                name=girl_name,
                extra=f"Новый диалог (@{username})" if username else "Новый диалог",
            )

        log.info("[%s][GIRL_MSG] Чат %d (@%s) найден в белом списке! Обработка сообщения...", self.name, chat_id, username)
        chat_info = self.active_chats.get(chat_id) or self.active_chats.get(str(chat_id)) or self.active_chats.get(username)
        self.active_chats[chat_id] = chat_info
        self.active_chats[str(chat_id)] = chat_info
        if username:
            self.active_chats[username] = chat_info

        if self.is_ai_frozen():
            log.info("[%s][GIRL_MSG] ИИ заморожен (режим сна или выключен). Время обновлено, ответ не генерируется", self.name)
            self._update_chat_state(chat_id, is_me=False)
            return

        # Проверка, является ли входящее сообщение медиафайлом (фото, видео, анимация, голосовое, кружок, стикер, аудио, документ)
        is_media = bool(
            message.photo
            or message.video
            or message.animation
            or message.voice
            or message.video_note
            or message.sticker
            or message.audio
            or message.document
            or (message.media and message.media != enums.MessageMediaType.WEB_PAGE)
        )

        raw_caption = (message.caption or "").strip()
        msg_text = (message.text or raw_caption).strip()

        if is_media:
            media_type_name = "фото" if message.photo else \
                              "видео" if message.video else \
                              "анимация/GIF" if message.animation else \
                              "голосовое сообщение" if message.voice else \
                              "видеосообщение (кружок)" if message.video_note else \
                              "стикер" if message.sticker else \
                              "аудио" if message.audio else \
                              "документ" if message.document else "медиафайл"
            log.info("[%s][GIRL_MSG] Входящий медиафайл (%s) в чате %d", self.name, media_type_name, chat_id)

            self.active_chats[chat_id]["last_msg_time"] = asyncio.get_event_loop().time()
            self.active_chats[chat_id]["pinged"] = False

            history_tag = f"[{media_type_name.capitalize()}]"
            history_entry = f"{history_tag} {raw_caption}".strip() if raw_caption else history_tag
            self._append_to_history(chat_id, "user", history_entry)
            self.active_chats[chat_id]["last_msg_is_me"] = False

            # Медиа отвечает сразу: очередь текста и ожидание мысли больше не нужны
            self._cancel_thought_debounce(chat_id, drop_pending=True)
            prev_reply_task: asyncio.Task | None = chat_info.get("reply_task")
            if prev_reply_task and not prev_reply_task.done():
                prev_reply_task.cancel()
                log.debug("[%s][GIRL_MSG] Отменена предыдущая задача ответа для чата %d в пользу медиа", self.name, chat_id)

            # Запускаем отложенный ответ на медиа (БЕЗ запроса к OpenRouter, пауза 4-8 с, статус TYPING)
            log.info("[%s][GIRL_MSG] Запуск задачи ответа на медиа (пауза 4-8 с, TYPING) для чата %d...", self.name, chat_id)
            reply_task = asyncio.create_task(self._delayed_media_reply(message, chat_id))
            chat_info["reply_task"] = reply_task
            return

        if not msg_text:
            log.debug("[%s][GIRL_MSG] Сообщение в чате %d не содержит текста — пропуск", self.name, chat_id)
            return

        self.active_chats[chat_id]["last_msg_time"] = asyncio.get_event_loop().time()
        self.active_chats[chat_id]["pinged"] = False
        self._append_to_history(chat_id, "user", msg_text)
        self.active_chats[chat_id]["last_msg_is_me"] = False

        # Если ответ уже генерируется, обрываем его: пришла ещё одна часть мысли
        prev_reply_task: asyncio.Task | None = chat_info.get("reply_task")
        if prev_reply_task and not prev_reply_task.done():
            prev_reply_task.cancel()
            log.debug("[%s][GIRL_MSG] Отменена предыдущая задача ответа для чата %d в пользу более свежего сообщения", self.name, chat_id)

        self._queue_debounced_thought(chat_id, message, msg_text)

    # ── Фоновые задачи аккаунта ─────────────────────────────────────────────

    async def conversation_resurrector(self) -> None:
        """
        Событийный планировщик ночного режима на основе расчета дельты времени (динамический сон):
        1. Логика при старте скрипта:
           - Бот берет текущее локальное время.
           - Определяет, находится ли он СЕЙЧАС в ночном диапазоне (с 21:00 до 09:00) или в дневном (с 09:00 до 21:00).
           - Немедленно включает нужный режим: если ночь — блокирует инициацию чатов, если день — разрешает.
        2. Расчет точного времени сна:
           - Бот вычисляет, сколько ровно часов, минут и секунд осталось до ближайшей смены режима
             (до 21:00, если сейчас день, или до 09:00, если сейчас ночь, с надежной обработкой перехода через полночь 00:00).
           - Переводит это время в секунды (seconds_to_sleep).
           - Выводит в лог четкое сообщение: [INFO] Режим определен. До смены режима осталось ХХ секунд (ХХ часов). Задача уходит в сон.
        3. Асинхронный цикл смены фаз:
           - Бот делает await asyncio.sleep(seconds_to_sleep).
           - Как только таймер истекает (наступает 09:00 или 21:00), бот просыпается, автоматически меняет
             статус ночного режима на противоположный, логирует это и заново запускает расчет секунд до следующей точки смены режима.
        """
        log.info("[%s][RESURRECTOR] Событийный планировщик ночного режима (динамический сон) запущен", self.name)

        while not self._stop_event.is_set():
            try:
                # Шаг 1: Берем текущее локальное время и определяем режим
                now = datetime.now()
                is_night, seconds_to_sleep, target_dt = self.calculate_mode_delta(now)

                # Шаг 2: Немедленно включаем нужный режим
                self.night_mode_active = is_night
                if is_night:
                    self.day_mode_event.clear()
                    mode_name = "НОЧНОЙ (21:00–09:00, инициация чатов и пинги ЗАБЛОКИРОВАНЫ)"
                else:
                    self.day_mode_event.set()
                    mode_name = "ДНЕВНОЙ (09:00–21:00, инициация чатов и пинги РАЗРЕШЕНЫ)"

                hours = seconds_to_sleep / 3600.0
                target_str = target_dt.strftime("%H:%M:%S (%d.%m)")

                # Шаг 3: Вывод в лог четкого сообщения по спецификации
                log.info(
                    "[%s][RESURRECTOR][INFO] Режим определен: %s. До смены режима осталось %.1f секунд (%.2f часов). Задача уходит в сон до %s.",
                    self.name, mode_name, seconds_to_sleep, hours, target_str
                )
                log.info(
                    "[%s][INFO] Режим определен. До смены режима осталось %.0f секунд (%.2f часов). Задача уходит в сон.",
                    self.name, seconds_to_sleep, hours
                )

                # Шаг 4: Асинхронный сон на точное расчетное время (seconds_to_sleep)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=seconds_to_sleep)
                    # Если получен сигнал завершения работы бота — выходим из цикла
                    break
                except asyncio.TimeoutError:
                    # Таймер истек — наступила точка смены режима (09:00 или 21:00)
                    pass

                if self._stop_event.is_set():
                    break

                # Шаг 5: Просыпаемся и автоматически меняем статус на противоположный
                now_after = datetime.now()
                new_is_night, _, _ = self.calculate_mode_delta(now_after)
                self.night_mode_active = new_is_night

                if self.night_mode_active:
                    self.day_mode_event.clear()
                    log.info(
                        "[%s][RESURRECTOR] Наступило 21:00! Смена фазы: Ночной режим АКТИВИРОВАН. "
                        "Инициация диалогов и пинг молчащих чатов заблокированы до 09:00 утра.",
                        self.name
                    )
                else:
                    self.day_mode_event.set()
                    log.info(
                        "[%s][RESURRECTOR] Наступило 09:00! Смена фазы: Дневной режим АКТИВИРОВАН. "
                        "Инициация диалогов и пинг молчащих чатов разрешены до 21:00 вечера.",
                        self.name
                    )

            except asyncio.CancelledError:
                log.debug("[%s][RESURRECTOR] Фоновая задача планировщика ночного режима остановлена", self.name)
                break
            except Exception as exc:
                log.error("[%s][RESURRECTOR] Ошибка в планировщике ночного режима: %s", self.name, exc, exc_info=True)
                try:
                    await asyncio.sleep(5)
                except (asyncio.CancelledError, Exception):
                    break

    async def silent_chats_resurrector_worker(self) -> None:
        """
        Фоновый дневной воркер реанимации молчащих диалогов:
        Активен строго в дневное время (когда day_mode_event взведен).
        В ночное время задача находится в ожидании day_mode_event без лишних просыпаний и без ежечасных циклов.
        """
        log.info("[%s][RESURRECTOR] Дневной воркер проверки молчащих чатов запущен", self.name)
        while not self._stop_event.is_set():
            try:
                # Ожидание дневного режима
                await self.day_mode_event.wait()
                if self._stop_event.is_set():
                    break

                # Интервал между дневными проверками (20 минут) с реакцией на смену режима/остановку
                CHECK_INTERVAL = 20 * 60
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=CHECK_INTERVAL)
                    break
                except asyncio.TimeoutError:
                    pass

                if self._stop_event.is_set() or not self.day_mode_event.is_set() or self.night_mode_active:
                    continue

                if self.is_ai_frozen():
                    continue

                log.debug("[%s][RESURRECTOR] Плановый дневной обход молчащих чатов...", self.name)
                await self._resurrect_silent_chats()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("[%s][RESURRECTOR] Ошибка в дневном воркере обхода чатов: %s", self.name, exc)
                try:
                    await asyncio.sleep(10)
                except (asyncio.CancelledError, Exception):
                    break

    def _collect_ping_targets(self) -> list[tuple[int, dict, float]]:
        """Чаты, где тишина уже дольше случайного порога 1–3 часа и пинг ещё не уходил."""
        now = datetime.now()
        targets: list[tuple[int, dict, float]] = []
        for user_id, chat_data in list(self.active_chats.items()):
            if isinstance(user_id, str):
                continue
            if chat_data.get("pinged") or chat_data.get("last_msg_is_me"):
                continue
            last_time = chat_data.get("last_msg_time")
            if not last_time:
                continue
            threshold_h = random.uniform(PING_MIN_HOURS, PING_MAX_HOURS)
            if isinstance(last_time, (int, float)):
                elapsed_h = (asyncio.get_event_loop().time() - last_time) / 3600
            else:
                elapsed_h = (now - last_time).total_seconds() / 3600
            if elapsed_h < threshold_h:
                continue
            targets.append((user_id, chat_data, elapsed_h))
        return targets

    async def _send_canned_ping(self, user_id: int, chat_data: dict, phrase: str) -> None:
        """Вход в сеть, печать 2–3 секунды и готовая фраза без генерации Qwen."""
        try:
            await self.client.read_chat_history(user_id)
        except Exception as exc:
            log.debug("[%s][RESURRECTOR] Не удалось отметить чат %s прочитанным: %s", self.name, user_id, exc)
        try:
            await self.client.send_chat_action(user_id, enums.ChatAction.TYPING)
        except Exception as exc:
            log.debug("[%s][RESURRECTOR] Не удалось показать TYPING в чате %s: %s", self.name, user_id, exc)
        await asyncio.sleep(random.uniform(2.0, 3.0))

        await self.client.send_message(user_id, phrase)
        chat_data["pinged"] = True
        self._update_chat_state(user_id, is_me=True)
        self._append_to_history(user_id, "assistant", phrase)
        await save_dialog_memory(
            user_id,
            "Инициатива Вани",
            phrase,
            embed_text=phrase,
            initiative=True,
        )
        log.info("[%s][RESURRECTOR] Пинг успешно отправлен клиенту %s: '%s'", self.name, user_id, phrase)

    async def _resurrect_silent_chats(self) -> None:
        """По одному пингует молчащие чаты готовой фразой. Между чатами пауза 2–5 минут."""
        if self.is_night_time() or self.night_mode_active or self.is_ai_frozen():
            log.info("[%s][RESURRECTOR] Проверка чатов отменена: активен ночной режим или ИИ заморожен", self.name)
            return

        targets = self._collect_ping_targets()
        if not targets:
            return

        log.info("[%s][RESURRECTOR] К пингу подходит чатов: %d. Отправка строго по одному", self.name, len(targets))
        for index, (user_id, chat_data, elapsed_h) in enumerate(targets):
            if index > 0:
                pause_sec = random.randint(120, 300)
                log.info(
                    "[%s][RESURRECTOR] Пауза %d сек перед следующим чатом, чтобы не слать пачку",
                    self.name, pause_sec,
                )
                await asyncio.sleep(pause_sec)

            if self._stop_event.is_set():
                break
            if self.is_night_time() or self.night_mode_active:
                log.info(
                    "[%s][RESURRECTOR] Наступило ночное время (%s). Прерываем пинг чатов.",
                    self.name,
                    datetime.now().time().strftime("%H:%M:%S"),
                )
                self.night_mode_active = True
                break
            reply_task = chat_data.get("reply_task")
            if (
                self.is_ai_frozen()
                or chat_data.get("pinged")
                or chat_data.get("last_msg_is_me")
                or (reply_task and not reply_task.done())
            ):
                continue

            phrase, period = choose_ping_phrase()
            log.info(
                "[%s][RESURRECTOR] Пинг чата %s — тишина %.1f ч, период '%s', фраза из базы",
                self.name, user_id, elapsed_h, period,
            )
            try:
                await self._send_canned_ping(user_id, chat_data, phrase)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("[%s][RESURRECTOR] Ошибка отправки пинга клиенту %s: %s", self.name, user_id, exc)

    def _extract_girl_id_from_leobot(self, message: Message) -> tuple[str | None, str]:
        """
        Извлечение уникального ID анкеты девушки из сообщения Дайвинчика.
        Проверяет entities (text_link), inline-кнопки (callback_data / URL), photo file_unique_id и текст.
        """
        text = (message.text or message.caption or "").strip()
        entities = (message.entities or []) + (getattr(message, "caption_entities", None) or [])

        # 1. Поиск ссылок в сущностях сообщения (text_link, mention, url)
        for ent in entities:
            url = getattr(ent, "url", None)
            if url:
                uid_m = re.search(r"tg://user\?id=(\d+)", url)
                if uid_m:
                    return uid_m.group(1), f"tg://user?id={uid_m.group(1)}"
                uname_m = re.search(r"t\.me/([A-Za-z0-9_]+)", url)
                if uname_m:
                    return uname_m.group(1), f"@{uname_m.group(1)}"

        # 2. Поиск в callback_data кнопок
        if message.reply_markup and getattr(message.reply_markup, "inline_keyboard", None):
            for row in message.reply_markup.inline_keyboard:
                for btn in row:
                    cb = getattr(btn, "callback_data", None)
                    if cb:
                        cb_str = cb.decode("utf-8", errors="ignore") if isinstance(cb, bytes) else str(cb)
                        m_cb = re.search(r"\b(\d{7,12})\b", cb_str)
                        if m_cb:
                            return m_cb.group(1), f"cb:{m_cb.group(1)}"
                    url = getattr(btn, "url", None)
                    if url:
                        uid_m = re.search(r"tg://user\?id=(\d+)", url)
                        if uid_m:
                            return uid_m.group(1), f"btn_tg:{uid_m.group(1)}"
                        uname_m = re.search(r"t\.me/([A-Za-z0-9_]+)", url)
                        if uname_m:
                            return uname_m.group(1), f"btn_url:@{uname_m.group(1)}"

        # 3. Фото file_unique_id
        if message.photo:
            first_line = text.split("\n")[0][:30].strip() if text else "Фото"
            return f"photo_{message.photo.file_unique_id}", first_line

        # 4. Хеш текста анкеты (имя, возраст, описание)
        if text:
            first_line = text.split("\n")[0][:30].strip()
            h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            return f"profile_{h}", first_line

        return None, ""

    async def _handle_leobot_menu(self, message: Message) -> bool:
        """Обработка стартового или сервисного меню Дайвинчика (кнопка 'Смотреть анкеты' / '1')."""
        if message.reply_markup and getattr(message.reply_markup, "inline_keyboard", None):
            for r_idx, row in enumerate(message.reply_markup.inline_keyboard):
                for c_idx, btn in enumerate(row):
                    t = (getattr(btn, "text", "") or "").strip()
                    if any(x in t for x in ["Смотреть анкеты", "1", "🚀", "Анкеты"]):
                        await message.click(r_idx, c_idx)
                        return True
        if message.reply_markup and getattr(message.reply_markup, "keyboard", None):
            for row in message.reply_markup.keyboard:
                for btn in row:
                    t = (getattr(btn, "text", str(btn)) or "").strip()
                    if any(x in t for x in ["Смотреть анкеты", "1", "🚀", "Анкеты"]):
                        await self.client.send_message(message.chat.id, t)
                        return True
        return False

    async def _get_unprocessed_leobot_message(self) -> Message | None:
        """Получение последнего неотвеченного сообщения из чата Дайвинчика."""
        try:
            async for m in self.client.get_chat_history("leomatchbot", limit=3):
                if m.outgoing:
                    return None
                if m.id == getattr(self, "last_handled_leobot_msg_id", None):
                    return None
                text = m.text or m.caption or ""
                if "Начинай общаться" in text:
                    return None
                if m.reply_markup or m.photo:
                    return m
        except Exception:
            pass
        return None

    async def leomatchbot_autoliker_worker(self) -> None:
        """
        Фоновый воркер умного автолайкера для Дайвинчика (@leomatchbot).
        1. Помечает чат прочитанным (read_chat_history).
        2. Извлекает user_id анкеты.
        3. Проверяет в seen_users: если уже видели — 👎 (дизлайк), если новая — ❤️ (лайк).
        4. Выдерживает анти-спам паузу 3-8 секунд перед кликом.
        5. Соблюдает суточный лимит (AUTOLIKE_DAILY_LIMIT, по умолчанию 120), затем засыпает на 12 часов.
        """
        log.info("[%s][AUTOLIKER] Воркер умного автолайкера Дайвинчика запущен", self.name)
        while not self._stop_event.is_set():
            try:
                global AUTOCLICKER_ENABLED
                if not self.bot_active or not AUTOLIKE_ENABLED or not AUTOCLICKER_ENABLED:
                    await asyncio.sleep(5)
                    continue

                # Сброс суточного счетчика лайков каждые 24 часа
                if datetime.now() >= self.likes_reset_time:
                    self.likes_today = 0
                    self.likes_reset_time = datetime.now() + timedelta(hours=24)
                    log.info("[%s][AUTOLIKER] Суточный счетчик лайков сброшен", self.name)

                # Проверка суточного лимита на лайки
                if self.likes_today >= AUTOLIKE_DAILY_LIMIT:
                    log.warning(
                        "[%s][AUTOLIKER] Достигнут суточный лимит лайков (%d/%d)! Воркер засыпает на 12 часов...",
                        self.name, self.likes_today, AUTOLIKE_DAILY_LIMIT
                    )
                    sleep_deadline = datetime.now() + timedelta(hours=12)
                    while datetime.now() < sleep_deadline and not self._stop_event.is_set():
                        await asyncio.sleep(30)
                    self.likes_today = 0
                    self.likes_reset_time = datetime.now() + timedelta(hours=24)
                    continue

                # Ожидание анкеты из очереди
                try:
                    message = await asyncio.wait_for(self.leobot_queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    message = await self._get_unprocessed_leobot_message()
                    if not message:
                        await asyncio.sleep(5)
                        continue

                # Помечаем чат Дайвинчика прочитанным
                try:
                    await self.client.read_chat_history("leomatchbot")
                except Exception as read_err:
                    log.debug("[%s][AUTOLIKER] read_chat_history: %s", self.name, read_err)

                msg_text = message.text or message.caption or ""
                if "Начинай общаться" in msg_text:
                    continue

                girl_id, info = self._extract_girl_id_from_leobot(message)
                if not girl_id:
                    # Проверяем, не главное ли это меню
                    if await self._handle_leobot_menu(message):
                        await asyncio.sleep(random.uniform(3.0, 5.0))
                    continue

                is_seen = girl_id in self.seen_users
                action = "dislike" if is_seen else "like"
                action_desc = "👎 (Дизлайк — повтор)" if is_seen else "❤️ (Лайк — новая)"

                log.info(
                    "[%s][AUTOLIKER] Анкета: ID=%s (%s). Статус в базе: %s -> Действие: %s",
                    self.name, girl_id, info, "найдена" if is_seen else "новая", action_desc
                )

                # Анти-спам защита: случайная пауза от 3 до 8 секунд
                pause = round(random.uniform(3.0, 8.0), 2)
                log.info("[%s][AUTOLIKER] Анти-спам задержка %.2f сек перед нажатием...", self.name, pause)
                await asyncio.sleep(pause)

                # Нажатие на кнопку
                success = await _click_leobot_button(self.client, message, action)
                if success:
                    self.last_handled_leobot_msg_id = message.id
                    if action == "like":
                        self.likes_today += 1
                        log.info(
                            "[%s][AUTOLIKER] Поставлен лайк ❤️ анкете %s! Всего за сегодня: %d/%d",
                            self.name, girl_id, self.likes_today, AUTOLIKE_DAILY_LIMIT
                        )
                    else:
                        log.info("[%s][AUTOLIKER] Поставлен дизлайк 👎 анкете %s (ранее уже встречалась)", self.name, girl_id)

                    # Фиксируем в памяти и дублируем в Excel под асинхронным локом
                    self.seen_users.add(girl_id)
                    async with self.seen_users_lock:
                        await asyncio.to_thread(_sync_append_seen_user, self.seen_users_file, girl_id, action, info)
                        try:
                            await asyncio.to_thread(_sync_append_seen_user, DATABASES_DIR / "seen_users.xlsx", girl_id, action, f"[{self.name}] {info}")
                        except Exception:
                            pass
                else:
                    log.warning("[%s][AUTOLIKER] Не удалось отправить нажатие для анкеты %s", self.name, girl_id)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("[%s][AUTOLIKER] Непредвиденная ошибка воркера: %s", self.name, exc, exc_info=True)
                await asyncio.sleep(5)

    # ── Жизненный цикл аккаунта ─────────────────────────────────────────────

    async def start(self) -> None:
        """Инициализация БД, проверка сессии, авторизация и запуск фоновых задач."""
        log.info("[%s] Шаг 1: Инициализация файла базы данных Excel...", self.name)
        await self.ensure_excel()
        log.info("[%s] Шаг 2: Загрузка белого списка чатов (Excel, JSON, .env)...", self.name)
        await self.load_whitelist()
        log.info("[%s] Шаг 2.1: Инициализация базы данных seen_users...", self.name)
        await self.ensure_seen_users_excel()
        self.seen_users = await asyncio.to_thread(_sync_load_seen_users, self.seen_users_file)
        log.info("[%s] Загружено %d просмотренных анкет из seen_users", self.name, len(self.seen_users))

        # Если файла сессии нет, Pyrogram запросит номер и код в консоли
        async with _AUTH_LOCK:
            if not self.session_file.exists():
                print("\n" + "=" * 60, flush=True)
                print(f"[АВТОРИЗАЦИЯ] Требуется авторизация для аккаунта: '{self.name}'", flush=True)
                print(f"Файл сессии '{self.session_file.name}' отсутствует в папке 'sessions/'.", flush=True)
                print(f"Пожалуйста, введите данные для входа в аккаунт [{self.name}] в консоли ниже:", flush=True)
                print("=" * 60 + "\n", flush=True)

            log.info("[%s] Шаг 3: Подключение клиента Pyrogram (await client.start())...", self.name)
            await self.client.start()
            log.info("[%s] Шаг 3: Клиент Pyrogram успешно запущен", self.name)

        # Быстрое неблокирующее получение ID и запуск фоновой задачи без тяжелых синхронных проверок
        self.my_id = (await self.client.get_me()).id
        GLOBAL_WHITELIST_IDS.add(self.my_id)
        username = self.client.me.username if self.client.me and self.client.me.username else "??"
        log.info("[%s] Аккаунт успешно авторизован как @%s (ID: %d)", self.name, username, self.my_id)
        log.info("[%s] База Excel: %s | Бот активен: %s | Workers: %s", self.name, self.excel_file, self.bot_active, self.client.workers)

        # Предварительное определение ID бота Дайвинчика (@leomatchbot)
        try:
            leobot_peer = await self.client.resolve_peer("leomatchbot")
            if hasattr(leobot_peer, "user_id"):
                _KNOWN_LEOMATCHBOT_IDS.add(leobot_peer.user_id)
                GLOBAL_WHITELIST_IDS.add(leobot_peer.user_id)
                self.active_chats[leobot_peer.user_id] = {"name": "Дайвинчик", "username": "leomatchbot"}
                self.active_chats[str(leobot_peer.user_id)] = {"name": "Дайвинчик", "username": "leomatchbot"}
                log.info("[%s] ID бота Дайвинчика (@leomatchbot) определен: %d", self.name, leobot_peer.user_id)
        except Exception as leobot_err:
            log.debug("[%s] Не удалось разрешить peer leomatchbot: %s", self.name, leobot_err)

        # Запуск событийного планировщика ночного режима (динамический расчет дельты сна)
        self.resurrector_task = asyncio.create_task(
            self.conversation_resurrector(), name=f"resurrector_{self.name}"
        )
        self.day_worker_task = asyncio.create_task(
            self.silent_chats_resurrector_worker(), name=f"day_worker_{self.name}"
        )

        # Запуск фонового воркера умного автолайкера Дайвинчика
        if AUTOLIKE_ENABLED:
            self.autoliker_task = asyncio.create_task(
                self.leomatchbot_autoliker_worker(), name=f"autoliker_{self.name}"
            )
            log.info("[%s] Фоновый воркер автолайкера Дайвинчика запущен", self.name)

        log.info("[%s] Все системы аккаунта готовы к приему сообщений", self.name)

    async def idle(self) -> None:
        """Ожидание завершения работы аккаунта."""
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Корректная остановка аккаунта и его фоновых задач."""
        self._stop_event.set()
        if self.autoliker_task and not self.autoliker_task.done():
            self.autoliker_task.cancel()
            try:
                await self.autoliker_task
            except (asyncio.CancelledError, Exception):
                pass

        if self.resurrector_task and not self.resurrector_task.done():
            self.resurrector_task.cancel()
            try:
                await self.resurrector_task
            except (asyncio.CancelledError, Exception):
                pass

        if self.day_worker_task and not self.day_worker_task.done():
            self.day_worker_task.cancel()
            try:
                await self.day_worker_task
            except (asyncio.CancelledError, Exception):
                pass

        for chat_id in list(self._debounce_chat_ids):
            task = debouncer_tasks.pop(chat_id, None)
            pending_messages.pop(chat_id, None)
            if task and not task.done():
                task.cancel()
            self._silence_writer(chat_id)
            self._stop_listener(chat_id)
        self._debounce_chat_ids.clear()

        if self.client and self.client.is_connected:
            try:
                await self.client.stop()
            except Exception as exc:
                log.debug("[%s] Ошибка при остановке клиента: %s", self.name, exc)
        log.info("[%s] Аккаунт остановлен", self.name)


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа — параллельный запуск всех аккаунтов
# ─────────────────────────────────────────────────────────────────────────────


async def run_account(account: AccountBot) -> None:
    """Полный жизненный цикл работы одного аккаунта."""
    try:
        await account.start()
        await account.idle()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        log.error("[%s] Критическая ошибка при работе аккаунта: %s", account.name, exc, exc_info=True)
    finally:
        await account.stop()


async def main() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_CHATS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Запуск системы мультиаккаунтности. Список аккаунтов: %s", ACCOUNTS)
    log.info("Telegram API: API_ID=%s (загружен из .env)", API_ID)
    log.info(
        "ИИ-канал: локальная Ollama (модель %s, %s)",
        OLLAMA_MODEL,
        OLLAMA_HOST,
    )
    log.info(
        "Автолайкер Дайвинчика: активен=%s, дневной лимит=%d",
        AUTOLIKE_ENABLED,
        AUTOLIKE_DAILY_LIMIT,
    )

    try:
        await init_vector_memory()
    except Exception as memory_exc:
        log.error("[QDRANT] Память недоступна, юзербот продолжит без RAG: %s", memory_exc, exc_info=True)

    account_bots = [AccountBot(name) for name in ACCOUNTS]

    # Цикл, который с помощью asyncio.create_task() одновременно запускает метод start() и idle() для каждого аккаунта
    tasks: list[asyncio.Task] = []
    for account in account_bots:
        task = asyncio.create_task(run_account(account), name=f"account_{account.name}")
        tasks.append(task)

    from pyrogram import idle
    try:
        await idle()
    finally:
        log.info("Получен сигнал завершения. Остановка всех аккаунтов...")
        for account in account_bots:
            await account.stop()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await close_vector_memory()
        log.info("Все аккаунты успешно остановлены. Завершение работы программы.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Бот завершил работу.")
