"""
Независимая выгрузка сегодняшних диалогов из Qdrant в JSONL для Unsloth.

Читает QDRANT_HOST и QDRANT_PORT из .env, через Scroll API забирает точки
коллекции vanya_memories за текущие сутки и пишет dataset_unsloth.jsonl
в корне проекта. Системный промпт каждой пары — актуальный SYSTEM_PROMPT из userbot.py.
"""

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "vanya_memories"
OUTPUT_PATH = BASE_DIR / "dataset_unsloth.jsonl"
SCROLL_PAGE = 128


def load_vanya_system_prompt() -> str:
    """Берёт тот же SYSTEM_PROMPT, которым Ваня отвечает в юзерботе."""
    from userbot import SYSTEM_PROMPT

    prompt = str(SYSTEM_PROMPT or "").strip()
    if not prompt:
        raise RuntimeError("SYSTEM_PROMPT в userbot.py пуст")
    return prompt


def _saved_on_today(payload: dict) -> bool:
    """Точка попадает в датасет, только если её timestamp — сегодняшние локальные сутки."""
    raw = str(payload.get("timestamp") or "").strip()
    if not raw:
        return False
    try:
        saved_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if saved_at.tzinfo is not None:
        saved_at = saved_at.astimezone().replace(tzinfo=None)
    return saved_at.date() == datetime.now().date()


async def export_unsloth_dataset(system_prompt: str) -> tuple[int, int, int]:
    """
    Листает коллекцию и пишет сегодняшние пары в JSONL.
    Возвращает (просмотрено, за сегодня, записано).
    """
    client = AsyncQdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=30.0,
        trust_env=False,
    )
    found = 0
    today = 0
    written = 0
    offset = None

    try:
        with OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
            while True:
                points, offset = await client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=SCROLL_PAGE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points or []:
                    found += 1
                    payload = getattr(point, "payload", None) or {}
                    if not isinstance(payload, dict) or not _saved_on_today(payload):
                        continue
                    today += 1
                    user_message = str(payload.get("user_message") or "").strip()
                    assistant_reply = str(payload.get("assistant_reply") or "").strip()
                    if not user_message or not assistant_reply:
                        continue
                    row = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": assistant_reply},
                        ]
                    }
                    outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                if offset is None:
                    break
    finally:
        await client.close()

    return found, today, written


async def main() -> None:
    today_label = datetime.now().strftime("%Y-%m-%d")
    print(f"Подключение к Qdrant {QDRANT_HOST}:{QDRANT_PORT}, коллекция '{COLLECTION_NAME}'...")
    print(f"В датасет попадут только диалоги за {today_label}")
    system_prompt = load_vanya_system_prompt()
    print(f"Системный промпт взят из userbot.py ({len(system_prompt)} симв.)")
    found, today, written = await export_unsloth_dataset(system_prompt)
    print(f"Всего точек в коллекции: {found}")
    print(f"Из них за сегодня: {today}")
    print(f"Успешно сконвертировано: {written}")
    print(f"Файл: {OUTPUT_PATH}")
    print("[SUCCESS] Датасет dataset_unsloth.jsonl готов к загрузке в Unsloth!")


if __name__ == "__main__":
    asyncio.run(main())
