"""
Выгрузка векторной памяти Qdrant в JSONL для дообучения Unsloth.

Подключается к коллекции vanya_memories, через Scroll API забирает все точки
и сохраняет dataset_unsloth.jsonl в корне проекта.
"""

import asyncio
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
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "vanya_memories"
OUTPUT_PATH = BASE_DIR / "dataset_unsloth.jsonl"

FALLBACK_SYSTEM_PROMPT = (
    "Ты — 17-летний парень по имени Иван (Ваня). "
    "Веди живой, естественный диалог с девушкой."
)


async def export_unsloth_dataset() -> int:
    system_prompt = (os.getenv("SYSTEM_PROMPT") or "").strip() or FALLBACK_SYSTEM_PROMPT
    client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    written = 0
    offset = None

    try:
        with OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
            while True:
                points, offset = await client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
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

    return written


async def main() -> None:
    print(f"Подключение к Qdrant {QDRANT_HOST}:{QDRANT_PORT}, коллекция '{COLLECTION_NAME}'...")
    count = await export_unsloth_dataset()
    print(f"Готово: {count} диалогов записано в {OUTPUT_PATH.name}")


if __name__ == "__main__":
    asyncio.run(main())
