"""
Скрипт авторизации Telegram через официальные API-ключи Telegram Desktop (api_id=6).
Запрашивает код авторизации строго внутрь приложения Telegram (сервисные уведомления).
Экспортирует готовую Session String и сохраняет файл сессии в папку sessions/{имя_аккаунта}.session.
"""

import asyncio
import getpass
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyrogram import Client, raw
from pyrogram.enums import SentCodeType
from pyrogram.errors import (
    FloodWait,
    PasswordHashInvalid,
    PhoneCodeExpired,
    PhoneCodeInvalid,
    PhoneNumberInvalid,
    SessionPasswordNeeded,
)

# Поддержка UTF-8 вывода в консоли Windows
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Загрузка конфигурации из .env
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

_raw_api_id = (os.getenv("API_ID") or "").strip().strip('"').strip("'")
_raw_api_hash = (os.getenv("API_HASH") or "").strip().strip('"').strip("'")

if not _raw_api_id or not _raw_api_hash:
    print("\n" + "=" * 65)
    print("[!] КРИТИЧЕСКАЯ ОШИБКА: API_ID и API_HASH не заданы в файле .env!")
    print("Пожалуйста, укажите в вашем файле .env:")
    print("API_ID=ваш_api_id")
    print("API_HASH=ваш_api_hash")
    print("=" * 65 + "\n")
    sys.exit(1)

try:
    API_ID = int(_raw_api_id)
except ValueError:
    print(f"\n[!] ОШИБКА: API_ID в файле .env должен быть целым числом, получено: '{_raw_api_id}'\n")
    sys.exit(1)

API_HASH = _raw_api_hash
DEVICE_MODEL = "Desktop"
SYSTEM_VERSION = "Windows 10"
APP_VERSION = "5.10.0"
LANG_CODE = "ru"
SYSTEM_LANG_CODE = "ru-RU"

DEFAULT_ACCOUNT_NAME = "acc_main"
SESSIONS_DIR = "sessions"
OUTPUT_FILE = "session_string.txt"
ENV_FILE = ".env"

# Патч InitConnection, чтобы отправлять ru-RU в MTProto как официальный Telegram Desktop
_orig_init_connection = raw.functions.InitConnection.__init__


def _patched_init_connection(self, *args, **kwargs):
    if kwargs.get("system_lang_code") in ("ru", None, ""):
        kwargs["system_lang_code"] = SYSTEM_LANG_CODE
    _orig_init_connection(self, *args, **kwargs)


raw.functions.InitConnection.__init__ = _patched_init_connection


def update_env_file(session_string: str) -> None:
    """Обновить или добавить SESSION_STRING, API_ID и API_HASH в .env."""
    env_path = Path(ENV_FILE)
    lines = []
    keys_found = {"SESSION_STRING": False, "API_ID": False, "API_HASH": False}

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("SESSION_STRING="):
                    lines.append(f"SESSION_STRING={session_string}\n")
                    keys_found["SESSION_STRING"] = True
                elif line.startswith("API_ID="):
                    lines.append(f"API_ID={API_ID}\n")
                    keys_found["API_ID"] = True
                elif line.startswith("API_HASH="):
                    lines.append(f"API_HASH={API_HASH}\n")
                    keys_found["API_HASH"] = True
                else:
                    lines.append(line)

    if not keys_found["SESSION_STRING"]:
        lines.append(f"\nSESSION_STRING={session_string}\n")
    if not keys_found["API_ID"]:
        lines.append(f"API_ID={API_ID}\n")
    if not keys_found["API_HASH"]:
        lines.append(f"API_HASH={API_HASH}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


async def authorize() -> None:
    print("=" * 65)
    print(f"      ГЕНЕРАТОР СЕССИИ TELEGRAM (API_ID = {API_ID})")
    print("=" * 65)
    print(f"Ключи загружены из .env: API_ID={API_ID}")
    print("Код подтверждения придёт в ваш Telegram (чат 'Служебные уведомления').\n")

    # Имя аккаунта (например, acc_main или acc_second)
    if len(sys.argv) > 1 and sys.argv[1].strip():
        account_name = sys.argv[1].strip()
        print(f"Имя аккаунта из параметров запуска: '{account_name}'\n")
    else:
        raw_account = input(f"Введите имя аккаунта (например, acc_main или acc_second) [{DEFAULT_ACCOUNT_NAME}]: ").strip()
        account_name = raw_account if raw_account else DEFAULT_ACCOUNT_NAME

    if account_name.endswith(".session"):
        account_name = account_name[:-8]

    raw_phone = input("Введите номер телефона (+79991234567): ").strip()
    phone_number = re.sub(r"[^\d+]", "", raw_phone)
    if not phone_number.startswith("+"):
        phone_number = "+" + phone_number

    # Автоматическое создание папки sessions/, если её ещё нет
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    # Принудительное создание файла сессии внутри папки sessions/
    session_path = os.path.join(SESSIONS_DIR, account_name)

    # Создаём клиент с параметрами Telegram Desktop
    app = Client(
        name=session_path,
        api_id=API_ID,
        api_hash=API_HASH,
        device_model=DEVICE_MODEL,
        system_version=SYSTEM_VERSION,
        app_version=APP_VERSION,
        lang_code=LANG_CODE,
        in_memory=False,
    )

    try:
        await app.connect()
        print("\nОтправка запроса на получение кода авторизации...")

        sent_code = await app.send_code(phone_number)

        if sent_code.type == SentCodeType.APP:
            print("-> [УСПЕХ] Код отправлен ВНУТРЬ приложения Telegram (чат 777000)!")
        elif sent_code.type == SentCodeType.SMS:
            print("-> [ВНИМАНИЕ] Telegram отправил код по SMS.")
        elif sent_code.type == SentCodeType.CALL:
            print("-> [ВНИМАНИЕ] Telegram запросил голосовой звонок.")
        else:
            print(f"-> Тип доставки кода: {sent_code.type}")

        raw_code = input("\nВведите код подтверждения из Telegram: ").strip()
        code = raw_code.replace(" ", "").replace("-", "")

        try:
            await app.sign_in(phone_number, sent_code.phone_code_hash, code)
        except SessionPasswordNeeded:
            print("\nВ аккаунте включён облачный пароль (2FA двухфакторная аутентификация).")
            password = getpass.getpass("Введите ваш облачный пароль (ввод скрыт): ")
            await app.check_password(password)

        # Получаем информацию об аккаунте
        me = await app.get_me()
        session_string = await app.export_session_string()

        print("\n" + "=" * 65)
        print("                 АВТОРИЗАЦИЯ УСПЕШНА!")
        print("=" * 65)
        print(f"Пользователь: {me.first_name} {me.last_name or ''}".strip())
        print(f"Username:     @{me.username}" if me.username else "Username:     нет")
        print(f"ID:           {me.id}")
        print(f"Телефон:      +{me.phone_number}")

        # 1. Сохраняем строку сессии в текстовый файл
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(session_string.strip())
        print(f"\n[+] Строка сессии сохранена в: {OUTPUT_FILE}")

        # 2. Файл сессии сохранён в папке sessions/
        session_file_name = f"{account_name}.session"
        session_file_path = os.path.join(SESSIONS_DIR, session_file_name)
        print(f"[+] Файл сессии сохранён: {session_file_path}")

        # 3. Сохраняем/обновляем в .env
        update_env_file(session_string.strip())
        print(f"[+] Значения SESSION_STRING, API_ID, API_HASH записаны в: {ENV_FILE}")

        print(f"\nТеперь вы можете запускать основной скрипт для аккаунта '{account_name}': python userbot.py")
        print("=" * 65)

    except PhoneNumberInvalid:
        print("\n[!] Ошибка: Неверный формат номера телефона.")
    except PhoneCodeInvalid:
        print("\n[!] Ошибка: Введён неверный код подтверждения.")
    except PhoneCodeExpired:
        print("\n[!] Ошибка: Срок действия кода подтверждения истёк. Повторите запуск.")
    except PasswordHashInvalid:
        print("\n[!] Ошибка: Неверный облачный пароль (2FA).")
    except FloodWait as e:
        print(f"\n[!] Ошибка FloodWait: Telegram временно ограничил запросы. Подождите {e.value} секунд.")
    except Exception as e:
        print(f"\n[!] Непредвиденная ошибка: {e}")
    finally:
        if app.is_connected:
            await app.disconnect()


if __name__ == "__main__":
    asyncio.run(authorize())
