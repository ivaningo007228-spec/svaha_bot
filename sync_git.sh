#!/usr/bin/env bash
set -e

echo "========================================================"
echo "       Svaha Bot - Git Auto-Sync"
echo "========================================================"

if ! command -v git &> /dev/null; then
    echo "[ERROR] Git не найден!"
    exit 1
fi

# Проверка на случайное отслеживание .env
if git ls-files .env 2>/dev/null | grep -q ".env"; then
    echo "[CRITICAL] Файл .env в индексе! Удаляем из кэша git..."
    git rm --cached .env
fi

MSG="${1:-chore: auto-sync update $(date '+%Y-%m-%d %H:%M:%S')}"

echo "[+] Добавление разрешенных файлов в индекс..."
git add .

if git diff --cached --quiet; then
    echo "[INFO] Нет новых изменений для коммита."
else
    echo "[+] Создание коммита: '$MSG'"
    git commit -m "$MSG"
fi

echo "[+] Отправка изменений в remote (ветка main)..."
git push origin main

echo "========================================================"
echo " [SUCCESS] Репозиторий успешно синхронизирован с GitHub!"
echo "========================================================"
