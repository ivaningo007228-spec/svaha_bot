@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================================
echo        Svaha Bot - Git Auto-Sync
echo ========================================================

:: Проверка наличия git
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git не найден в переменной PATH!
    exit /b 1
)

:: Проверка статуса
git status >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Текущая директория не является git-репозиторием!
    exit /b 1
)

:: Проверка на случайное добавление .env
git ls-files .env 2>nul | findstr /i ".env" >nul
if %ERRORLEVEL% equ 0 (
    echo [CRITICAL ERROR] Файл .env отслеживается в git! Удаляем из индекса...
    git rm --cached .env
)

:: Формирование описания коммита
set "MSG=%~1"
if "%MSG%"=="" (
    set "TIMESTAMP=%date% %time:~0,8%"
    set "MSG=chore: auto-sync update !TIMESTAMP!"
)

echo [+] Добавление разрешенных файлов в индекс...
git add .

:: Проверка наличия изменений для коммита
git diff --cached --quiet
if %ERRORLEVEL% equ 0 (
    echo [INFO] Нет новых изменений для коммита.
    goto push_check
)

echo [+] Создание коммита: "%MSG%"
git commit -m "%MSG%"

:push_check
echo [+] Отправка изменений в remote (ветка main)...
git push origin main
if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo  [SUCCESS] Все изменения успешно закоммичены и отправлены!
    echo ========================================================
) else (
    echo.
    echo [WARNING] git push завершился с ошибкой. Проверьте подключение к GitHub или удалённый репозиторий.
)

exit /b 0
