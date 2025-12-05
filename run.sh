# /top_eye/run.sh
#!/bin/bash

# Активируем виртуальное окружение (если есть)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Устанавливаем PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 Запуск системы распознавания лиц..."

# Проверяем аргументы
MODE="web"
CAMERA_URL=""
PORT=8000

# Парсим аргументы
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --camera)
            CAMERA_URL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --deduplicate)
            MODE="deduplicate"
            shift
            ;;
        --stats)
            MODE="stats"
            shift
            ;;
        --help|-h)
            echo "Использование: $0 [опции]"
            echo ""
            echo "Опции:"
            echo "  --mode MODE       Режим запуска (web, test, deduplicate, stats)"
            echo "  --camera URL      URL камеры RTSP"
            echo "  --port PORT       Порт веб-сервера (по умолчанию: 8000)"
            echo "  --test            Тест камеры"
            echo "  --deduplicate     Запуск дедупликации базы"
            echo "  --stats           Показать статистику"
            echo "  --help, -h        Показать эту справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --test"
            echo "  $0 --mode web --camera rtsp://admin:123456@192.168.1.100:554/live"
            echo "  $0 --deduplicate"
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $1"
            exit 1
            ;;
    esac
done

# Запускаем Python скрипт с нужными аргументами
if [ -n "$CAMERA_URL" ]; then
    python main.py --mode "$MODE" --camera "$CAMERA_URL" --port "$PORT"
else
    python main.py --mode "$MODE" --port "$PORT"
fi