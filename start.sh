#!/bin/bash
# start.sh

# Проверка наличия GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU не обнаружена!"
    exit 1
fi

# Создание необходимых директорий
mkdir -p data/faces models/yolo models/facenet

# Проверка зависимостей
if [ ! -f "requirements.txt" ]; then
    echo "Файл requirements.txt не найден!"
    exit 1
fi

# Установка Python зависимостей
pip install -r requirements.txt

# Запуск приложения
python src/main.py --mode both