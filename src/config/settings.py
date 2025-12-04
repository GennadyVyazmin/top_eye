# src/config/settings.py
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Параметры камеры
    RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream1"
    CAMERA_ID = "trassir_tr-d1415_1"

    # Настройки обработки
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 25
    PROCESS_EVERY_N_FRAMES = 2

    # Пути к моделям
    YOLO_MODEL_PATH = "models/yolo/yolov8n.pt"
    FACENET_MODEL_PATH = "models/facenet/facenet.pb"
    DEEPSORT_CONFIG = "config/deep_sort.yaml"

    # Настройки базы данных
    DB_PATH = "data/database.db"
    FACES_DIR = "data/faces/"

    # Пороги
    CONFIDENCE_THRESHOLD = 0.5
    FACE_MATCH_THRESHOLD = 0.6
    TRACKING_MAX_AGE = 30

    # Интервалы статистики
    STATS_INTERVALS = [180, 1440, 2880, 10080, 43200]  # минуты

    # Настройки веб-сервера
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8000
    API_PORT = 8080


settings = Settings()