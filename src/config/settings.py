# /top_eye/src/config/settings.py
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Параметры камеры Trassir TR-D1415
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:admin@10.0.0.242:554/live/main")
    CAMERA_ID = "trassir_tr-d1415_1"

    # Настройки обработки
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 25
    PROCESS_EVERY_N_FRAMES = 2

    # Пути к моделям
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/yolo/yolov8n.pt")
    FACENET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/facenet/facenet.pb")

    # Пороги - ИСПРАВЛЕНО ИМЯ ПЕРЕМЕННОЙ
    CONFIDENCE_THRESHOLD = 0.5  # было CONFENCE_THRESHOLD
    FACE_MATCH_THRESHOLD = 0.6
    TRACKING_MAX_AGE = 30

    # Веб-сервер
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8000
    API_PORT = 8080


settings = Settings()