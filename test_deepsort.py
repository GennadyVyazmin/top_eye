# /top_eye/test_deepsort.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

print("Тестирование DeepSORT...")

# Создаем тестовые детекции
test_detections = [
    [100, 100, 50, 150, 0.8],  # [x, y, w, h, confidence]
    [200, 200, 60, 160, 0.7],
    [300, 300, 70, 170, 0.9]
]

print(f"Тестовые детекции: {test_detections}")

# Инициализируем трекер
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=None
)

# Создаем тестовый кадр
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

try:
    # Пробуем обновить треки
    tracks = tracker.update_tracks(test_detections, frame=test_frame)
    print(f"✅ DeepSORT работает! Получено треков: {len(tracks)}")

    for track in tracks:
        print(f"  Трек ID: {track.track_id}, bbox: {track.to_tlbr()}")

except Exception as e:
    print(f"❌ Ошибка DeepSORT: {e}")
    import traceback

    traceback.print_exc()