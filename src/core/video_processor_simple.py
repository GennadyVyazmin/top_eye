# /top_eye/src/core/video_processor_simple.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import time
import os


class SimpleVideoProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Инициализация УПРОЩЕННОГО процессора для камеры: {config.CAMERA_ID}")

        # Камера
        self.cap = None
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()

        # YOLO модель
        self.model = None

        # История
        self.last_detections = []
        self.frame_counter = 0

        print("✓ Упрощенный видеопроцессор инициализирован")

    def init_model(self):
        """Инициализация только YOLO модели"""
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # Автоматически скачает если нет
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ YOLO модель загружена на {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        except Exception as e:
            print(f"✗ Ошибка загрузки YOLO: {e}")
            self.model = None

    def start(self):
        """Запуск обработки"""
        self.running = True
        self.init_model()
        Thread(target=self._process_loop, daemon=True).start()
        print("✓ Обработка запущена")

    def _process_loop(self):
        """Основной цикл обработки"""
        self.cap = cv2.VideoCapture(self.config.RTSP_URL)

        if not self.cap.isOpened():
            print("✗ Не удалось подключиться к камере")
            return

        print("✓ Камера подключена")

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ Ошибка чтения кадра")
                    time.sleep(1)
                    continue

                # Обрабатываем каждый 5-й кадр для экономии ресурсов
                self.frame_counter += 1
                if self.frame_counter % 5 != 0:
                    time.sleep(0.01)
                    continue

                # Детекция если модель загружена
                detections = []
                if self.model is not None:
                    try:
                        results = self.model(frame, conf=0.5, classes=[0], verbose=False)

                        if results and len(results) > 0:
                            result = results[0]
                            if result.boxes is not None:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                confidences = result.boxes.conf.cpu().numpy()

                                for i in range(len(boxes)):
                                    x1, y1, x2, y2 = boxes[i].astype(int)
                                    conf = float(confidences[i])

                                    # Простой "трекинг" - считаем каждую детекцию новым человеком
                                    track_id = hash((x1, y1, x2, y2)) % 10000

                                    detections.append({
                                        'track_id': int(track_id),
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': conf
                                    })

                                    # Рисуем bounding box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person {i + 1}",
                                                (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Ошибка детекции: {e}")

                # Обновляем статистику
                self.current_count = len(detections)
                self.last_detections = detections

                # Сохраняем кадр для веб-интерфейса
                self.last_frame = frame
                self.last_frame_time = datetime.now()

                # Небольшая пауза
                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                print(f"Ошибка в основном цикле: {e}")
                time.sleep(1)

    def get_current_data(self):
        """Получить текущие данные"""
        if hasattr(self, 'last_frame'):
            return {
                'frame': self.last_frame,
                'detections': self.last_detections,
                'people_count': self.current_count,
                'timestamp': self.last_frame_time if hasattr(self, 'last_frame_time') else datetime.now()
            }
        return None

    def get_statistics(self):
        """Получить статистику"""
        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique)
        }

    def stop(self):
        """Остановка"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("✓ Обработка остановлена")