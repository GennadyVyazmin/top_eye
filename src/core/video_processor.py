# src/core/video_processor.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
from ultralytics import YOLO
import time


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        self.cap = cv2.VideoCapture(config.RTSP_URL)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.frame_queue = Queue(maxsize=10)
        self.processed_queue = Queue(maxsize=10)
        self.lock = Lock()
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()

    def start(self):
        self.running = True
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()
        Thread(target=self._update_statistics, daemon=True).start()

    def _capture_frames(self):
        frame_count = 0
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.config.RTSP_URL)
                continue

            if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            frame_count += 1

    def _process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Детекция объектов
                results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD,
                                     device='cuda' if torch.cuda.is_available() else 'cpu')

                # Фильтрация только людей
                people_detections = []
                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == 0:  # класс 'person' в YOLO
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            people_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf
                            })

                # Трекинг с DeepSORT
                tracked_objects = self.tracker.update(people_detections, frame)

                # Извлечение лиц для идентификации
                faces_data = self.face_extractor.extract_faces(frame, tracked_objects)

                processed_data = {
                    'frame': frame,
                    'detections': tracked_objects,
                    'faces': faces_data,
                    'timestamp': datetime.now()
                }

                if not self.processed_queue.full():
                    self.processed_queue.put(processed_data)

    def get_current_frame(self):
        """Получить текущий обработанный кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def stop(self):
        self.running = False
        self.cap.release()