# /top_eye/src/core/video_processor.py - ИСПРАВЛЕННАЯ СТРОКА 128
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime, timedelta
import time
import os


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Инициализация процессора для камеры: {config.CAMERA_ID}")

        # Инициализация камеры
        self.cap = None
        self.last_reconnect = time.time()
        self.reconnect_interval = 5  # секунд

        # Очереди для обработки
        self.frame_queue = Queue(maxsize=5)
        self.processed_queue = Queue(maxsize=5)
        self.lock = Lock()
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()
        self.detections_history = []

        # Инициализация YOLO (позже)
        self.model = None
        self.tracker = None
        self.face_recognizer = None

        print("✓ Видеопроцессор инициализирован")

    def init_models(self):
        """Инициализация моделей YOLO, DeepSORT и Face Recognition"""
        try:
            print("Загрузка моделей...")

            # YOLO
            from ultralytics import YOLO
            self.model = YOLO(self.config.YOLO_MODEL_PATH)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ YOLO модель загружена на {'CUDA' if torch.cuda.is_available() else 'CPU'}")

            # DeepSORT
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self.tracker = DeepSort(
                    max_age=self.config.TRACKING_MAX_AGE,
                    n_init=3,
                    max_cosine_distance=0.2,
                    nn_budget=None
                )
                print("✓ DeepSORT инициализирован")
            except ImportError:
                print("⚠ DeepSORT не установлен, используется простой трекинг")
                self.tracker = SimpleTracker()

            # Face Recognition (пропускаем если есть проблемы с tensorflow)
            try:
                # Пробуем установить tf-keras если нужно
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-keras"])
                from deepface import DeepFace
                self.face_recognizer = DeepFace
                print("✓ DeepFace инициализирован")
            except Exception as e:
                print(f"⚠ DeepFace не доступен: {e}")
                print("Идентификация лиц отключена")

        except Exception as e:
            print(f"✗ Ошибка загрузки моделей: {e}")
            print("⚠ Режим работы без моделей - только захват видео")

    def start(self):
        """Запуск обработки видео"""
        self.running = True

        # Инициализация моделей
        self.init_models()

        # Запуск потоков
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()

        print("✓ Обработка видео запущена")

    def _capture_frames(self):
        """Поток захвата кадров с камеры"""
        frame_count = 0
        reconnect_attempts = 0

        while self.running:
            try:
                # Проверяем, нужно ли переподключиться
                if self.cap is None or not self.cap.isOpened():
                    if time.time() - self.last_reconnect > self.reconnect_interval:
                        print(f"Подключение к камере... (попытка {reconnect_attempts + 1})")
                        self._reconnect_camera()
                        reconnect_attempts += 1
                        time.sleep(1)
                        continue
                    else:
                        time.sleep(0.1)
                        continue

                # Чтение кадра
                success, frame = self.cap.read()

                if not success:
                    print("✗ Ошибка чтения кадра, переподключение...")
                    self.cap.release()
                    self.cap = None
                    continue

                # Сбрасываем счетчик попыток при успешном чтении
                reconnect_attempts = 0

                # Добавляем кадр в очередь (каждый N-й кадр для обработки)
                if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)

                frame_count += 1

                # Небольшая пауза для контроля FPS
                time.sleep(0.01)

            except Exception as e:
                print(f"Ошибка захвата кадра: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(1)

    def _reconnect_camera(self):
        """Переподключение к камере"""
        try:
            if self.cap:
                self.cap.release()

            print(f"Подключение к: {self.config.RTSP_URL}")
            self.cap = cv2.VideoCapture(self.config.RTSP_URL)

            # Настройка параметров
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)

            # Даем камере время на инициализацию
            time.sleep(0.5)

            if self.cap.isOpened():
                print("✓ Камера подключена успешно")
                self.last_reconnect = time.time()
                return True
            else:
                print("✗ Не удалось подключиться к камере")
                return False

        except Exception as e:
            print(f"Ошибка подключения к камере: {e}")
            return False

    def _process_frames(self):
        """Поток обработки кадров"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Обработка кадра
                    processed_data = self._process_single_frame(frame)

                    # Сохранение в очередь обработанных данных
                    if not self.processed_queue.full():
                        self.processed_queue.put(processed_data)

            except Exception as e:
                print(f"Ошибка обработки кадра: {e}")

    def _process_single_frame(self, frame):
        """Обработка одного кадра"""
        result = {
            'frame': frame,
            'detections': [],
            'faces': [],
            'timestamp': datetime.now(),
            'people_count': 0
        }

        try:
            # Если модели загружены, делаем детекцию
            if self.model is not None:
                # YOLO детекция - ИСПРАВЛЕН ПАРАМЕТР CONFIDENCE_THRESHOLD
                yolo_results = self.model(frame,
                                          conf=self.config.CONFIDENCE_THRESHOLD,  # ИСПРАВЛЕНО
                                          device='cuda' if torch.cuda.is_available() else 'cpu',
                                          verbose=False)

                # Фильтруем только людей (класс 0 в YOLO)
                people_detections = []
                for r in yolo_results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            if int(box.cls) == 0:  # person class
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                people_detections.append({
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                                    'confidence': conf
                                })

                # Трекинг
                if self.tracker is not None and people_detections:
                    tracks = self.tracker.update_tracks(people_detections, frame=frame)

                    for track in tracks:
                        if track.is_confirmed():
                            bbox = track.to_tlbr()
                            track_id = track.track_id

                            result['detections'].append({
                                'track_id': track_id,
                                'bbox': bbox,
                                'confidence': track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
                            })

                # Обновляем счетчик людей
                result['people_count'] = len(result['detections'])
                self.current_count = result['people_count']

        except Exception as e:
            print(f"Ошибка в обработке кадра: {e}")

        return result

    def get_current_frame(self):
        """Получить последний обработанный кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_current_people_count(self):
        """Получить текущее количество людей в кадре"""
        return self.current_count

    def stop(self):
        """Остановка обработки"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("✓ Обработка видео остановлена")


class SimpleTracker:
    """Простой трекер если DeepSORT не установлен"""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update_tracks(self, detections, frame=None):
        class SimpleTrack:
            def __init__(self, track_id, bbox):
                self.track_id = track_id
                self.bbox = bbox

            def to_tlbr(self):
                x, y, w, h = self.bbox
                return [x, y, x + w, y + h]

            def is_confirmed(self):
                return True

            def get_det_conf(self):
                return 0.5

        tracks = []
        for det in detections:
            track_id = self.next_id
            self.next_id += 1
            tracks.append(SimpleTrack(track_id, det['bbox']))

        return tracks