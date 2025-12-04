# /top_eye/src/core/video_processor.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import time
import os
import hashlib


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Инициализация процессора для камеры: {config.CAMERA_ID}")

        # Инициализация камеры
        self.cap = None
        self.last_reconnect = time.time()
        self.reconnect_interval = 5

        # Очереди для обработки
        self.frame_queue = Queue(maxsize=10)
        self.processed_queue = Queue(maxsize=10)
        self.lock = Lock()
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()
        self.detections_history = []

        # Трекинг
        self.tracks = {}  # {track_id: {bbox, last_seen, appearance}}
        self.next_track_id = 1
        self.track_max_age = 30  # кадров

        # YOLO модель
        self.model = None
        self.face_recognizer = None

        print("✓ Видеопроцессор инициализирован")

    def init_models(self):
        """Инициализация моделей"""
        try:
            print("Загрузка моделей...")

            # YOLOv8
            from ultralytics import YOLO
            model_path = self.config.YOLO_MODEL_PATH

            # Скачиваем модель если нет
            if not os.path.exists(model_path):
                print(f"Модель {model_path} не найдена, скачиваем...")
                model_path = 'yolov8n.pt'  # Автоскачивание

            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ YOLO модель загружена на {'CUDA' if torch.cuda.is_available() else 'CPU'}")

            # DeepFace (опционально)
            try:
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
                # Переподключение если нужно
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
                    print("✗ Ошибка чтения кадра")
                    self.cap.release()
                    self.cap = None
                    continue

                reconnect_attempts = 0

                # Добавляем кадр в очередь (обрабатываем каждый 3-й кадр)
                if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())

                frame_count += 1

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

                    # Сохранение в очередь
                    if not self.processed_queue.full():
                        self.processed_queue.put(processed_data)

            except Exception as e:
                print(f"Ошибка обработки кадра: {e}")

    def _process_single_frame(self, frame):
        """Обработка одного кадра с простым трекингом"""
        result = {
            'frame': frame.copy(),
            'detections': [],
            'faces': [],
            'timestamp': datetime.now(),
            'people_count': 0,
            'fps': 0
        }

        try:
            if self.model is not None:
                start_time = time.time()

                # YOLO детекция
                with torch.no_grad():
                    yolo_results = self.model(
                        frame,
                        conf=self.config.CONFIDENCE_THRESHOLD,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        verbose=False,
                        classes=[0]
                    )

                # Извлекаем текущие детекции
                current_detections = []
                if yolo_results and len(yolo_results) > 0:
                    yolo_result = yolo_results[0]

                    if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                        boxes = yolo_result.boxes.xyxy.cpu().numpy()
                        confidences = yolo_result.boxes.conf.cpu().numpy()

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[i].astype(int)
                            conf = float(confidences[i])

                            # Центр bounding box
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2

                            # Размер bounding box (для фильтрации слишком маленьких объектов)
                            width = x2 - x1
                            height = y2 - y1

                            if width > 20 and height > 40:  # Фильтр по размеру
                                current_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'center': [center_x, center_y],
                                    'confidence': conf,
                                    'width': width,
                                    'height': height
                                })

                # Простой трекинг (сопоставление по расстоянию)
                tracked_detections = self._simple_tracking(current_detections)

                # Обновляем треки
                for det in tracked_detections:
                    track_id = det['track_id']
                    self.tracks[track_id] = {
                        'bbox': det['bbox'],
                        'last_seen': time.time(),
                        'appearance': det.get('appearance_hash', '')
                    }

                    # Добавляем в результат
                    result['detections'].append({
                        'track_id': track_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })

                # Удаляем старые треки
                current_time = time.time()
                tracks_to_delete = []
                for track_id, track_data in self.tracks.items():
                    if current_time - track_data['last_seen'] > self.track_max_age:
                        tracks_to_delete.append(track_id)

                for track_id in tracks_to_delete:
                    del self.tracks[track_id]

                # Обновляем статистику
                result['people_count'] = len(result['detections'])
                self.current_count = result['people_count']

                # Обновляем уникальных посетителей
                for det in result['detections']:
                    track_id = det['track_id']
                    self.session_unique.add(track_id)

                    # Для сегодняшних уникальных
                    today = datetime.now().date().isoformat()
                    self.today_unique.add(f"{today}_{track_id}")

                # FPS
                end_time = time.time()
                result['fps'] = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

                # Рисуем bounding boxes на кадре
                self._draw_detections(frame, result['detections'])

        except Exception as e:
            print(f"Ошибка обработки: {e}")

        return result

    def _simple_tracking(self, current_detections):
        """Простейший трекинг по расстоянию между центрами"""
        tracked_detections = []

        # Если нет текущих детекций, возвращаем пустой список
        if not current_detections:
            return tracked_detections

        # Если нет активных треков, создаем новые
        if not self.tracks:
            for i, det in enumerate(current_detections):
                track_id = self.next_track_id
                self.next_track_id += 1
                det['track_id'] = track_id
                tracked_detections.append(det)
            return tracked_detections

        # Матрица расстояний между текущими детекциями и существующими треками
        distances = []
        for i, det in enumerate(current_detections):
            for track_id, track_data in self.tracks.items():
                # Получаем центр последнего известного положения трека
                last_bbox = track_data['bbox']
                last_center = [
                    (last_bbox[0] + last_bbox[2]) / 2,
                    (last_bbox[1] + last_bbox[3]) / 2
                ]

                # Расстояние между центрами
                dist = np.sqrt(
                    (det['center'][0] - last_center[0]) ** 2 +
                    (det['center'][1] - last_center[1]) ** 2
                )

                distances.append((i, track_id, dist))

        # Сортируем по расстоянию
        distances.sort(key=lambda x: x[2])

        # Сопоставляем
        matched_detections = set()
        matched_tracks = set()

        for i, track_id, dist in distances:
            # Если расстояние меньше порога и детекция/трек еще не сопоставлены
            if dist < 100 and i not in matched_detections and track_id not in matched_tracks:
                current_detections[i]['track_id'] = track_id
                tracked_detections.append(current_detections[i])
                matched_detections.add(i)
                matched_tracks.add(track_id)

        # Для несопоставленных детекций создаем новые треки
        for i, det in enumerate(current_detections):
            if i not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                det['track_id'] = track_id
                tracked_detections.append(det)

        return tracked_detections

    def _draw_detections(self, frame, detections):
        """Рисует bounding boxes и ID на кадре"""
        for det in detections:
            bbox = det['bbox']
            track_id = det['track_id']
            confidence = det['confidence']

            x1, y1, x2, y2 = map(int, bbox)

            # Цвет в зависимости от ID (для визуального различия)
            color_hash = hash(str(track_id)) % 256
            color = (
                (color_hash * 7) % 256,
                (color_hash * 13) % 256,
                (color_hash * 17) % 256
            )

            # Прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Фон для текста
            text = f"ID: {track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 5),
                          (x1 + text_size[0], y1), color, -1)

            # Текст
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Confidence
            conf_text = f"{confidence:.1%}"
            cv2.putText(frame, conf_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Счетчик людей
        count_text = f"Людей: {len(detections)}"
        cv2.putText(frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_current_frame(self):
        """Получить последний обработанный кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_statistics(self):
        """Получить статистику"""
        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'active_tracks': len(self.tracks),
            'total_detections': len(self.detections_history)
        }

    def get_detection_history(self, limit=50):
        """Получить историю детекций"""
        if hasattr(self, 'detections_history'):
            return self.detections_history[-limit:]
        return []

    def stop(self):
        """Остановка"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("✓ Обработка остановлена")