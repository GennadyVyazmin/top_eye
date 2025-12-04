# /top_eye/src/core/video_processor.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
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

        # Инициализация моделей
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

            # DeepSORT - ИСПРАВЛЕННАЯ ИНИЦИАЛИЗАЦИЯ
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self.tracker = DeepSort(
                    max_age=self.config.TRACKING_MAX_AGE,
                    n_init=3,
                    max_cosine_distance=0.2,
                    nn_budget=None,
                    override_track_class=None,
                    embedder="mobilenet"  # Используем более легкий embedder
                )
                print("✓ DeepSORT инициализирован (mobilenet)")
            except ImportError:
                print("⚠ DeepSORT не установлен, используется простой трекинг")
                self.tracker = SimpleTracker()

            # Face Recognition
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
                        self.frame_queue.put(frame.copy())  # Копируем кадр

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
            'frame': frame.copy() if frame is not None else None,
            'detections': [],
            'faces': [],
            'timestamp': datetime.now(),
            'people_count': 0,
            'fps': 0
        }

        try:
            # Если модели загружены, делаем детекцию
            if self.model is not None and frame is not None:
                start_time = time.time()

                # YOLO детекция
                with torch.no_grad():
                    yolo_results = self.model(
                        frame,
                        conf=self.config.CONFIDENCE_THRESHOLD,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        verbose=False,
                        classes=[0]  # Только люди
                    )

                # Извлекаем детекции людей - ИСПРАВЛЕННЫЙ КОД
                people_detections = []
                if yolo_results and len(yolo_results) > 0:
                    yolo_result = yolo_results[0]

                    if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                        # Получаем данные в numpy формате
                        boxes = yolo_result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                        confidences = yolo_result.boxes.conf.cpu().numpy()
                        classes = yolo_result.boxes.cls.cpu().numpy()

                        for i in range(len(boxes)):
                            if classes[i] == 0:  # person class
                                x1, y1, x2, y2 = boxes[i].astype(int)
                                conf = float(confidences[i])

                                people_detections.append({
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                                    'confidence': conf
                                })

                # Трекинг с DeepSORT - ИСПРАВЛЕННЫЙ КОД
                if self.tracker is not None and people_detections:
                    try:
                        # Конвертируем детекции в формат для DeepSORT
                        # DeepSORT ожидает список списков: [[x, y, w, h, confidence], ...]
                        deepsort_detections = []
                        for det in people_detections:
                            bbox = det['bbox']
                            # Убеждаемся, что все значения float
                            deepsort_detections.append([
                                float(bbox[0]),  # x
                                float(bbox[1]),  # y
                                float(bbox[2]),  # w
                                float(bbox[3]),  # h
                                float(det['confidence'])  # confidence
                            ])

                        # Обновляем трекер
                        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)

                        # Обрабатываем треки
                        for track in tracks:
                            if track.is_confirmed():
                                bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                                track_id = track.track_id

                                # Получаем confidence если доступен
                                confidence = 0.5
                                if hasattr(track, 'get_det_conf'):
                                    try:
                                        confidence = track.get_det_conf()
                                    except:
                                        pass
                                elif hasattr(track, 'det_conf'):
                                    confidence = track.det_conf

                                result['detections'].append({
                                    'track_id': int(track_id),
                                    'bbox': [float(coord) for coord in bbox],
                                    'confidence': float(confidence)
                                })

                    except Exception as e:
                        print(f"Ошибка трекинга DeepSORT: {str(e)[:100]}")
                        # Используем простой трекинг как fallback
                        result['detections'] = self._simple_tracking(people_detections)

                # Если трекер не инициализирован, используем простой
                elif people_detections:
                    result['detections'] = self._simple_tracking(people_detections)

                # Вычисляем FPS
                end_time = time.time()
                processing_time = end_time - start_time
                result['fps'] = 1.0 / processing_time if processing_time > 0 else 0

                # Обновляем счетчик людей
                result['people_count'] = len(result['detections'])
                self.current_count = result['people_count']

                # Обновляем статистику уникальных
                for det in result['detections']:
                    track_id = det['track_id']
                    self.session_unique.add(track_id)

                    # Для today_unique добавляем дату
                    today = datetime.now().date().isoformat()
                    self.today_unique.add(f"{today}_{track_id}")

                # Добавляем в историю
                if result['people_count'] > 0:
                    self.detections_history.append({
                        'timestamp': result['timestamp'],
                        'count': result['people_count'],
                        'detections': result['detections']
                    })

                    # Ограничиваем историю последними 1000 записей
                    if len(self.detections_history) > 1000:
                        self.detections_history = self.detections_history[-1000:]

        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка в обработке кадра: {error_msg[:100]}")

        return result

    def _simple_tracking(self, detections):
        """Простой трекинг без DeepSORT"""
        simple_detections = []
        for i, det in enumerate(detections):
            simple_detections.append({
                'track_id': i + 1,  # Простой ID
                'bbox': det['bbox'],  # [x, y, w, h]
                'confidence': det['confidence']
            })
        return simple_detections

    def get_current_frame(self):
        """Получить последний обработанный кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_current_people_count(self):
        """Получить текущее количество людей в кадре"""
        return self.current_count

    def get_statistics(self):
        """Получить статистику"""
        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'total_detections': len(self.detections_history),
            'avg_people_count': np.mean([d['count'] for d in self.detections_history[-100:]])
            if self.detections_history else 0
        }

    def get_detection_history(self, limit=100):
        """Получить историю детекций"""
        return self.detections_history[-limit:]

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
                return [float(x), float(y), float(x + w), float(y + h)]

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