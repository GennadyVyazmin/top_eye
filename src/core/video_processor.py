# /top_eye/src/core/video_processor.py
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from threading import Thread, Lock, Event
from queue import Queue
from datetime import datetime, timedelta
import time
import os
import pickle
import json
import hashlib
from collections import deque, defaultdict, OrderedDict
from sklearn.neighbors import NearestNeighbors, BallTree
from scipy.spatial.distance import cosine, euclidean
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings('ignore')

# Импорт дополнительных модулей
from .kalman_tracker import KalmanTracker
from .reid_model import StrongReIDModel


class Track:
    """Класс для хранения данных о треке"""
    __slots__ = ['track_id', 'bbox', 'kalman', 'embeddings', 'features',
                 'last_seen', 'first_seen', 'hits', 'age', 'time_since_update',
                 'color', 'is_confirmed', 'consecutive_invisible', 'visible',
                 'velocity', 'predicted_bbox', 'appearance', 'class_id']

    def __init__(self, track_id, bbox, embedding=None, class_id=0):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.kalman = KalmanTracker(bbox)
        self.embeddings = deque(maxlen=50)  # История эмбеддингов
        self.features = deque(maxlen=20)  # Дополнительные фичи
        if embedding is not None:
            self.embeddings.append(embedding)
            self.features.append(self._extract_features(bbox, embedding))

        self.last_seen = time.time()
        self.first_seen = time.time()
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.color = self._generate_color(track_id)
        self.is_confirmed = False
        self.consecutive_invisible = 0
        self.visible = True
        self.velocity = np.zeros(2, dtype=np.float32)
        self.predicted_bbox = None
        self.appearance = None
        self.class_id = class_id

    def _generate_color(self, track_id):
        """Генерация уникального цвета для трека"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(50, 205, 3)))

    def _extract_features(self, bbox, embedding):
        """Извлечение дополнительных признаков"""
        x1, y1, x2, y2 = bbox
        return {
            'width': x2 - x1,
            'height': y2 - y1,
            'aspect_ratio': (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0,
            'area': (x2 - x1) * (y2 - y1),
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'embedding_norm': np.linalg.norm(embedding) if embedding is not None else 0
        }

    def update(self, bbox, embedding=None, is_matched=True):
        """Обновление трека"""
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kalman.update(bbox)

        if embedding is not None:
            self.embeddings.append(embedding)
            self.features.append(self._extract_features(bbox, embedding))

        self.last_seen = time.time()
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        if is_matched:
            self.consecutive_invisible = 0
            if self.hits >= 3 and not self.is_confirmed:
                self.is_confirmed = True
        else:
            self.consecutive_invisible += 1

        # Обновляем скорость
        if len(self.features) >= 2:
            last_feat = list(self.features)[-1]
            prev_feat = list(self.features)[-2]
            self.velocity = np.array([
                last_feat['center_x'] - prev_feat['center_x'],
                last_feat['center_y'] - prev_feat['center_y']
            ], dtype=np.float32)

    def predict(self):
        """Предсказание следующего положения"""
        self.predicted_bbox = self.kalman.predict()
        self.time_since_update += 1
        return self.predicted_bbox

    def get_embedding(self):
        """Получение среднего эмбеддинга"""
        if not self.embeddings:
            return None
        return np.mean(list(self.embeddings), axis=0)

    def get_features(self):
        """Получение последних признаков"""
        if not self.features:
            return None
        return list(self.features)[-1]

    def mark_missing(self):
        """Отметить как пропавший"""
        self.consecutive_invisible += 1
        self.time_since_update += 1
        self.visible = False

    def is_reliable(self):
        """Проверка надежности трека"""
        return self.is_confirmed and self.hits >= 5 and self.consecutive_invisible < 10

    def get_state(self):
        """Получение состояния трека"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox.tolist(),
            'age': self.age,
            'hits': self.hits,
            'is_confirmed': self.is_confirmed,
            'consecutive_invisible': self.consecutive_invisible,
            'velocity': self.velocity.tolist(),
            'last_seen': self.last_seen
        }


class AppearanceDatabase:
    """База данных внешнего вида для долгосрочной ReID"""

    def __init__(self, max_size=1000, similarity_threshold=0.85):
        self.embeddings = []
        self.track_ids = []
        self.timestamps = []
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.kdtree = None
        self._needs_rebuild = True

    def add(self, track_id, embedding, timestamp=None):
        """Добавление эмбеддинга"""
        if timestamp is None:
            timestamp = time.time()

        self.embeddings.append(embedding)
        self.track_ids.append(track_id)
        self.timestamps.append(timestamp)
        self._needs_rebuild = True

        # Ограничиваем размер
        if len(self.embeddings) > self.max_size:
            self.embeddings.pop(0)
            self.track_ids.pop(0)
            self.timestamps.pop(0)

    def query(self, embedding, max_results=5):
        """Поиск похожих эмбеддингов"""
        if not self.embeddings:
            return []

        if self._needs_rebuild:
            self._rebuild_index()

        # Поиск K ближайших соседей
        distances, indices = self.kdtree.query([embedding], k=min(max_results, len(self.embeddings)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - dist  # Предполагаем нормализованные эмбеддинги
            if similarity >= self.similarity_threshold:
                results.append({
                    'track_id': self.track_ids[idx],
                    'similarity': float(similarity),
                    'timestamp': self.timestamps[idx]
                })

        return results

    def _rebuild_index(self):
        """Перестроение индекса поиска"""
        if self.embeddings:
            self.kdtree = NearestNeighbors(
                n_neighbors=min(10, len(self.embeddings)),
                metric='cosine',
                algorithm='ball_tree'
            )
            self.kdtree.fit(self.embeddings)
            self._needs_rebuild = False


class VideoProcessor:
    """Финальная улучшенная версия видео процессора"""

    def __init__(self, config):
        self.config = config
        print(f"🚀 Инициализация УЛУЧШЕННОГО процессора для камеры: {config.CAMERA_ID}")

        # Камера
        self.cap = None
        self.last_reconnect = time.time()
        self.reconnect_interval = 3
        self.frame_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

        # Очереди и синхронизация
        self.frame_queue = Queue(maxsize=30)
        self.processed_queue = Queue(maxsize=30)
        self.lock = Lock()
        self.running = False
        self.stop_event = Event()

        # Модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 Устройство: {self.device}")

        self.model = None  # YOLO
        self.reid_model = None  # ReID модель
        self.transform = self._create_transforms()

        # Трекинг
        self.tracks = OrderedDict()  # {track_id: Track}
        self.lost_tracks = OrderedDict()  # Потерянные треки
        self.next_track_id = 1000
        self.appearance_db = AppearanceDatabase(max_size=2000)

        # Настройки трекинга
        self.max_age = 60  # Максимальный возраст без обновления
        self.min_hits = 3  # Минимум попаданий для подтверждения
        self.iou_threshold = 0.3
        self.reid_threshold = 0.75
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()
        self.total_detections = 0
        self.total_frames = 0
        self.fps_history = deque(maxlen=100)

        # Кэши
        self.embedding_cache = OrderedDict()
        self.feature_cache = OrderedDict()
        self.max_cache_size = 500

        # Инициализация
        self._init_models()
        self._load_persistent_data()

        print("✅ Улучшенный видео процессор инициализирован")
        print(f"   • ReID порог: {self.reid_threshold}")
        print(f"   • Max age: {self.max_age} кадров")
        print(f"   • Min hits: {self.min_hits}")

    def _init_models(self):
        """Инициализация моделей"""
        try:
            # YOLOv8
            from ultralytics import YOLO
            model_path = self.config.YOLO_MODEL_PATH

            if not os.path.exists(model_path):
                print("📥 Скачиваем YOLOv8n...")
                model_path = 'yolov8n.pt'

            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✅ YOLOv8 загружен на {self.device}")

            # ReID модель
            self.reid_model = StrongReIDModel(device=self.device)
            print("✅ ReID модель загружена")

            # Оптимизация для inference
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

        except Exception as e:
            print(f"❌ Ошибка инициализации моделей: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_transforms(self):
        """Создание трансформов для ReID"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_persistent_data(self):
        """Загрузка сохраненных данных"""
        try:
            db_path = os.path.join(os.path.dirname(__file__), "../../data/visitors.db")
            if os.path.exists(db_path):
                with open(db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.appearance_db = data.get('appearance_db', AppearanceDatabase())
                    self.next_track_id = data.get('next_track_id', 1000)
                    self.today_unique = set(data.get('today_unique', []))
                print(f"📂 Загружено {len(self.appearance_db.embeddings)} эмбеддингов из базы")
        except:
            print("📂 Новая база данных")

    def _save_persistent_data(self):
        """Сохранение данных"""
        try:
            os.makedirs(os.path.dirname(self.config.DB_PATH), exist_ok=True)
            data = {
                'appearance_db': self.appearance_db,
                'next_track_id': self.next_track_id,
                'today_unique': list(self.today_unique),
                'saved_at': datetime.now().isoformat()
            }
            with open(self.config.DB_PATH, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠ Ошибка сохранения базы: {e}")

    def start(self):
        """Запуск обработки"""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()

        # Запуск потоков
        threads = [
            Thread(target=self._capture_frames, daemon=True, name="CaptureThread"),
            Thread(target=self._process_frames, daemon=True, name="ProcessThread"),
            Thread(target=self._manage_tracks, daemon=True, name="TrackManagerThread"),
            Thread(target=self._monitor_system, daemon=True, name="MonitorThread")
        ]

        for thread in threads:
            thread.start()

        print("▶ Обработка видео запущена")

    def _capture_frames(self):
        """Поток захвата кадров"""
        print("🎥 Захват кадров запущен")

        while self.running and not self.stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    if time.time() - self.last_reconnect > self.reconnect_interval:
                        self._reconnect_camera()
                    time.sleep(0.1)
                    continue

                success, frame = self.cap.read()
                if not success:
                    print("⚠ Ошибка чтения кадра")
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.5)
                    continue

                # Ресайз если нужно
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    frame = cv2.resize(frame, self.frame_size)

                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), time.time()))

                # Контроль FPS
                time.sleep(max(0, 1 / self.config.FPS - 0.001))

            except Exception as e:
                print(f"❌ Ошибка захвата: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)

    def _reconnect_camera(self):
        """Переподключение к камере"""
        print(f"🔌 Подключение к: {self.config.RTSP_URL}")

        try:
            if self.cap:
                self.cap.release()

            # Пробуем разные опции для RTSP
            self.cap = cv2.VideoCapture(self.config.RTSP_URL)

            # Оптимизации для RTSP
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

            # Таймауты
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)

            if self.cap.isOpened():
                print("✅ Камера подключена")
                self.last_reconnect = time.time()
                return True
            else:
                print("❌ Не удалось подключиться")
                return False

        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False

    def _process_frames(self):
        """Поток обработки кадров"""
        print("⚙ Обработка кадров запущена")

        while self.running and not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get()

                    # Обработка с таймингом
                    process_start = time.time()
                    result = self._process_single_frame(frame, timestamp)
                    process_time = time.time() - process_start

                    # Сохраняем FPS
                    if process_time > 0:
                        self.fps_history.append(1 / process_time)

                    if not self.processed_queue.full():
                        self.processed_queue.put(result)

                    self.total_frames += 1

                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"❌ Ошибка обработки: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _process_single_frame(self, frame, timestamp):
        """Обработка одного кадра"""
        result = {
            'frame': frame.copy(),
            'detections': [],
            'timestamp': timestamp,
            'people_count': 0,
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'processing_time': 0,
            'tracks_info': []
        }

        process_start = time.time()

        try:
            # 1. Детекция YOLO
            detections = self._yolo_detection(frame)

            # 2. Извлечение ReID эмбеддингов
            detections_with_embeddings = self._extract_embeddings(frame, detections)

            # 3. Прогнозирование для существующих треков
            for track in self.tracks.values():
                track.predict()

            # 4. Сопоставление треков
            matched_pairs = self._match_tracks(detections_with_embeddings)

            # 5. Обновление треков
            updated_tracks = self._update_tracks(matched_pairs, detections_with_embeddings)

            # 6. Создание новых треков
            new_tracks = self._create_new_tracks(detections_with_embeddings, matched_pairs)

            # 7. Обновление потерянных треков
            self._update_lost_tracks()

            # 8. Восстановление треков из базы
            recovered_tracks = self._recover_tracks(detections_with_embeddings, matched_pairs)

            # 9. Сбор результатов
            all_tracks = list(self.tracks.values()) + recovered_tracks
            result['people_count'] = len([t for t in all_tracks if t.visible])
            self.current_count = result['people_count']

            for track in all_tracks:
                if track.visible:
                    result['detections'].append({
                        'track_id': track.track_id,
                        'bbox': track.bbox.tolist(),
                        'confidence': 0.9 if track.is_confirmed else 0.5,
                        'age': track.age,
                        'hits': track.hits,
                        'is_confirmed': track.is_confirmed,
                        'velocity': track.velocity.tolist(),
                        'color': track.color
                    })

                    # Обновляем статистику уникальных
                    if track.is_confirmed and track.hits > 10:
                        self.session_unique.add(track.track_id)
                        today = datetime.now().date().isoformat()
                        self.today_unique.add(f"{today}_{track.track_id}")

            # 10. Рисование на кадре
            self._draw_detections(frame, result['detections'])

            # 11. Информация о треках
            result['tracks_info'] = [track.get_state() for track in all_tracks[:10]]

            self.total_detections += len(result['detections'])

        except Exception as e:
            print(f"⚠ Ошибка обработки кадра: {e}")

        result['processing_time'] = time.time() - process_start

        return result

    def _yolo_detection(self, frame):
        """Детекция YOLO"""
        detections = []

        try:
            with torch.no_grad():
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False,
                    classes=[0],  # Только люди
                    imgsz=640,
                    agnostic_nms=True,
                    max_det=50
                )

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        conf = float(confidences[i])

                        # Фильтрация по размеру
                        width = x2 - x1
                        height = y2 - y1

                        if width > 40 and height > 80 and width < 500 and height < 800:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': conf,
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                'width': width,
                                'height': height,
                                'area': width * height
                            })

        except Exception as e:
            print(f"⚠ Ошибка YOLO детекции: {e}")

        return detections

    def _extract_embeddings(self, frame, detections):
        """Извлечение ReID эмбеддингов"""
        detections_with_emb = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])

            # Вырезаем и обрабатываем ROI
            roi = frame[max(0, y1):min(frame.shape[0], y2),
            max(0, x1):min(frame.shape[1], x2)]

            if roi.size == 0:
                continue

            # Проверяем кэш
            roi_hash = hashlib.md5(roi.tobytes()).hexdigest()
            if roi_hash in self.embedding_cache:
                embedding = self.embedding_cache[roi_hash]
            else:
                embedding = self.reid_model.extract_embedding(roi)
                if embedding is not None:
                    self.embedding_cache[roi_hash] = embedding
                    if len(self.embedding_cache) > self.max_cache_size:
                        self.embedding_cache.popitem(last=False)

            if embedding is not None:
                det['embedding'] = embedding
                detections_with_emb.append(det)

        return detections_with_emb

    def _match_tracks(self, detections):
        """Сопоставление треков с детекциями"""
        matched_pairs = []

        if not self.tracks or not detections:
            return matched_pairs

        # Создаем матрицу стоимости
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]

            for j, det in enumerate(detections):
                # 1. IoU стоимость
                iou = self._compute_iou(track.bbox, det['bbox'])
                iou_cost = 1 - iou

                # 2. ReID стоимость
                reid_cost = 1
                if 'embedding' in det and track.embeddings:
                    track_embedding = track.get_embedding()
                    if track_embedding is not None:
                        similarity = 1 - cosine(det['embedding'], track_embedding)
                        reid_cost = 1 - similarity

                # 3. Расстояние между центрами
                center_dist = self._compute_center_distance(track.bbox, det['bbox'])
                dist_cost = min(1, center_dist / 200)

                # 4. Комбинированная стоимость с весами
                weight_iou = 0.4 if iou > 0.1 else 0.1
                weight_reid = 0.4 if track.is_confirmed else 0.3
                weight_dist = 0.2

                total_cost = (weight_iou * iou_cost +
                              weight_reid * reid_cost +
                              weight_dist * dist_cost)

                # 5. Штраф за долгое отсутствие
                if track.consecutive_invisible > 0:
                    total_cost *= (1 + track.consecutive_invisible * 0.1)

                cost_matrix[i, j] = total_cost

        # Венгерский алгоритм для оптимального сопоставления
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.7:  # Порог сопоставления
                matched_pairs.append((track_ids[i], j))

        return matched_pairs

    def _update_tracks(self, matched_pairs, detections):
        """Обновление сопоставленных треков"""
        updated_tracks = []

        for track_id, det_idx in matched_pairs:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                det = detections[det_idx]

                # Обновляем трек
                track.update(det['bbox'], det.get('embedding'))

                # Обновляем базу внешнего вида
                if 'embedding' in det:
                    self.appearance_db.add(track_id, det['embedding'])

                updated_tracks.append(track)

                # Удаляем использованную детекцию
                detections[det_idx]['matched'] = True

        return updated_tracks

    def _create_new_tracks(self, detections, matched_pairs):
        """Создание новых треков"""
        new_tracks = []
        matched_indices = set(j for _, j in matched_pairs)

        for i, det in enumerate(detections):
            if i not in matched_indices and not det.get('matched', False):
                # Создаем новый трек
                track_id = self.next_track_id
                self.next_track_id += 1

                new_track = Track(
                    track_id=track_id,
                    bbox=det['bbox'],
                    embedding=det.get('embedding')
                )

                self.tracks[track_id] = new_track
                new_tracks.append(new_track)

                # Добавляем в базу внешнего вида
                if 'embedding' in det:
                    self.appearance_db.add(track_id, det['embedding'])

        return new_tracks

    def _update_lost_tracks(self):
        """Обновление потерянных треков"""
        tracks_to_remove = []

        for track_id, track in list(self.tracks.items()):
            if not track.visible:
                track.mark_missing()

                # Проверяем, не пора ли удалить трек
                if (track.consecutive_invisible > self.max_age or
                        (not track.is_confirmed and track.consecutive_invisible > 10)):
                    # Перемещаем в lost_tracks
                    self.lost_tracks[track_id] = track
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def _recover_tracks(self, detections, matched_pairs):
        """Восстановление треков из базы внешнего вида"""
        recovered_tracks = []
        matched_indices = set(j for _, j in matched_pairs)

        for i, det in enumerate(detections):
            if i in matched_indices or det.get('matched', False):
                continue

            if 'embedding' in det:
                # Ищем в базе внешнего вида
                matches = self.appearance_db.query(det['embedding'], max_results=3)

                for match in matches:
                    if match['similarity'] > self.reid_threshold:
                        # Восстанавливаем трек
                        track_id = match['track_id']

                        if track_id in self.lost_tracks:
                            # Возвращаем из lost_tracks
                            track = self.lost_tracks[track_id]
                            track.update(det['bbox'], det['embedding'])
                            track.visible = True

                            self.tracks[track_id] = track
                            del self.lost_tracks[track_id]

                            recovered_tracks.append(track)
                            print(f"🔄 Восстановлен трек {track_id} (схожесть: {match['similarity']:.3f})")

                            det['matched'] = True
                            break

        return recovered_tracks

    def _compute_iou(self, box1, box2):
        """Вычисление IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        return inter_area / (box1_area + box2_area - inter_area + 1e-10)

    def _compute_center_distance(self, box1, box2):
        """Вычисление расстояния между центрами"""
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]

        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def _manage_tracks(self):
        """Управление треками"""
        print("📊 Менеджер треков запущен")

        while self.running and not self.stop_event.is_set():
            try:
                # Периодическое сохранение
                if int(time.time()) % 300 == 0:  # Каждые 5 минут
                    self._save_persistent_data()

                # Очистка кэшей
                if len(self.embedding_cache) > self.max_cache_size * 2:
                    for _ in range(self.max_cache_size):
                        if self.embedding_cache:
                            self.embedding_cache.popitem(last=False)

                time.sleep(1)

            except Exception as e:
                print(f"⚠ Ошибка менеджера треков: {e}")
                time.sleep(5)

    def _monitor_system(self):
        """Мониторинг системы"""
        print("📈 Мониторинг системы запущен")

        while self.running and not self.stop_event.is_set():
            try:
                # Логирование статистики
                if self.total_frames % 100 == 0:
                    print(f"📊 Статистика: "
                          f"Кадров: {self.total_frames}, "
                          f"Треков: {len(self.tracks)}, "
                          f"Потерянных: {len(self.lost_tracks)}, "
                          f"FPS: {np.mean(self.fps_history):.1f}")

                time.sleep(1)

            except Exception as e:
                print(f"⚠ Ошибка мониторинга: {e}")
                time.sleep(5)

    def _draw_detections(self, frame, detections):
        """Рисование детекций на кадре"""
        for det in detections:
            bbox = det['bbox']
            track_id = det['track_id']
            confidence = det['confidence']
            age = det['age']
            hits = det['hits']
            color = det['color']
            is_confirmed = det['is_confirmed']

            x1, y1, x2, y2 = map(int, bbox)

            # Цвет в зависимости от статуса
            if not is_confirmed:
                color = (0, 165, 255)  # Оранжевый для неподтвержденных
            elif hits < 10:
                color = (0, 255, 255)  # Желтый для новых
            else:
                color = color  # Уникальный цвет для стабильных

            # Прямоугольник
            thickness = 3 if is_confirmed else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Фон для текста
            status = "✓" if is_confirmed else "?"
            text = f"{status}{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1), color, -1)

            # ID
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Статистика
            stats = f"{hits}h"
            if confidence < 0.9:
                stats += f" {confidence:.0%}"

            cv2.putText(frame, stats, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Общая статистика
        stats_text = (f"Людей: {len(detections)} | "
                      f"Треков: {len(self.tracks)} | "
                      f"FPS: {np.mean(self.fps_history):.1f}" if self.fps_history else "0")

        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_current_frame(self):
        """Получить текущий обработанный кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_statistics(self):
        """Получить статистику"""
        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'total_detections': self.total_detections,
            'total_frames': self.total_frames,
            'active_tracks': len(self.tracks),
            'lost_tracks': len(self.lost_tracks),
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'appearance_db_size': len(self.appearance_db.embeddings)
        }

    def get_detailed_statistics(self):
        """Подробная статистика"""
        tracks_info = []
        for track_id, track in list(self.tracks.items())[:20]:
            tracks_info.append({
                'id': track_id,
                'age': track.age,
                'hits': track.hits,
                'confirmed': track.is_confirmed,
                'invisible': track.consecutive_invisible,
                'reliable': track.is_reliable()
            })

        return {
            'summary': self.get_statistics(),
            'tracks': tracks_info,
            'timestamp': datetime.now().isoformat()
        }

    def stop(self):
        """Остановка"""
        print("🛑 Остановка видео процессора...")

        self.running = False
        self.stop_event.set()

        # Сохраняем данные
        self._save_persistent_data()

        # Освобождаем ресурсы
        if self.cap:
            self.cap.release()

        # Очищаем очереди
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.processed_queue.empty():
            self.processed_queue.get()

        print("✅ Видео процессор остановлен")