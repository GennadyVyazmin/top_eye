# /top_eye/src/core/video_processor_reid.py
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import time
import os
import pickle
from collections import deque, defaultdict
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')


class ReIDModel(nn.Module):
    """Упрощенная ReID модель на основе ResNet"""

    def __init__(self):
        super(ReIDModel, self).__init__()
        # Используем предобученный ResNet
        from torchvision import models
        self.backbone = models.resnet18(pretrained=True)
        # Убираем последний слой
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # Замораживаем часть слоев
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        # Дополнительные слои для ReID
        self.reid_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)  # 128-мерный эмбеддинг
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.reid_head(features)
        # Нормализуем эмбеддинги
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Инициализация процессора для камеры: {config.CAMERA_ID}")

        # Камера
        self.cap = None
        self.last_reconnect = time.time()
        self.reconnect_interval = 5

        # Очереди
        self.frame_queue = Queue(maxsize=20)
        self.processed_queue = Queue(maxsize=20)
        self.lock = Lock()
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()
        self.visitor_embeddings = {}  # {track_id: embeddings_history}
        self.visitor_appearances = {}  # {track_id: appearance_samples}

        # Модели
        self.model = None  # YOLO
        self.reid_model = None  # ReID модель
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Трансформы для ReID
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Стандартный размер для ReID
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Продвинутый трекинг
        self.active_tracks = {}
        self.lost_tracks = {}  # Потерянные треки (кратковременно)
        self.next_track_id = 1000
        self.track_counter = 0

        # Настройки трекинга
        self.max_age = 30  # Максимальный возраст трека без обновления
        self.min_hits = 3  # Минимальное количество попаданий для подтверждения
        self.iou_threshold = 0.3
        self.reid_threshold = 0.7  # Порог для ReID совпадения
        self.max_features_per_track = 10  # Максимум эмбеддингов на трек

        # Кэш для эмбеддингов
        self.embedding_cache = {}

        # Инициализация
        self.init_models()
        print(f"✓ Инициализация завершена на устройстве: {self.device}")

    def init_models(self):
        """Инициализация моделей"""
        try:
            print("Загрузка YOLO...")
            from ultralytics import YOLO
            model_path = self.config.YOLO_MODEL_PATH

            if not os.path.exists(model_path):
                print("Скачиваем YOLOv8n...")
                model_path = 'yolov8n.pt'

            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✓ YOLO загружен на {self.device}")

            print("Загрузка ReID модели...")
            self.reid_model = ReIDModel().to(self.device)
            self.reid_model.eval()  # Режим inference
            print("✓ ReID модель загружена")

            # Загружаем веса если есть
            reid_weights = os.path.join(os.path.dirname(__file__), "../../models/reid_weights.pth")
            if os.path.exists(reid_weights):
                self.reid_model.load_state_dict(torch.load(reid_weights, map_location=self.device))
                print("✓ Веса ReID модели загружены")

        except Exception as e:
            print(f"✗ Ошибка загрузки моделей: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        """Запуск"""
        self.running = True
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()
        Thread(target=self._manage_tracks, daemon=True).start()
        print("✓ Обработка запущена")

    def _capture_frames(self):
        """Захват кадров"""
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if time.time() - self.last_reconnect > self.reconnect_interval:
                        self._reconnect_camera()
                        time.sleep(1)
                    continue

                success, frame = self.cap.read()
                if not success:
                    self.cap.release()
                    self.cap = None
                    continue

                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                time.sleep(0.01)

            except Exception as e:
                print(f"Ошибка захвата: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)

    def _reconnect_camera(self):
        """Переподключение"""
        try:
            if self.cap:
                self.cap.release()

            print(f"Подключение к: {self.config.RTSP_URL}")
            self.cap = cv2.VideoCapture(self.config.RTSP_URL)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)

            if self.cap.isOpened():
                print("✓ Камера подключена")
                self.last_reconnect = time.time()
                return True
            else:
                print("✗ Не удалось подключиться")
                return False

        except Exception as e:
            print(f"Ошибка подключения: {e}")
            return False

    def _process_frames(self):
        """Обработка кадров"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    processed = self._process_single_frame(frame)

                    if not self.processed_queue.full():
                        self.processed_queue.put(processed)

            except Exception as e:
                print(f"Ошибка обработки: {e}")

    def _manage_tracks(self):
        """Управление треками"""
        while self.running:
            try:
                current_time = time.time()

                # Обновляем возраст треков
                tracks_to_remove = []
                for track_id, track in list(self.active_tracks.items()):
                    if current_time - track['last_seen'] > self.max_age / 10:
                        # Перемещаем в потерянные
                        self.lost_tracks[track_id] = {
                            **track,
                            'lost_since': current_time
                        }
                        tracks_to_remove.append(track_id)
                        print(f"Трек {track_id} перемещен в потерянные")

                for track_id in tracks_to_remove:
                    del self.active_tracks[track_id]

                # Очищаем старые потерянные треки
                lost_to_remove = []
                for track_id, track in list(self.lost_tracks.items()):
                    if current_time - track['lost_since'] > 5:  # 5 секунд
                        lost_to_remove.append(track_id)

                for track_id in lost_to_remove:
                    del self.lost_tracks[track_id]

                time.sleep(0.5)

            except Exception as e:
                print(f"Ошибка управления треками: {e}")
                time.sleep(1)

    def _process_single_frame(self, frame):
        """Обработка кадра"""
        result = {
            'frame': frame.copy(),
            'detections': [],
            'timestamp': datetime.now(),
            'people_count': 0,
            'fps': 0,
            'track_info': []
        }

        try:
            start_time = time.time()

            # YOLO детекция
            with torch.no_grad():
                yolo_results = self.model(
                    frame,
                    conf=self.config.CONFIDENCE_THRESHOLD,
                    device=self.device,
                    verbose=False,
                    classes=[0]
                )

            # Извлекаем детекции
            current_detections = []
            if yolo_results and len(yolo_results) > 0:
                yolo_result = yolo_results[0]

                if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                    boxes = yolo_result.boxes.xyxy.cpu().numpy()
                    confidences = yolo_result.boxes.conf.cpu().numpy()

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        conf = float(confidences[i])

                        width = x2 - x1
                        height = y2 - y1

                        if width > 40 and height > 80:  # Фильтр размера
                            # Вырезаем и нормализуем
                            person_roi = frame[y1:y2, x1:x2]

                            # Получаем ReID эмбеддинг
                            embedding = self._get_embedding(person_roi)

                            if embedding is not None:
                                current_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                    'confidence': conf,
                                    'embedding': embedding,
                                    'roi': person_roi,
                                    'width': width,
                                    'height': height
                                })

            # Продвинутый трекинг
            tracked = self._advanced_tracking(current_detections)

            # Обновляем активные треки
            current_time = time.time()
            for det in tracked:
                track_id = det['track_id']

                if track_id not in self.active_tracks:
                    # Новый трек
                    self.active_tracks[track_id] = {
                        'bbox': det['bbox'],
                        'last_seen': current_time,
                        'first_seen': current_time,
                        'embeddings': [det['embedding']],
                        'hits': 1,
                        'age': 1,
                        'color': self._get_random_color(track_id)
                    }

                    # Сохраняем внешний вид
                    self.visitor_appearances[track_id] = [det['roi']]
                else:
                    # Обновляем существующий трек
                    track = self.active_tracks[track_id]

                    # Обновляем эмбеддинги (скользящее окно)
                    if len(track['embeddings']) >= self.max_features_per_track:
                        track['embeddings'].pop(0)
                    track['embeddings'].append(det['embedding'])

                    # Обновляем внешний вид
                    if track_id in self.visitor_appearances:
                        if len(self.visitor_appearances[track_id]) < 5:
                            self.visitor_appearances[track_id].append(det['roi'])

                    track.update({
                        'bbox': det['bbox'],
                        'last_seen': current_time,
                        'hits': track['hits'] + 1,
                        'age': track['age'] + 1
                    })

                # Добавляем в результат
                result['detections'].append({
                    'track_id': track_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'age': self.active_tracks[track_id]['age'],
                    'hits': self.active_tracks[track_id]['hits']
                })

            # Обновляем статистику
            result['people_count'] = len(result['detections'])
            self.current_count = result['people_count']

            # Обновляем уникальных
            for det in result['detections']:
                track_id = det['track_id']
                if self.active_tracks[track_id]['hits'] > 10:
                    self.session_unique.add(track_id)
                    today = datetime.now().date().isoformat()
                    self.today_unique.add(f"{today}_{track_id}")

            # FPS
            end_time = time.time()
            result['fps'] = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

            # Рисуем
            self._draw_detections(frame, result['detections'])

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _get_embedding(self, image):
        """Получает эмбеддинг из изображения"""
        if image is None or image.size == 0:
            return None

        try:
            # Кэширование по хешу изображения
            img_hash = hashlib.md5(image.tobytes()).hexdigest()
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]

            # Препроцессинг
            transformed = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                embedding = self.reid_model(transformed)
                embedding = embedding.cpu().numpy().flatten()

            # Кэшируем
            self.embedding_cache[img_hash] = embedding
            if len(self.embedding_cache) > 1000:
                # Очищаем старые записи
                keys = list(self.embedding_cache.keys())
                for key in keys[:-500]:
                    del self.embedding_cache[key]

            return embedding

        except Exception as e:
            print(f"Ошибка получения эмбеддинга: {e}")
            return None

    def _advanced_tracking(self, current_detections):
        """Продвинутый трекинг с ReID"""
        tracked_detections = []

        if not current_detections:
            return tracked_detections

        # 1. Сопоставление с активными треками
        matched_detections = set()
        matched_tracks = set()

        if self.active_tracks:
            # Создаем матрицу схожести
            similarity_matrix = []

            for i, det in enumerate(current_detections):
                for track_id, track in self.active_tracks.items():
                    # Вычисляем IoU
                    iou = self._compute_iou(det['bbox'], track['bbox'])

                    # Вычисляем ReID схожесть
                    reid_similarity = 0
                    if det['embedding'] is not None and track['embeddings']:
                        # Сравниваем со всеми эмбеддингами трека
                        similarities = []
                        for track_emb in track['embeddings']:
                            sim = 1 - cosine(det['embedding'], track_emb)
                            similarities.append(sim)
                        reid_similarity = max(similarities) if similarities else 0

                    # Комбинированный score
                    if iou > self.iou_threshold:
                        # При высоком IoU используем его
                        score = 0.7 * min(1, iou / 0.5) + 0.3 * reid_similarity
                    else:
                        # При низком IoU больше полагаемся на ReID
                        score = 0.3 * min(1, iou / 0.5) + 0.7 * reid_similarity

                    similarity_matrix.append((i, track_id, score, iou, reid_similarity))

            # Сортируем по score
            similarity_matrix.sort(key=lambda x: x[2], reverse=True)

            # Жадное сопоставление
            for i, track_id, score, iou, reid_sim in similarity_matrix:
                if score > 0.4:  # Порог сопоставления
                    if i not in matched_detections and track_id not in matched_tracks:
                        det = current_detections[i]
                        det['track_id'] = track_id
                        det['match_score'] = score
                        det['iou'] = iou
                        det['reid_sim'] = reid_sim
                        tracked_detections.append(det)

                        matched_detections.add(i)
                        matched_tracks.add(track_id)

        # 2. Восстановление потерянных треков
        for i, det in enumerate(current_detections):
            if i in matched_detections:
                continue

            best_track_id = None
            best_score = 0

            for track_id, track in self.lost_tracks.items():
                if 'embeddings' in track and track['embeddings']:
                    # Сравниваем эмбеддинги
                    similarities = []
                    for track_emb in track['embeddings']:
                        if det['embedding'] is not None:
                            sim = 1 - cosine(det['embedding'], track_emb)
                            similarities.append(sim)

                    if similarities:
                        score = max(similarities)
                        if score > self.reid_threshold and score > best_score:
                            best_score = score
                            best_track_id = track_id

            if best_track_id:
                # Восстанавливаем трек
                det['track_id'] = best_track_id
                det['match_score'] = best_score
                det['recovered'] = True
                tracked_detections.append(det)

                matched_detections.add(i)

                # Возвращаем в активные
                self.active_tracks[best_track_id] = self.lost_tracks[best_track_id]
                self.active_tracks[best_track_id]['last_seen'] = time.time()
                self.active_tracks[best_track_id]['hits'] += 1
                self.active_tracks[best_track_id]['age'] += 1

                # Обновляем эмбеддинги
                if det['embedding'] is not None:
                    if len(self.active_tracks[best_track_id]['embeddings']) >= self.max_features_per_track:
                        self.active_tracks[best_track_id]['embeddings'].pop(0)
                    self.active_tracks[best_track_id]['embeddings'].append(det['embedding'])

                del self.lost_tracks[best_track_id]
                print(f"✅ Восстановлен трек {best_track_id} (ReID score: {best_score:.3f})")

        # 3. Новые детекции
        for i, det in enumerate(current_detections):
            if i not in matched_detections:
                track_id = self._get_new_track_id()
                det['track_id'] = track_id
                det['new'] = True
                tracked_detections.append(det)

        return tracked_detections

    def _compute_iou(self, box1, box2):
        """Вычисляет IoU"""
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

    def _get_random_color(self, track_id):
        """Генерирует цвет на основе ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    def _get_new_track_id(self):
        """Новый ID"""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _draw_detections(self, frame, detections):
        """Рисует детекции"""
        for det in detections:
            bbox = det['bbox']
            track_id = det['track_id']
            confidence = det['confidence']
            age = det.get('age', 1)
            hits = det.get('hits', 1)

            x1, y1, x2, y2 = map(int, bbox)

            # Получаем цвет трека
            color = self.active_tracks.get(track_id, {}).get('color', (0, 255, 0))

            # Прямоугольник
            thickness = 2 if hits > 10 else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # ID и confidence
            text = f"ID: {track_id}"
            if det.get('recovered'):
                text = f"↻{track_id}"
            elif det.get('new'):
                text = f"NEW {track_id}"

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Статистика
            stats_text = f"{confidence:.0%} ({hits}h)"
            cv2.putText(frame, stats_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Дополнительная информация при наведении
            if 'match_score' in det:
                match_text = f"M: {det['match_score']:.2f}"
                cv2.putText(frame, match_text, (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Общая статистика
        stats = f"Людей: {len(detections)} | Активных треков: {len(self.active_tracks)} | Восстановлено: {sum(1 for d in detections if d.get('recovered', False))}"
        cv2.putText(frame, stats, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Легенда
        legend = [
            ("🟢 Стабильный (>10 попаданий)", (0, 255, 0)),
            ("🟡 Новый (<10 попаданий)", (255, 255, 0)),
            ("↻ Восстановленный", (255, 0, 255))
        ]

        y_offset = 60
        for text, color in legend:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

    def get_current_frame(self):
        """Получить кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_statistics(self):
        """Статистика"""
        stable_tracks = sum(1 for t in self.active_tracks.values() if t['hits'] > 10)

        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'active_tracks': len(self.active_tracks),
            'stable_tracks': stable_tracks,
            'lost_tracks': len(self.lost_tracks),
            'avg_track_age': np.mean([t['age'] for t in self.active_tracks.values()])
            if self.active_tracks else 0
        }

    def stop(self):
        """Остановка"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("✓ Обработка остановлена")