# /top_eye/src/core/video_processor_occlusion.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import time
import os
import pickle
import hashlib
from collections import deque, defaultdict
import json


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Инициализация процессора для камеры: {config.CAMERA_ID}")

        # Инициализация камеры
        self.cap = None
        self.last_reconnect = time.time()
        self.reconnect_interval = 5

        # Очереди
        self.frame_queue = Queue(maxsize=15)
        self.processed_queue = Queue(maxsize=15)
        self.lock = Lock()
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()
        self.visitor_history = {}
        self.total_visitors = 0

        # Продвинутый трекинг с окклюзиями
        self.active_tracks = {}  # {track_id: TrackObject}
        self.occluded_tracks = {}  # Треки в окклюзии
        self.group_manager = GroupManager()  # Менеджер групп
        self.next_track_id = 1000

        # Настройки
        self.max_occlusion_time = 45  # кадров (1.5 сек при 30 FPS)
        self.occlusion_threshold = 0.3  # порог перекрытия для окклюзии
        self.group_threshold = 50  # расстояние для объединения в группу

        # YOLO модель
        self.model = None

        # База данных
        self.visitors_db_path = "data/visitors_advanced.db"
        self._load_visitors_db()

        # История для сглаживания
        self.track_history = defaultdict(lambda: deque(maxlen=10))

        print("✓ Продвинутый трекинг с обработкой окклюзий")

    def _load_visitors_db(self):
        """Загрузка базы данных"""
        try:
            if os.path.exists(self.visitors_db_path):
                with open(self.visitors_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.visitor_history = data.get('visitor_history', {})
                    self.next_track_id = data.get('next_track_id', 1000)
                print(f"✓ Загружена история {len(self.visitor_history)} посетителей")
        except:
            print("⚠ Новая база данных")
            self.visitor_history = {}

    def _save_visitors_db(self):
        """Сохранение базы данных"""
        try:
            os.makedirs(os.path.dirname(self.visitors_db_path), exist_ok=True)
            data = {
                'visitor_history': self.visitor_history,
                'next_track_id': self.next_track_id,
                'saved_at': datetime.now().isoformat()
            }
            with open(self.visitors_db_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Ошибка сохранения: {e}")

    def init_models(self):
        """Инициализация моделей"""
        try:
            print("Загрузка моделей...")

            from ultralytics import YOLO
            model_path = self.config.YOLO_MODEL_PATH

            if not os.path.exists(model_path):
                print("Скачиваем YOLOv8n...")
                model_path = 'yolov8n.pt'

            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ YOLO модель загружена на {'CUDA' if torch.cuda.is_available() else 'CPU'}")

            print("✓ Используется продвинутый трекинг с окклюзиями")

        except Exception as e:
            print(f"✗ Ошибка загрузки: {e}")

    def start(self):
        """Запуск"""
        self.running = True
        self.init_models()
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()
        Thread(target=self._manage_tracks, daemon=True).start()
        print("✓ Обработка запущена")

    def _capture_frames(self):
        """Захват кадров"""
        frame_count = 0

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if time.time() - self.last_reconnect > self.reconnect_interval:
                        print("Переподключение к камере...")
                        self._reconnect_camera()
                        time.sleep(1)
                        continue
                    else:
                        time.sleep(0.1)
                        continue

                success, frame = self.cap.read()

                if not success:
                    print("✗ Ошибка чтения кадра")
                    self.cap.release()
                    self.cap = None
                    continue

                if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())

                frame_count += 1
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
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)

            time.sleep(0.5)

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
                    processed_data = self._process_single_frame(frame)

                    if not self.processed_queue.full():
                        self.processed_queue.put(processed_data)

            except Exception as e:
                print(f"Ошибка обработки: {e}")

    def _manage_tracks(self):
        """Управление треками и окклюзиями"""
        while self.running:
            try:
                current_time = time.time()

                # Обновляем окклюзированные треки
                occluded_to_remove = []
                for track_id, track_data in list(self.occluded_tracks.items()):
                    if current_time - track_data['occluded_since'] > self.max_occlusion_time / 30:
                        occluded_to_remove.append(track_id)

                for track_id in occluded_to_remove:
                    del self.occluded_tracks[track_id]

                # Сохраняем базу каждые 2 минуты
                if int(current_time) % 120 == 0:
                    self._save_visitors_db()

                time.sleep(0.5)

            except Exception as e:
                print(f"Ошибка управления треками: {e}")
                time.sleep(2)

    def _process_single_frame(self, frame):
        """Обработка кадра с учетом окклюзий"""
        result = {
            'frame': frame.copy(),
            'detections': [],
            'occluded': [],
            'timestamp': datetime.now(),
            'people_count': 0,
            'occluded_count': 0,
            'fps': 0
        }

        try:
            if self.model is not None:
                start_time = time.time()

                # Детекция
                with torch.no_grad():
                    yolo_results = self.model(
                        frame,
                        conf=self.config.CONFIDENCE_THRESHOLD,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        verbose=False,
                        classes=[0]
                    )

                # Текущие детекции
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

                            if width > 30 and height > 60:
                                person_roi = frame[y1:y2, x1:x2]
                                appearance_hash = self._get_appearance_hash(person_roi)

                                current_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                    'confidence': conf,
                                    'width': width,
                                    'height': height,
                                    'appearance_hash': appearance_hash,
                                    'area': width * height
                                })

                # Обработка окклюзий
                processed_detections = self._handle_occlusions(current_detections)

                # Обновляем активные треки
                current_time = time.time()
                for det in processed_detections:
                    track_id = det['track_id']
                    is_occluded = det.get('is_occluded', False)

                    if is_occluded:
                        # Трек в окклюзии
                        if track_id not in self.occluded_tracks:
                            self.occluded_tracks[track_id] = {
                                'last_bbox': det['bbox'],
                                'occluded_since': current_time,
                                'appearance_hash': det['appearance_hash']
                            }
                    else:
                        # Активный трек
                        if track_id not in self.active_tracks:
                            self.active_tracks[track_id] = {
                                'bbox': det['bbox'],
                                'last_seen': current_time,
                                'first_seen': current_time,
                                'appearance_hash': det['appearance_hash'],
                                'age': 1,
                                'occlusion_count': 0,
                                'velocity': [0, 0]
                            }
                        else:
                            # Обновляем с прогнозом движения
                            old_bbox = self.active_tracks[track_id]['bbox']
                            new_bbox = det['bbox']

                            # Прогноз положения на основе скорости
                            velocity = self.active_tracks[track_id]['velocity']
                            predicted_bbox = self._predict_bbox(old_bbox, velocity)

                            # Сглаживание
                            smoothed_bbox = self._smooth_bbox(old_bbox, new_bbox, predicted_bbox)

                            # Обновляем скорость
                            dx = (smoothed_bbox[0] - old_bbox[0] + smoothed_bbox[2] - old_bbox[2]) / 2
                            dy = (smoothed_bbox[1] - old_bbox[1] + smoothed_bbox[3] - old_bbox[3]) / 2
                            velocity = [velocity[0] * 0.7 + dx * 0.3,
                                        velocity[1] * 0.7 + dy * 0.3]

                            self.active_tracks[track_id].update({
                                'bbox': smoothed_bbox,
                                'last_seen': current_time,
                                'appearance_hash': det['appearance_hash'],
                                'age': self.active_tracks[track_id]['age'] + 1,
                                'velocity': velocity
                            })

                        # Добавляем в результат
                        result['detections'].append({
                            'track_id': track_id,
                            'bbox': self.active_tracks[track_id]['bbox'],
                            'confidence': det['confidence'],
                            'age': self.active_tracks[track_id]['age'],
                            'velocity': self.active_tracks[track_id]['velocity']
                        })

                # Обрабатываем окклюзированные треки
                for track_id, track_data in self.occluded_tracks.items():
                    # Пробуем восстановить окклюзированные треки
                    restored = self._try_restore_occluded(track_id, current_detections)
                    if not restored:
                        result['occluded'].append({
                            'track_id': track_id,
                            'last_bbox': track_data['last_bbox'],
                            'occluded_since': track_data['occluded_since']
                        })

                # Обновляем статистику
                result['people_count'] = len(result['detections'])
                result['occluded_count'] = len(result['occluded'])
                self.current_count = result['people_count']

                # Обновляем уникальных
                for det in result['detections']:
                    track_id = det['track_id']
                    if self.active_tracks[track_id]['age'] > 10:
                        self.session_unique.add(track_id)
                        today = datetime.now().date().isoformat()
                        self.today_unique.add(f"{today}_{track_id}")

                # FPS
                end_time = time.time()
                result['fps'] = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

                # Рисуем
                self._draw_detections(frame, result['detections'], result['occluded'])

        except Exception as e:
            print(f"Ошибка обработки: {e}")

        return result

    def _handle_occlusions(self, current_detections):
        """Обработка окклюзий и группового движения"""
        processed = []

        if not current_detections:
            return processed

        # 1. Анализ пересечений
        intersections = self._analyze_intersections(current_detections)

        # 2. Группировка близких детекций
        groups = self._group_detections(current_detections)

        # 3. Сопоставление с активными треками с учетом групп
        matched_detections = set()
        matched_tracks = set()

        if self.active_tracks:
            # Для каждого активного трека ищем лучшую детекцию
            for track_id, track_data in self.active_tracks.items():
                best_det_idx = -1
                best_score = 0

                for i, det in enumerate(current_detections):
                    if i in matched_detections:
                        continue

                    # Учет окклюзии
                    is_occluded = self._check_occlusion(det['bbox'], intersections)

                    # Score с учетом различных факторов
                    score = self._compute_occlusion_score(
                        det, track_data, is_occluded, groups
                    )

                    if score > best_score and score > 0.3:
                        best_score = score
                        best_det_idx = i

                if best_det_idx != -1:
                    # Сопоставляем
                    det = current_detections[best_det_idx]
                    det['track_id'] = track_id
                    det['is_occluded'] = self._check_occlusion(det['bbox'], intersections)
                    processed.append(det)

                    matched_detections.add(best_det_idx)
                    matched_tracks.add(track_id)

        # 4. Восстановление из окклюзий
        for track_id, track_data in list(self.occluded_tracks.items()):
            if track_id in matched_tracks:
                continue

            best_det_idx = -1
            best_score = 0

            for i, det in enumerate(current_detections):
                if i in matched_detections:
                    continue

                # Проверяем схожесть с окклюзированным треком
                score = self._compare_with_occluded(det, track_data)

                if score > best_score and score > 0.5:
                    best_score = score
                    best_det_idx = i

            if best_det_idx != -1:
                # Восстанавливаем трек
                det = current_detections[best_det_idx]
                det['track_id'] = track_id
                det['is_occluded'] = False
                processed.append(det)

                matched_detections.add(best_det_idx)

                # Перемещаем из окклюзированных в активные
                self.active_tracks[track_id] = {
                    'bbox': det['bbox'],
                    'last_seen': time.time(),
                    'first_seen': time.time(),
                    'appearance_hash': det['appearance_hash'],
                    'age': track_data.get('age', 0) + 1,
                    'occlusion_count': track_data.get('occlusion_count', 0),
                    'velocity': [0, 0]
                }
                del self.occluded_tracks[track_id]

                print(f"🔄 Восстановлен трек {track_id} после окклюзии")

        # 5. Новые детекции
        for i, det in enumerate(current_detections):
            if i not in matched_detections:
                track_id = self._get_new_track_id()
                det['track_id'] = track_id
                det['is_occluded'] = self._check_occlusion(det['bbox'], intersections)
                processed.append(det)

        return processed

    def _analyze_intersections(self, detections):
        """Анализирует пересечения между bounding boxes"""
        intersections = []
        n = len(detections)

        for i in range(n):
            for j in range(i + 1, n):
                iou = self._compute_iou(detections[i]['bbox'], detections[j]['bbox'])
                if iou > self.occlusion_threshold:
                    intersections.append({
                        'det1_idx': i,
                        'det2_idx': j,
                        'iou': iou,
                        'center_dist': self._compute_distance(
                            detections[i]['center'], detections[j]['center']
                        )
                    })

        return intersections

    def _group_detections(self, detections):
        """Группирует близко расположенные детекции"""
        groups = []
        n = len(detections)
        visited = [False] * n

        for i in range(n):
            if not visited[i]:
                group = [i]
                visited[i] = True

                # Находим всех близких соседей
                queue = [i]
                while queue:
                    current = queue.pop(0)
                    for j in range(n):
                        if not visited[j]:
                            dist = self._compute_distance(
                                detections[current]['center'],
                                detections[j]['center']
                            )
                            if dist < self.group_threshold:
                                group.append(j)
                                visited[j] = True
                                queue.append(j)

                if len(group) > 1:
                    groups.append(group)

        return groups

    def _compute_occlusion_score(self, det, track_data, is_occluded, groups):
        """Вычисляет score для сопоставления с учетом окклюзий"""
        score = 0

        # 1. Расстояние (основной фактор)
        center1 = det['center']
        bbox = track_data['bbox']
        center2 = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        distance = self._compute_distance(center1, center2)

        # Учитываем скорость для прогноза
        velocity = track_data.get('velocity', [0, 0])
        predicted_center = [center2[0] + velocity[0], center2[1] + velocity[1]]
        predicted_distance = self._compute_distance(center1, predicted_center)

        distance_score = max(0, 1 - min(distance, predicted_distance) / 100)
        score += 0.5 * distance_score

        # 2. Визуальная схожесть
        if det['appearance_hash'] and track_data.get('appearance_hash'):
            hash_sim = self._compare_hashes(det['appearance_hash'],
                                            track_data['appearance_hash'])
            score += 0.3 * hash_sim

        # 3. Учет групп
        group_bonus = self._get_group_bonus(det, track_data, groups)
        score += 0.2 * group_bonus

        # 4. Штраф за окклюзию
        if is_occluded:
            score *= 0.8  # Штрафуем окклюзированные

        return score

    def _check_occlusion(self, bbox, intersections):
        """Проверяет, находится ли детекция в окклюзии"""
        for inter in intersections:
            # Упрощенная проверка - если есть значительное пересечение
            if inter['iou'] > 0.5:
                return True
        return False

    def _compare_with_occluded(self, det, track_data):
        """Сравнивает с окклюзированным треком"""
        score = 0

        # Расстояние до последней известной позиции
        last_bbox = track_data.get('last_bbox', [0, 0, 0, 0])
        last_center = [(last_bbox[0] + last_bbox[2]) / 2,
                       (last_bbox[1] + last_bbox[3]) / 2]

        distance = self._compute_distance(det['center'], last_center)
        distance_score = max(0, 1 - distance / 150)
        score += 0.6 * distance_score

        # Визуальная схожесть
        if det['appearance_hash'] and track_data.get('appearance_hash'):
            hash_sim = self._compare_hashes(det['appearance_hash'],
                                            track_data['appearance_hash'])
            score += 0.4 * hash_sim

        return score

    def _try_restore_occluded(self, track_id, current_detections):
        """Пытается восстановить окклюзированный трек"""
        track_data = self.occluded_tracks.get(track_id)
        if not track_data:
            return False

        for det in current_detections:
            score = self._compare_with_occluded(det, track_data)
            if score > 0.6:  # Высокий порог для восстановления
                return True

        return False

    def _predict_bbox(self, bbox, velocity):
        """Прогнозирует bbox на основе скорости"""
        x1, y1, x2, y2 = bbox
        dx, dy = velocity
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def _smooth_bbox(self, old_bbox, new_bbox, predicted_bbox):
        """Сглаживает bbox"""
        alpha = 0.7  # Коэффициент сглаживания

        smoothed = []
        for o, n, p in zip(old_bbox, new_bbox, predicted_bbox):
            # Используем комбинацию нового значения и прогноза
            value = alpha * n + (1 - alpha) * p
            smoothed.append(value)

        return smoothed

    def _get_group_bonus(self, det, track_data, groups):
        """Вычисляет бонус за нахождение в одной группе"""
        # Здесь можно реализовать логику для группового движения
        # Например, если два человека движутся вместе, они могут сохранять свои ID
        return 0.5  # Фиксированный бонус

    def _get_appearance_hash(self, image):
        """Визуальный хеш"""
        if image is None or image.size == 0:
            return "0"

        try:
            resized = cv2.resize(image, (32, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Простой хеш на основе яркости
            avg = np.mean(gray)
            binary = (gray > avg).flatten()
            return ''.join(['1' if b else '0' for b in binary])
        except:
            return "0"

    def _compare_hashes(self, hash1, hash2):
        """Сравнивает два хеша"""
        if len(hash1) != len(hash2):
            return 0

        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return matches / len(hash1)

    def _compute_iou(self, box1, box2):
        """IoU"""
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

    def _compute_distance(self, point1, point2):
        """Расстояние между точками"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _get_new_track_id(self):
        """Новый ID"""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _draw_detections(self, frame, detections, occluded):
        """Рисует детекции с разметкой окклюзий"""
        # Активные треки
        for det in detections:
            bbox = det['bbox']
            track_id = det['track_id']
            age = det.get('age', 1)
            velocity = det.get('velocity', [0, 0])

            x1, y1, x2, y2 = map(int, bbox)

            # Цвет в зависимости от возраста и скорости
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

            if speed < 2:
                color = (0, 255, 0)  # зеленый - неподвижный
                status = "STABLE"
            elif speed < 10:
                color = (0, 255, 255)  # желтый - движется
                status = "MOVING"
            else:
                color = (0, 0, 255)  # красный - быстро движется
                status = "FAST"

            # Прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID и статус
            text = f"ID: {track_id} ({status})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Возраст трека
            age_text = f"Age: {age}f"
            cv2.putText(frame, age_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Окклюзированные треки (полупрозрачные)
        for occ in occluded:
            bbox = occ['last_bbox']
            track_id = occ['track_id']
            occluded_time = time.time() - occ['occluded_since']

            x1, y1, x2, y2 = map(int, bbox)

            # Полупрозрачный красный для окклюзированных
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Текст
            text = f"OCC: {track_id}"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Время окклюзии
            time_text = f"{occluded_time:.1f}s"
            cv2.putText(frame, time_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Статистика
        stats_text = f"Активных: {len(detections)} | Окклюзий: {len(occluded)} | Всего треков: {len(self.active_tracks) + len(self.occluded_tracks)}"
        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Легенда
        legend = [
            ("🟢 Неподвижный", (0, 255, 0)),
            ("🟡 Движется", (0, 255, 255)),
            ("🔴 Быстро движется", (0, 0, 255)),
            ("⚫ Окклюзия", (0, 0, 255))
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
        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'active_tracks': len(self.active_tracks),
            'occluded_tracks': len(self.occluded_tracks),
            'total_visitors': len(self.visitor_history)
        }

    def stop(self):
        """Остановка"""
        self.running = False
        self._save_visitors_db()
        if self.cap:
            self.cap.release()
        print("✓ Обработка остановлена")


class GroupManager:
    """Менеджер группового движения"""

    def __init__(self):
        self.groups = {}  # {group_id: [track_ids]}
        self.next_group_id = 1

    def update_groups(self, detections):
        """Обновляет группы"""
        # Простая группировка по расстоянию
        groups = []
        visited = set()

        for i, det1 in enumerate(detections):
            if i in visited:
                continue

            group = [i]
            visited.add(i)

            for j, det2 in enumerate(detections):
                if j in visited:
                    continue

                # Проверяем расстояние
                dist = np.sqrt(
                    (det1['center'][0] - det2['center'][0]) ** 2 +
                    (det1['center'][1] - det2['center'][1]) ** 2
                )

                if dist < 100:  # Порог для группы
                    group.append(j)
                    visited.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    # В класс VideoProcessor добавьте:

    def _use_kalman_filter(self):
        """Использование фильтра Калмана для прогнозирования"""
        # Фильтр Калмана для каждого трека
        pass

    def _deep_sort_integration(self):
        """Интеграция с DeepSORT для лучшего трекинга"""