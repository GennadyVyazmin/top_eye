# /top_eye/src/core/video_processor_advanced.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime, timedelta
import time
import os
import pickle
import hashlib
from collections import deque
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
        self.frame_queue = Queue(maxsize=10)
        self.processed_queue = Queue(maxsize=10)
        self.lock = Lock()
        self.running = False

        # Статистика
        self.current_count = 0
        self.today_unique = set()
        self.session_unique = set()
        self.visitor_history = {}  # История посетителей
        self.total_visitors = 0

        # Долгосрочная память
        self.active_tracks = {}  # Активные треки {track_id: data}
        self.inactive_tracks = {}  # Неактивные треки (еще в памяти)
        self.known_visitors = {}  # Известные посетители (долговременная память)
        self.next_track_id = 1000  # Начинаем с большого числа

        # Настройки трекинга
        self.max_disappeared = 60  # кадров до перехода в неактивные
        self.max_forget = 300  # кадров до полного забывания
        self.reid_threshold = 0.6  # порог для re-identification

        # YOLO модель
        self.model = None

        # ReID модель (упрощенная)
        self.reid_model = None

        # База данных посетителей
        self.visitors_db_path = "data/visitors.db"
        self._load_visitors_db()

        print("✓ Видеопроцессор с долгосрочной памятью инициализирован")

    def _load_visitors_db(self):
        """Загрузка базы данных посетителей"""
        try:
            if os.path.exists(self.visitors_db_path):
                with open(self.visitors_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_visitors = data.get('known_visitors', {})
                    self.next_track_id = data.get('next_track_id', 1000)
                print(f"✓ Загружено {len(self.known_visitors)} известных посетителей")
        except:
            print("⚠ Новая база данных посетителей")
            self.known_visitors = {}

    def _save_visitors_db(self):
        """Сохранение базы данных посетителей"""
        try:
            os.makedirs(os.path.dirname(self.visitors_db_path), exist_ok=True)
            data = {
                'known_visitors': self.known_visitors,
                'next_track_id': self.next_track_id,
                'saved_at': datetime.now().isoformat()
            }
            with open(self.visitors_db_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Ошибка сохранения базы данных: {e}")

    def init_models(self):
        """Инициализация моделей"""
        try:
            print("Загрузка моделей...")

            # YOLOv8
            from ultralytics import YOLO
            model_path = self.config.YOLO_MODEL_PATH

            if not os.path.exists(model_path):
                print("Скачиваем YOLOv8n...")
                model_path = 'yolov8n.pt'

            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ YOLO модель загружена на {'CUDA' if torch.cuda.is_available() else 'CPU'}")

            # Упрощенный ReID (цветовые гистограммы + CNN фичи)
            self._init_reid_model()

        except Exception as e:
            print(f"✗ Ошибка загрузки моделей: {e}")

    def _init_reid_model(self):
        """Инициализация упрощенной ReID модели"""
        print("✓ Используется упрощенный ReID на основе цветовых гистограмм")

    def _extract_reid_features(self, image):
        """Извлечение признаков для ReID"""
        if image is None or image.size == 0:
            return None

        try:
            features = {}

            # 1. Цветовые гистограммы (HSV)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])

            features['color_hist'] = np.concatenate([
                h_hist.flatten() / np.sum(h_hist),
                s_hist.flatten() / np.sum(s_hist),
                v_hist.flatten() / np.sum(v_hist)
            ])

            # 2. Текстура (LBP упрощенный)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp_features = self._simplified_lbp(gray)
            features['texture'] = lbp_features

            # 3. Размер и пропорции
            h, w = image.shape[:2]
            features['aspect_ratio'] = w / h if h > 0 else 1
            features['area'] = w * h

            # 4. Доминирующие цвета
            pixels = image.reshape(-1, 3)
            colors, counts = np.unique(pixels, axis=0, return_counts=True)
            top_colors = colors[np.argsort(-counts)[:3]]
            features['dominant_colors'] = top_colors.flatten()

            return features

        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return None

    def _simplified_lbp(self, gray):
        """Упрощенный LBP для текстуры"""
        try:
            # Ресайз для ускорения
            small = cv2.resize(gray, (32, 64))

            # Простой градиентный признак
            sobelx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            orientation = np.arctan2(sobely, sobelx)

            # Бинаризация и хеширование
            mag_mean = np.mean(magnitude)
            mag_binary = (magnitude > mag_mean).flatten()
            ori_mean = np.mean(orientation)
            ori_binary = (orientation > ori_mean).flatten()

            # Комбинированный хеш
            combined = np.concatenate([mag_binary, ori_binary])
            return combined.astype(np.float32)

        except:
            return np.zeros(32 * 64 * 2, dtype=np.float32)

    def _compare_features(self, features1, features2):
        """Сравнение двух наборов признаков"""
        if features1 is None or features2 is None:
            return 0

        similarity = 0
        weights = {'color': 0.4, 'texture': 0.3, 'appearance': 0.3}

        try:
            # 1. Сравнение цветовых гистограмм (корреляция)
            if 'color_hist' in features1 and 'color_hist' in features2:
                color_sim = np.corrcoef(features1['color_hist'], features2['color_hist'])[0, 1]
                similarity += weights['color'] * max(0, color_sim)

            # 2. Сравнение текстуры (cosine similarity)
            if 'texture' in features1 and 'texture' in features2:
                vec1 = features1['texture'].flatten()
                vec2 = features2['texture'].flatten()
                if len(vec1) > 0 and len(vec2) > 0:
                    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
                    similarity += weights['texture'] * max(0, cos_sim)

            # 3. Сравнение пропорций
            if 'aspect_ratio' in features1 and 'aspect_ratio' in features2:
                ar_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
                ar_sim = max(0, 1 - ar_diff)
                similarity += 0.1 * ar_sim

            # 4. Сравнение доминирующих цветов
            if 'dominant_colors' in features1 and 'dominant_colors' in features2:
                colors1 = features1['dominant_colors'].reshape(-1, 3)
                colors2 = features2['dominant_colors'].reshape(-1, 3)
                color_dists = []
                for c1 in colors1:
                    for c2 in colors2:
                        dist = np.linalg.norm(c1 - c2)
                        color_dists.append(dist)
                if color_dists:
                    color_sim = max(0, 1 - min(color_dists) / 100)
                    similarity += 0.2 * color_sim

        except Exception as e:
            print(f"Ошибка сравнения признаков: {e}")

        return min(1.0, similarity)

    def start(self):
        """Запуск обработки"""
        self.running = True
        self.init_models()
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()
        Thread(target=self._manage_memory, daemon=True).start()  # Поток управления памятью
        print("✓ Обработка видео запущена")

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
        """Переподключение к камере"""
        try:
            if self.cap:
                self.cap.release()

            print(f"Подключение к: {self.config.RTSP_URL}")
            self.cap = cv2.VideoCapture(self.config.RTSP_URL)
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

    def _manage_memory(self):
        """Управление долгосрочной памятью"""
        while self.running:
            try:
                # Обновляем счетчики неактивности
                current_time = time.time()

                # Перемещаем неактивные треки в известные посетители
                tracks_to_remove = []
                for track_id, track_data in list(self.active_tracks.items()):
                    if current_time - track_data['last_seen'] > self.max_disappeared / 30:
                        # Перемещаем в неактивные
                        self.inactive_tracks[track_id] = track_data
                        tracks_to_remove.append(track_id)

                        # Сохраняем в известные посетители если трек был долгим
                        if track_data['age'] > 30:
                            visitor_id = f"visitor_{track_id}"
                            self.known_visitors[visitor_id] = {
                                'features': track_data['features'],
                                'last_seen': current_time,
                                'first_seen': track_data['first_seen'],
                                'visit_count': track_data.get('visit_count', 0) + 1
                            }

                for track_id in tracks_to_remove:
                    del self.active_tracks[track_id]

                # Очищаем старые неактивные треки
                inactive_to_remove = []
                for track_id, track_data in list(self.inactive_tracks.items()):
                    if current_time - track_data['last_seen'] > self.max_forget / 30:
                        inactive_to_remove.append(track_id)

                for track_id in inactive_to_remove:
                    del self.inactive_tracks[track_id]

                # Периодически сохраняем базу данных
                if int(time.time()) % 60 == 0:  # Каждую минуту
                    self._save_visitors_db()

                time.sleep(1)  # Проверяем каждую секунду

            except Exception as e:
                print(f"Ошибка управления памятью: {e}")
                time.sleep(5)

    def _process_single_frame(self, frame):
        """Обработка одного кадра"""
        result = {
            'frame': frame.copy(),
            'detections': [],
            'timestamp': datetime.now(),
            'people_count': 0,
            'fps': 0,
            'known_visitors': 0
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
                                # Вырезаем регион человека
                                person_roi = frame[y1:y2, x1:x2]

                                # Извлекаем признаки для ReID
                                features = self._extract_reid_features(person_roi)

                                current_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                    'confidence': conf,
                                    'width': width,
                                    'height': height,
                                    'roi': person_roi,
                                    'features': features,
                                    'area': width * height
                                })

                # Продвинутый трекинг с долгосрочной памятью
                tracked_detections = self._advanced_tracking(current_detections)

                # Обновляем активные треки
                current_time = time.time()
                known_count = 0

                for det in tracked_detections:
                    track_id = det['track_id']

                    if track_id not in self.active_tracks:
                        # Новый трек - проверяем, не известный ли это посетитель
                        matched_visitor = self._find_matching_visitor(det['features'])
                        if matched_visitor:
                            # Нашли известного посетителя!
                            track_id = matched_visitor['id']
                            known_count += 1
                            print(f"✅ Распознан известный посетитель ID: {track_id}")

                        # Создаем новый трек
                        self.active_tracks[track_id] = {
                            'bbox': det['bbox'],
                            'last_seen': current_time,
                            'first_seen': current_time,
                            'features': det['features'],
                            'age': 1,
                            'visit_count': 1,
                            'is_known': matched_visitor is not None
                        }
                    else:
                        # Обновляем существующий трек
                        self.active_tracks[track_id].update({
                            'bbox': det['bbox'],
                            'last_seen': current_time,
                            'features': det['features'],
                            'age': self.active_tracks[track_id]['age'] + 1
                        })
                        if self.active_tracks[track_id].get('is_known'):
                            known_count += 1

                    # Добавляем в результат
                    result['detections'].append({
                        'track_id': track_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'age': self.active_tracks[track_id]['age'],
                        'is_known': self.active_tracks[track_id].get('is_known', False)
                    })

                # Обновляем статистику
                result['people_count'] = len(result['detections'])
                result['known_visitors'] = known_count
                self.current_count = result['people_count']

                # Обновляем уникальных посетителей
                for det in result['detections']:
                    track_id = det['track_id']
                    if self.active_tracks[track_id]['age'] > 15:  # Устойчивый трек
                        self.session_unique.add(track_id)
                        today = datetime.now().date().isoformat()
                        self.today_unique.add(f"{today}_{track_id}")

                        # Записываем в историю
                        if track_id not in self.visitor_history:
                            self.visitor_history[track_id] = {
                                'first_seen': current_time,
                                'last_seen': current_time,
                                'visit_count': 1,
                                'total_time': 0
                            }
                        else:
                            self.visitor_history[track_id]['last_seen'] = current_time
                            self.visitor_history[track_id]['visit_count'] += 1

                # FPS
                end_time = time.time()
                result['fps'] = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

                # Рисуем на кадре
                self._draw_detections(frame, result['detections'])

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")

        return result

    def _find_matching_visitor(self, features):
        """Поиск совпадения с известными посетителями"""
        if features is None or not self.known_visitors:
            return None

        best_match = None
        best_score = 0

        for visitor_id, visitor_data in self.known_visitors.items():
            if 'features' in visitor_data:
                score = self._compare_features(features, visitor_data['features'])
                if score > self.reid_threshold and score > best_score:
                    best_score = score
                    best_match = {
                        'id': int(visitor_id.split('_')[1]) if '_' in visitor_id else visitor_id,
                        'score': score,
                        'data': visitor_data
                    }

        return best_match

    def _advanced_tracking(self, current_detections):
        """Продвинутый трекинг с учетом долгосрочной памяти"""
        if not current_detections:
            return []

        result = []

        # 1. Сначала пытаемся сопоставить с активными треками
        matched_detections = set()
        matched_tracks = set()

        if self.active_tracks:
            # Матрица схожести
            similarity_matrix = []

            for i, det in enumerate(current_detections):
                for track_id, track_data in self.active_tracks.items():
                    # Вычисляем IoU
                    iou = self._compute_iou(det['bbox'], track_data['bbox'])

                    # Вычисляем расстояние между центрами
                    center1 = det['center']
                    bbox = track_data['bbox']
                    center2 = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

                    # Вычисляем схожесть признаков если есть
                    feature_sim = 0
                    if det['features'] is not None and 'features' in track_data:
                        feature_sim = self._compare_features(det['features'], track_data['features'])

                    # Общий score
                    score = 0.4 * min(1, 1 - distance / 100) + 0.4 * iou + 0.2 * feature_sim

                    similarity_matrix.append((i, track_id, score))

            # Сортируем по score
            similarity_matrix.sort(key=lambda x: x[2], reverse=True)

            # Сопоставляем
            for i, track_id, score in similarity_matrix:
                if score > 0.3 and i not in matched_detections and track_id not in matched_tracks:
                    current_detections[i]['track_id'] = track_id
                    result.append(current_detections[i])
                    matched_detections.add(i)
                    matched_tracks.add(track_id)

        # 2. Пробуем сопоставить с неактивными треками (недавно вышедшими)
        for i, det in enumerate(current_detections):
            if i not in matched_detections and det['features'] is not None:
                best_match = None
                best_score = 0

                for track_id, track_data in self.inactive_tracks.items():
                    if 'features' in track_data:
                        score = self._compare_features(det['features'], track_data['features'])
                        if score > self.reid_threshold and score > best_score:
                            best_score = score
                            best_match = track_id

                if best_match:
                    # Возвращение известного посетителя!
                    det['track_id'] = best_match
                    result.append(det)
                    matched_detections.add(i)

                    # Возвращаем в активные
                    self.active_tracks[best_match] = self.inactive_tracks[best_match]
                    self.active_tracks[best_match]['last_seen'] = time.time()
                    self.active_tracks[best_match]['age'] += 1
                    self.active_tracks[best_match]['visit_count'] = self.active_tracks[best_match].get('visit_count',
                                                                                                       0) + 1

                    # Удаляем из неактивных
                    if best_match in self.inactive_tracks:
                        del self.inactive_tracks[best_match]

                    print(f"🔄 Возвращение посетителя ID: {best_match} (схожесть: {best_score:.2f})")

        # 3. Новые детекции (не сопоставленные)
        for i, det in enumerate(current_detections):
            if i not in matched_detections:
                det['track_id'] = self._get_new_track_id()
                result.append(det)

        return result

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

    def _get_new_track_id(self):
        """Новый ID трека"""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _draw_detections(self, frame, detections):
        """Рисует детекции с разными цветами для известных/неизвестных"""
        for det in detections:
            bbox = det['bbox']
            track_id = det['track_id']
            confidence = det['confidence']
            age = det.get('age', 1)
            is_known = det.get('is_known', False)

            x1, y1, x2, y2 = map(int, bbox)

            # Цвет в зависимости от типа
            if is_known:
                color = (255, 0, 255)  # фиолетовый для известных
                label = f"KNOWN {track_id}"
            elif age < 20:
                color = (0, 165, 255)  # оранжевый для новых
                label = f"NEW {track_id}"
            else:
                color = (0, 255, 0)  # зеленый для постоянных
                label = f"ID {track_id}"

            # Прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Фон для текста
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1), color, -1)

            # Текст
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Дополнительная информация
            info_text = f"{confidence:.0%}"
            if age > 1:
                info_text += f" ({age}f)"
            cv2.putText(frame, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Статистика
        known_count = sum(1 for d in detections if d.get('is_known', False))
        stats_text = f"Людей: {len(detections)} | Известных: {known_count} | Всего в памяти: {len(self.known_visitors)}"
        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Подсказка по цветам
        legend_y = 60
        legends = [
            ("🟣 Известный посетитель", (255, 0, 255)),
            ("🟠 Новый (менее 20 кадров)", (0, 165, 255)),
            ("🟢 Постоянный", (0, 255, 0))
        ]

        for text, color in legends:
            cv2.putText(frame, text, (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            legend_y += 20

    def get_current_frame(self):
        """Получить кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_statistics(self):
        """Получить статистику"""
        known_active = sum(1 for t in self.active_tracks.values() if t.get('is_known', False))

        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'known_visitors': len(self.known_visitors),
            'known_active': known_active,
            'active_tracks': len(self.active_tracks),
            'inactive_tracks': len(self.inactive_tracks),
            'total_visitors': len(self.visitor_history)
        }

    def get_visitor_details(self, limit=20):
        """Получить детали посетителей"""
        visitors = []
        current_time = time.time()

        for track_id, data in list(self.active_tracks.items())[:limit]:
            visitors.append({
                'id': track_id,
                'age': data['age'],
                'is_known': data.get('is_known', False),
                'visit_count': data.get('visit_count', 1),
                'time_in_frame': current_time - data.get('first_seen', current_time)
            })

        return visitors

    def stop(self):
        """Остановка"""
        self.running = False
        # Сохраняем базу данных
        self._save_visitors_db()
        if self.cap:
            self.cap.release()
        print("✓ Обработка остановлена, база данных сохранена")