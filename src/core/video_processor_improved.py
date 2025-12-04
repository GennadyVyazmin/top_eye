# /top_eye/src/core/video_processor_improved.py
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

        # Улучшенный трекинг
        self.tracks = {}  # {track_id: {bbox, last_seen, appearance_hash, features, age}}
        self.next_track_id = 1
        self.track_max_age = 60  # кадров (2 секунды при 30 FPS)
        self.max_distance = 150  # максимальное расстояние для сопоставления
        self.min_iou = 0.3  # минимальное пересечение для сопоставления

        # YOLO модель
        self.model = None

        print("✓ Видеопроцессор инициализирован")

    def init_models(self):
        """Инициализация моделей"""
        try:
            print("Загрузка моделей...")

            # YOLOv8
            from ultralytics import YOLO
            model_path = self.config.YOLO_MODEL_PATH

            if not os.path.exists(model_path):
                print(f"Модель {model_path} не найдена, скачиваем...")
                model_path = 'yolov8n.pt'

            self.model = YOLO(model_path)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ YOLO модель загружена на {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        except Exception as e:
            print(f"✗ Ошибка загрузки моделей: {e}")
            print("⚠ Режим работы без моделей - только захват видео")

    def start(self):
        """Запуск обработки видео"""
        self.running = True
        self.init_models()
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()
        print("✓ Обработка видео запущена")

    def _capture_frames(self):
        """Поток захвата кадров с камеры"""
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

                # Добавляем кадр в очередь
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
                    processed_data = self._process_single_frame(frame)

                    if not self.processed_queue.full():
                        self.processed_queue.put(processed_data)

            except Exception as e:
                print(f"Ошибка обработки кадра: {e}")

    def _process_single_frame(self, frame):
        """Обработка одного кадра с улучшенным трекингом"""
        result = {
            'frame': frame.copy(),
            'detections': [],
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

                            # Фильтр по размеру
                            width = x2 - x1
                            height = y2 - y1

                            if width > 30 and height > 60:  # Игнорируем слишком маленькие объекты
                                # Извлекаем регион для визуального хеша
                                person_roi = frame[y1:y2, x1:x2]
                                appearance_hash = self._get_appearance_hash(person_roi)

                                # Вычисляем цветовые характеристики
                                color_features = self._get_color_features(person_roi)

                                current_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                    'confidence': conf,
                                    'width': width,
                                    'height': height,
                                    'appearance_hash': appearance_hash,
                                    'color_features': color_features,
                                    'area': width * height
                                })

                # Улучшенный трекинг с несколькими критериями
                tracked_detections = self._improved_tracking(current_detections)

                # Обновляем треки
                current_time = time.time()
                for det in tracked_detections:
                    track_id = det['track_id']

                    if track_id not in self.tracks:
                        # Новый трек
                        self.tracks[track_id] = {
                            'bbox': det['bbox'],
                            'last_seen': current_time,
                            'appearance_hash': det['appearance_hash'],
                            'color_features': det['color_features'],
                            'age': 1,
                            'first_seen': current_time
                        }
                    else:
                        # Обновляем существующий трек
                        self.tracks[track_id].update({
                            'bbox': det['bbox'],
                            'last_seen': current_time,
                            'appearance_hash': det['appearance_hash'],
                            'color_features': det['color_features'],
                            'age': self.tracks[track_id]['age'] + 1
                        })

                    # Добавляем в результат
                    result['detections'].append({
                        'track_id': track_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'age': self.tracks[track_id]['age']
                    })

                # Удаляем старые треки
                tracks_to_delete = []
                for track_id, track_data in self.tracks.items():
                    if current_time - track_data['last_seen'] > self.track_max_age / 30:  # конвертируем в секунды
                        tracks_to_delete.append(track_id)

                for track_id in tracks_to_delete:
                    del self.tracks[track_id]

                # Обновляем статистику
                result['people_count'] = len(result['detections'])
                self.current_count = result['people_count']

                # Обновляем уникальных посетителей (только треки с возрастом > 10 кадров)
                for det in result['detections']:
                    track_id = det['track_id']
                    if self.tracks[track_id]['age'] > 10:  # Только устойчивые треки
                        self.session_unique.add(track_id)
                        today = datetime.now().date().isoformat()
                        self.today_unique.add(f"{today}_{track_id}")

                # FPS
                end_time = time.time()
                result['fps'] = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

                # Рисуем на кадре
                self._draw_detections(frame, result['detections'])

        except Exception as e:
            print(f"Ошибка обработки: {e}")

        return result

    def _get_appearance_hash(self, image):
        """Получает визуальный хеш для изображения"""
        if image is None or image.size == 0:
            return "0"

        try:
            # Ресайзим до фиксированного размера
            resized = cv2.resize(image, (32, 64))

            # Конвертируем в grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

            # Упрощенный хеш: средняя яркость по 8 зонам
            h, w = gray.shape
            zones = []
            for i in range(4):
                for j in range(2):
                    zone = gray[i * h // 4:(i + 1) * h // 4, j * w // 2:(j + 1) * w // 2]
                    zones.append(np.mean(zone))

            # Бинаризуем относительно медианы
            median = np.median(zones)
            hash_str = ''.join(['1' if zone > median else '0' for zone in zones])
            return hash_str

        except:
            return "0"

    def _get_color_features(self, image):
        """Получает цветовые характеристики"""
        if image is None or image.size == 0:
            return [0, 0, 0]

        try:
            # Средний цвет в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mean_color = np.mean(hsv, axis=(0, 1))
            return mean_color.tolist()
        except:
            return [0, 0, 0]

    def _compute_similarity(self, det1, det2):
        """Вычисляет схожесть между двумя детекциями"""
        similarity_score = 0
        weights = {'distance': 0.3, 'iou': 0.4, 'appearance': 0.3}

        # 1. Расстояние между центрами (нормализованное)
        center1 = det1['center']
        center2 = det2['center']
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        distance_score = max(0, 1 - distance / self.max_distance)

        # 2. IoU (Intersection over Union)
        iou = self._compute_iou(det1['bbox'], det2['bbox'])
        iou_score = max(0, min(1, iou / 0.5))  # нормализуем

        # 3. Визуальная схожесть
        if det1['appearance_hash'] != "0" and det2['appearance_hash'] != "0":
            # Сравниваем хеши
            hash1 = det1['appearance_hash']
            hash2 = det2['appearance_hash']
            if len(hash1) == len(hash2):
                matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
                appearance_score = matches / len(hash1)
            else:
                appearance_score = 0
        else:
            appearance_score = 0

        # 4. Цветовая схожесть
        color1 = det1.get('color_features', [0, 0, 0])
        color2 = det2.get('color_features', [0, 0, 0])
        if len(color1) == 3 and len(color2) == 3:
            color_diff = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
            color_score = max(0, 1 - color_diff / 50)  # эмпирический порог
        else:
            color_score = 0

        # Итоговый score
        similarity_score = (
                weights['distance'] * distance_score +
                weights['iou'] * iou_score +
                0.2 * appearance_score +  # часть веса для appearance
                0.1 * color_score  # часть веса для цвета
        )

        return similarity_score

    def _compute_iou(self, box1, box2):
        """Вычисляет Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Вычисляем координаты пересечения
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Площадь пересечения
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Площади каждого бокса
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # IoU
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _improved_tracking(self, current_detections):
        """Улучшенный трекинг с несколькими критериями"""
        if not current_detections:
            return []

        # Если нет активных треков, создаем новые
        if not self.tracks:
            return [{'track_id': self._get_new_track_id(), **det} for det in current_detections]

        # Создаем матрицу схожести
        similarity_matrix = []
        track_ids = list(self.tracks.keys())

        for i, det in enumerate(current_detections):
            for track_id in track_ids:
                track_data = self.tracks[track_id]
                # Создаем объект детекции из данных трека
                track_detection = {
                    'bbox': track_data['bbox'],
                    'center': [(track_data['bbox'][0] + track_data['bbox'][2]) / 2,
                               (track_data['bbox'][1] + track_data['bbox'][3]) / 2],
                    'appearance_hash': track_data.get('appearance_hash', '0'),
                    'color_features': track_data.get('color_features', [0, 0, 0])
                }

                similarity = self._compute_similarity(det, track_detection)
                similarity_matrix.append((i, track_id, similarity))

        # Сортируем по убыванию схожести
        similarity_matrix.sort(key=lambda x: x[2], reverse=True)

        # Венгерский алгоритм (упрощенный)
        matched_detections = set()
        matched_tracks = set()
        matches = []

        for i, track_id, similarity in similarity_matrix:
            if similarity > 0.4:  # порог схожести
                if i not in matched_detections and track_id not in matched_tracks:
                    matches.append((i, track_id))
                    matched_detections.add(i)
                    matched_tracks.add(track_id)

        # Собираем результат
        result = []

        # Сопоставленные детекции
        for i, track_id in matches:
            current_detections[i]['track_id'] = track_id
            result.append(current_detections[i])

        # Новые детекции (не сопоставленные)
        for i, det in enumerate(current_detections):
            if i not in matched_detections:
                det['track_id'] = self._get_new_track_id()
                result.append(det)

        return result

    def _get_new_track_id(self):
        """Получить новый ID трека"""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _draw_detections(self, frame, detections):
        """Рисует детекции на кадре"""
        for det in detections:
            bbox = det['bbox']
            track_id = det['track_id']
            confidence = det['confidence']
            age = det.get('age', 1)

            x1, y1, x2, y2 = map(int, bbox)

            # Цвет в зависимости от возраста трека
            if age < 10:
                color = (0, 165, 255)  # оранжевый для новых
            elif age < 30:
                color = (0, 255, 255)  # желтый для средних
            else:
                color = (0, 255, 0)  # зеленый для старых

            # Прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Фон для текста
            text = f"ID: {track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1), color, -1)

            # ID
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Confidence и возраст
            info_text = f"{confidence:.0%} ({age}f)"
            cv2.putText(frame, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Статистика
        stats_text = f"Людей: {len(detections)} | Треков: {len(self.tracks)}"
        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_current_frame(self):
        """Получить последний обработанный кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_statistics(self):
        """Получить статистику"""
        stable_tracks = sum(1 for t in self.tracks.values() if t['age'] > 10)

        return {
            'current_count': self.current_count,
            'today_unique': len(self.today_unique),
            'session_unique': len(self.session_unique),
            'active_tracks': len(self.tracks),
            'stable_tracks': stable_tracks,
            'avg_track_age': np.mean([t['age'] for t in self.tracks.values()]) if self.tracks else 0
        }

    def stop(self):
        """Остановка"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("✓ Обработка остановлена")