# /top_eye/src/core/video_processor_final.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock
from queue import Queue
from datetime import datetime, timedelta
import time
import os
import json
import hashlib
from collections import OrderedDict, defaultdict

from .face_database import FaceDatabase
from .reid_model import StrongReIDModel


class LongTermVideoProcessor:
    """Видео процессор с улучшенной дедупликацией"""

    def __init__(self, config):
        self.config = config
        print("🚀 Инициализация системы с улучшенной дедупликацией")

        # Камера
        self.cap = None
        self.last_reconnect = time.time()
        self.frame_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

        # Очереди
        self.frame_queue = Queue(maxsize=20)
        self.processed_queue = Queue(maxsize=20)
        self.lock = Lock()
        self.running = False

        # База данных лиц
        self.face_db = FaceDatabase(config.DB_PATH)

        # Модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # YOLO
        self.reid_model = StrongReIDModel(device=str(self.device))

        # Трекинг с улучшенной дедупликацией
        self.active_tracks = OrderedDict()  # {track_id: TrackInfo}
        self.embedding_history = defaultdict(list)  # {track_id: [embeddings]}
        self.next_track_id = 1000
        self.sessions = {}  # {track_id: session_id}

        # Хеши для предотвращения дублирования в одном кадре
        self.frame_hashes = {}
        self.hash_expiry = {}  # Время истечения хеша

        # Настройки
        self.reid_threshold = 0.75  # Увеличенный порог для распознавания
        self.new_person_threshold = 0.85  # Порог для создания нового человека
        self.min_face_size = (100, 100)
        self.max_absent_time = 3600

        # Статистика
        self.stats = defaultdict(int)
        self.stats['start_time'] = time.time()
        self.stats['duplicates_prevented'] = 0
        self.stats['persons_merged'] = 0

        # Инициализация
        self._init_yolo()

        print(f"✅ Система инициализирована")
        print(f"   • ReID порог: {self.reid_threshold}")
        print(f"   • Порог нового человека: {self.new_person_threshold}")
        print(f"   • В базе: {len(self.face_db.face_cache)} лиц")

    def _init_yolo(self):
        """Инициализация YOLO"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.YOLO_MODEL_PATH)
            self.model.to(self.device)
            print(f"✅ YOLO загружен на {self.device}")
        except Exception as e:
            print(f"❌ Ошибка загрузки YOLO: {e}")
            raise

    def start(self):
        """Запуск системы"""
        self.running = True
        Thread(target=self._capture_thread, daemon=True, name="Capture").start()
        Thread(target=self._process_thread, daemon=True, name="Process").start()
        Thread(target=self._maintenance_thread, daemon=True, name="Maintenance").start()
        print("▶ Система запущена")

    def _capture_thread(self):
        """Поток захвата видео"""
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    self._reconnect_camera()
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.5)
                    continue

                # Ресайз если нужно
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    frame = cv2.resize(frame, self.frame_size)

                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), time.time()))

                time.sleep(max(0, 1 / self.config.FPS - 0.01))

            except Exception as e:
                print(f"❌ Ошибка захвата: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)

    def _reconnect_camera(self):
        """Переподключение к камере"""
        try:
            print(f"🔌 Подключение к: {self.config.RTSP_URL}")

            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.config.RTSP_URL)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    def _process_thread(self):
        """Поток обработки"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get()
                    result = self._process_frame(frame, timestamp)

                    if not self.processed_queue.full():
                        self.processed_queue.put(result)

                time.sleep(0.001)

            except Exception as e:
                print(f"❌ Ошибка обработки: {e}")
                time.sleep(0.1)

    def _process_frame(self, frame, timestamp):
        """Обработка кадра с улучшенной дедупликацией"""
        result = {
            'frame': frame.copy(),
            'detections': [],
            'known_faces': [],
            'timestamp': timestamp,
            'people_count': 0,
            'stats': self._get_stats()
        }

        try:
            # 1. Детекция людей
            people_detections = self._detect_people(frame)

            # 2. Обработка каждого обнаруженного человека
            for det in people_detections:
                # Извлекаем ROI лица
                face_roi = self._extract_face_roi(frame, det['bbox'])

                if face_roi is not None:
                    # Получаем эмбеддинг
                    embedding = self.reid_model.extract_embedding(face_roi)

                    if embedding is not None:
                        # Вычисляем хеш эмбеддинга для предотвращения дублирования в одном кадре
                        emb_hash = self._get_embedding_hash(embedding)

                        # Проверяем, не видели ли мы это лицо в текущем кадре
                        current_time = time.time()
                        if emb_hash in self.frame_hashes:
                            if current_time - self.frame_hashes[emb_hash] < 2.0:  # 2 секунды
                                print(f"⚠️ Пропускаем дубликат в текущем кадре (хеш: {emb_hash[:8]})")
                                self.stats['duplicates_prevented'] += 1
                                continue

                        # Обновляем хеш
                        self.frame_hashes[emb_hash] = current_time

                        # Очищаем старые хеши
                        self._cleanup_old_hashes(current_time)

                        # Оцениваем качество лица
                        quality_score = self._assess_face_quality(face_roi)

                        # Ищем в базе данных с улучшенной логикой
                        face_id, person_id, similarity = self.face_db.find_similar_face(
                            embedding,
                            threshold=self.reid_threshold,
                            min_matches=1
                        )

                        if face_id:  # Нашли в базе
                            # Проверяем уверенность
                            confidence = det['confidence'] * similarity

                            if similarity >= 0.9:
                                # Очень высокая уверенность - точно известный человек
                                status = 'KNOWN_HIGH'
                                color = (0, 255, 0)  # Зеленый
                            elif similarity >= 0.8:
                                # Средняя уверенность
                                status = 'KNOWN_MED'
                                color = (0, 255, 255)  # Желтый
                            else:
                                # Низкая уверенность, но выше порога
                                status = 'KNOWN_LOW'
                                color = (255, 165, 0)  # Оранжевый

                            # Обновляем информацию в базе если качество хорошее
                            if quality_score > 0.5:
                                self.face_db.update_face(
                                    face_id,
                                    embedding=embedding if quality_score > 0.7 else None,
                                    confidence=max(similarity, det['confidence']),
                                    seen_now=True
                                )

                            # Добавляем детекцию
                            detection_id = self.face_db.add_detection(
                                face_id,
                                self.config.CAMERA_ID,
                                confidence,
                                det['bbox'],
                                emb_hash[:16]  # Сохраняем часть хеша для отслеживания
                            )

                            # Обновляем активный трек
                            self._update_active_track(
                                face_id, person_id, det['bbox'],
                                embedding, confidence, status
                            )

                            result['detections'].append({
                                'track_id': face_id,
                                'person_id': person_id,
                                'bbox': det['bbox'],
                                'confidence': confidence,
                                'similarity': similarity,
                                'status': status,
                                'quality': quality_score,
                                'color': color
                            })

                            result['known_faces'].append(person_id)
                            self.stats['known_detections'] += 1

                        else:  # Новое или потенциально новое лицо
                            # Проверяем с более высоким порогом для нового человека
                            face_id2, person_id2, similarity2 = self.face_db.find_similar_face(
                                embedding,
                                threshold=self.new_person_threshold,
                                min_matches=1
                            )

                            if face_id2 and similarity2 >= self.new_person_threshold:
                                # Это был ложный отрицательный результат, на самом деле известный
                                print(f"🔄 Исправление: {person_id2} ранее не распознан (схожесть: {similarity2:.3f})")

                                confidence = det['confidence'] * similarity2
                                self.face_db.update_face(
                                    face_id2,
                                    embedding=embedding if quality_score > 0.7 else None,
                                    confidence=max(similarity2, det['confidence']),
                                    seen_now=True
                                )

                                self._update_active_track(
                                    face_id2, person_id2, det['bbox'],
                                    embedding, confidence, 'KNOWN_CORRECTED'
                                )

                                result['detections'].append({
                                    'track_id': face_id2,
                                    'person_id': person_id2,
                                    'bbox': det['bbox'],
                                    'confidence': confidence,
                                    'similarity': similarity2,
                                    'status': 'KNOWN_CORRECTED',
                                    'quality': quality_score,
                                    'color': (0, 200, 255)  # Светло-оранжевый
                                })

                                self.stats['false_negatives_corrected'] += 1

                            else:
                                # Действительно новое лицо
                                confidence = det['confidence']

                                # Добавляем с проверкой на дубликаты
                                new_face_id, new_person_id = self.face_db.add_face(
                                    embedding=embedding,
                                    person_id=None,
                                    name=f"Person_{self.next_track_id}",
                                    confidence=confidence,
                                    metadata={
                                        'first_detection': datetime.now().isoformat(),
                                        'camera_id': self.config.CAMERA_ID,
                                        'bbox': det['bbox'],
                                        'quality_score': quality_score,
                                        'embedding_hash': emb_hash[:16]
                                    },
                                    quality_score=quality_score,
                                    check_duplicates=True
                                )

                                if new_face_id:
                                    # Добавляем детекцию
                                    self.face_db.add_detection(
                                        new_face_id,
                                        self.config.CAMERA_ID,
                                        confidence,
                                        det['bbox'],
                                        emb_hash[:16]
                                    )

                                    # Начинаем сессию
                                    session_id = self.face_db.start_session(
                                        new_face_id, self.config.CAMERA_ID
                                    )
                                    self.sessions[new_face_id] = session_id

                                    # Добавляем в активные треки
                                    self.active_tracks[new_face_id] = {
                                        'person_id': new_person_id,
                                        'first_seen': time.time(),
                                        'last_seen': time.time(),
                                        'detection_count': 1,
                                        'bbox': det['bbox'],
                                        'embedding': embedding,
                                        'quality': quality_score,
                                        'confidence': confidence
                                    }

                                    result['detections'].append({
                                        'track_id': new_face_id,
                                        'person_id': new_person_id,
                                        'bbox': det['bbox'],
                                        'confidence': confidence,
                                        'similarity': 0.0,
                                        'status': 'NEW',
                                        'quality': quality_score,
                                        'color': (255, 0, 0)  # Красный для новых
                                    })

                                    self.stats['new_detections'] += 1
                                    self.next_track_id += 1

            # 3. Проверяем активные треки (не появившиеся в этом кадре)
            self._update_missing_tracks()

            # 4. Обновляем статистику
            result['people_count'] = len(result['detections'])

            # 5. Рисуем результат
            self._draw_detections(frame, result['detections'])

        except Exception as e:
            print(f"⚠ Ошибка обработки кадра: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _get_embedding_hash(self, embedding):
        """Вычисляет хеш эмбеддинга"""
        # Используем первые 16 значений для хеша
        emb_flat = embedding.flatten()[:16]
        emb_bytes = emb_flat.tobytes()
        return hashlib.md5(emb_bytes).hexdigest()

    def _cleanup_old_hashes(self, current_time, max_age=5.0):
        """Очистка старых хешей"""
        hashes_to_remove = []
        for emb_hash, timestamp in list(self.frame_hashes.items()):
            if current_time - timestamp > max_age:
                hashes_to_remove.append(emb_hash)

        for emb_hash in hashes_to_remove:
            del self.frame_hashes[emb_hash]

    def _assess_face_quality(self, face_roi):
        """Оценка качества лица"""
        try:
            if face_roi is None or face_roi.size == 0:
                return 0.0

            h, w = face_roi.shape[:2]

            # 1. Размер
            size_score = min(1.0, (h * w) / (200 * 200))

            # 2. Контраст
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi

            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50.0)

            # 3. Резкость (лапласиан)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian / 100.0)

            # 4. Освещение (не должно быть слишком темным или пересвеченным)
            mean_brightness = np.mean(gray)
            if mean_brightness < 30 or mean_brightness > 220:
                brightness_score = 0.3
            elif mean_brightness < 50 or mean_brightness > 200:
                brightness_score = 0.6
            else:
                brightness_score = 1.0

            # Итоговый score
            quality = 0.2 * size_score + 0.2 * contrast_score + \
                      0.3 * sharpness_score + 0.3 * brightness_score

            return min(1.0, max(0.0, quality))

        except Exception as e:
            print(f"⚠ Ошибка оценки качества: {e}")
            return 0.5  # Среднее значение по умолчанию

    def _detect_people(self, frame):
        """Детекция людей YOLO"""
        detections = []

        try:
            with torch.no_grad():
                results = self.model(
                    frame,
                    conf=self.config.CONFIDENCE_THRESHOLD,
                    device=self.device,
                    verbose=False,
                    classes=[0],  # Только люди
                    imgsz=640
                )

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        conf = float(confidences[i])

                        width = x2 - x1
                        height = y2 - y1

                        # Фильтр размера и соотношения сторон (типичные для людей)
                        if width > 40 and height > 80 and height / width > 1.5:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': conf,
                                'width': width,
                                'height': height,
                                'area': width * height
                            })

        except Exception as e:
            print(f"⚠ Ошибка детекции: {e}")

        return detections

    def _extract_face_roi(self, frame, bbox):
        """Извлечение ROI лица с улучшенной логикой"""
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Увеличиваем ROI для захвата большего контекста
            padding_w = int((x2 - x1) * 0.1)
            padding_h = int((y2 - y1) * 0.2)

            x1 = max(0, x1 - padding_w)
            x2 = min(frame.shape[1], x2 + padding_w)
            y1 = max(0, y1 - padding_h)
            y2 = min(frame.shape[0], y2 + padding_h)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                return None

            # Проверяем размер
            if face_roi.shape[0] < self.min_face_size[0] or face_roi.shape[1] < self.min_face_size[1]:
                return None

            return face_roi

        except Exception as e:
            print(f"⚠ Ошибка извлечения лица: {e}")
            return None

    def _update_active_track(self, face_id, person_id, bbox, embedding, confidence, status):
        """Обновление активного трека"""
        current_time = time.time()

        if face_id not in self.active_tracks:
            # Новый трек
            self.active_tracks[face_id] = {
                'person_id': person_id,
                'first_seen': current_time,
                'last_seen': current_time,
                'detection_count': 1,
                'bbox': bbox,
                'embedding': embedding,
                'confidence': confidence,
                'status': status
            }

            # Начинаем сессию если еще не начата
            if face_id not in self.sessions:
                session_id = self.face_db.start_session(face_id, self.config.CAMERA_ID)
                self.sessions[face_id] = session_id
        else:
            # Обновляем существующий трек
            self.active_tracks[face_id].update({
                'last_seen': current_time,
                'detection_count': self.active_tracks[face_id]['detection_count'] + 1,
                'bbox': bbox,
                'confidence': confidence,
                'status': status
            })

            # Обновляем эмбеддинг в истории
            if face_id not in self.embedding_history:
                self.embedding_history[face_id] = []

            self.embedding_history[face_id].append(embedding)
            if len(self.embedding_history[face_id]) > 10:
                self.embedding_history[face_id].pop(0)

    def _update_missing_tracks(self):
        """Обновление пропавших треков"""
        current_time = time.time()
        tracks_to_remove = []

        for face_id, track_info in list(self.active_tracks.items()):
            if current_time - track_info['last_seen'] > 2:  # Не появлялся 2 секунды
                # Завершаем сессию если есть
                if face_id in self.sessions:
                    self.face_db.end_session(self.sessions[face_id])
                    del self.sessions[face_id]

                # Удаляем из активных если долго нет
                if current_time - track_info['last_seen'] > 10:  # 10 секунд
                    tracks_to_remove.append(face_id)

        for face_id in tracks_to_remove:
            del self.active_tracks[face_id]
            if face_id in self.embedding_history:
                del self.embedding_history[face_id]

    def _draw_detections(self, frame, detections):
        """Рисование детекций с информацией о статусе"""
        for det in detections:
            bbox = det['bbox']
            track_id = det.get('track_id', '?')
            person_id = det.get('person_id', 'Unknown')
            confidence = det['confidence']
            status = det['status']
            similarity = det.get('similarity', 0)
            quality = det.get('quality', 0)
            color = det.get('color', (0, 255, 0))

            x1, y1, x2, y2 = map(int, bbox)

            # Прямоугольник
            thickness = 3 if 'KNOWN' in status else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Текст в зависимости от статуса
            if status == 'NEW':
                text = f"NEW: {person_id[-6:]}"
                subtext = f"Conf: {confidence:.1%}"
            elif 'KNOWN' in status:
                short_pid = person_id[-6:] if len(person_id) > 8 else person_id
                text = f"{short_pid}"
                subtext = f"Sim: {similarity:.1%}"
            else:
                text = f"{person_id[-6:]}"
                subtext = f"{status}"

            # Фон для текста
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 25),
                          (x1 + text_size[0], y1), color, -1)

            # Основной текст
            cv2.putText(frame, text, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Дополнительный текст
            cv2.putText(frame, subtext, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Качество
            quality_text = f"Q: {quality:.2f}"
            cv2.putText(frame, quality_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Статистика
        status_counts = defaultdict(int)
        for det in detections:
            status_counts[det['status']] += 1

        stats_parts = []
        if status_counts.get('KNOWN_HIGH', 0) > 0:
            stats_parts.append(f"Высокая: {status_counts['KNOWN_HIGH']}")
        if status_counts.get('KNOWN_MED', 0) > 0:
            stats_parts.append(f"Средняя: {status_counts['KNOWN_MED']}")
        if status_counts.get('KNOWN_LOW', 0) > 0:
            stats_parts.append(f"Низкая: {status_counts['KNOWN_LOW']}")
        if status_counts.get('NEW', 0) > 0:
            stats_parts.append(f"Новые: {status_counts['NEW']}")

        stats_text = f"Людей: {len(detections)} ({', '.join(stats_parts)})"

        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Информация о базе
        db_stats = self.face_db.get_statistics(period_hours=1)
        db_info = f"В базе: {db_stats.get('total_people', 0)} людей, {db_stats.get('total_faces', 0)} лиц"
        cv2.putText(frame, db_info, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Статистика предотвращения дубликатов
        dup_info = f"Дубликатов предотвращено: {self.stats.get('duplicates_prevented', 0)}"
        cv2.putText(frame, dup_info, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _maintenance_thread(self):
        """Поток обслуживания с дедупликацией"""
        print("🔧 Поток обслуживания запущен")

        while self.running:
            try:
                current_time = time.time()

                # Каждые 5 минут сохраняем статистику
                if int(current_time) % 300 == 0:
                    self._save_statistics()

                # Каждый час выполняем дедупликацию
                if int(current_time) % 3600 == 0:
                    print("🔄 Запуск дедупликации...")
                    duplicates_removed = self.face_db.deduplicate_faces(similarity_threshold=0.85)
                    if duplicates_removed > 0:
                        print(f"🧹 Удалено {duplicates_removed} дубликатов лиц")
                        self.stats['duplicates_removed'] += duplicates_removed

                    # Выводим статистику дубликатов
                    db_stats = self.face_db.get_statistics(period_hours=24)
                    if db_stats.get('duplicates'):
                        print("📊 Текущие дубликаты в базе:")
                        for dup in db_stats['duplicates'][:5]:
                            print(f"  • {dup['person_id'][-8:]}: {dup['face_count']} лиц, "
                                  f"уверенность: {dup['avg_confidence']:.2f}")

                # Каждые 30 минут очищаем старые данные
                if int(current_time) % 1800 == 0:
                    self.face_db.cleanup_old_data(days_to_keep=7)

                # Ежечасная статистика
                if int(current_time) % 3600 == 0:
                    self._print_hourly_stats()

                time.sleep(1)

            except Exception as e:
                print(f"⚠ Ошибка обслуживания: {e}")
                time.sleep(5)

    def _save_statistics(self):
        """Сохранение статистики"""
        try:
            stats_file = "data/statistics.json"
            stats = {
                'timestamp': datetime.now().isoformat(),
                'total_frames': self.stats['total_frames'],
                'total_detections': self.stats['total_detections'],
                'known_detections': self.stats.get('known_detections', 0),
                'new_detections': self.stats.get('new_detections', 0),
                'duplicates_prevented': self.stats.get('duplicates_prevented', 0),
                'duplicates_removed': self.stats.get('duplicates_removed', 0),
                'false_negatives_corrected': self.stats.get('false_negatives_corrected', 0),
                'active_tracks': len(self.active_tracks),
                'database_stats': self.face_db.get_statistics(period_hours=24),
                'uptime_hours': (time.time() - self.stats['start_time']) / 3600
            }

            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"📊 Статистика сохранена: {stats_file}")

        except Exception as e:
            print(f"⚠ Ошибка сохранения статистики: {e}")

    def _print_hourly_stats(self):
        """Вывод ежечасной статистики"""
        db_stats = self.face_db.get_statistics(period_hours=1)

        print(f"\n{'=' * 60}")
        print(f"📈 ЕЖЕЧАСНАЯ СТАТИСТИКА")
        print(f"{'=' * 60}")
        print(f"• Всего людей в базе: {db_stats.get('total_people', 0)}")
        print(f"• Всего лиц в базе: {db_stats.get('total_faces', 0)}")
        print(f"• Уникальных за час: {db_stats.get('recent_people', 0)}")
        print(f"• Детекций за час: {db_stats.get('recent_detections', 0)}")
        print(f"• Активных треков: {len(self.active_tracks)}")
        print(f"• Дубликатов предотвращено: {self.stats.get('duplicates_prevented', 0)}")

        if db_stats.get('duplicates'):
            print(f"• Людей с дубликатами: {len(db_stats['duplicates'])}")
            for dup in db_stats['duplicates'][:3]:
                print(f"  - {dup['person_id'][-8:]}: {dup['face_count']} лиц")

        print(f"{'=' * 60}\n")

    def _get_stats(self):
        """Получение текущей статистики"""
        return {
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'active_tracks': len(self.active_tracks),
            'database_size': len(self.face_db.face_cache),
            'unique_people': len(self.face_db.person_cache),
            'duplicates_prevented': self.stats.get('duplicates_prevented', 0),
            'uptime': (time.time() - self.stats['start_time']) / 3600
        }

    def get_current_frame(self):
        """Получить текущий кадр"""
        if not self.processed_queue.empty():
            return self.processed_queue.get()
        return None

    def get_detailed_statistics(self):
        """Получить детальную статистику"""
        db_stats = self.face_db.get_statistics(period_hours=24)

        return {
            'system': self._get_stats(),
            'database': db_stats,
            'active_sessions': len(self.sessions),
            'tracking_info': {
                'active_tracks': len(self.active_tracks),
                'embedding_history': {k: len(v) for k, v in self.embedding_history.items()}
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_person_info(self, person_id):
        """Получить информацию о человеке"""
        return self.face_db.get_person_history(person_id)

    def register_person(self, name, embedding=None, images=None):
        """Регистрация нового человека с проверкой на дубликаты"""
        if embedding is None and images:
            # Создаем эмбеддинг из изображений
            embeddings = []
            quality_scores = []

            for img in images[:3]:  # Используем до 3 изображений
                # Оцениваем качество
                quality = self._assess_face_quality(img)

                # Получаем эмбеддинг
                emb = self.reid_model.extract_embedding(img)

                if emb is not None and quality > 0.4:
                    embeddings.append(emb)
                    quality_scores.append(quality)

            if embeddings:
                # Взвешенное среднее по качеству
                weights = np.array(quality_scores) / sum(quality_scores)
                embedding = np.average(embeddings, axis=0, weights=weights)

                # Среднее качество
                avg_quality = np.mean(quality_scores)

        if embedding is not None:
            # Ищем возможные дубликаты перед регистрацией
            face_id, existing_person_id, similarity = self.face_db.find_similar_face(
                embedding, threshold=0.85, min_matches=1
            )

            if face_id and similarity >= 0.9:
                return {
                    'success': False,
                    'message': f'Person already exists as {existing_person_id} (similarity: {similarity:.3f})',
                    'existing_person_id': existing_person_id,
                    'similarity': similarity
                }

            # Регистрируем нового человека
            face_id, person_id = self.face_db.add_face(
                embedding=embedding,
                name=name,
                confidence=0.9,
                quality_score=avg_quality if 'avg_quality' in locals() else 0.7,
                check_duplicates=True
            )

            if face_id:
                # Добавляем дополнительные изображения как отдельные лица
                for i, (img, quality) in enumerate(zip(images[1:3], quality_scores[1:])):
                    emb = self.reid_model.extract_embedding(img)
                    if emb is not None and quality > 0.4:
                        self.face_db.add_face(
                            embedding=emb,
                            person_id=person_id,
                            name=f"{name}_view{i}",
                            confidence=0.8,
                            quality_score=quality,
                            check_duplicates=False
                        )

                return {
                    'success': True,
                    'face_id': face_id,
                    'person_id': person_id,
                    'message': f'Person {name} registered successfully',
                    'quality': avg_quality if 'avg_quality' in locals() else 0.7
                }

        return {
            'success': False,
            'message': 'Failed to extract embedding or poor quality images'
        }

    def stop(self):
        """Остановка системы"""
        print("🛑 Остановка системы...")

        self.running = False

        # Завершаем все активные сессии
        for face_id, session_id in list(self.sessions.items()):
            self.face_db.end_session(session_id)

        # Выполняем финальную дедупликацию
        print("🔍 Финальная проверка на дубликаты...")
        duplicates = self.face_db.deduplicate_faces(similarity_threshold=0.85)
        print(f"🧹 Удалено {duplicates} дубликатов")

        # Сохраняем статистику
        self._save_statistics()

        # Закрываем базу данных
        self.face_db.close()

        # Освобождаем ресурсы камеры
        if self.cap:
            self.cap.release()

        print("✅ Система остановлена")