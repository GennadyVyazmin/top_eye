# /top_eye/src/core/video_processor_final.py
import cv2
import torch
import numpy as np
from threading import Thread, Lock, Event
from queue import Queue
from datetime import datetime, timedelta
import time
import os
import json
from collections import OrderedDict, defaultdict

from .face_database import FaceDatabase
from .reid_model import StrongReIDModel
from .kalman_tracker import KalmanTracker


class LongTermVideoProcessor:
    """Видео процессор с долговременным хранением лиц"""

    def __init__(self, config):
        self.config = config
        print("🚀 Инициализация системы с ДОЛГОВРЕМЕННЫМ хранением")

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

        # Трекинг
        self.active_tracks = OrderedDict()  # {track_id: TrackInfo}
        self.long_term_memory = {}  # Долговременная память лиц
        self.next_track_id = 1000
        self.sessions = {}  # {track_id: session_id}

        # Настройки
        self.reid_threshold = 0.65  # Порог для распознавания
        self.min_face_size = (100, 100)  # Минимальный размер лица
        self.max_absent_time = 3600  # 1 час - считаем новым человеком

        # Статистика
        self.stats = defaultdict(int)
        self.stats['start_time'] = time.time()

        # Инициализация
        self._init_yolo()

        print(f"✅ Система инициализирована с долговременным хранением")
        print(f"   • База лиц: {len(self.face_db.face_cache)} записей")
        print(f"   • ReID порог: {self.reid_threshold}")
        print(f"   • Макс. время отсутствия: {self.max_absent_time // 3600}ч")

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
        """Обработка кадра"""
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

            # 2. Для каждого человека извлекаем лицо и эмбеддинг
            for det in people_detections:
                # Извлекаем ROI лица
                face_roi = self._extract_face_roi(frame, det['bbox'])

                if face_roi is not None:
                    # Получаем эмбеддинг
                    embedding = self.reid_model.extract_embedding(face_roi)

                    if embedding is not None:
                        # Ищем в базе данных
                        face_id, person_id, similarity = self.face_db.find_similar_face(
                            embedding,
                            threshold=self.reid_threshold
                        )

                        if face_id:  # Нашли в базе
                            # Обновляем информацию
                            self.face_db.update_face(face_id, embedding=embedding,
                                                     confidence=similarity, seen_now=True)

                            # Добавляем детекцию в базу
                            detection_id = self.face_db.add_detection(
                                face_id,
                                self.config.CAMERA_ID,
                                similarity,
                                det['bbox']
                            )

                            # Начинаем/продолжаем сессию
                            if face_id not in self.sessions:
                                session_id = self.face_db.start_session(face_id,
                                                                        self.config.CAMERA_ID)
                                self.sessions[face_id] = session_id

                            # Обновляем активный трек
                            if face_id not in self.active_tracks:
                                self.active_tracks[face_id] = {
                                    'person_id': person_id,
                                    'first_seen': time.time(),
                                    'last_seen': time.time(),
                                    'detection_count': 1,
                                    'bbox': det['bbox'],
                                    'embedding': embedding
                                }
                            else:
                                self.active_tracks[face_id].update({
                                    'last_seen': time.time(),
                                    'detection_count': self.active_tracks[face_id]['detection_count'] + 1,
                                    'bbox': det['bbox']
                                })

                            result['detections'].append({
                                'track_id': face_id,
                                'person_id': person_id,
                                'bbox': det['bbox'],
                                'confidence': similarity,
                                'status': 'KNOWN',
                                'detection_count': self.active_tracks[face_id]['detection_count'],
                                'color': (0, 255, 0)  # Зеленый для известных
                            })

                            result['known_faces'].append(person_id)
                            self.stats['known_detections'] += 1

                        else:  # Новое лицо
                            # Добавляем в базу
                            new_face_id, new_person_id = self.face_db.add_face(
                                embedding=embedding,
                                person_id=None,
                                name=f"Person_{self.next_track_id}",
                                confidence=det['confidence'],
                                metadata={
                                    'first_detection': datetime.now().isoformat(),
                                    'camera_id': self.config.CAMERA_ID,
                                    'bbox': det['bbox']
                                }
                            )

                            if new_face_id:
                                # Добавляем детекцию
                                self.face_db.add_detection(
                                    new_face_id,
                                    self.config.CAMERA_ID,
                                    det['confidence'],
                                    det['bbox']
                                )

                                # Начинаем сессию
                                session_id = self.face_db.start_session(new_face_id,
                                                                        self.config.CAMERA_ID)
                                self.sessions[new_face_id] = session_id

                                # Добавляем в активные треки
                                self.active_tracks[new_face_id] = {
                                    'person_id': new_person_id,
                                    'first_seen': time.time(),
                                    'last_seen': time.time(),
                                    'detection_count': 1,
                                    'bbox': det['bbox'],
                                    'embedding': embedding
                                }

                                result['detections'].append({
                                    'track_id': new_face_id,
                                    'person_id': new_person_id,
                                    'bbox': det['bbox'],
                                    'confidence': det['confidence'],
                                    'status': 'NEW',
                                    'detection_count': 1,
                                    'color': (0, 165, 255)  # Оранжевый для новых
                                })

                                self.stats['new_detections'] += 1
                                self.next_track_id += 1

            # 3. Проверяем активные треки (не появившиеся в этом кадре)
            self._update_missing_tracks()

            # 4. Обновляем статистику
            result['people_count'] = len(result['detections'])
            self.stats['total_frames'] += 1
            self.stats['total_detections'] += len(result['detections'])

            # 5. Рисуем результат
            self._draw_detections(frame, result['detections'])

        except Exception as e:
            print(f"⚠ Ошибка обработки кадра: {e}")
            import traceback
            traceback.print_exc()

        return result

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

                        if width > 40 and height > 80:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': conf,
                                'width': width,
                                'height': height
                            })

        except Exception as e:
            print(f"⚠ Ошибка детекции: {e}")

        return detections

    def _extract_face_roi(self, frame, bbox):
        """Извлечение ROI лица"""
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Вырезаем верхнюю часть тела (где обычно лицо)
            face_height = int((y2 - y1) * 0.4)  # 40% от высоты тела
            face_y1 = y1
            face_y2 = y1 + face_height

            # Корректируем координаты
            face_y1 = max(0, face_y1)
            face_y2 = min(frame.shape[0], face_y2)

            face_roi = frame[face_y1:face_y2, x1:x2]

            if face_roi.size == 0:
                return None

            # Проверяем размер
            if face_roi.shape[0] < self.min_face_size[0] or face_roi.shape[1] < self.min_face_size[1]:
                return None

            return face_roi

        except Exception as e:
            print(f"⚠ Ошибка извлечения лица: {e}")
            return None

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

    def _draw_detections(self, frame, detections):
        """Рисование детекций"""
        for det in detections:
            bbox = det['bbox']
            track_id = det.get('track_id', '?')
            person_id = det.get('person_id', 'Unknown')
            confidence = det['confidence']
            status = det['status']
            color = det.get('color', (0, 255, 0))

            x1, y1, x2, y2 = map(int, bbox)

            # Прямоугольник
            thickness = 3 if status == 'KNOWN' else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # ID и статус
            if status == 'KNOWN':
                text = f"ID: {person_id}"
                subtext = f"Conf: {confidence:.1%}"
            else:
                text = f"NEW: {person_id}"
                subtext = f"New person"

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

        # Статистика
        active_known = len([d for d in detections if d['status'] == 'KNOWN'])
        active_new = len([d for d in detections if d['status'] == 'NEW'])

        stats_text = (f"Людей: {len(detections)} "
                      f"(Известных: {active_known}, Новых: {active_new})")

        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Информация о базе
        db_info = f"В базе: {len(self.face_db.face_cache)} лиц"
        cv2.putText(frame, db_info, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _maintenance_thread(self):
        """Поток обслуживания"""
        print("🔧 Поток обслуживания запущен")

        while self.running:
            try:
                # Каждые 5 минут сохраняем статистику
                if int(time.time()) % 300 == 0:
                    self._save_statistics()

                # Каждые 30 минут очищаем старые данные
                if int(time.time()) % 1800 == 0:
                    self.face_db.cleanup_old_data(days_to_keep=7)

                # Ежечасная статистика
                if int(time.time()) % 3600 == 0:
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
                'active_tracks': len(self.active_tracks),
                'database_size': len(self.face_db.face_cache),
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

        print(f"\n{'=' * 50}")
        print(f"📈 ЕЖЕЧАСНАЯ СТАТИСТИКА")
        print(f"{'=' * 50}")
        print(f"• Всего лиц в базе: {db_stats.get('total_people', 0)}")
        print(f"• Уникальных за час: {db_stats.get('recent_people', 0)}")
        print(f"• Детекций за час: {db_stats.get('recent_detections', 0)}")
        print(f"• Активных треков: {len(self.active_tracks)}")
        print(f"• Всего кадров: {self.stats['total_frames']}")
        print(f"{'=' * 50}\n")

    def _get_stats(self):
        """Получение текущей статистики"""
        return {
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'active_tracks': len(self.active_tracks),
            'database_size': len(self.face_db.face_cache),
            'known_in_frame': len([d for d in self.active_tracks.values()]),
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
            'timestamp': datetime.now().isoformat()
        }

    def get_person_info(self, person_id):
        """Получить информацию о человеке"""
        return self.face_db.get_person_history(person_id)

    def register_person(self, name, embedding=None, images=None):
        """Регистрация нового человека"""
        if embedding is None and images:
            # Создаем эмбеддинг из изображений
            embeddings = []
            for img in images[:3]:  # Используем до 3 изображений
                emb = self.reid_model.extract_embedding(img)
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                embedding = np.mean(embeddings, axis=0)

        if embedding is not None:
            face_id, person_id = self.face_db.add_face(
                embedding=embedding,
                name=name,
                confidence=0.9
            )

            return {
                'success': face_id is not None,
                'face_id': face_id,
                'person_id': person_id,
                'message': f'Person {name} registered successfully'
            }

        return {
            'success': False,
            'message': 'Failed to extract embedding'
        }

    def stop(self):
        """Остановка системы"""
        print("🛑 Остановка системы...")

        self.running = False

        # Завершаем все активные сессии
        for face_id, session_id in list(self.sessions.items()):
            self.face_db.end_session(session_id)

        # Сохраняем статистику
        self._save_statistics()

        # Закрываем базу данных
        self.face_db.close()

        # Освобождаем ресурсы камеры
        if self.cap:
            self.cap.release()

        print("✅ Система остановлена")