# /top_eye/src/core/face_database.py
import sqlite3
import numpy as np
import json
import os
import pickle
import face_recognition
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib


class FaceDatabase:
    """База данных для долговременного хранения лиц"""

    def __init__(self, db_path="data/face_database.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

        # Кэш для быстрого доступа
        self.face_cache = {}
        self.embedding_cache = {}
        self.load_cache()

        print(f"📊 База данных лиц инициализирована: {db_path}")

    def _init_database(self):
        """Инициализация базы данных"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        # Таблица известных лиц
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                name TEXT,
                embedding BLOB NOT NULL,
                metadata TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                total_time INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                confidence REAL DEFAULT 0.0
            )
        ''')

        # Таблица детекций (для статистики)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                confidence REAL,
                bbox TEXT,
                FOREIGN KEY (face_id) REFERENCES known_faces (face_id)
            )
        ''')

        # Таблица сессий
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration INTEGER,
                FOREIGN KEY (face_id) REFERENCES known_faces (face_id)
            )
        ''')

        # Индексы для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_person ON known_faces(person_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(start_time)')

        self.conn.commit()

    def load_cache(self):
        """Загрузка кэша из базы"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT face_id, person_id, embedding, confidence 
                FROM known_faces 
                WHERE is_active = 1
            ''')

            for row in cursor.fetchall():
                face_id, person_id, embedding_blob, confidence = row
                embedding = pickle.loads(embedding_blob)

                self.face_cache[face_id] = {
                    'person_id': person_id,
                    'embedding': embedding,
                    'confidence': confidence
                }

                if person_id not in self.embedding_cache:
                    self.embedding_cache[person_id] = []
                self.embedding_cache[person_id].append(embedding)

            print(f"📂 Загружено {len(self.face_cache)} лиц в кэш")

        except Exception as e:
            print(f"⚠ Ошибка загрузки кэша: {e}")

    def add_face(self, embedding, person_id=None, name="Unknown", confidence=0.0, metadata=None):
        """Добавление нового лица в базу"""
        try:
            if person_id is None:
                person_id = self._generate_person_id()

            # Сериализация эмбеддинга
            embedding_blob = pickle.dumps(embedding)

            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO known_faces 
                (person_id, name, embedding, metadata, confidence, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (person_id, name, embedding_blob,
                  json.dumps(metadata) if metadata else None,
                  confidence, datetime.now()))

            face_id = cursor.lastrowid

            # Обновляем кэш
            self.face_cache[face_id] = {
                'person_id': person_id,
                'embedding': embedding,
                'confidence': confidence
            }

            if person_id not in self.embedding_cache:
                self.embedding_cache[person_id] = []
            self.embedding_cache[person_id].append(embedding)

            self.conn.commit()

            print(f"✅ Добавлено новое лицо: {person_id} (ID: {face_id})")

            return face_id, person_id

        except Exception as e:
            print(f"❌ Ошибка добавления лица: {e}")
            return None, None

    def update_face(self, face_id, embedding=None, confidence=None, seen_now=True):
        """Обновление информации о лице"""
        try:
            cursor = self.conn.cursor()

            updates = []
            params = []

            if embedding is not None:
                updates.append("embedding = ?")
                params.append(pickle.dumps(embedding))

                # Обновляем кэш
                if face_id in self.face_cache:
                    self.face_cache[face_id]['embedding'] = embedding

            if confidence is not None:
                updates.append("confidence = ?")
                params.append(confidence)

                if face_id in self.face_cache:
                    self.face_cache[face_id]['confidence'] = confidence

            if seen_now:
                updates.append("last_seen = ?, visit_count = visit_count + 1")
                params.append(datetime.now())

            if updates:
                query = f"UPDATE known_faces SET {', '.join(updates)} WHERE face_id = ?"
                params.append(face_id)
                cursor.execute(query, params)
                self.conn.commit()

                print(f"📝 Обновлено лицо ID: {face_id}")

            return True

        except Exception as e:
            print(f"❌ Ошибка обновления лица: {e}")
            return False

    def find_similar_face(self, query_embedding, threshold=0.6):
        """Поиск похожего лица в базе"""
        if not self.face_cache:
            return None, None, 0.0

        best_match_id = None
        best_person_id = None
        best_similarity = 0.0

        query_embedding = np.array(query_embedding).flatten()

        for face_id, face_data in self.face_cache.items():
            stored_embedding = np.array(face_data['embedding']).flatten()

            # Вычисляем косинусное сходство
            similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding) + 1e-10
            )

            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match_id = face_id
                best_person_id = face_data['person_id']

        if best_similarity >= threshold:
            # Обновляем время последнего визита
            self.update_face(best_match_id, seen_now=True)

            print(f"🔍 Найдено похожее лицо: {best_person_id} "
                  f"(схожесть: {best_similarity:.3f}, порог: {threshold})")

            return best_match_id, best_person_id, best_similarity

        return None, None, 0.0

    def add_detection(self, face_id, camera_id, confidence, bbox):
        """Добавление записи о детекции"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO detections (face_id, camera_id, confidence, bbox)
                VALUES (?, ?, ?, ?)
            ''', (face_id, camera_id, confidence, json.dumps(bbox)))

            self.conn.commit()
            return cursor.lastrowid

        except Exception as e:
            print(f"❌ Ошибка добавления детекции: {e}")
            return None

    def start_session(self, face_id, camera_id):
        """Начало сессии для лица"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (face_id, start_time)
                VALUES (?, ?)
            ''', (face_id, datetime.now()))

            session_id = cursor.lastrowid
            self.conn.commit()

            print(f"⏱️ Начата сессия {session_id} для лица {face_id}")
            return session_id

        except Exception as e:
            print(f"❌ Ошибка начала сессии: {e}")
            return None

    def end_session(self, session_id):
        """Завершение сессии"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, duration = strftime('%s', ?) - strftime('%s', start_time)
                WHERE session_id = ?
            ''', (datetime.now(), datetime.now(), session_id))

            self.conn.commit()

            print(f"⏱️ Завершена сессия {session_id}")
            return True

        except Exception as e:
            print(f"❌ Ошибка завершения сессии: {e}")
            return False

    def get_statistics(self, period_hours=24):
        """Получение статистики"""
        try:
            cursor = self.conn.cursor()

            # Общая статистика
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT person_id) as total_people,
                    COUNT(*) as total_detections,
                    SUM(visit_count) as total_visits
                FROM known_faces
            ''')
            total_stats = cursor.fetchone()

            # Статистика за период
            time_threshold = datetime.now() - timedelta(hours=period_hours)
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT d.face_id) as recent_people,
                    COUNT(*) as recent_detections
                FROM detections d
                JOIN known_faces kf ON d.face_id = kf.face_id
                WHERE d.timestamp > ?
            ''', (time_threshold,))
            recent_stats = cursor.fetchone()

            # Самые частые посетители
            cursor.execute('''
                SELECT 
                    kf.person_id,
                    kf.name,
                    kf.visit_count,
                    MAX(d.timestamp) as last_seen
                FROM known_faces kf
                LEFT JOIN detections d ON kf.face_id = d.face_id
                GROUP BY kf.person_id
                ORDER BY kf.visit_count DESC
                LIMIT 10
            ''')
            top_visitors = cursor.fetchall()

            return {
                'total_people': total_stats[0],
                'total_detections': total_stats[1],
                'total_visits': total_stats[2],
                'recent_people': recent_stats[0],
                'recent_detections': recent_stats[1],
                'top_visitors': [
                    {
                        'person_id': row[0],
                        'name': row[1],
                        'visit_count': row[2],
                        'last_seen': row[3]
                    }
                    for row in top_visitors
                ]
            }

        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return {}

    def get_person_history(self, person_id, limit=50):
        """Получение истории посещений для человека"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    d.timestamp,
                    d.camera_id,
                    d.confidence,
                    d.bbox,
                    s.duration
                FROM known_faces kf
                LEFT JOIN detections d ON kf.face_id = d.face_id
                LEFT JOIN sessions s ON kf.face_id = s.face_id
                WHERE kf.person_id = ?
                ORDER BY d.timestamp DESC
                LIMIT ?
            ''', (person_id, limit))

            history = cursor.fetchall()

            return [
                {
                    'timestamp': row[0],
                    'camera_id': row[1],
                    'confidence': row[2],
                    'bbox': json.loads(row[3]) if row[3] else None,
                    'duration': row[4]
                }
                for row in history
            ]

        except Exception as e:
            print(f"❌ Ошибка получения истории: {e}")
            return []

    def _generate_person_id(self):
        """Генерация уникального ID для человека"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_str = hashlib.md5(str(os.urandom(16)).encode()).hexdigest()[:8]
        return f"PERSON_{timestamp}_{random_str}"

    def cleanup_old_data(self, days_to_keep=30):
        """Очистка старых данных"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            cursor = self.conn.cursor()

            # Удаляем старые детекции
            cursor.execute('DELETE FROM detections WHERE timestamp < ?', (cutoff_date,))
            deleted_detections = cursor.rowcount

            # Удаляем старые сессии
            cursor.execute('DELETE FROM sessions WHERE start_time < ?', (cutoff_date,))
            deleted_sessions = cursor.rowcount

            # Деактивируем лица, которые не появлялись давно
            cursor.execute('''
                UPDATE known_faces 
                SET is_active = 0 
                WHERE last_seen < ? AND is_active = 1
            ''', (cutoff_date,))
            deactivated_faces = cursor.rowcount

            self.conn.commit()

            print(f"🧹 Очистка данных: "
                  f"удалено {deleted_detections} детекций, "
                  f"{deleted_sessions} сессий, "
                  f"деактивировано {deactivated_faces} лиц")

            # Перезагружаем кэш
            self.load_cache()

        except Exception as e:
            print(f"❌ Ошибка очистки данных: {e}")

    def close(self):
        """Закрытие соединения с базой"""
        if self.conn:
            self.conn.close()
            print("📂 База данных закрыта")