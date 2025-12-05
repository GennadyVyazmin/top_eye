# /top_eye/src/core/face_database.py
import sqlite3
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib


class FaceDatabase:
    """База данных для долговременного хранения лиц с дедупликацией"""

    def __init__(self, db_path="data/face_database.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

        # Кэш для быстрого доступа
        self.face_cache = {}  # {face_id: face_data}
        self.person_cache = defaultdict(list)  # {person_id: [face_ids]}
        self.embedding_cache = {}  # {person_id: [embeddings]}
        self.load_cache()

        print(f"📊 База данных лиц инициализирована: {db_path}")
        print(f"   Загружено {len(self.face_cache)} лиц, {len(self.person_cache)} уникальных людей")

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
                confidence REAL DEFAULT 0.0,
                is_primary BOOLEAN DEFAULT 0,
                quality_score REAL DEFAULT 0.0
            )
        ''')

        # Таблица детекций (для статистики)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                person_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                confidence REAL,
                bbox TEXT,
                embedding_hash TEXT,
                FOREIGN KEY (face_id) REFERENCES known_faces (face_id)
            )
        ''')

        # Таблица сессий
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                person_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration INTEGER,
                FOREIGN KEY (face_id) REFERENCES known_faces (face_id)
            )
        ''')

        # Таблица для отслеживания слияний
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS merges (
                merge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_person_id TEXT,
                new_person_id TEXT,
                merge_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reason TEXT
            )
        ''')

        # Индексы для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_person ON known_faces(person_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_active ON known_faces(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(start_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(person_id)')

        self.conn.commit()

    def load_cache(self):
        """Загрузка кэша из базы"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT face_id, person_id, embedding, confidence, quality_score, is_primary
                FROM known_faces 
                WHERE is_active = 1
                ORDER BY last_seen DESC
            ''')

            for row in cursor.fetchall():
                face_id, person_id, embedding_blob, confidence, quality_score, is_primary = row
                embedding = pickle.loads(embedding_blob)

                self.face_cache[face_id] = {
                    'person_id': person_id,
                    'embedding': embedding,
                    'confidence': confidence,
                    'quality_score': quality_score,
                    'is_primary': is_primary
                }

                self.person_cache[person_id].append(face_id)

                if person_id not in self.embedding_cache:
                    self.embedding_cache[person_id] = []
                self.embedding_cache[person_id].append(embedding)

            print(f"📂 Загружено {len(self.face_cache)} лиц, {len(self.person_cache)} людей в кэш")

        except Exception as e:
            print(f"⚠ Ошибка загрузки кэша: {e}")

    def _normalize_embedding(self, embedding):
        """Нормализация эмбеддинга"""
        emb = np.array(embedding).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def find_similar_face(self, query_embedding, threshold=0.75, min_matches=1):
        """Улучшенный поиск похожего лица с несколькими проверками"""
        if not self.face_cache:
            return None, None, 0.0

        query_emb = self._normalize_embedding(query_embedding)

        best_match_id = None
        best_person_id = None
        best_similarity = 0.0
        all_matches = []

        # Первый проход: быстрый поиск
        for face_id, face_data in self.face_cache.items():
            stored_emb = self._normalize_embedding(face_data['embedding'])

            similarity = np.dot(query_emb, stored_emb)

            if similarity >= threshold:
                all_matches.append({
                    'face_id': face_id,
                    'person_id': face_data['person_id'],
                    'similarity': similarity,
                    'confidence': face_data['confidence'],
                    'quality': face_data['quality_score']
                })

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = face_id
                    best_person_id = face_data['person_id']

        if not all_matches:
            return None, None, 0.0

        # Проверяем, достаточно ли матчей для этого человека
        person_matches = [m for m in all_matches if m['person_id'] == best_person_id]

        if len(person_matches) < min_matches:
            if best_similarity < 0.85:  # Очень высокая уверенность может быть с одним матчем
                return None, None, 0.0

        # Проверяем качество совпадения
        avg_similarity = np.mean([m['similarity'] for m in person_matches])
        max_similarity = best_similarity

        # Финальная проверка
        if max_similarity >= threshold and avg_similarity >= threshold - 0.1:
            print(f"✅ Найдено лицо: {best_person_id} "
                  f"(схожесть: {max_similarity:.3f}, матчей: {len(person_matches)})")

            # Обновляем время последнего визита
            self.update_face(best_match_id, seen_now=True)

            return best_match_id, best_person_id, max_similarity

        return None, None, 0.0

    def add_face(self, embedding, person_id=None, name="Unknown", confidence=0.0,
                 metadata=None, quality_score=0.0, check_duplicates=True):
        """Добавление нового лица с проверкой на дубликаты"""
        try:
            # Проверяем на дубликаты перед добавлением
            if check_duplicates:
                # Сначала с нормальным порогом
                face_id, existing_person_id, similarity = self.find_similar_face(
                    embedding, threshold=0.75, min_matches=1
                )

                if not face_id and similarity >= 0.7:  # Если средняя схожесть
                    # Проверяем с более строгим порогом для одиночных матчей
                    face_id, existing_person_id, similarity = self.find_similar_face(
                        embedding, threshold=0.85, min_matches=1
                    )

                if face_id:
                    print(f"⚠️ Найден возможный дубликат для {existing_person_id} (схожесть: {similarity:.3f})")

                    # Если очень высокая схожесть, используем существующего
                    if similarity >= 0.9:
                        print(f"✅ Используем существующего человека {existing_person_id}")
                        self.update_face(face_id, embedding=embedding, confidence=max(confidence, similarity))
                        return face_id, existing_person_id

                    # Если средняя схожесть, проверяем дополнительные критерии
                    elif similarity >= 0.8:
                        # Получаем все лица этого человека
                        person_faces = self.get_person_faces(existing_person_id)
                        if len(person_faces) >= 2:
                            # У человека уже есть несколько лиц, доверяем базе
                            print(f"✅ Добавляем к существующему человеку {existing_person_id}")
                            new_face_id = self._add_new_face_instance(
                                embedding, existing_person_id, name, confidence,
                                metadata, quality_score, is_primary=False
                            )
                            return new_face_id, existing_person_id

            # Создаем нового человека
            if person_id is None:
                person_id = self._generate_person_id()

            new_face_id = self._add_new_face_instance(
                embedding, person_id, name, confidence, metadata,
                quality_score, is_primary=True
            )

            # Проверяем нового человека на дубликаты с другими людьми
            self._check_new_person_for_duplicates(new_face_id, person_id)

            return new_face_id, person_id

        except Exception as e:
            print(f"❌ Ошибка добавления лица: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _add_new_face_instance(self, embedding, person_id, name, confidence,
                               metadata, quality_score, is_primary):
        """Внутренний метод добавления лица"""
        embedding_blob = pickle.dumps(embedding)

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO known_faces 
            (person_id, name, embedding, metadata, confidence, last_seen, 
             quality_score, is_primary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, name, embedding_blob,
              json.dumps(metadata) if metadata else None,
              confidence, datetime.now(), quality_score, 1 if is_primary else 0))

        face_id = cursor.lastrowid
        self.conn.commit()

        # Обновляем кэш
        self.face_cache[face_id] = {
            'person_id': person_id,
            'embedding': embedding,
            'confidence': confidence,
            'quality_score': quality_score,
            'is_primary': is_primary
        }
        self.person_cache[person_id].append(face_id)

        if person_id not in self.embedding_cache:
            self.embedding_cache[person_id] = []
        self.embedding_cache[person_id].append(embedding)

        print(f"✅ Добавлено лицо: {person_id} (ID: {face_id}, качество: {quality_score:.2f})")

        return face_id

    def _check_new_person_for_duplicates(self, new_face_id, new_person_id):
        """Проверка нового человека на дубликаты с существующими"""
        try:
            if new_face_id not in self.face_cache:
                return

            new_embedding = self.face_cache[new_face_id]['embedding']
            new_emb_norm = self._normalize_embedding(new_embedding)

            # Ищем похожие лица среди других людей
            for person_id, face_ids in self.person_cache.items():
                if person_id == new_person_id:
                    continue  # Пропускаем самого себя

                for face_id in face_ids:
                    if face_id in self.face_cache:
                        existing_emb = self.face_cache[face_id]['embedding']
                        existing_emb_norm = self._normalize_embedding(existing_emb)

                        similarity = np.dot(new_emb_norm, existing_emb_norm)

                        if similarity >= 0.85:  # Очень высокая схожесть
                            print(f"⚠️ Обнаружен возможный дубликат между {new_person_id} и {person_id} "
                                  f"(схожесть: {similarity:.3f})")

                            # Можно автоматически объединить или пометить для ручной проверки
                            self._mark_for_review(new_person_id, person_id, similarity)
                            break

        except Exception as e:
            print(f"⚠️ Ошибка проверки на дубликаты: {e}")

    def _mark_for_review(self, person1, person2, similarity):
        """Пометка пары людей для ручной проверки"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO merges (old_person_id, new_person_id, reason)
                VALUES (?, ?, ?)
            ''', (person1, person2, f"Auto-detected duplicate, similarity: {similarity:.3f}"))

            self.conn.commit()
            print(f"📝 Пара {person1} - {person2} помечена для проверки")

        except Exception as e:
            print(f"❌ Ошибка пометки для проверки: {e}")

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

    def add_detection(self, face_id, camera_id, confidence, bbox, embedding_hash=None):
        """Добавление записи о детекции"""
        try:
            cursor = self.conn.cursor()

            # Получаем person_id для лица
            cursor.execute('SELECT person_id FROM known_faces WHERE face_id = ?', (face_id,))
            result = cursor.fetchone()
            person_id = result[0] if result else "unknown"

            cursor.execute('''
                INSERT INTO detections (face_id, person_id, camera_id, confidence, bbox, embedding_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (face_id, person_id, camera_id, confidence, json.dumps(bbox), embedding_hash))

            self.conn.commit()
            return cursor.lastrowid

        except Exception as e:
            print(f"❌ Ошибка добавления детекции: {e}")
            return None

    def start_session(self, face_id, camera_id):
        """Начало сессии для лица"""
        try:
            cursor = self.conn.cursor()

            # Получаем person_id
            cursor.execute('SELECT person_id FROM known_faces WHERE face_id = ?', (face_id,))
            result = cursor.fetchone()
            person_id = result[0] if result else "unknown"

            cursor.execute('''
                INSERT INTO sessions (face_id, person_id, start_time)
                VALUES (?, ?, ?)
            ''', (face_id, person_id, datetime.now()))

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

    def deduplicate_faces(self, similarity_threshold=0.85):
        """Удаление дубликатов лиц в базе"""
        try:
            cursor = self.conn.cursor()

            # Получаем все активные лица
            cursor.execute('''
                SELECT face_id, person_id, embedding, quality_score, confidence
                FROM known_faces 
                WHERE is_active = 1
                ORDER BY person_id, quality_score DESC, confidence DESC
            ''')

            all_faces = cursor.fetchall()

            # Группируем по person_id
            faces_by_person = defaultdict(list)
            for face_id, person_id, embedding_blob, quality_score, confidence in all_faces:
                embedding = pickle.loads(embedding_blob)
                faces_by_person[person_id].append({
                    'face_id': face_id,
                    'embedding': embedding,
                    'quality_score': quality_score,
                    'confidence': confidence
                })

            # Для каждого человека оставляем только уникальные лица
            faces_to_deactivate = []
            updated_primary = []

            for person_id, faces in faces_by_person.items():
                if len(faces) <= 1:
                    # Отмечаем как primary если еще не отмечен
                    if faces:
                        cursor.execute('''
                            UPDATE known_faces SET is_primary = 1 
                            WHERE face_id = ? AND is_primary = 0
                        ''', (faces[0]['face_id'],))
                        updated_primary.append(faces[0]['face_id'])
                    continue

                # Сортируем по качеству
                faces.sort(key=lambda x: (x['quality_score'], x['confidence']), reverse=True)

                # Первое лицо всегда уникально и становится primary
                primary_face_id = faces[0]['face_id']
                unique_faces = [faces[0]]

                # Обновляем primary если нужно
                cursor.execute('''
                    UPDATE known_faces SET is_primary = 1 
                    WHERE face_id = ? AND is_primary = 0
                ''', (primary_face_id,))
                updated_primary.append(primary_face_id)

                # Сбрасываем primary для остальных
                cursor.execute('''
                    UPDATE known_faces SET is_primary = 0 
                    WHERE person_id = ? AND face_id != ?
                ''', (person_id, primary_face_id))

                # Сравниваем остальные лица с уникальными
                for i in range(1, len(faces)):
                    current_face = faces[i]
                    is_duplicate = False

                    current_emb_norm = self._normalize_embedding(current_face['embedding'])

                    for unique_face in unique_faces:
                        unique_emb_norm = self._normalize_embedding(unique_face['embedding'])
                        similarity = np.dot(current_emb_norm, unique_emb_norm)

                        if similarity > similarity_threshold:
                            # Это дубликат
                            faces_to_deactivate.append(current_face['face_id'])
                            is_duplicate = True
                            print(f"  🗑️ Дубликат: лицо {current_face['face_id']} "
                                  f"похоже на {unique_face['face_id']} "
                                  f"(схожесть: {similarity:.3f})")
                            break

                    if not is_duplicate:
                        unique_faces.append(current_face)

            # Деактивируем дубликаты
            if faces_to_deactivate:
                placeholders = ','.join(['?'] * len(faces_to_deactivate))
                cursor.execute(f'''
                    UPDATE known_faces 
                    SET is_active = 0, is_primary = 0
                    WHERE face_id IN ({placeholders})
                ''', faces_to_deactivate)

                # Обновляем детекции
                cursor.execute(f'''
                    UPDATE detections 
                    SET person_id = (
                        SELECT person_id FROM known_faces 
                        WHERE is_active = 1 AND person_id = detections.person_id 
                        LIMIT 1
                    )
                    WHERE face_id IN ({placeholders})
                ''', faces_to_deactivate)

                self.conn.commit()
                print(f"🧹 Деактивировано {len(faces_to_deactivate)} дубликатов")

            if updated_primary:
                print(f"📌 Обновлено {len(updated_primary)} primary лиц")

            # Обновляем кэш
            self.load_cache()

            return len(faces_to_deactivate)

        except Exception as e:
            print(f"❌ Ошибка дедупликации: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def merge_persons(self, old_person_id, new_person_id):
        """Объединение двух персон"""
        try:
            cursor = self.conn.cursor()

            # Проверяем существование людей
            cursor.execute('SELECT COUNT(*) FROM known_faces WHERE person_id = ?', (old_person_id,))
            old_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM known_faces WHERE person_id = ?', (new_person_id,))
            new_count = cursor.fetchone()[0]

            if old_count == 0 or new_count == 0:
                print(f"⚠️ Один из людей не найден: {old_person_id} ({old_count}), {new_person_id} ({new_count})")
                return False

            # Выбираем основного человека (того, у кого больше лиц или более свежий)
            cursor.execute('''
                SELECT person_id, COUNT(*) as cnt, MAX(last_seen) as last_seen
                FROM known_faces 
                WHERE person_id IN (?, ?)
                GROUP BY person_id
                ORDER BY cnt DESC, last_seen DESC
                LIMIT 1
            ''', (old_person_id, new_person_id))

            result = cursor.fetchone()
            target_person_id = result[0] if result else new_person_id
            source_person_id = old_person_id if target_person_id == new_person_id else new_person_id

            # Обновляем person_id во всех таблицах
            cursor.execute('''
                UPDATE known_faces 
                SET person_id = ? 
                WHERE person_id = ?
            ''', (target_person_id, source_person_id))

            cursor.execute('''
                UPDATE detections 
                SET person_id = ? 
                WHERE person_id = ?
            ''', (target_person_id, source_person_id))

            cursor.execute('''
                UPDATE sessions 
                SET person_id = ? 
                WHERE person_id = ?
            ''', (target_person_id, source_person_id))

            # Записываем слияние
            cursor.execute('''
                INSERT INTO merges (old_person_id, new_person_id, reason)
                VALUES (?, ?, ?)
            ''', (source_person_id, target_person_id, "Manual merge"))

            self.conn.commit()

            # Обновляем кэш
            self.load_cache()

            print(f"🔗 Объединены {source_person_id} -> {target_person_id}")

            return True

        except Exception as e:
            print(f"❌ Ошибка объединения персон: {e}")
            return False

    def get_person_faces(self, person_id):
        """Получение всех лиц человека"""
        if person_id not in self.person_cache:
            return []

        faces = []
        for face_id in self.person_cache[person_id]:
            if face_id in self.face_cache:
                faces.append({
                    'face_id': face_id,
                    **self.face_cache[face_id]
                })

        return faces

    def get_statistics(self, period_hours=24):
        """Получение статистики"""
        try:
            cursor = self.conn.cursor()

            # Общая статистика
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT person_id) as total_people,
                    COUNT(*) as total_faces,
                    SUM(visit_count) as total_visits,
                    AVG(confidence) as avg_confidence
                FROM known_faces
                WHERE is_active = 1
            ''')
            total_stats = cursor.fetchone()

            # Статистика за период
            time_threshold = datetime.now() - timedelta(hours=period_hours)
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT d.person_id) as recent_people,
                    COUNT(*) as recent_detections
                FROM detections d
                WHERE d.timestamp > ?
            ''', (time_threshold,))
            recent_stats = cursor.fetchone()

            # Статистика по дубликатам
            cursor.execute('''
                SELECT 
                    person_id,
                    COUNT(*) as face_count,
                    AVG(confidence) as avg_conf,
                    MAX(last_seen) as last_seen
                FROM known_faces
                WHERE is_active = 1
                GROUP BY person_id
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                LIMIT 10
            ''')
            duplicates_stats = cursor.fetchall()

            return {
                'total_people': total_stats[0] or 0,
                'total_faces': total_stats[1] or 0,
                'total_visits': total_stats[2] or 0,
                'avg_confidence': float(total_stats[3] or 0),
                'recent_people': recent_stats[0] or 0,
                'recent_detections': recent_stats[1] or 0,
                'duplicates': [
                    {
                        'person_id': row[0],
                        'face_count': row[1],
                        'avg_confidence': float(row[2] or 0),
                        'last_seen': row[3]
                    }
                    for row in duplicates_stats
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
                    s.duration,
                    kf.name
                FROM known_faces kf
                LEFT JOIN detections d ON kf.face_id = d.face_id
                LEFT JOIN sessions s ON kf.face_id = s.face_id
                WHERE kf.person_id = ? AND kf.is_active = 1
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
                    'duration': row[4],
                    'name': row[5]
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

            # Удаляем старые записи о слияниях
            cursor.execute('DELETE FROM merges WHERE merge_time < ?', (cutoff_date,))
            deleted_merges = cursor.rowcount

            self.conn.commit()

            print(f"🧹 Очистка данных: "
                  f"удалено {deleted_detections} детекций, "
                  f"{deleted_sessions} сессий, "
                  f"{deleted_merges} слияний, "
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