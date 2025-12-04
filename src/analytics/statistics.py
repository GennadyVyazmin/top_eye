# src/analytics/statistics.py
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import json


class StatisticsManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    track_id INTEGER,
                    identity TEXT,
                    confidence REAL,
                    bbox TEXT,
                    face_image BLOB
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    unique_count INTEGER,
                    total_detections INTEGER
                )
            ''')

    def save_detection(self, detection_data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO detections 
                (timestamp, track_id, identity, confidence, bbox, face_image)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                detection_data['timestamp'],
                detection_data['track_id'],
                detection_data['identity'],
                detection_data['confidence'],
                json.dumps(detection_data['bbox']),
                detection_data['face_image']
            ))

    def get_statistics(self, interval_minutes):
        """Получить статистику за указанный интервал в минутах"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=interval_minutes)

        with sqlite3.connect(self.db_path) as conn:
            # Уникальные посетители
            unique_query = '''
                SELECT COUNT(DISTINCT identity) 
                FROM detections 
                WHERE timestamp BETWEEN ? AND ?
                AND identity != 'unknown'
            '''
            unique_count = conn.execute(unique_query,
                                        (start_time, end_time)).fetchone()[0]

            # Общее количество детекций
            total_query = '''
                SELECT COUNT(*) 
                FROM detections 
                WHERE timestamp BETWEEN ? AND ?
            '''
            total_count = conn.execute(total_query,
                                       (start_time, end_time)).fetchone()[0]

            # Статистика по часам
            hourly_query = '''
                SELECT strftime('%H', timestamp) as hour,
                       COUNT(DISTINCT identity) as unique_count
                FROM detections
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY hour
                ORDER BY hour
            '''
            hourly_stats = pd.read_sql_query(hourly_query, conn,
                                             params=(start_time, end_time))

        return {
            'period': f"Last {interval_minutes} minutes",
            'unique_visitors': unique_count,
            'total_detections': total_count,
            'hourly_breakdown': hourly_stats.to_dict('records')
        }

    def generate_report(self, intervals=None):
        if intervals is None:
            intervals = [180, 1440, 2880, 10080, 43200]  # 3ч, день, 2 дня, неделя, месяц

        report = {}
        for interval in intervals:
            stats = self.get_statistics(interval)
            report[f"{interval}_minutes"] = stats

        # Сохранение отчета в файл
        filename = f"statistics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report