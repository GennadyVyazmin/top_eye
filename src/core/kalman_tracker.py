# /top_eye/src/core/kalman_tracker.py
import numpy as np
import cv2


class KalmanTracker:
    """Калман фильтр для трекинга"""

    def __init__(self, bbox):
        # Инициализация фильтра Калмана
        self.kalman = cv2.KalmanFilter(8, 4)

        # Матрица перехода состояния (предполагаем постоянную скорость)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)

        # Матрица измерения
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        # Ковариация процесса
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03

        # Ковариация измерения
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        # Ковариация ошибки
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)

        # Инициализация состояния
        x, y, w, h = self._bbox_to_z(bbox)
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def _bbox_to_z(self, bbox):
        """Конвертация bbox в формат для фильтра"""
        x = bbox[0] + bbox[2] / 2
        y = bbox[1] + bbox[3] / 2
        w = bbox[2]
        h = bbox[3]
        return x, y, w, h

    def _z_to_bbox(self, z):
        """Конвертация из формата фильтра в bbox"""
        x, y, w, h = z[0], z[1], z[2], z[3]
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    def update(self, bbox):
        """Обновление фильтра"""
        x, y, w, h = self._bbox_to_z(bbox)
        measurement = np.array([x, y, w, h], dtype=np.float32)

        self.kalman.correct(measurement)
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        """Предсказание следующего состояния"""
        prediction = self.kalman.predict()
        self.time_since_update += 1
        self.age += 1

        bbox = self._z_to_bbox(prediction[:4])
        return bbox

    def get_state(self):
        """Получение текущего состояния"""
        state = self.kalman.statePost
        bbox = self._z_to_bbox(state[:4])
        velocity = state[4:6]

        return {
            'bbox': bbox,
            'velocity': velocity,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update
        }