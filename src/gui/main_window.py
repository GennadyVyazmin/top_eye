# src/gui/main_window.py
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTabWidget)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from datetime import datetime


class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    update_stats = pyqtSignal(dict)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.running = True

    def run(self):
        while self.running:
            data = self.processor.get_current_frame()
            if data:
                # Конвертация кадра для отображения
                frame = data['frame']
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line,
                                  QImage.Format_RGB888)

                self.change_pixmap.emit(qt_image)

                # Отправка статистики
                stats = {
                    'current': self.processor.current_count,
                    'today_unique': len(self.processor.today_unique),
                    'session_unique': len(self.processor.session_unique)
                }
                self.update_stats.emit(stats)

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Video Analytics System')
        self.setGeometry(100, 100, 1400, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QHBoxLayout(central_widget)

        # Левая панель - видео
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(1280, 720)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # Правая панель - статистика
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)

        # Текущая статистика
        self.current_label = QLabel("Сейчас в кадре: 0")
        self.today_label = QLabel("Уникальных сегодня: 0")
        self.session_label = QLabel("Уникальных за тренировку: 0")

        stats_layout.addWidget(QLabel("<h2>Статистика</h2>"))
        stats_layout.addWidget(self.current_label)
        stats_layout.addWidget(self.today_label)
        stats_layout.addWidget(self.session_label)
        stats_layout.addStretch()

        # Кнопки управления
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)

        self.start_btn = QPushButton("Запуск")
        self.stop_btn = QPushButton("Остановка")
        self.export_btn = QPushButton("Экспорт статистики")

        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.export_btn)

        stats_layout.addWidget(buttons_widget)

        # Добавление виджетов в главный layout
        main_layout.addWidget(video_widget, 70)
        main_layout.addWidget(stats_widget, 30)

        # Подключение сигналов
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.export_btn.clicked.connect(self.export_statistics)

        # Запуск потока видео
        self.video_thread = VideoThread(self.processor)
        self.video_thread.change_pixmap.connect(self.set_image)
        self.video_thread.update_stats.connect(self.update_stats_display)

    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_stats_display(self, stats):
        self.current_label.setText(f"Сейчас в кадре: {stats['current']}")
        self.today_label.setText(f"Уникальных сегодня: {stats['today_unique']}")
        self.session_label.setText(f"Уникальных за тренировку: {stats['session_unique']}")

    def start_processing(self):
        self.processor.start()
        self.video_thread.start()

    def stop_processing(self):
        self.processor.stop()
        self.video_thread.stop()

    def export_statistics(self):
        # Экспорт статистики в файл
        import pandas as pd
        from datetime import datetime

        stats = self.processor.get_all_statistics()
        df = pd.DataFrame(stats)
        filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()