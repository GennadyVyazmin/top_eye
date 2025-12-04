# /top_eye/main.py
import sys
import os

# Добавляем текущую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Python path: {sys.path}")
print(f"Текущая директория: {current_dir}")
print(f"Содержимое директории: {os.listdir(current_dir)}")

try:
    # Попробуем разные варианты импорта
    if os.path.exists(os.path.join(current_dir, 'src')):
        print("Обнаружена директория 'src'")
        from src.config.settings import settings

        print("✓ Импорт из src.config.settings успешен")
    elif os.path.exists(os.path.join(current_dir, 'config')):
        print("Обнаружена директория 'config'")
        from config.settings import settings

        print("✓ Импорт из config.settings успешен")
    else:
        print("✗ Не найдены директории config или src")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("Проверяем доступные модули...")

    # Покажем все доступные модули
    import pkgutil

    for module in pkgutil.iter_modules([current_dir]):
        print(f"  - {module.name}")

    sys.exit(1)

import argparse
import signal


def main():
    print("=" * 50)
    print(f"Система видеоаналитики запущена")
    print(f"Камера: {settings.RTSP_URL}")
    print(f"Разрешение: {settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}")
    print("=" * 50)

    parser = argparse.ArgumentParser(description='Video Analytics System')
    parser.add_argument('--mode', choices=['web', 'gui', 'both', 'test'],
                        default='test', help='Режим запуска')

    args = parser.parse_args()

    if args.mode == 'test':
        # Тестовый режим - проверка камеры
        test_camera_connection()
    else:
        # Полный режим
        run_full_system(args.mode)


def test_camera_connection():
    """Тест подключения к камере"""
    print("\n🔍 Тестирование подключения к камере...")

    try:
        import cv2

        print(f"Подключение к: {settings.RTSP_URL}")
        cap = cv2.VideoCapture(settings.RTSP_URL)

        if not cap.isOpened():
            print("✗ Не удалось открыть камеру")
            return

        ret, frame = cap.read()
        if ret:
            print(f"✓ Камера подключена успешно")
            print(f"  Размер кадра: {frame.shape}")
            print(f"  Тип данных: {frame.dtype}")

            # Покажем несколько кадров
            print("\n📹 Получение 5 кадров...")
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"  Кадр {i + 1}: {frame.shape}")
                else:
                    print(f"  ✗ Ошибка чтения кадра {i + 1}")
                import time
                time.sleep(0.1)
        else:
            print("✗ Не удалось прочитать кадр")

        cap.release()

    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


def run_full_system(mode):
    """Запуск полной системы"""
    try:
        # Импортируем здесь, чтобы не падать при тестовом режиме
        from src.core.video_processor import VideoProcessor

        processor = VideoProcessor(settings)

        def signal_handler(sig, frame):
            print("\nЗавершение работы...")
            processor.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if mode == 'web':
            from src.web.app import app
            import uvicorn

            processor.start()
            app.state.processor = processor
            print(f"\n🌐 Запуск веб-сервера на {settings.WEB_HOST}:{settings.WEB_PORT}")
            uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)

        elif mode == 'gui':
            from PyQt5.QtWidgets import QApplication
            from src.gui.main_window import MainWindow

            processor.start()
            qt_app = QApplication(sys.argv)
            window = MainWindow(processor)
            window.show()
            print("\n🖥️ Запуск GUI приложения")
            sys.exit(qt_app.exec_())

        elif mode == 'both':
            import threading
            from src.web.app import app
            from PyQt5.QtWidgets import QApplication
            from src.gui.main_window import MainWindow
            import uvicorn

            processor.start()

            # Веб-сервер в отдельном потоке
            def run_web():
                app.state.processor = processor
                uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)

            web_thread = threading.Thread(target=run_web, daemon=True)
            web_thread.start()

            # GUI в основном потоке
            qt_app = QApplication(sys.argv)
            window = MainWindow(processor)
            window.show()
            print("\n🚀 Запуск комбинированного режима (Web + GUI)")
            sys.exit(qt_app.exec_())

    except ImportError as e:
        print(f"\n✗ Ошибка импорта модуля: {e}")
        print("\nСоздаем базовые модули...")
        create_basic_modules()
        print("Попробуйте запустить снова: python main.py --mode test")


def create_basic_modules():
    """Создание базовых модулей если их нет"""

    # Создаем базовый config/settings.py если не существует
    config_dir = os.path.join(os.path.dirname(__file__), 'src', 'config')
    os.makedirs(config_dir, exist_ok=True)

    config_file = os.path.join(config_dir, 'settings.py')
    if not os.path.exists(config_file):
        print(f"Создаем {config_file}")
        with open(config_file, 'w') as f:
            f.write('''import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Параметры камеры Trassir TR-D1415
    RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:admin@10.0.0.242:554/live/main")
    CAMERA_ID = "trassir_tr-d1415_1"

    # Настройки обработки
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 25
    PROCESS_EVERY_N_FRAMES = 2

    # Пороги
    CONFIDENCE_THRESHOLD = 0.5
    FACE_MATCH_THRESHOLD = 0.6

    # Веб-сервер
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8000
    API_PORT = 8080

settings = Settings()
''')

    # Создаем базовый video_processor.py
    core_dir = os.path.join(os.path.dirname(__file__), 'src', 'core')
    os.makedirs(core_dir, exist_ok=True)

    processor_file = os.path.join(core_dir, 'video_processor.py')
    if not os.path.exists(processor_file):
        print(f"Создаем {processor_file}")
        with open(processor_file, 'w') as f:
            f.write('''import cv2
import time
from threading import Thread

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.running = False

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.config.RTSP_URL)
        print(f"Камера подключена: {self.config.RTSP_URL}")

    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            print("Камера отключена")
''')


if __name__ == "__main__":
    main()