# /top_eye/debug_main.py
import sys
import os

# Добавляем текущую директорию
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"Python path: {sys.path[0]}")
print(f"Текущая директория: {os.getcwd()}")

try:
    from src.config.settings import settings

    print(f"✓ Настройки загружены: {settings.CAMERA_ID}")
except ImportError as e:
    print(f"✗ Ошибка импорта настроек: {e}")
    sys.exit(1)

import argparse
import signal
import time
import cv2


def test_camera():
    """Простой тест камеры"""
    print("\n🔍 Тестирование подключения к камере")
    print("-" * 40)

    print(f"Подключение к: {settings.RTSP_URL}")

    cap = cv2.VideoCapture(settings.RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("✗ Не удалось открыть камеру")
        return False

    print("✓ Камера подключена")

    # Пробуем получить несколько кадров
    frames_received = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            frames_received += 1
            print(f"  Кадр {i + 1}: {frame.shape}")

            # Сохраняем первый кадр
            if i == 0:
                cv2.imwrite('/tmp/test_camera_frame.jpg', frame)
                print(f"  Сохранен: /tmp/test_camera_frame.jpg")
        else:
            print(f"  ✗ Ошибка кадра {i + 1}")

        time.sleep(0.1)

    cap.release()

    if frames_received > 0:
        print(f"\n✅ Успешно получено {frames_received} кадров")
        return True
    else:
        print("\n✗ Не удалось получить ни одного кадра")
        return False


def run_simple_server():
    """Запуск простого веб-сервера"""
    print("\n🌐 Запуск веб-сервера...")

    try:
        from src.web.app import app
        import uvicorn

        # Простой видео процессор для теста
        class SimpleProcessor:
            def __init__(self, config):
                self.config = config
                self.cap = None
                self.current_count = 0
                self.today_unique = set()
                self.session_unique = set()

            def get_current_frame(self):
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.config.RTSP_URL)

                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Имитация детекции
                        import random
                        self.current_count = random.randint(0, 5)

                        return {
                            'frame': frame,
                            'people_count': self.current_count,
                            'detections': [],
                            'fps': 25.0
                        }

                return None

            def get_statistics(self):
                return {
                    'current_count': self.current_count,
                    'today_unique': len(self.today_unique),
                    'session_unique': len(self.session_unique),
                    'detections_history': 0
                }

        # Создаем и настраиваем процессор
        processor = SimpleProcessor(settings)
        app.state.processor = processor

        print(f"Сервер запущен на http://{settings.WEB_HOST}:{settings.WEB_PORT}")
        print("Откройте в браузере:")
        print(f"  http://localhost:{settings.WEB_PORT}")
        print(f"  или http://ваш_ip:{settings.WEB_PORT}")
        print("\nНажмите Ctrl+C для остановки")

        uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)

    except Exception as e:
        print(f"✗ Ошибка запуска сервера: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("🔧 ОТЛАДОЧНЫЙ РЕЖИМ СИСТЕМЫ ВИДЕОАНАЛИТИКИ")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Video Analytics Debug Mode')
    parser.add_argument('--mode', choices=['test', 'server'],
                        default='test', help='Режим работы')

    args = parser.parse_args()

    if args.mode == 'test':
        test_camera()
    else:
        # Обработка Ctrl+C
        def signal_handler(sig, frame):
            print("\n🛑 Завершение работы...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        run_simple_server()


if __name__ == "__main__":
    main()