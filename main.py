# /top_eye/main.py
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


def main():
    print("=" * 60)
    print("🚀 СИСТЕМА ВИДЕОАНАЛИТИКИ ДЛЯ СПОРТИВНОГО ЗАЛА")
    print("=" * 60)
    print(f"Камера: {settings.RTSP_URL}")
    print(f"Разрешение: {settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}")
    print(f"Веб-интерфейс: http://{settings.WEB_HOST}:{settings.WEB_PORT}")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Video Analytics System')
    # ИСПРАВЛЕНО: добавлен режим 'simple'
    parser.add_argument('--mode', choices=['web', 'gui', 'both', 'test', 'simple'],
                        default='simple', help='Режим запуска')

    args = parser.parse_args()

    if args.mode == 'test':
        test_camera_only()
    elif args.mode == 'simple':
        run_simple_mode()
    else:
        run_full_mode(args.mode)


def test_camera_only():
    """Только тест камеры без моделей"""
    print("\n🔍 ТЕСТ ПОДКЛЮЧЕНИЯ К КАМЕРЕ")
    print("-" * 40)

    import cv2

    print(f"Подключение к: {settings.RTSP_URL}")

    # Пробуем разные варианты подключения
    rtsp_urls = [
        settings.RTSP_URL,
        settings.RTSP_URL.replace("rtsp://", "rtsp://admin:admin@"),
        "rtsp://admin:admin@10.0.0.242:554/stream1",
        "rtsp://admin:admin@10.0.0.242:554/h264",
        "rtsp://admin:admin@10.0.0.242:554/mjpeg"
    ]

    for url in rtsp_urls:
        print(f"\nПопытка подключения: {url}")
        cap = cv2.VideoCapture(url)

        if cap.isOpened():
            print(f"✓ Успешно!")

            # Пробуем получить кадр
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"  Кадр {i + 1}: {frame.shape if frame is not None else 'None'}")
                    # Сохраним первый кадр для проверки
                    if i == 0 and frame is not None:
                        cv2.imwrite('/tmp/test_frame.jpg', frame)
                        print(f"  Кадр сохранен: /tmp/test_frame.jpg")
                else:
                    print(f"  ✗ Ошибка чтения кадра")

                time.sleep(0.1)

            cap.release()
            break
        else:
            print("✗ Не удалось подключиться")

    print("\n✅ Тест завершен")


def run_simple_mode():
    """Простой режим - только захват видео"""
    print("\n🎥 ПРОСТОЙ РЕЖИМ - ЗАХВАТ ВИДЕО")
    print("-" * 40)

    from src.core.video_processor import VideoProcessor

    processor = VideoProcessor(settings)

    def signal_handler(sig, frame):
        print("\n🛑 Завершение работы...")
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Запускаем без моделей
    processor.running = True
    processor._reconnect_camera()

    print("\n📹 Захват видео запущен. Нажмите Ctrl+C для остановки.")
    print("Откройте в браузере: http://localhost:8000")

    # Простой веб-сервер для отображения видео
    start_simple_webserver(processor)


def start_simple_webserver(processor):
    """Простой веб-сервер для отображения видео"""
    try:
        from flask import Flask, Response, render_template_string
        import cv2
        import threading

        app = Flask(__name__)

        def generate_frames():
            while processor.running:
                if processor.cap and processor.cap.isOpened():
                    ret, frame = processor.cap.read()
                    if ret:
                        # Ресайз для веб-стрима
                        frame = cv2.resize(frame, (640, 360))
                        ret, buffer = cv2.imencode('.jpg', frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' +
                                   buffer.tobytes() + b'\r\n')
                    time.sleep(0.03)  # ~30 FPS
                else:
                    time.sleep(1)

        @app.route('/')
        def index():
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Видео с камеры</title>
                    <style>
                        body { margin: 0; padding: 20px; background: #222; }
                        .container { max-width: 800px; margin: 0 auto; }
                        h1 { color: white; text-align: center; }
                        .stats { background: #333; color: white; padding: 10px; 
                                border-radius: 5px; margin: 10px 0; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📹 Видео с камеры Trassir TR-D1415</h1>
                        <div class="stats">
                            <p>Камера: {{ camera_url }}</p>
                            <p>Статус: <span id="status">🟢 Активен</span></p>
                        </div>
                        <img src="/video_feed" width="640" height="360" 
                             style="border: 2px solid #555; border-radius: 5px;">
                    </div>
                </body>
                </html>
            ''', camera_url=settings.RTSP_URL)

        @app.route('/video_feed')
        def video_feed():
            return Response(generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # Запуск Flask в отдельном потоке
        flask_thread = threading.Thread(
            target=lambda: app.run(
                host=settings.WEB_HOST,
                port=settings.WEB_PORT,
                debug=False,
                use_reloader=False
            ),
            daemon=True
        )
        flask_thread.start()

        print(f"🌐 Веб-интерфейс доступен по адресу: http://{settings.WEB_HOST}:{settings.WEB_PORT}")

        # Держим основную программу активной
        while processor.running:
            time.sleep(1)

    except ImportError as e:
        print(f"⚠ Flask не установлен: {e}")
        print("Установите: pip install flask")

        # Просто ждем Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            processor.stop()


def run_full_mode(mode):
    """Полный режим с моделями"""
    print(f"\n🚀 ЗАПУСК ПОЛНОГО РЕЖИМА ({mode.upper()})")
    print("-" * 40)

    try:
        # Проверяем наличие моделей
        print("Проверка зависимостей...")
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA доступна: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

        from src.core.video_processor import VideoProcessor

        processor = VideoProcessor(settings)

        def signal_handler(sig, frame):
            print("\n🛑 Завершение работы...")
            processor.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        if mode == 'web':
            from src.web.app import app
            import uvicorn

            processor.start()
            app.state.processor = processor
            print(f"\n🌐 Запуск FastAPI на {settings.WEB_HOST}:{settings.WEB_PORT}")
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
        print(f"\n✗ Не хватает зависимостей: {e}")
        print("\nУстановите необходимые пакеты:")
        print("pip install torch torchvision ultralytics opencv-python flask")
        print("\nИли запустите в простом режиме:")
        print("python main.py --mode simple")


if __name__ == "__main__":
    main()