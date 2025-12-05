# /top_eye/main.py - ОБНОВЛЕННЫЙ
import sys
import os

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


def main():
    print("=" * 60)
    print("🚀 УЛУЧШЕННАЯ СИСТЕМА ВИДЕОАНАЛИТИКИ")
    print("=" * 60)
    print(f"Камера: {settings.RTSP_URL}")
    print(f"Разрешение: {settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}")
    print(f"Веб-интерфейс: http://{settings.WEB_HOST}:{settings.WEB_PORT}")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Video Analytics System')
    parser.add_argument('--mode', choices=['web', 'simple', 'test'],
                        default='web', help='Режим запуска')

    args = parser.parse_args()

    if args.mode == 'test':
        test_camera()
    elif args.mode == 'simple':
        run_simple_mode()
    else:
        run_improved_mode()


def test_camera():
    """Тест камеры"""
    import cv2
    print(f"\n🔍 Тестирование камеры: {settings.RTSP_URL}")

    cap = cv2.VideoCapture(settings.RTSP_URL)
    if not cap.isOpened():
        print("✗ Не удалось подключиться к камере")
        return

    print("✓ Камера подключена")

    # Читаем несколько кадров
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Кадр {i + 1}: {frame.shape}")
            if i == 0:
                cv2.imwrite('/tmp/test_frame.jpg', frame)
                print(f"  Сохранен: /tmp/test_frame.jpg")
        time.sleep(0.1)

    cap.release()
    print("✅ Тест завершен")


def run_simple_mode():
    """Простой режим (старая версия)"""
    from src.core.video_processor_final import LongTermVideoProcessor
    from src.web.app import app
    import uvicorn
    import signal as sig

    processor = LongTermVideoProcessor(settings)

    def signal_handler(sig, frame):
        print("\n🛑 Завершение работы...")
        processor.stop()
        sys.exit(0)

    sig.signal(sig.SIGINT, signal_handler)

    processor.start()
    app.state.processor = processor

    print(f"\n🌐 Запуск веб-сервера на {settings.WEB_HOST}:{settings.WEB_PORT}")
    uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)


def run_improved_mode():
    """Улучшенный режим с лучшим трекингом"""
    # Импортируем улучшенный процессор
    from src.core.video_processor_improved import VideoProcessor
    from src.web.app import app
    import uvicorn
    import signal as sig

    processor = VideoProcessor(settings)

    def signal_handler(sig, frame):
        print("\n🛑 Завершение работы...")
        processor.stop()
        sys.exit(0)

    sig.signal(sig.SIGINT, signal_handler)

    processor.start()
    app.state.processor = processor

    print(f"\n🌐 Запуск УЛУЧШЕННОЙ системы на {settings.WEB_HOST}:{settings.WEB_PORT}")
    print("📊 Особенности улучшенного трекинга:")
    print("  • Устойчивые ID при поворотах и движении")
    print("  • Визуальные хеши для идентификации")
    print("  • Цветовые характеристики")
    print("  • Несколько критериев схожести")
    print("  • Фильтрация кратковременных треков")

    uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)


if __name__ == "__main__":
    import time

    main()