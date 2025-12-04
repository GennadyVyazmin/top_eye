# src/main.py
import argparse
import signal
import sys
from src.config.settings import settings
from src.core.video_processor import VideoProcessor
from src.web.app import app
from src.gui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication


def run_web_server(processor):
    """Запуск веб-сервера"""
    app.state.processor = processor
    import uvicorn
    uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)


def run_gui(processor):
    """Запуск GUI приложения"""
    qt_app = QApplication(sys.argv)
    window = MainWindow(processor)
    window.show()
    return qt_app.exec_()


def main():
    parser = argparse.ArgumentParser(description='Video Analytics System')
    parser.add_argument('--mode', choices=['web', 'gui', 'both'],
                        default='both', help='Режим запуска')

    args = parser.parse_args()

    # Инициализация процессора
    processor = VideoProcessor(settings)

    # Обработка сигналов для корректного завершения
    def signal_handler(sig, frame):
        print("Завершение работы...")
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.mode == 'web':
        processor.start()
        run_web_server(processor)
    elif args.mode == 'gui':
        processor.start()
        sys.exit(run_gui(processor))
    else:  # both
        processor.start()

        # Запуск в отдельных потоках
        import threading
        web_thread = threading.Thread(target=run_web_server, args=(processor,))
        web_thread.daemon = True
        web_thread.start()

        # GUI запускается в основном потоке
        run_gui(processor)


if __name__ == "__main__":
    main()