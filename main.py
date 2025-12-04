# main.py (в корне проекта)
import sys
import os

# Добавляем корневую директорию в путь Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import signal
from config.settings import settings
from core.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description='Video Analytics System')
    parser.add_argument('--mode', choices=['web', 'gui', 'both'],
                        default='both', help='Режим запуска')

    args = parser.parse_args()

    # Инициализация процессора
    processor = VideoProcessor(settings)

    # Обработка сигналов
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

        import threading
        web_thread = threading.Thread(target=run_web_server, args=(processor,))
        web_thread.daemon = True
        web_thread.start()

        run_gui(processor)


def run_web_server(processor):
    from web.app import app
    import uvicorn

    app.state.processor = processor
    uvicorn.run(app, host=settings.WEB_HOST, port=settings.WEB_PORT)


def run_gui(processor):
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow

    qt_app = QApplication(sys.argv)
    window = MainWindow(processor)
    window.show()
    return qt_app.exec_()


if __name__ == "__main__":
    main()