# /top_eye/main.py
import sys
import os
import argparse
import signal
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"🚀 Python path: {sys.path[0]}")
print(f"📁 Текущая директория: {os.getcwd()}")

try:
    from src.config.settings import settings

    print(f"✅ Настройки загружены: {settings.CAMERA_ID}")
except ImportError as e:
    print(f"❌ Ошибка импорта настроек: {e}")
    sys.exit(1)


def main():
    print("=" * 60)
    print("🚀 СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ С ДЕДУПЛИКАЦИЕЙ")
    print("=" * 60)
    print(f"📹 Камера: {settings.CAMERA_ID}")
    print(f"🔗 RTSP: {settings.RTSP_URL}")
    print(f"📊 Разрешение: {settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}")
    print(f"🌐 Веб-интерфейс: http://{settings.WEB_HOST}:{settings.WEB_PORT}")
    print(f"🗄️ База данных: {settings.DB_PATH}")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', choices=['web', 'test', 'deduplicate', 'stats'],
                        default='web', help='Режим запуска')
    parser.add_argument('--camera', type=str, help='URL камеры (переопределяет настройки)')
    parser.add_argument('--port', type=int, default=settings.WEB_PORT, help='Порт веб-сервера')

    args = parser.parse_args()

    # Переопределение URL камеры если указано
    if args.camera:
        settings.RTSP_URL = args.camera
        print(f"📹 Используется камера из аргументов: {args.camera}")

    if args.mode == 'test':
        test_camera()
    elif args.mode == 'deduplicate':
        run_deduplication()
    elif args.mode == 'stats':
        show_statistics()
    else:
        run_web_mode(args.port)


def test_camera():
    """Тест камеры"""
    import cv2
    print(f"\n🔍 Тестирование камеры: {settings.RTSP_URL}")

    # Создаем необходимые директории
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(settings.RTSP_URL)
    if not cap.isOpened():
        print("❌ Не удалось подключиться к камере")
        print("   Проверьте:")
        print("   1. URL камеры: ", settings.RTSP_URL)
        print("   2. Сетевое подключение")
        print("   3. Порт 554 (RTSP)")
        return

    print("✅ Камера подключена")

    # Читаем несколько кадров
    success_count = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            print(f"  Кадр {i + 1}: {frame.shape}")
            if i == 0:
                # Сохраняем тестовый кадр
                test_path = os.path.join(settings.DATA_DIR, 'test_frame.jpg')
                cv2.imwrite(test_path, frame)
                print(f"  📸 Сохранен: {test_path}")
        else:
            print(f"  ❌ Ошибка чтения кадра {i + 1}")

        time.sleep(0.1)

    cap.release()

    if success_count >= 5:
        print(f"✅ Тест завершен успешно ({success_count}/10 кадров)")
    else:
        print(f"⚠️ Тест завершен с проблемами ({success_count}/10 кадров)")


def run_deduplication():
    """Запуск дедупликации"""
    print("\n🧹 Запуск дедупликации базы данных")

    try:
        from src.core.face_database import FaceDatabase

        # Инициализируем базу данных
        db = FaceDatabase(settings.DB_PATH)

        # Запускаем дедупликацию
        removed = db.deduplicate_faces(similarity_threshold=settings.DEDUPLICATION_THRESHOLD)

        print(f"✅ Дедупликация завершена")
        print(f"   Удалено дубликатов: {removed}")

        # Показываем статистику
        stats = db.get_statistics(period_hours=24)
        print(f"\n📊 Статистика после дедупликации:")
        print(f"   • Всего людей: {stats.get('total_people', 0)}")
        print(f"   • Всего лиц: {stats.get('total_faces', 0)}")
        print(f"   • Средняя уверенность: {stats.get('avg_confidence', 0):.2f}")

        if stats.get('duplicates'):
            print(f"   • Людей с дубликатами: {len(stats['duplicates'])}")
            for dup in stats['duplicates'][:3]:
                print(f"     - {dup['person_id'][-8:]}: {dup['face_count']} лиц")

        db.close()

    except Exception as e:
        print(f"❌ Ошибка дедупликации: {e}")
        import traceback
        traceback.print_exc()


def show_statistics():
    """Показать статистику"""
    print("\n📊 Статистика системы")

    try:
        from src.core.face_database import FaceDatabase

        db = FaceDatabase(settings.DB_PATH)
        stats = db.get_statistics(period_hours=24)

        print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
        print(f"   • Всего людей в базе: {stats.get('total_people', 0)}")
        print(f"   • Всего лиц в базе: {stats.get('total_faces', 0)}")
        print(f"   • Всего посещений: {stats.get('total_visits', 0)}")
        print(f"   • Средняя уверенность: {stats.get('avg_confidence', 0):.2f}")

        print(f"\n⏰ ЗА ПОСЛЕДНИЕ 24 ЧАСА:")
        print(f"   • Уникальных людей: {stats.get('recent_people', 0)}")
        print(f"   • Детекций: {stats.get('recent_detections', 0)}")

        if stats.get('duplicates'):
            print(f"\n⚠️ ЛЮДИ С ДУБЛИКАТАМИ (всего {len(stats['duplicates'])}):")
            for i, dup in enumerate(stats['duplicates'][:5], 1):
                print(f"   {i}. {dup['person_id'][-12:]} - {dup['face_count']} лиц, "
                      f"уверенность: {dup['avg_confidence']:.2f}")

        # Проверяем файл статистики
        stats_file = os.path.join(settings.STATISTICS_DIR, 'statistics.json')
        if os.path.exists(stats_file):
            print(f"\n💾 Файл статистики: {stats_file}")

        db.close()

    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")


def run_web_mode(port):
    """Запуск веб-режима с новой системой"""
    try:
        # Импортируем новую систему
        from src.core.video_processor_final import LongTermVideoProcessor
        from src.web.app import app
        from src.web.app_extended import app as extended_app

        # Объединяем маршруты
        app.include_router(extended_app.router)

        import uvicorn

        # Создаем директории если их нет
        os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
        os.makedirs(settings.EXPORTS_DIR, exist_ok=True)
        os.makedirs(settings.STATISTICS_DIR, exist_ok=True)

        print(f"\n🔄 Инициализация системы...")

        # Инициализируем процессор
        processor = LongTermVideoProcessor(settings)

        def signal_handler(sig, frame):
            print("\n🛑 Получен сигнал завершения...")
            processor.stop()
            print("✅ Система завершена корректно")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Запускаем процессор
        processor.start()

        # Сохраняем процессор в состоянии приложения
        app.state.processor = processor

        print(f"\n🌐 ЗАПУСК ВЕБ-СЕРВЕРА")
        print(f"   Адрес: http://{settings.WEB_HOST}:{port}")
        print(f"   Основная панель: http://{settings.WEB_HOST}:{port}/")
        print(f"   Админ панель: http://{settings.WEB_HOST}:{port}/admin")
        print(f"   Управление дубликатами: http://{settings.WEB_HOST}:{port}/admin/duplicates")
        print(f"\n📊 КОНТРОЛЬНЫЕ ТОЧКИ:")
        print(f"   • Проверка здоровья: http://{settings.WEB_HOST}:{port}/health")
        print(f"   • Статистика (JSON): http://{settings.WEB_HOST}:{port}/api/stats")
        print(f"   • Тестовая страница: http://{settings.WEB_HOST}:{port}/test")
        print(f"\n🎯 ОСОБЕННОСТИ СИСТЕМЫ:")
        print(f"   • Улучшенная дедупликация лиц")
        print(f"   • Многоуровневая проверка похожести")
        print(f"   • Предотвращение дублирования в реальном времени")
        print(f"   • Регулярная автоматическая очистка дубликатов")
        print("=" * 60)

        # Запускаем веб-сервер
        uvicorn.run(app, host=settings.WEB_HOST, port=port, log_level="info")

    except ImportError as e:
        print(f"❌ Ошибка импорта модулей: {e}")
        print("\n🔧 Устранение проблем:")
        print("1. Проверьте структуру проекта:")
        print("   /top_eye/")
        print("   ├── src/")
        print("   │   ├── core/")
        print("   │   │   ├── face_database.py")
        print("   │   │   ├── video_processor_final.py")
        print("   │   │   └── reid_model.py")
        print("   │   ├── config/")
        print("   │   │   └── settings.py")
        print("   │   └── web/")
        print("   │       ├── app.py")
        print("   │       └── app_extended.py")
        print("   └── main.py")
        print("\n2. Установите зависимости:")
        print("   pip install ultralytics opencv-python numpy scikit-learn")

    except Exception as e:
        print(f"❌ Критическая ошибка запуска: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()