# /top_eye/src/web/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import asyncio
import json
import base64
from datetime import datetime
import os

app = FastAPI(title="Video Analytics System", version="1.0.0")


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)


manager = ConnectionManager()


# Простой HTML интерфейс без статических файлов
@app.get("/")
async def get_dashboard():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Видеоаналитика - Детский спортивный зал</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: Arial, sans-serif;
            }

            body {
                background: #f0f2f5;
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }

            .video-panel {
                flex: 3;
                min-width: 300px;
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            .stats-panel {
                flex: 1;
                min-width: 300px;
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            h1, h2 {
                color: #333;
                margin-bottom: 20px;
            }

            .video-container {
                background: #000;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 20px;
                position: relative;
            }

            #video {
                width: 100%;
                display: block;
            }

            .connection-status {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 12px;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 30px;
            }

            .stat-card {
                background: #4a6fa5;
                color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }

            .stat-value {
                font-size: 24px;
                font-weight: bold;
                margin: 5px 0;
            }

            .stat-label {
                font-size: 12px;
                opacity: 0.9;
            }

            .controls {
                margin-top: 20px;
                display: flex;
                gap: 10px;
            }

            button {
                padding: 10px 15px;
                background: #4a6fa5;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                flex: 1;
            }

            button:hover {
                background: #3a5a80;
            }

            #detectionsHistory {
                margin-top: 20px;
                max-height: 200px;
                overflow-y: auto;
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }

            .detection-item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <h1>🏆 Система видеоаналитики спортивного зала</h1>

        <div class="container">
            <div class="video-panel">
                <h2>📹 Видеопоток с камеры</h2>
                <div class="video-container">
                    <img id="video" alt="Видео с камеры">
                    <div class="connection-status">
                        <span id="status">🟢 Подключение...</span>
                    </div>
                </div>

                <div class="controls">
                    <button onclick="exportData('today')">📥 Экспорт за день</button>
                    <button onclick="takeSnapshot()">📷 Снимок</button>
                    <button onclick="toggleDetection()" id="detectionBtn">⏸️ Пауза детекции</button>
                </div>
            </div>

            <div class="stats-panel">
                <h2>📊 Статистика</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Сейчас в зале</div>
                        <div class="stat-value" id="currentCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Уникальных сегодня</div>
                        <div class="stat-value" id="todayCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">За тренировку</div>
                        <div class="stat-value" id="sessionCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">FPS обработки</div>
                        <div class="stat-value" id="fpsCount">0</div>
                    </div>
                </div>

                <h3>История детекций</h3>
                <div id="detectionsHistory">
                    <!-- История будет здесь -->
                </div>
            </div>
        </div>

        <script>
            let ws;
            let detectionEnabled = true;

            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/video`;

                ws = new WebSocket(wsUrl);

                ws.onopen = function() {
                    console.log('WebSocket подключен');
                    document.getElementById('status').innerHTML = '🟢 Подключено';
                    document.getElementById('status').style.color = '#28a745';
                };

                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === 'frame' && data.frame) {
                            // Обновляем видео
                            document.getElementById('video').src = 'data:image/jpeg;base64,' + data.frame;

                            // Обновляем статистику
                            document.getElementById('currentCount').textContent = data.current_count || 0;
                            document.getElementById('todayCount').textContent = data.today_unique || 0;
                            document.getElementById('sessionCount').textContent = data.session_unique || 0;
                            document.getElementById('fpsCount').textContent = data.fps ? data.fps.toFixed(1) : '0.0';

                            // Обновляем историю
                            if (data.detections && data.detections.length > 0) {
                                updateDetectionHistory(data.detections, data.timestamp);
                            }
                        }
                    } catch (error) {
                        console.error('Ошибка обработки сообщения:', error);
                    }
                };

                ws.onclose = function() {
                    console.log('WebSocket отключен, переподключение через 3 сек...');
                    document.getElementById('status').innerHTML = '🔴 Отключено';
                    document.getElementById('status').style.color = '#dc3545';
                    setTimeout(connectWebSocket, 3000);
                };

                ws.onerror = function(error) {
                    console.error('WebSocket ошибка:', error);
                };
            }

            function updateDetectionHistory(detections, timestamp) {
                const container = document.getElementById('detectionsHistory');
                const time = new Date(timestamp).toLocaleTimeString();

                detections.forEach(det => {
                    const item = document.createElement('div');
                    item.className = 'detection-item';
                    item.innerHTML = `
                        <strong>${time}</strong> - ID ${det.track_id} (${det.confidence ? (det.confidence * 100).toFixed(0) + '%' : '?'})
                    `;
                    container.prepend(item);

                    // Ограничиваем количество записей
                    if (container.children.length > 20) {
                        container.removeChild(container.lastChild);
                    }
                });
            }

            function exportData(period) {
                fetch(`/api/export/${period}`)
                    .then(response => response.blob())
                    .then(blob => {
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `статистика_${period}_${new Date().toISOString().split('T')[0]}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);

                        alert(`Статистика за ${period} экспортирована`);
                    })
                    .catch(error => {
                        console.error('Ошибка экспорта:', error);
                        alert('Ошибка при экспорте');
                    });
            }

            function takeSnapshot() {
                const video = document.getElementById('video');
                if (video.src) {
                    const link = document.createElement('a');
                    link.href = video.src;
                    link.download = `снимок_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    alert('Снимок сохранен');
                } else {
                    alert('Нет видео для сохранения');
                }
            }

            function toggleDetection() {
                detectionEnabled = !detectionEnabled;
                const btn = document.getElementById('detectionBtn');
                btn.textContent = detectionEnabled ? '⏸️ Пауза детекции' : '▶️ Возобновить детекцию';

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ 
                        type: 'control', 
                        command: detectionEnabled ? 'enable_detection' : 'disable_detection' 
                    }));
                }
            }

            // Запускаем при загрузке
            document.addEventListener('DOMContentLoaded', connectWebSocket);

            // Автоматический ping каждые 30 секунд
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """)


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Ждем сообщения от клиента
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get('type') == 'ping' or message.get('type') == 'control':
                    # Получаем текущий кадр из процессора
                    if hasattr(app.state, 'processor'):
                        processor = app.state.processor
                        frame_data = processor.get_current_frame()

                        if frame_data and frame_data.get('frame') is not None:
                            try:
                                frame = frame_data['frame']

                                # Если детекция включена и есть детекции, рисуем их
                                detections_enabled = message.get('command') != 'disable_detection'
                                if detections_enabled and frame_data.get('detections'):
                                    for det in frame_data.get('detections', []):
                                        bbox = det['bbox']
                                        if len(bbox) >= 4:
                                            # Рисуем прямоугольник
                                            cv2.rectangle(frame,
                                                          (int(bbox[0]), int(bbox[1])),
                                                          (int(bbox[2]), int(bbox[3])),
                                                          (0, 255, 0), 2)

                                            # Подпись с ID
                                            cv2.putText(frame, f"ID: {det.get('track_id', '?')}",
                                                        (int(bbox[0]), int(bbox[1]) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                        (0, 255, 0), 2)

                                # Ресайз для веб-стрима (опционально, для экономии трафика)
                                if frame.shape[1] > 1280:
                                    frame = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))

                                # Кодируем в JPEG
                                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                                # Получаем статистику
                                stats = {}
                                if hasattr(processor, 'get_statistics'):
                                    stats = processor.get_statistics()

                                # Отправляем кадр клиенту
                                response = {
                                    'type': 'frame',
                                    'frame': frame_base64,
                                    'current_count': frame_data.get('people_count', 0),
                                    'today_unique': stats.get('today_unique', 0),
                                    'session_unique': stats.get('session_unique', 0),
                                    'fps': frame_data.get('fps', 0),
                                    'detections': frame_data.get('detections', []),
                                    'timestamp': datetime.now().isoformat()
                                }

                                await websocket.send_json(response)

                            except Exception as e:
                                print(f"Ошибка обработки кадра: {e}")
                                # Отправляем сообщение об ошибке
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': str(e)
                                })
                    else:
                        # Если процессор не инициализирован
                        await websocket.send_json({
                            'type': 'error',
                            'message': 'Процессор видео не инициализирован'
                        })

            except json.JSONDecodeError:
                # Игнорируем не-JSON сообщения
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Ошибка WebSocket: {e}")
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_statistics():
    """Получить статистику"""
    try:
        if hasattr(app.state, 'processor'):
            processor = app.state.processor
            if hasattr(processor, 'get_statistics'):
                return JSONResponse(processor.get_statistics())
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")

    # Возвращаем данные по умолчанию
    return JSONResponse({
        "current_count": 0,
        "today_unique": 0,
        "session_unique": 0,
        "detections_history": 0
    })


@app.get("/api/export/{period}")
async def export_statistics(period: str):
    """Экспорт статистики"""
    from datetime import datetime

    # Генерируем тестовые данные
    data = {
        "period": period,
        "exported_at": datetime.now().isoformat(),
        "camera": "trassir_tr-d1415_1",
        "statistics": {
            "current_count": 0,
            "today_unique": 0,
            "session_unique": 0
        },
        "detections": []
    }

    # Пытаемся получить реальные данные
    try:
        if hasattr(app.state, 'processor'):
            processor = app.state.processor
            data["camera"] = processor.config.CAMERA_ID
            data["statistics"] = {
                "current_count": processor.current_count,
                "today_unique": len(processor.today_unique),
                "session_unique": len(processor.session_unique)
            }
    except:
        pass

    return JSONResponse(data)


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/test")
async def test_page():
    """Тестовая страница"""
    return HTMLResponse("""
    <h1>Система видеоаналитики работает! 🎉</h1>
    <p><a href="/">Перейти к основной панели</a></p>
    <p><a href="/health">Проверка здоровья</a></p>
    <p><a href="/api/stats">Статистика</a></p>
    """)