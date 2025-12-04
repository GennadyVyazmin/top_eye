# /top_eye/src/web/app.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import asyncio
import json
import base64
from datetime import datetime
import time

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
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }

            .stats-panel {
                flex: 1;
                min-width: 300px;
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }

            h1 {
                color: white;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            .video-container {
                position: relative;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 20px;
            }

            #video {
                width: 100%;
                display: block;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 30px;
            }

            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                transition: transform 0.3s ease;
            }

            .stat-card:hover {
                transform: translateY(-5px);
            }

            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }

            .stat-label {
                font-size: 0.9em;
                opacity: 0.9;
            }

            .connection-status {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 0.8em;
            }

            .controls {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }

            button {
                padding: 10px 15px;
                background: #4a6fa5;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                flex: 1;
                transition: background 0.3s;
            }

            button:hover {
                background: #3a5a80;
            }

            #detectionsHistory {
                margin-top: 20px;
                max-height: 300px;
                overflow-y: auto;
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }

            .detection-item {
                padding: 8px;
                border-bottom: 1px solid #ddd;
                display: flex;
                justify-content: space-between;
            }

            .time {
                color: #666;
                font-size: 0.9em;
            }

            .track-id {
                font-weight: bold;
                color: #4a6fa5;
            }

            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }

                .stats-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <h1>🏆 Система видеоаналитики детского спортивного зала</h1>

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
                    <button onclick="location.reload()">🔄 Обновить</button>
                </div>
            </div>

            <div class="stats-panel">
                <h2>📊 Статистика зала</h2>
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
                        <div class="stat-label">FPS</div>
                        <div class="stat-value" id="fpsCount">0</div>
                    </div>
                </div>

                <h3>🎯 Активные посетители</h3>
                <div id="activeVisitors" style="margin-bottom: 20px;">
                    <!-- Активные посетители будут здесь -->
                </div>

                <h3>📋 История детекций</h3>
                <div id="detectionsHistory">
                    <div class="loading">Загрузка истории...</div>
                </div>
            </div>
        </div>

        <script>
            let ws;
            let lastFrameTime = Date.now();
            let frameCount = 0;
            let actualFPS = 0;

            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/video`;

                ws = new WebSocket(wsUrl);

                ws.onopen = function() {
                    console.log('WebSocket подключен');
                    document.getElementById('status').innerHTML = '🟢 Подключено';
                    document.getElementById('status').style.color = '#28a745';

                    // Запрашиваем первый кадр
                    requestFrame();
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

                            // Обновляем активных посетителей
                            updateActiveVisitors(data.detections || []);

                            // Обновляем историю
                            updateDetectionHistory(data.detections || [], data.timestamp);

                            // Вычисляем фактический FPS
                            frameCount++;
                            const now = Date.now();
                            if (now - lastFrameTime >= 1000) {
                                actualFPS = frameCount;
                                frameCount = 0;
                                lastFrameTime = now;
                            }

                            // Автоматически запрашиваем следующий кадр через 40мс (~25 FPS)
                            setTimeout(requestFrame, 40);
                        }
                        else if (data.type === 'error') {
                            console.error('Ошибка сервера:', data.message);
                            document.getElementById('status').innerHTML = '🔴 Ошибка';
                            document.getElementById('status').style.color = '#dc3545';

                            // Пробуем переподключиться через 3 секунды
                            setTimeout(requestFrame, 3000);
                        }
                    } catch (error) {
                        console.error('Ошибка обработки сообщения:', error);
                    }
                };

                ws.onclose = function() {
                    console.log('WebSocket отключен, переподключение через 3 секунды...');
                    document.getElementById('status').innerHTML = '🔴 Отключено';
                    document.getElementById('status').style.color = '#dc3545';
                    setTimeout(connectWebSocket, 3000);
                };

                ws.onerror = function(error) {
                    console.error('WebSocket ошибка:', error);
                };
            }

            function requestFrame() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'get_frame' }));
                }
            }

            function updateActiveVisitors(detections) {
                const container = document.getElementById('activeVisitors');

                if (detections.length === 0) {
                    container.innerHTML = '<div style="color: #666; text-align: center; padding: 10px;">Нет активных посетителей</div>';
                    return;
                }

                let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 10px;">';

                detections.forEach(det => {
                    const confidencePercent = det.confidence ? Math.round(det.confidence * 100) : '?';
                    html += `
                        <div style="text-align: center; background: #f0f2f5; padding: 10px; border-radius: 8px;">
                            <div style="font-size: 24px; font-weight: bold; color: #4a6fa5;">${det.track_id}</div>
                            <div style="font-size: 11px; color: #666;">ID посетителя</div>
                            <div style="font-size: 10px; color: #28a745; margin-top: 5px;">${confidencePercent}%</div>
                        </div>
                    `;
                });

                html += '</div>';
                container.innerHTML = html;
            }

            function updateDetectionHistory(detections, timestamp) {
                const container = document.getElementById('detectionsHistory');

                if (detections.length === 0) {
                    return;
                }

                const time = new Date(timestamp).toLocaleTimeString();

                detections.forEach(det => {
                    const confidencePercent = det.confidence ? Math.round(det.confidence * 100) : '?';

                    const item = document.createElement('div');
                    item.className = 'detection-item';
                    item.innerHTML = `
                        <div>
                            <span class="time">${time}</span> - 
                            <span class="track-id">ID ${det.track_id}</span>
                        </div>
                        <div style="color: #28a745;">${confidencePercent}%</div>
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
                    .then(response => response.json())
                    .then(data => {
                        // Создаем JSON для скачивания
                        const jsonStr = JSON.stringify(data, null, 2);
                        const blob = new Blob([jsonStr], { type: 'application/json' });
                        const url = window.URL.createObjectURL(blob);

                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `статистика_${period}_${new Date().toISOString().split('T')[0]}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);

                        alert(`Статистика за ${period} успешно экспортирована!`);
                    })
                    .catch(error => {
                        console.error('Ошибка экспорта:', error);
                        alert('Ошибка при экспорте данных');
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

                    alert('Снимок сохранен!');
                } else {
                    alert('Нет видео для сохранения');
                }
            }

            // Запускаем при загрузке
            document.addEventListener('DOMContentLoaded', connectWebSocket);

            // Обновляем статистику каждые 30 секунд
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

                if message.get('type') in ['get_frame', 'ping']:
                    # Получаем текущий кадр из процессора
                    if hasattr(app.state, 'processor'):
                        processor = app.state.processor
                        frame_data = processor.get_current_frame()

                        if frame_data and frame_data.get('frame') is not None:
                            try:
                                frame = frame_data['frame']

                                # Ресайз для веб-стрима (опционально)
                                max_width = 1280
                                if frame.shape[1] > max_width:
                                    scale = max_width / frame.shape[1]
                                    new_height = int(frame.shape[0] * scale)
                                    frame = cv2.resize(frame, (max_width, new_height))

                                # Кодируем в JPEG
                                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                                # Получаем статистику
                                stats = processor.get_statistics() if hasattr(processor, 'get_statistics') else {}

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
                                print(f"Ошибка кодирования кадра: {e}")
                                await websocket.send_json({
                                    'type': 'error',
                                    'message': str(e)
                                })
                        else:
                            # Если нет кадра, ждем немного
                            await asyncio.sleep(0.1)
                            await websocket.send_json({
                                'type': 'frame',
                                'frame': '',
                                'current_count': 0,
                                'timestamp': datetime.now().isoformat()
                            })
                    else:
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

    return JSONResponse({
        "current_count": 0,
        "today_unique": 0,
        "session_unique": 0,
        "active_tracks": 0
    })


@app.get("/api/export/{period}")
async def export_statistics(period: str):
    """Экспорт статистики"""
    from datetime import datetime, timedelta

    # Базовые данные
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

            stats = processor.get_statistics() if hasattr(processor, 'get_statistics') else {}
            data["statistics"] = {
                "current_count": stats.get('current_count', 0),
                "today_unique": stats.get('today_unique', 0),
                "session_unique": stats.get('session_unique', 0),
                "active_tracks": stats.get('active_tracks', 0)
            }

            # История детекций
            if hasattr(processor, 'get_detection_history'):
                history = processor.get_detection_history(limit=100)
                data["detections"] = history
    except Exception as e:
        print(f"Ошибка при экспорте: {e}")

    return JSONResponse(data)


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/test")
async def test_page():
    """Тестовая страница"""
    return HTMLResponse("""
    <h1>✅ Система видеоаналитики работает!</h1>
    <p><a href="/">Перейти к основной панели управления</a></p>
    <p><a href="/health">Проверка здоровья системы</a></p>
    <p><a href="/api/stats">Текущая статистика (JSON)</a></p>
    """)