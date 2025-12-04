# /top_eye/src/web/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import asyncio
import json
import base64
from datetime import datetime, timedelta
import numpy as np
import io
from PIL import Image

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


# Простой HTML интерфейс
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
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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

            .faces-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                gap: 10px;
                margin-top: 20px;
            }

            .face-card {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
                border: 2px solid #e9ecef;
                transition: all 0.3s ease;
            }

            .face-card:hover {
                border-color: #667eea;
                transform: scale(1.05);
            }

            .face-card img {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                object-fit: cover;
                margin-bottom: 5px;
                border: 3px solid #667eea;
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

            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }

            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <h1>🏆 Видеоаналитика детского спортивного зала</h1>

        <div class="container">
            <div class="video-panel">
                <h2>📹 Видеопоток с камеры</h2>
                <div class="video-container">
                    <img id="video" alt="Видео с камеры">
                    <div class="connection-status">
                        <span id="status">🟢 Подключение...</span>
                    </div>
                </div>
                <div class="faces-container" id="facesContainer">
                    <!-- Здесь будут отображаться лица -->
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
                        <div class="stat-label">За сегодня</div>
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

                <h3>⚡ Быстрый экспорт</h3>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <button onclick="exportData('3h')" style="flex: 1; padding: 10px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        3 часа
                    </button>
                    <button onclick="exportData('today')" style="flex: 1; padding: 10px; background: #17a2b8; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Сегодня
                    </button>
                    <button onclick="exportData('week')" style="flex: 1; padding: 10px; background: #ffc107; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Неделя
                    </button>
                </div>

                <h3 style="margin-top: 30px;">📋 История детекций</h3>
                <div id="detectionsHistory" style="margin-top: 10px; max-height: 200px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 5px;">
                    <!-- История будет здесь -->
                </div>
            </div>
        </div>

        <script>
            let ws;
            let reconnectInterval = 3000;

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

                            // Обновляем лица
                            updateFaces(data.faces || []);

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
                    console.log('WebSocket отключен, переподключение...');
                    document.getElementById('status').innerHTML = '🔴 Отключено';
                    document.getElementById('status').style.color = '#dc3545';
                    setTimeout(connectWebSocket, reconnectInterval);
                };

                ws.onerror = function(error) {
                    console.error('WebSocket ошибка:', error);
                };
            }

            function updateFaces(faces) {
                const container = document.getElementById('facesContainer');
                if (faces.length === 0) {
                    container.innerHTML = '<div class="loading">👥 Нет людей в кадре</div>';
                    return;
                }

                container.innerHTML = faces.map(face => `
                    <div class="face-card">
                        <img src="data:image/jpeg;base64,${face.face_image || ''}" alt="Лицо">
                        <div>ID: ${face.track_id || '?'}</div>
                        <small>${face.identity || 'Неизвестный'}</small>
                    </div>
                `).join('');
            }

            function updateDetectionHistory(detections, timestamp) {
                const container = document.getElementById('detectionsHistory');
                const time = new Date(timestamp).toLocaleTimeString();

                detections.forEach(det => {
                    const item = document.createElement('div');
                    item.style.padding = '5px';
                    item.style.borderBottom = '1px solid #dee2e6';
                    item.innerHTML = `
                        <small>${time}</small> - 
                        <strong>ID ${det.track_id}</strong> 
                        (${det.confidence ? (det.confidence * 100).toFixed(0) + '%' : '?'})
                    `;
                    container.prepend(item);

                    // Ограничиваем количество записей
                    if (container.children.length > 10) {
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
                    })
                    .catch(error => console.error('Ошибка экспорта:', error));
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
            # Ждем сообщения от клиента (ping)
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('type') == 'ping':
                # Отправляем текущий кадр
                processor = app.state.processor
                frame_data = processor.get_current_frame()

                if frame_data and frame_data.get('frame') is not None:
                    try:
                        # Кодируем кадр в JPEG
                        frame = frame_data['frame']
                        if frame is not None and frame.size > 0:
                            # Рисуем bounding boxes
                            for det in frame_data.get('detections', []):
                                bbox = det['bbox']
                                cv2.rectangle(frame,
                                              (int(bbox[0]), int(bbox[1])),
                                              (int(bbox[2]), int(bbox[3])),
                                              (0, 255, 0), 2)

                                # ID трека
                                cv2.putText(frame, f"ID: {det['track_id']}",
                                            (int(bbox[0]), int(bbox[1]) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2)

                            # Ресайз для веб-стрима
                            frame_resized = cv2.resize(frame, (1280, 720))

                            # Кодируем в base64
                            _, buffer = cv2.imencode('.jpg', frame_resized,
                                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')

                            # Получаем статистику
                            stats = processor.get_statistics()

                            # Формируем ответ
                            response = {
                                'type': 'frame',
                                'frame': frame_base64,
                                'current_count': stats['current_count'],
                                'today_unique': stats['today_unique'],
                                'session_unique': stats['session_unique'],
                                'fps': frame_data.get('fps', 0),
                                'detections': frame_data.get('detections', []),
                                'faces': frame_data.get('faces', []),
                                'timestamp': datetime.now().isoformat()
                            }

                            await websocket.send_json(response)
                    except Exception as e:
                        print(f"Ошибка кодирования кадра: {e}")
                        # Отправляем пустой кадр
                        await websocket.send_json({
                            'type': 'frame',
                            'frame': '',
                            'current_count': 0,
                            'today_unique': 0,
                            'session_unique': 0,
                            'fps': 0,
                            'timestamp': datetime.now().isoformat()
                        })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket ошибка: {e}")
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_statistics():
    """Получить статистику"""
    try:
        processor = app.state.processor
        if processor:
            return JSONResponse(processor.get_statistics())
    except:
        pass

    return JSONResponse({
        "current_count": 0,
        "today_unique": 0,
        "session_unique": 0,
        "detections_history": 0
    })


@app.get("/api/export/{period}")
async def export_statistics(period: str):
    """Экспорт статистики"""
    from datetime import datetime, timedelta

    # Генерируем тестовые данные
    data = {
        "period": period,
        "exported_at": datetime.now().isoformat(),
        "camera": app.state.processor.config.CAMERA_ID if hasattr(app.state, 'processor') else "unknown",
        "statistics": {
            "current_count": app.state.processor.current_count if hasattr(app.state, 'processor') else 0,
            "today_unique": len(app.state.processor.today_unique) if hasattr(app.state, 'processor') else 0,
            "session_unique": len(app.state.processor.session_unique) if hasattr(app.state, 'processor') else 0
        },
        "detections": []
    }

    return JSONResponse(data)


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# Добавляем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")