# src/web/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import asyncio
import json
import base64
from datetime import datetime, timedelta

app = FastAPI(title="Video Analytics System")


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/")
async def get_dashboard():
    return HTMLResponse("""
    <html>
        <head>
            <title>Video Analytics Dashboard</title>
            <script>
                let ws = new WebSocket("ws://" + window.location.host + "/ws/video");

                ws.onmessage = function(event) {
                    let data = JSON.parse(event.data);
                    if(data.type === 'frame') {
                        document.getElementById('video').src = 'data:image/jpeg;base64,' + data.frame;
                    }
                    document.getElementById('current').innerText = data.current_count;
                    document.getElementById('today').innerText = data.today_unique;
                    document.getElementById('session').innerText = data.session_unique;
                };

                function updateStats() {
                    fetch('/api/stats')
                        .then(response => response.json())
                        .then(data => {
                            // Обновление статистики на странице
                        });
                }

                setInterval(updateStats, 5000);
            </script>
        </head>
        <body>
            <div class="container">
                <div class="video-container">
                    <img id="video" width="1280" height="720">
                </div>
                <div class="stats-container">
                    <h2>Статистика</h2>
                    <p>Сейчас в кадре: <span id="current">0</span></p>
                    <p>Уникальных сегодня: <span id="today">0</span></p>
                    <p>Уникальных за тренировку: <span id="session">0</span></p>
                </div>
            </div>
        </body>
    </html>
    """)


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Отправка кадров через WebSocket
            processor = app.state.processor
            data = processor.get_current_frame()

            if data:
                # Кодирование кадра в base64
                _, buffer = cv2.imencode('.jpg', data['frame'])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                stats = {
                    'type': 'frame',
                    'frame': frame_base64,
                    'current_count': processor.current_count,
                    'today_unique': len(processor.today_unique),
                    'session_unique': len(processor.session_unique),
                    'timestamp': datetime.now().isoformat()
                }

                await websocket.send_json(stats)

            await asyncio.sleep(0.04)  # ~25 FPS
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_statistics():
    # Возвращает статистику за различные периоды
    return {
        "last_3_hours": get_stats_for_period(hours=3),
        "yesterday": get_stats_for_period(days=1),
        "day_before_yesterday": get_stats_for_period(days=2),
        "week": get_stats_for_period(days=7),
        "month": get_stats_for_period(days=30)
    }


@app.get("/api/current_faces")
async def get_current_faces():
    # Возвращает список лиц, находящихся сейчас в кадре
    faces = app.state.processor.get_current_faces()
    return {"faces": faces}