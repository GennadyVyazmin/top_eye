# /top_eye/src/web/app_extended.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import json
import os
from datetime import datetime

app = FastAPI(title="Long-term Face Recognition System")


# Дополнительные endpoint'ы для управления базой лиц

@app.post("/api/register_person")
async def register_person(
        name: str = Form(...),
        images: list[UploadFile] = File(...)
):
    """Регистрация нового человека в системе"""
    try:
        processor = app.state.processor

        # Конвертируем изображения
        image_list = []
        for img_file in images:
            contents = await img_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                image_list.append(img)

        # Регистрируем человека
        result = processor.register_person(name, images=image_list)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/persons")
async def get_persons(limit: int = 50, offset: int = 0):
    """Получение списка известных людей"""
    try:
        processor = app.state.processor
        db_stats = processor.face_db.get_statistics()

        # Здесь можно добавить логику получения списка людей
        persons = []
        #TODO: Реализовать получение списка из базы

        return JSONResponse({
            'persons': persons,
            'total': db_stats.get('total_people', 0),
            'limit': limit,
            'offset': offset
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/person/{person_id}")
async def get_person(person_id: str):
    """Получение информации о конкретном человеке"""
    try:
        processor = app.state.processor

        # Получаем историю
        history = processor.get_person_info(person_id)

        # Получаем статистику по человеку
        #TODO: Добавить получение статистики из базы

        return JSONResponse({
            'person_id': person_id,
            'history': history,
            'total_visits': len(history),
            'last_seen': history[0]['timestamp'] if history else None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export_statistics")
async def export_statistics(period: str = "day"):
    """Экспорт статистики"""
    try:
        processor = app.state.processor

        # Получаем статистику
        stats = processor.get_detailed_statistics()

        # Создаем файл для скачивания
        filename = f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"data/exports/{filename}"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        return FileResponse(
            filepath,
            media_type='application/json',
            filename=filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin")
async def admin_panel():
    """Панель администратора"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Admin Panel - Face Recognition</title>
        <style>
            body { font-family: Arial; padding: 20px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
            button { padding: 10px 15px; margin: 5px; }
            .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
            .stat-card { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>📊 Панель управления системой распознавания</h1>

        <div class="section">
            <h2>Регистрация нового человека</h2>
            <form id="registerForm" enctype="multipart/form-data">
                <input type="text" id="personName" placeholder="Имя человека" required>
                <input type="file" id="personImages" multiple accept="image/*">
                <button type="submit">Зарегистрировать</button>
            </form>
        </div>

        <div class="section">
            <h2>Статистика</h2>
            <div class="stats-grid" id="statsContainer">
                <!-- Статистика будет здесь -->
            </div>
            <button onclick="exportStats()">📥 Экспорт статистики</button>
            <button onclick="refreshStats()">🔄 Обновить</button>
        </div>

        <div class="section">
            <h2>Известные люди</h2>
            <div id="personsList">
                <!-- Список людей будет здесь -->
            </div>
        </div>

        <script>
            // Регистрация человека
            document.getElementById('registerForm').onsubmit = async (e) => {
                e.preventDefault();

                const formData = new FormData();
                formData.append('name', document.getElementById('personName').value);

                const files = document.getElementById('personImages').files;
                for (let i = 0; i < files.length; i++) {
                    formData.append('images', files[i]);
                }

                const response = await fetch('/api/register_person', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                alert(result.message);

                if (result.success) {
                    refreshStats();
                }
            };

            // Загрузка статистики
            async function refreshStats() {
                const response = await fetch('/api/stats');
                const stats = await response.json();

                const container = document.getElementById('statsContainer');
                container.innerHTML = `
                    <div class="stat-card">
                        <h3>👥 Всего в базе</h3>
                        <p>${stats.database?.total_people || 0} человек</p>
                    </div>
                    <div class="stat-card">
                        <h3>📈 Детекций сегодня</h3>
                        <p>${stats.database?.recent_detections || 0}</p>
                    </div>
                    <div class="stat-card">
                        <h3>⏱️ Время работы</h3>
                        <p>${stats.system?.uptime?.toFixed(1) || 0} часов</p>
                    </div>
                `;
            }

            // Экспорт статистики
            async function exportStats() {
                const response = await fetch('/api/export_statistics?period=day');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `статистика_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }

            // Загрузка при старте
            document.addEventListener('DOMContentLoaded', refreshStats);
        </script>
    </body>
    </html>
    """)