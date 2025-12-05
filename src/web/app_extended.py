# /top_eye/src/web/app_extended.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import cv2
import numpy as np
import json
import os
from datetime import datetime

app = FastAPI(title="Long-term Face Recognition System")


# Дополнительные endpoint'ы для управления дубликатами

@app.get("/api/duplicates")
async def get_duplicates(threshold: float = 0.85):
    """Получение списка потенциальных дубликатов"""
    try:
        processor = app.state.processor

        # Получаем статистику с дубликатами
        stats = processor.face_db.get_statistics(period_hours=24)

        duplicates = stats.get('duplicates', [])

        # Фильтруем по порогу количества лиц
        filtered_dups = [dup for dup in duplicates if dup['face_count'] > 1]

        return JSONResponse({
            'duplicates': filtered_dups,
            'total': len(filtered_dups),
            'threshold': threshold
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/merge_persons")
async def merge_persons(
        person1_id: str = Form(...),
        person2_id: str = Form(...),
        reason: str = Form("Manual merge")
):
    """Объединение двух людей в одного"""
    try:
        processor = app.state.processor

        success = processor.face_db.merge_persons(person1_id, person2_id)

        if success:
            return JSONResponse({
                'success': True,
                'message': f'Successfully merged {person1_id} and {person2_id}',
                'merged_at': datetime.now().isoformat()
            })
        else:
            return JSONResponse({
                'success': False,
                'message': 'Failed to merge persons'
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/deduplicate")
async def run_deduplication(
        threshold: float = Form(0.85),
        auto_merge: bool = Form(False)
):
    """Запуск дедупликации"""
    try:
        processor = app.state.processor

        removed = processor.face_db.deduplicate_faces(similarity_threshold=threshold)

        return JSONResponse({
            'success': True,
            'duplicates_removed': removed,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/person/{person_id}/faces")
async def get_person_faces(person_id: str):
    """Получение всех лиц конкретного человека"""
    try:
        processor = app.state.processor

        faces = processor.face_db.get_person_faces(person_id)

        return JSONResponse({
            'person_id': person_id,
            'faces': faces,
            'total_faces': len(faces),
            'primary_faces': len([f for f in faces if f.get('is_primary', False)])
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/face/{face_id}")
async def delete_face(face_id: int, deactivate_only: bool = True):
    """Удаление или деактивация лица"""
    try:
        processor = app.state.processor

        cursor = processor.face_db.conn.cursor()

        if deactivate_only:
            cursor.execute('''
                UPDATE known_faces 
                SET is_active = 0, is_primary = 0
                WHERE face_id = ?
            ''', (face_id,))
            action = "deactivated"
        else:
            cursor.execute('DELETE FROM known_faces WHERE face_id = ?', (face_id,))
            action = "deleted"

        processor.face_db.conn.commit()

        # Обновляем кэш
        processor.face_db.load_cache()

        return JSONResponse({
            'success': True,
            'message': f'Face {face_id} {action}',
            'face_id': face_id,
            'action': action
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/duplicates")
async def duplicates_admin_panel():
    """Панель управления дубликатами"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Duplicate Management - Face Recognition</title>
        <style>
            body { font-family: Arial; padding: 20px; background: #f5f5f5; }
            .section { 
                margin: 20px 0; 
                padding: 25px; 
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; border-bottom: 2px solid #4a6fa5; padding-bottom: 10px; }
            h2 { color: #444; margin-top: 0; }
            button { 
                padding: 10px 20px; 
                background: #4a6fa5; 
                color: white; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                margin: 5px;
                transition: background 0.3s;
            }
            button:hover { background: #3a5a80; }
            button.danger { background: #dc3545; }
            button.danger:hover { background: #c82333; }
            button.success { background: #28a745; }
            button.success:hover { background: #218838; }
            .stats-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; 
                margin: 20px 0;
            }
            .stat-card { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 20px; 
                border-radius: 8px;
                text-align: center;
            }
            .stat-value { 
                font-size: 2em; 
                font-weight: bold; 
                margin: 10px 0;
            }
            .stat-label { 
                font-size: 0.9em; 
                opacity: 0.9;
            }
            .duplicates-list { 
                margin: 20px 0;
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .duplicate-item { 
                padding: 15px; 
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .duplicate-item:hover { background: #f8f9fa; }
            .person-info { flex: 1; }
            .person-id { 
                font-weight: bold; 
                color: #4a6fa5;
                font-family: monospace;
            }
            .face-count { 
                background: #ffc107; 
                color: #333;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 0.9em;
                margin-left: 10px;
            }
            .actions { display: flex; gap: 10px; }
            .loading { 
                text-align: center; 
                padding: 20px; 
                color: #666;
            }
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
            }
            .modal-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 30px;
                border-radius: 10px;
                min-width: 400px;
            }
        </style>
    </head>
    <body>
        <h1>🔄 Управление дубликатами лиц</h1>

        <div class="section">
            <h2>📊 Статистика дубликатов</h2>
            <div class="stats-grid" id="statsContainer">
                <div class="loading">Загрузка статистики...</div>
            </div>
            <div style="margin-top: 20px;">
                <button onclick="loadDuplicates()">🔄 Обновить список</button>
                <button onclick="runDeduplication()" class="success">🧹 Запустить дедупликацию</button>
                <button onclick="exportDuplicatesReport()" class="success">📥 Экспорт отчета</button>
            </div>
        </div>

        <div class="section">
            <h2>👥 Список дубликатов</h2>
            <div class="duplicates-list" id="duplicatesList">
                <div class="loading">Загрузка списка дубликатов...</div>
            </div>
            <div style="margin-top: 15px; font-size: 0.9em; color: #666;">
                <i>💡 Дубликаты - это люди у которых в базе более одного лица с высокой схожестью</i>
            </div>
        </div>

        <div class="section">
            <h2>⚙️ Настройки дедупликации</h2>
            <div style="display: flex; gap: 15px; align-items: center;">
                <div>
                    <label for="threshold">Порог схожести:</label>
                    <input type="range" id="threshold" min="0.7" max="0.95" step="0.01" value="0.85" 
                           style="width: 200px; margin: 0 10px;">
                    <span id="thresholdValue">0.85</span>
                </div>
                <div>
                    <label for="minFaces">Минимальное количество лиц:</label>
                    <input type="number" id="minFaces" min="2" max="10" value="2" 
                           style="width: 60px; margin: 0 10px;">
                </div>
            </div>
            <div style="margin-top: 15px;">
                <button onclick="testThreshold()" class="success">🎯 Протестировать порог</button>
                <button onclick="showMergeModal()" class="success">🔗 Объединить вручную</button>
            </div>
        </div>

        <!-- Модальное окно для ручного объединения -->
        <div id="mergeModal" class="modal">
            <div class="modal-content">
                <h2>🔗 Объединение людей</h2>
                <div style="margin: 20px 0;">
                    <div>
                        <label>ID первого человека:</label>
                        <input type="text" id="mergePerson1" style="width: 100%; padding: 8px; margin: 5px 0;">
                    </div>
                    <div>
                        <label>ID второго человека:</label>
                        <input type="text" id="mergePerson2" style="width: 100%; padding: 8px; margin: 5px 0;">
                    </div>
                    <div>
                        <label>Причина:</label>
                        <input type="text" id="mergeReason" value="Manual merge" style="width: 100%; padding: 8px; margin: 5px 0;">
                    </div>
                </div>
                <div style="text-align: right;">
                    <button onclick="closeMergeModal()">Отмена</button>
                    <button onclick="performMerge()" class="success">Объединить</button>
                </div>
            </div>
        </div>

        <script>
            // Загрузка статистики
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();

                    const container = document.getElementById('statsContainer');
                    if (stats.database) {
                        container.innerHTML = `
                            <div class="stat-card">
                                <div class="stat-label">Всего людей</div>
                                <div class="stat-value">${stats.database.total_people || 0}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Всего лиц</div>
                                <div class="stat-value">${stats.database.total_faces || 0}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Дубликатов</div>
                                <div class="stat-value">${stats.database.duplicates?.length || 0}</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-label">Средняя уверенность</div>
                                <div class="stat-value">${(stats.database.avg_confidence || 0).toFixed(2)}</div>
                            </div>
                        `;
                    }
                } catch (error) {
                    console.error('Ошибка загрузки статистики:', error);
                }
            }

            // Загрузка списка дубликатов
            async function loadDuplicates() {
                try {
                    const threshold = document.getElementById('threshold').value;
                    const response = await fetch(\`/api/duplicates?threshold=\${threshold}\`);
                    const data = await response.json();

                    const container = document.getElementById('duplicatesList');

                    if (data.duplicates && data.duplicates.length > 0) {
                        let html = '';
                        data.duplicates.forEach(dup => {
                            const shortId = dup.person_id.length > 12 ? 
                                dup.person_id.substring(dup.person_id.length - 12) : 
                                dup.person_id;

                            html += \`
                                <div class="duplicate-item">
                                    <div class="person-info">
                                        <span class="person-id">\${shortId}</span>
                                        <span class="face-count">\${dup.face_count} лиц</span>
                                        <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                                            Уверенность: \${dup.avg_confidence.toFixed(2)} | 
                                            Последний раз: \${new Date(dup.last_seen).toLocaleString()}
                                        </div>
                                    </div>
                                    <div class="actions">
                                        <button onclick="viewPersonFaces('\${dup.person_id}')">👁️ Просмотр</button>
                                        <button onclick="autoFixDuplicate('\${dup.person_id}')" class="success">🔄 Исправить</button>
                                    </div>
                                </div>
                            \`;
                        });
                        container.innerHTML = html;
                    } else {
                        container.innerHTML = '<div class="loading">🎉 Дубликатов не найдено!</div>';
                    }

                    // Обновляем статистику
                    loadStats();

                } catch (error) {
                    console.error('Ошибка загрузки дубликатов:', error);
                    document.getElementById('duplicatesList').innerHTML = 
                        '<div class="loading" style="color: #dc3545;">Ошибка загрузки</div>';
                }
            }

            // Запуск дедупликации
            async function runDeduplication() {
                try {
                    const threshold = document.getElementById('threshold').value;
                    const response = await fetch('/api/deduplicate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: \`threshold=\${threshold}&auto_merge=true\`
                    });

                    const result = await response.json();

                    if (result.success) {
                        alert(\`✅ Дедупликация завершена! Удалено \${result.duplicates_removed} дубликатов.\`);
                        loadDuplicates();
                        loadStats();
                    } else {
                        alert('❌ Ошибка дедупликации');
                    }

                } catch (error) {
                    console.error('Ошибка дедупликации:', error);
                    alert('❌ Ошибка при выполнении дедупликации');
                }
            }

            // Просмотр лиц конкретного человека
            async function viewPersonFaces(personId) {
                try {
                    const response = await fetch(\`/api/person/\${personId}/faces\`);
                    const data = await response.json();

                    let message = \`Человек \${personId} имеет \${data.total_faces} лиц:\n\n\`;

                    data.faces.forEach((face, index) => {
                        message += \`\${index + 1}. ID лица: \${face.face_id}\n\`;
                        message += \`   Уверенность: \${(face.confidence || 0).toFixed(2)}\n\`;
                        message += \`   Качество: \${(face.quality_score || 0).toFixed(2)}\n\`;
                        if (face.is_primary) message += \`   ⭐ Основное лицо\n\`;
                        message += '\n';
                    });

                    alert(message);

                } catch (error) {
                    console.error('Ошибка просмотра лиц:', error);
                    alert('❌ Ошибка при получении информации о лицах');
                }
            }

            // Автоматическое исправление дубликата
            async function autoFixDuplicate(personId) {
                if (confirm(\`Вы уверены, что хотите автоматически исправить дубликаты для человека \${personId}?\`)) {
                    try {
                        // Здесь можно добавить логику автоматического исправления
                        // Например, оставить только лучшее лицо
                        alert(\`Функция автоматического исправления для \${personId} в разработке\`);

                    } catch (error) {
                        console.error('Ошибка исправления:', error);
                        alert('❌ Ошибка при исправлении дубликата');
                    }
                }
            }

            // Тестирование порога
            async function testThreshold() {
                const threshold = document.getElementById('threshold').value;
                const minFaces = document.getElementById('minFaces').value;

                try {
                    const response = await fetch(\`/api/duplicates?threshold=\${threshold}\`);
                    const data = await response.json();

                    const filtered = data.duplicates.filter(d => d.face_count >= parseInt(minFaces));

                    alert(\`При пороге \${threshold} и минимуме \${minFaces} лиц найдено \${filtered.length} дубликатов.\`);

                } catch (error) {
                    console.error('Ошибка тестирования:', error);
                    alert('❌ Ошибка при тестировании порога');
                }
            }

            // Экспорт отчета
            async function exportDuplicatesReport() {
                try {
                    const response = await fetch('/api/duplicates?threshold=0.8');
                    const data = await response.json();

                    const report = {
                        generated_at: new Date().toISOString(),
                        threshold: 0.8,
                        total_duplicates: data.total,
                        duplicates: data.duplicates
                    };

                    const jsonStr = JSON.stringify(report, null, 2);
                    const blob = new Blob([jsonStr], { type: 'application/json' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = \`duplicates_report_\${new Date().toISOString().split('T')[0]}.json\`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);

                    alert('✅ Отчет о дубликатах успешно экспортирован!');

                } catch (error) {
                    console.error('Ошибка экспорта:', error);
                    alert('❌ Ошибка при экспорте отчета');
                }
            }

            // Управление модальным окном объединения
            function showMergeModal() {
                document.getElementById('mergeModal').style.display = 'block';
            }

            function closeMergeModal() {
                document.getElementById('mergeModal').style.display = 'none';
            }

            async function performMerge() {
                const person1 = document.getElementById('mergePerson1').value.trim();
                const person2 = document.getElementById('mergePerson2').value.trim();
                const reason = document.getElementById('mergeReason').value.trim();

                if (!person1 || !person2) {
                    alert('❌ Пожалуйста, заполните оба ID');
                    return;
                }

                if (confirm(\`Вы уверены, что хотите объединить \${person1} и \${person2}?\`)) {
                    try {
                        const response = await fetch('/api/merge_persons', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/x-www-form-urlencoded',
                            },
                            body: \`person1_id=\${person1}&person2_id=\${person2}&reason=\${encodeURIComponent(reason)}\`
                        });

                        const result = await response.json();

                        if (result.success) {
                            alert(\`✅ Люди успешно объединены!\n\${result.message}\`);
                            closeMergeModal();
                            loadDuplicates();
                            loadStats();
                        } else {
                            alert(\`❌ Ошибка объединения: \${result.message}\`);
                        }

                    } catch (error) {
                        console.error('Ошибка объединения:', error);
                        alert('❌ Ошибка при объединении людей');
                    }
                }
            }

            // Обновление значения порога
            document.getElementById('threshold').addEventListener('input', function() {
                document.getElementById('thresholdValue').textContent = this.value;
            });

            // Инициализация при загрузке
            document.addEventListener('DOMContentLoaded', function() {
                loadStats();
                loadDuplicates();

                // Закрытие модального окна по клику вне его
                window.addEventListener('click', function(event) {
                    const modal = document.getElementById('mergeModal');
                    if (event.target === modal) {
                        closeMergeModal();
                    }
                });
            });
        </script>
    </body>
    </html>
    """)