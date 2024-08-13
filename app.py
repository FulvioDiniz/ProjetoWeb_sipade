from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from detection import detect_objects
from ultralytics import YOLO
from flask import send_from_directory
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
INFO_FILE = 'image_info.json'  # Arquivo para armazenar informações sobre as imagens
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

model = YOLO("yolov8m.pt")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def save_image_detection_info(filename, count, colors):
    """Salvar as informações de detecção de cada imagem em um arquivo JSON."""
    info = {}
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, 'r') as f:
            info = json.load(f)
    
    info[filename] = {
        'count': count,
        'colors': colors
    }

    with open(INFO_FILE, 'w') as f:
        json.dump(info, f)

def get_image_detection_info(filename):
    """Recuperar as informações de detecção para uma imagem específica."""
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, 'r') as f:
            info = json.load(f)
            return info.get(filename, {'count': 0, 'colors': []})
    return {'count': 0, 'colors': []}

def get_latest_files_info(directory, limit=6):
    """Função para obter os últimos arquivos processados com informações adicionais."""
    files_info = []
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    
    for filename in files[:limit]:
        detection_info = get_image_detection_info(filename)
        files_info.append({
            'filename': filename,
            'count': detection_info['count'],
            'colors': detection_info['colors']
        })
    
    return files_info

@app.route('/')
def index():
    # Obtenha as últimas imagens processadas com suas informações
    latest_files_info = get_latest_files_info(app.config['PROCESSED_FOLDER'])
    return render_template('index.html', latest_files_info=latest_files_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Detecte objetos na imagem e salve a imagem processada
        processed_image_filename, count, colors = detect_objects(file_path, app.config['PROCESSED_FOLDER'], model)

        # Salve as informações de detecção para a imagem processada
        save_image_detection_info(processed_image_filename, count, colors)

        return redirect(url_for('index'))

@app.route('/img/<path:filename>')
def serve_images(filename):
    return send_from_directory('templates/img', filename)

if __name__ == "__main__":
    app.run(debug=True)
