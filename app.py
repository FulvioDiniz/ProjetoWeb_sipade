from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from detection import detect_objects
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO("yolov8m.pt")

# Verifique se as pastas de upload e processed existem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

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

       
        processed_image_filename = detect_objects(file_path, app.config['PROCESSED_FOLDER'], model)
        
        return redirect(url_for('uploaded_file', filename=processed_image_filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('index.html', filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
