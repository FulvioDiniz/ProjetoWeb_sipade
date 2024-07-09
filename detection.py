import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple

def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega uma imagem do caminho especificado."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def draw_detections(img: np.ndarray, boxes, model: YOLO) -> np.ndarray:
    """Desenha caixas delimitadoras na imagem para os objetos detectados."""
    for box in boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = box.conf.item()
        bbox = box.xyxy[0].tolist()  # Converter tensor para lista

        if class_name in ['cow', 'bull']:  # Filtrar para detectar vacas e bois
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def resize_image(image: np.ndarray, width: int) -> np.ndarray:
    """Redimensiona a imagem mantendo a proporção."""
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def detect_objects(image_path: str, output_folder: str, model: YOLO, conf_thres: float = 0.25, imgsz: int = 640) -> str:
    """Detecta objetos em uma imagem e salva a imagem com detecções desenhadas."""
    img, img_rgb = load_image(image_path)
    
    # Realizar a detecção usando o modelo YOLOv8 com parâmetros ajustados
    results = model.predict(img_rgb, conf=conf_thres, imgsz=imgsz)
    boxes = results[0].boxes
    
    # Desenhar caixas delimitadoras na imagem original
    img_with_detections = draw_detections(img, boxes, model)
    
    # Redimensionar a imagem
    img_with_detections = resize_image(img_with_detections, width=800)
    
    # Salvar a imagem com as detecções
    output_path = os.path.join(output_folder, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img_with_detections)
    return os.path.basename(output_path)
