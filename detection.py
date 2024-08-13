import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
from typing import Tuple, List

def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def draw_detections(img: np.ndarray, boxes, model: YOLO) -> Tuple[np.ndarray, int, List[str]]:
    count = 0
    colors = []
    for box in boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = box.conf.item()
        bbox = box.xyxy[0].tolist()

        if class_name in ['cow', 'bull']:
            count += 1
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            roi = img[y1:y2, x1:x2]
            dominant_color = get_dominant_color(roi)
            colors.append(dominant_color)
    
    return img, count, colors

def get_dominant_color(image: np.ndarray) -> str:
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 5
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1), 
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    counts = Counter(labels.flatten())
    dominant_color = palette[np.argmax(counts)]
    
    return f'RGB({int(dominant_color[0])}, {int(dominant_color[1])}, {int(dominant_color[2])})'

def resize_image(image: np.ndarray, width: int) -> np.ndarray:
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def detect_objects(image_path: str, output_folder: str, model: YOLO, conf_thres: float = 0.25, imgsz: int = 640) -> Tuple[str, int, List[str]]:
    img, img_rgb = load_image(image_path)
    results = model.predict(img_rgb, conf=conf_thres, imgsz=imgsz)
    boxes = results[0].boxes
    
    img_with_detections, count, colors = draw_detections(img, boxes, model)
    img_with_detections = resize_image(img_with_detections, width=800)
    
    output_path = os.path.join(output_folder, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img_with_detections)
    return os.path.basename(output_path), count, colors
