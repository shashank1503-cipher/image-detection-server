from ultralytics import YOLO
import easyocr
from urllib.parse import urlparse
import os
import requests

def get_objects_in_image(file_name):
    model = YOLO('yolov8n.pt')
    predictions = model.predict(file_name,device='cpu')
    prediction = predictions[0]
    result = []
    for box in prediction.boxes:
        class_id = prediction.names[box.cls[0].item()]
        
        conf = round(box.conf[0].item(), 2)
        result.append({
            "Object type": class_id,
            "Probability": conf
        })
    return result

def get_text_from_images(url):
    reader = easyocr.Reader(['en','hi'], gpu = False)
    textValues = reader.readtext(url)
    result = []
    
    for text in textValues:
        result.append({
            'text': text[1],
            'probability': text[2]
        })
        
    return result

def download_data_from_FTP(url):
    a = urlparse(url)
    name = os.path.basename(a.path)                     
    request_obj = requests.get(url)
    os.chdir(os.getcwd() + "/PartialLabelingCSL/pics")
    with open(name, "wb") as file:
        file.write(request_obj.content)
    file_path = os.getcwd() + "\\" + name
    return file_path, name