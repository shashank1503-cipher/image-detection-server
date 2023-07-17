from ultralytics import YOLO
import easyocr
from urllib.parse import urlparse
import os
import requests
import torch
from paddleocr import PaddleOCR

def get_objects_in_image(file_name):
    torch.cuda.empty_cache()
    model = YOLO('yolov8n.pt')
    predictions = model.predict(file_name)
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

def get_text_from_images(file_name):
        reader = easyocr.Reader(['hi'], gpu = False)
        textValues = reader.readtext(file_name)
        
        hiResults = []
        for text in textValues:
            if text[2] >= 0.2:
                hiResults.append({
                        'text': text[1],
                        'probability': text[2]
                })
        
        
        engResults = []
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        img_path = file_name
        result = ocr.ocr(img_path, cls=True)
        for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    if line[1][1] >= 0.5:
                        engResults.append({
                            'text': line[1][0],
                            'probability': line[1][1]
                        })
        
        results = hiResults + engResults
        # result = engResults if engAvg > hiAvg else hiResults
        # print(results)
        for result in results:
            print(result)
        return results

def download_data_from_FTP(url):
    a = urlparse(url)
    name = os.path.basename(a.path)                     
    request_obj = requests.get(url)
    os.chdir(os.getcwd() + "/PartialLabelingCSL/pics")
    with open(name, "wb") as file:
        file.write(request_obj.content)
    file_path = os.getcwd() + "\\" + name
    return file_path, name