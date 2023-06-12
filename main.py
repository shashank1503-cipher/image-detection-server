from fastapi import FastAPI, Request
from os import chdir, getcwd,remove
from utils import get_objects_in_image,download_data_from_FTP,get_text_from_images
import subprocess
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    routes = {
        "/": "This page",
        "/getdata": "Get Image Data from the server",
    }
    return routes

@app.post("/getdata")
async def getdata(req: Request):
    data = await req.json()
    print(data)
    url = data['url']
    try:
        file_path, file_name = download_data_from_FTP(url)
    except Exception as e:
        return {"meta": {"error": [str(e)]}, "data": {}}
    result = {
        "meta": {
            "file_name": file_name,
            "file_path": file_path,
            "error": []
        },
        "data" :{
            "objects": [],
            "labels": [],
            "texts": []
        }
    }

    chdir(getcwd() + "/..")

    try:
        texts = get_text_from_images(file_path)
        result['data']['texts'] = texts
    except Exception as e:
        result['meta']['error'].append({"task":"text detection", "error": str(e)})
    try:
        labels = subprocess.check_output(f"python infer.py --dataset_type=OpenImages --model_name=tresnet_m --model_path=./models_local/mtresnet_opim_86.72.pth --pic_path=./pics/{file_name} --input_size=224", shell=True)
        labels = labels.decode()
        labels = labels.strip()
        labels = labels.split(",")
        result['data']['labels'] = labels
    except Exception as e:
        result['meta']['error'].append({"task":"labeling", "error": str(e)})
    
    chdir(getcwd() + "/..")
    try:
        objects = get_objects_in_image(file_path)
        result['data']['objects'] = objects
    except Exception as e:
        result['meta']['error'].append({"task":"object detection", "error": str(e)})

    remove(file_path)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4500)
