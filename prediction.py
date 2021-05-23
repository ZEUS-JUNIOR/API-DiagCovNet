import cv2
from fastapi import FastAPI, UploadFile, File
import uvicorn
from fastapi.responses import FileResponse
from server import *
from io import BytesIO
import os
from pathlib import Path

app = FastAPI(title="API Intelligence for Medecin")


@app.get("/")
async def hello_world():
    return FileResponse("1.png")


@app.put('/api/predict_detection')
async def predict_image_detection(file: bytes = File(...)):
    #path = os.getcwd() + '\\' + file.filename
    # lire le fichier venant de l'utilisateur
    #image = read_image_all(path)
    image = read_image(file)
    # faire du preprocessing
    image = preprocess(image)
    # lire le model
    # faire des prediction
    result = predict(image)
    return str(result)


@app.put('/api/predict_segmentation/{file_path:path}')
async def predict_image_segmentation(file_path: str):
    result = predict_segmentation(file_path)
    return FileResponse("1.png")

@app.post("/upload")
async def root(file: UploadFile = File(...)):

    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
