from fastapi import FastAPI, UploadFile, File
import uvicorn
import onnxruntime as ort
import numpy as np
import cv2
import io

app = FastAPI()

# Carregar o modelo RVM
session = ort.InferenceSession("rvm_mobilenetv3_fp32.onnx")

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    contents = await file.read()
    np_video = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_video, cv2.IMREAD_COLOR)

    # Processamento do RVM (ainda precisa ser implementado)

    return {"message": "Processado com sucesso"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
