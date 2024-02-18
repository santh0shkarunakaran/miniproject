from io import BytesIO
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
app = FastAPI()
model = tf.keras.models.load_model(f"C:/Users/Deva/Downloads/version_2/models/1")
CLASS_NAMES = ["Monkeypox","Others"]
def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert('RGB')
    img = img.resize((224, 224))
    image = np.array(img)
    return image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = np.array(model.predict(img_batch))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8009)