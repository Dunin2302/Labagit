# Создание REST API с помощью FastAPI для предсказания цен ноутбуков на основе обученной модели
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

# Загружаем обученную модель из Google Диска
model_path = "/content/drive/MyDrive/laptop_price_model.pkl"
model = joblib.load(model_path)

# Роут для POST-запросов, принимающих CSV-файлы с признаками ноутбука
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
# Читаем входящий файл в байтовый поток
content = await file.read()

# Парсим полученные данные в DataFrame
df = pd.read_csv(BytesIO(content))

# Прогоняем данные через модель и получаем прогнозы
predictions = model.predict(df)

# Возвращаем спрогнозированные цены в виде списка
return {"predictions": predictions.tolist()}
