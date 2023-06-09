from time import time
from fastapi import FastAPI, __version__, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/model", StaticFiles(directory="../model"), name="model")

# Load the trained model
model_knn = joblib.load('../model/knn_diabetes_prediction.pkl')

# Load the scaler and encoder used for preprocessing
scaler = joblib.load('../model/scaler.pkl')


html = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Diabetes Predictions</title>
        <link rel="icon" href="/static/favicon.ico" type="image/x-icon" />
    </head>
    <body>
        <div class="bg-gray-200 p-4 rounded-lg shadow-lg">
            <h1>Diabetes Predictions</h1>
            <ul>
                <li><a href="/docs">/docs</a></li>
                <li><a href="/redoc">/redoc</a></li>
            </ul>
            <p>Powered by <a href="https://vercel.com" target="_blank">Vercel</a></p>
        </div>
    </body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(html)

@app.get('/ping')
async def hello():
    return {'res': 'pong', 'version': __version__, "time": time()}

@app.post('/predict')
async def predict(request: Request):
    # Get the input data from the request
    json = await request.json()
    df = pd.DataFrame(json['data'])

    # Preprocess the categorical features
    df['gender']            = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2 })
    df['smoking_history']   = df['smoking_history'].map({'never': 0, 'No Info': 1, 'current': 2, 'former':3, 'ever': 4, 'not current': 5 })

    # Preprocess the numerical features
    desired_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender",
        "smoking_history"
    ]

    df_reindex = df.reindex(columns=desired_columns)
    preprocessed_data = scaler.transform(df_reindex)

    # Make the prediction
    prediction = model_knn.predict(preprocessed_data)

    # Return the prediction
    return {'prediction': prediction.tolist()}
