from time import time
from fastapi import FastAPI, __version__, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np
import pandas as pd
import time
import os

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

@app.post('/predict/knn')
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
    return {'message': 'Predict success', 'prediction': prediction.tolist()}

@app.post('/predict/knn/csv')
async def predict_csv(request: Request):
    # Get the uploaded file from the request
    file = await request.form()
    csv_file = file['dataset'].file

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Preprocess the categorical features
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['smoking_history'] = df['smoking_history'].map({'never': 0, 'No Info': 1, 'current': 2, 'former':3, 'ever': 4, 'not current': 5})

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

    # Add the prediction column to the DataFrame
    df['stroke_prediction'] = prediction

    # Generate a unique filename based on timestamp
    timestamp = str(int(time.time()))
    filename = f"prediction_{timestamp}.csv"

    # Save the DataFrame as a CSV file
    current_dir = os.getcwd()
    save_path = f"{current_dir}/static/predict/{filename}"
    df.to_csv(save_path, index=False)

    # Return the URL of the saved file
    return {'message': 'Predict success', 'csv_url': f"/static/predict/{filename}"}
