from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesInput(BaseModel):
    data: List[float]

app=FastAPI()

# Load the saved model
model = load_model('my_model.h5')

# Create the scaler object
scaler = MinMaxScaler()

@app.post("/forecast")
async def forecast_time_series(input_data: TimeSeriesInput):
    # Scale the input data
    input_array = np.array(input_data.data).reshape(1, 10, 1)
    scaled_input = scaler.fit_transform(input_array.reshape(-1, 1)).reshape(1, 10, 1)
    
    # Predict using the model
    forecast = model.predict(scaled_input)
    
    # Inverse transform the predicted values to get actual values
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).reshape(-1)
    
    # Return the actual values as forecast
    return {"forecast": forecast.tolist()}


@app.get('/')
async def scoring_endpoint():
    return {'hello':'world'}
