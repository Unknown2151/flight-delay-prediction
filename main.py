import os
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

try:
    model_pipeline = joblib.load('artifacts/flight_delay_pipeline.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file not found. Make sure 'flight_delay_pipeline.pkl' is in the 'artifacts' folder.")

app = FastAPI(title="Flight Delay Prediction API")


def get_amadeus_access_token():
    """Gets a temporary access token from the Amadeus API."""
    token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET,
    }
    try:
        response = requests.post(token_url, data=token_data)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Amadeus authentication error: {e}")


class FlightInput(BaseModel):
    carrierCode: str  # e.g., "AA"
    flightNumber: str  # e.g., "123"
    scheduledDepartureDate: str  # e.g., "2025-08-23"


@app.post("/predict")
def predict_delay(flight_input: FlightInput):
    access_token = get_amadeus_access_token()

    flight_status_url = "https://test.api.amadeus.com/v2/schedule/flights"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "carrierCode": flight_input.carrierCode,
        "flightNumber": flight_input.flightNumber,
        "scheduledDepartureDate": flight_input.scheduledDepartureDate,
    }

    try:
        api_result = requests.get(flight_status_url, headers=headers, params=params)
        api_result.raise_for_status()
        flight_response = api_result.json()

        if not flight_response.get("data"):
            raise HTTPException(status_code=404, detail="Flight not found with Amadeus API.")

        flight_data = flight_response["data"][0]["flightSegments"][0]

        airline = flight_data["carrierCode"]
        origin_airport = flight_data["departure"]["iataCode"]
        destination_airport = flight_data["arrival"]["iataCode"]
        scheduled_departure_str = flight_data["departure"]["at"]

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Amadeus API error: {e}")

    try:
        scheduled_departure_dt = datetime.fromisoformat(scheduled_departure_str)

        features = {
            'MKT_UNIQUE_CARRIER': airline,
            'ORIGIN': origin_airport,
            'DEST': destination_airport,
            'DISTANCE': 2475.0,  # Placeholder
            'MONTH': scheduled_departure_dt.month,
            'DAY_OF_WEEK': scheduled_departure_dt.weekday(),
            'DEP_HOUR': scheduled_departure_dt.hour
        }

        features_df = pd.DataFrame([features])

        prediction_proba = model_pipeline.predict_proba(features_df)[0][1]
        prediction = model_pipeline.predict(features_df)[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction pipeline: {e}")

    return {
        "flight_details": flight_input.dict(),
        "prediction_input_features": features,
        "is_delayed_prediction": int(prediction),
        "delay_probability": f"{prediction_proba:.2%}"
    }


@app.get("/")
def read_root():
    return {"message": "Welcome to the Flight Delay Prediction API!"}