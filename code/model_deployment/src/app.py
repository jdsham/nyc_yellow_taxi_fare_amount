from fastapi import FastAPI, HTTPException
import numpy as np
import lightgbm as lgb

app = FastAPI(title="NYC Yello Taxicab Base Fare Amount Predictor")


# Load LightGBM Model
model = lgb.Booster(model_file="lgbm.txt")

# Get
@app.get("/")
async def home():
    return {"message": "NYC Taxi Cab Fare Prediction Service"} 


# Predict
@app.post("/predict", status_code=200)
async def predict(data:dict):
    try:
        # Unpack dictionary values
        distance = float(data["trip_distance"])
        duration = float(data["trip_duration"])

        # Create polar coordinates
        r = np.sqrt(distance**2 + duration**2)
        theta = np.arccos(distance/r) * (360/(2*np.pi))

        # Make numpy array
        output = np.array((distance, duration, r, theta)).reshape(1,4)

        # Get prediction rounded to 2 decimal places
        prediction = round(model.predict(output)[0], 2)
        return {"result":prediction}
    except:
         raise HTTPException(
            status_code=404, detail="An error occurred during prediction."
        )
