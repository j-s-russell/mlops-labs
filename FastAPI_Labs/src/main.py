from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class HousingResponse(BaseModel):
    response: float


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=HousingResponse)
async def predict_housing(housing_features: HousingData):
    try:
        features = [[housing_features.MedInc, housing_features.HouseAge,
                    housing_features.AveRooms, housing_features.AveBedrms,
                    housing_features.Population, housing_features.AveOccup,
                    housing_features.Latitude, housing_features.Longitude]]

        prediction = predict_data(features)
        return HousingResponse(response=float(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
