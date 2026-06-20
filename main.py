from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schema import CancerFeature, Predict
from model import predict

app = FastAPI(
    title="Cancer Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Cancer Detection API Running"}


import traceback

@app.post("/predict", response_model=Predict)
def predict_cancer(features: CancerFeature):
    try:
        diagnosis = predict(features.model_dump())
        return Predict(diagnosis=diagnosis)

    except Exception as e:
        traceback.print_exc()
        raise e