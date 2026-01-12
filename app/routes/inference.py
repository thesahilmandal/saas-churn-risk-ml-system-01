from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import pandas as pd

from app.schemas.request import ChurnRequest
from app.schemas.response import ChurnResponse
from app.dependencies import get_prediction_service
from app.services.predictor import PredictionService

router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post("/predict", response_model=ChurnResponse)
def predict(
    request: ChurnRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    try:
        return service.predict_single(request.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Inference failed"
        )


@router.post("/predict-batch", response_model=list[dict])
def predict_batch(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
):
    if file.content_type != "text/csv":
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    try:
        df = pd.read_csv(file.file)
        predictions = service.predict_batch(df)
        return predictions.to_dict(orient="records")
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Batch inference failed"
        )
