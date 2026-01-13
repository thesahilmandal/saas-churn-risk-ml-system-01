"""
Prediction API routes.

Responsibilities:
- Expose online (real-time) prediction endpoint
- Expose batch prediction endpoint via CSV upload
- Validate inputs at the API boundary
- Delegate inference to the prediction service
"""

from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.responses import StreamingResponse

import pandas as pd

from src.api.schemas.request import ChurnRequest
from src.api.schemas.response import ChurnResponse
from src.service.prediction_service import PredictionService
from src.logging import logging
from src.exception import CustomerChurnException

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ============================================================
# Online Inference
# ============================================================
@router.post(
    "/",
    response_model=ChurnResponse,
    summary="Online churn prediction",
    description=(
        "Generate churn prediction for a single customer record. "
        "Designed for low-latency, real-time inference."
    ),
)
def predict_online(
    request: Request,
    payload: ChurnRequest,
):
    """
    Perform online (real-time) churn prediction.

    Args:
        request (Request): FastAPI request object
        payload (ChurnRequest): Customer feature payload

    Returns:
        ChurnResponse: Prediction result with probability
    """
    logging.info(
        "[API REQUEST] Online prediction requested | "
        f"path={request.url.path}"
    )

    try:
        service = PredictionService(request.app.state.predictor)

        df = pd.DataFrame([payload.model_dump()])
        result = service.predict_dataframe(df).iloc[0]

        response = {
            "prediction": int(result["churn_prediction"]),
            "prediction_probability": float(
                result["churn_probability"]
            ),
        }

        logging.info(
            "[API RESPONSE] Online prediction completed successfully"
        )

        return response

    except CustomerChurnException:
        logging.error(
            "[API ERROR] CustomerChurnException during online prediction",
            exc_info=True,
        )
        # Let centralized handler deal with response
        raise

    except Exception as e:
        logging.error(
            "[API ERROR] Unexpected error during online prediction",
            exc_info=True,
        )
        raise CustomerChurnException(e, sys)


# ============================================================
# Batch Inference
# ============================================================
@router.post(
    "/batch",
    summary="Batch churn prediction",
    description=(
        "Generate churn predictions for multiple records via CSV upload. "
        "Returns a downloadable CSV with prediction results."
    ),
)
async def predict_batch(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Perform batch churn prediction using a CSV file.

    Args:
        request (Request): FastAPI request object
        file (UploadFile): Uploaded CSV file

    Returns:
        StreamingResponse: CSV file with predictions appended
    """
    logging.info(
        "[API REQUEST] Batch prediction requested | "
        f"path={request.url.path}, filename={file.filename}"
    )

    # -------------------------
    # Client-side validation
    # -------------------------
    if not file.filename or not file.filename.endswith(".csv"):
        logging.warning(
            "[API VALIDATION] Invalid file type for batch prediction"
        )
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )

    try:
        df = pd.read_csv(file.file)
    except Exception:
        logging.warning(
            "[API VALIDATION] Failed to parse uploaded CSV file"
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file",
        )

    try:
        service = PredictionService(request.app.state.predictor)

        output_df = service.predict_dataframe(df)
        csv_buffer = service.to_csv_buffer(output_df)

        logging.info(
            "[API RESPONSE] Batch prediction completed successfully | "
            f"rows_processed={len(output_df)}"
        )

        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=predictions_{file.filename}"
                )
            },
        )

    except CustomerChurnException:
        logging.error(
            "[API ERROR] CustomerChurnException during batch prediction",
            exc_info=True,
        )
        raise

    except Exception as e:
        logging.error(
            "[API ERROR] Unexpected error during batch prediction",
            exc_info=True,
        )
        raise CustomerChurnException(e, sys)
