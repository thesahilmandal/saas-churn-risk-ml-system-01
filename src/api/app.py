"""
FastAPI application for Customer Churn batch inference (CSV-based).

Features:
- Upload CSV for batch prediction
- Return preview of processed results
- Allow full CSV download
"""

import sys
import io
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse

from src.logging import logging
from src.exception import CustomerChurnException
from src.pipeline.prediction_pipeline import CustomerChurnPredictor

# ============================================================
# App Initialization
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Batch CSV inference service with preview and download",
    version="1.2.0",
)

predictor: CustomerChurnPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    """
    Load model artifacts once at application startup.
    """
    global predictor
    try:
        logging.info("[API STARTUP] Loading churn prediction pipeline")
        predictor = CustomerChurnPredictor()
        logging.info("[API STARTUP] Model loaded successfully")
    except Exception as e:
        logging.exception("[API STARTUP] Model loading failed")
        raise CustomerChurnException(e, sys)


# ============================================================
# Health Check
# ============================================================

@app.get("/health", tags=["system"])
def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# CSV Batch Prediction with Preview + Download
# ============================================================

@app.post(
    "/predict/batch-csv",
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
)
async def batch_predict_csv(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Perform batch churn prediction using uploaded CSV file.

    Returns:
    - Preview of processed results (first N rows)
    - Summary metrics
    - Downloadable CSV file
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported",
        )

    try:
        logging.info(
            "[CSV PREDICTION] File received | "
            f"filename={file.filename}"
        )

        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))

        if input_df.empty:
            raise ValueError("Uploaded CSV file is empty")

        logging.info(
            "[CSV PREDICTION] CSV loaded | "
            f"rows={len(input_df)}, columns={len(input_df.columns)}"
        )

        # -------------------------------
        # Run prediction pipeline
        # -------------------------------
        predictions_df = predictor.predict(input_df)

        churn_rate = round(
            predictions_df["churn_prediction"].mean(), 4
        )

        # -------------------------------
        # Preview (safe limit)
        # -------------------------------
        PREVIEW_ROWS = 20
        preview_df = predictions_df.head(PREVIEW_ROWS)

        # -------------------------------
        # Prepare CSV for download
        # -------------------------------
        output_buffer = io.StringIO()
        predictions_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        output_filename = (
            f"churn_predictions_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        )

        logging.info(
            "[CSV PREDICTION] Completed | "
            f"rows={len(predictions_df)}, churn_rate={churn_rate}"
        )

        return {
            "summary": {
                "total_records": len(predictions_df),
                "preview_rows": len(preview_df),
                "churn_rate": churn_rate,
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
            "preview": preview_df.to_dict(orient="records"),
            "download": {
                "filename": output_filename,
                "content_type": "text/csv",
                "data": output_buffer.getvalue(),
            },
        }

    except CustomerChurnException as e:
        logging.exception("[CSV PREDICTION] Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    except Exception as e:
        logging.exception("[CSV PREDICTION] Unexpected failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
