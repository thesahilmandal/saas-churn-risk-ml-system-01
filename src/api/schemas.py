from typing import List, Dict, Any
from pydantic import BaseModel, Field


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch churn prediction.
    Each item represents one customer record.
    """
    records: List[Dict[str, Any]] = Field(
        ..., description="List of customer feature dictionaries"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch churn prediction.
    """
    total_records: int
    churn_rate: float
    predictions: List[Dict[str, Any]]
