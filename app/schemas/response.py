from pydantic import BaseModel


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float
    model_name: str
    