from pydantic import BaseModel


class ChurnResponse(BaseModel):
    prediction: int
    prediction_probability: float
