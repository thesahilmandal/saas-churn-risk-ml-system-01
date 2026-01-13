import pandas as pd
from io import BytesIO

from src.pipeline.prediction_pipeline import CustomerChurnPredictor


class PredictionService:
    """
    Shared prediction logic for online and batch inference.
    """

    def __init__(self, predictor: CustomerChurnPredictor):
        self.predictor = predictor

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions on a DataFrame and return enriched DataFrame.
        """
        return self.predictor.predict(df)

    @staticmethod
    def to_csv_buffer(df: pd.DataFrame) -> BytesIO:
        """
        Convert DataFrame to in-memory CSV buffer.
        """
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
