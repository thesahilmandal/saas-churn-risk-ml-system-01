import pandas as pd

from src.pipeline.batch_prediction import CustomerChurnBatchPrediction
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)


class PredictionService:
    def __init__(
        self,
        transformation_artifact: DataTransformationArtifact,
        trainer_artifact: ModelTrainerArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
    ):
        self.pipeline = CustomerChurnBatchPrediction(
            transformation_artifact,
            trainer_artifact,
            evaluation_artifact,
        )

        self.model_name = evaluation_artifact.selected_model_name
        self.threshold = evaluation_artifact.operating_threshold

    def predict_single(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])
        result = self.pipeline.predict(df)

        return {
            "churn_probability": float(result["churn_probability"].iloc[0]),
            "churn_prediction": int(result["churn_prediction"].iloc[0]),
            "threshold": self.threshold,
            "model_name": self.model_name,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.predict(df)
