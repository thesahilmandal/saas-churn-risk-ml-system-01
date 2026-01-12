from app.services.predictor import PredictionService
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

_prediction_service = None

def get_prediction_service() -> PredictionService:
    if _prediction_service is None:
        raise RuntimeError("Prediction service not initialized")
    return _prediction_service


def init_prediction_service(
        transformation_artifact: DataTransformationArtifact,
        trainer_artifact: ModelTrainerArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
):
    global _prediction_service
    _prediction_service = PredictionService(
        transformation_artifact,
        trainer_artifact,
        evaluation_artifact
    )
    