from fastapi import FastAPI
from app.routes.inference import router
from app.dependencies import init_prediction_service
from app.core.config import SERVICE_NAME

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

app = FastAPI(
    title=SERVICE_NAME,
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    # Load real artifacts here
    transformation_artifact = DataTransformationArtifact(...)
    trainer_artifact = ModelTrainerArtifact(...)
    evaluation_artifact = ModelEvaluationArtifact(...)

    init_prediction_service(
        transformation_artifact,
        trainer_artifact,
        evaluation_artifact,
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(router)
