"""
Application lifespan management.

Responsibilities:
- Load ML artifacts at application startup
- Initialize the prediction pipeline once
- Attach predictor to application state
- Ensure clean startup and shutdown logging

This module is critical for production stability:
the API must fail fast if required ML artifacts
are unavailable or invalid.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.pipeline.prediction_pipeline import CustomerChurnPredictor
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.logging import logging
from src.exception import CustomerChurnException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Loads the trained ML model and related artifacts at startup
    and attaches the predictor to the application state.

    If artifact loading fails, the application will not start.
    This is intentional and required for production ML systems.
    """
    logging.info("[API STARTUP] Lifespan startup initiated")

    try:
        logging.info(
            "[API STARTUP] Loading model evaluation artifacts"
        )

        evaluation_artifact = ModelEvaluationArtifact(
            report_file_path=(
                "artifacts/01_13_2026_08_07_03/"
                "06_model_evaluation/report.json"
            ),
            selected_trained_model_file_path=(
                "artifacts/01_13_2026_08_07_03/"
                "05_model_training/trained_models/"
                "gradient_boosting.pkl"
            ),
            operating_threshold=0.1,
            metadata_file_path=(
                "artifacts/01_13_2026_08_07_03/"
                "06_model_evaluation/metadata.json"
            ),
        )

        logging.info(
            "[API STARTUP] Initializing CustomerChurnPredictor"
        )

        app.state.predictor = CustomerChurnPredictor(
            evaluation_artifact=evaluation_artifact
        )

        logging.info(
            "[API STARTUP] Predictor initialized successfully | "
            f"model_path={evaluation_artifact.selected_trained_model_file_path}"
        )

        # Application is now ready to serve requests
        yield

    except CustomerChurnException:
        logging.error(
            "[API STARTUP] Failed to initialize predictor due to "
            "CustomerChurnException",
            exc_info=True,
        )
        # Re-raise to prevent application from starting
        raise

    except Exception as e:
        logging.error(
            "[API STARTUP] Unexpected error during application startup",
            exc_info=True,
        )
        # Wrap and re-raise as domain-specific exception
        raise CustomerChurnException(e, sys)

    finally:
        logging.info("[API SHUTDOWN] Lifespan shutdown completed")
