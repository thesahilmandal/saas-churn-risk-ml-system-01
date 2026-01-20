"""
Customer Churn Training Pipeline Orchestrator.

Responsibilities:
- Coordinate execution of all ML pipeline stages
- Enforce strict execution order and artifact dependencies
- Manage run-level identity and logging
- Sync generated artifacts to cloud storage
- Serve as the single entry point for training workflows

NOTE:
This module intentionally contains NO business logic.
"""

import sys
import uuid
from typing import Optional

from src.exception import CustomerChurnException
from src.logging import logging

from src.components.etl import CustomerChurnETL
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import (
    TrainingPipelineConfig,
    ETLconfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)

from src.entity.artifact_entity import (
    ETLArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

from src.cloud.s3_syncer import S3Sync
from src.constants.training_pipeline import TRAINING_BUCKET_NAME


class TrainingPipeline:
    """
    Orchestrates the end-to-end Customer Churn ML training pipeline.

    Execution Order:
        1. ETL
        2. Data Ingestion
        3. Data Validation
        4. Data Transformation
        5. Model Training
        6. Model Evaluation
        7. Artifact Sync (Cloud)
    """

    def __init__(self) -> None:
        """
        Initialize pipeline configuration, artifacts, and run identity.
        """
        try:
            self.pipeline_config = TrainingPipelineConfig()

            # Strong, globally unique run identity
            self.run_id: str = f"run_{uuid.uuid4().hex}"

            # Artifacts (populated sequentially)
            self.etl_artifact: Optional[ETLArtifact] = None
            self.ingestion_artifact: Optional[DataIngestionArtifact] = None
            self.validation_artifact: Optional[DataValidationArtifact] = None
            self.transformation_artifact: Optional[
                DataTransformationArtifact
            ] = None
            self.trainer_artifact: Optional[ModelTrainerArtifact] = None
            self.evaluation_artifact: Optional[ModelEvaluationArtifact] = None

            self.s3_sync = S3Sync()

            logging.info(
                "[PIPELINE INIT] Initialized | "
                f"run_id={self.run_id}, "
                f"timestamp={self.pipeline_config.timestamp}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Pipeline Stages
    # ============================================================
    def _run_etl(self) -> None:
        logging.info("[PIPELINE] ETL stage started")

        etl = CustomerChurnETL(ETLconfig(self.pipeline_config))
        self.etl_artifact = etl.initiate_etl()

        logging.info("[PIPELINE] ETL stage completed")

    def _run_data_ingestion(self) -> None:
        logging.info("[PIPELINE] Data ingestion stage started")

        ingestion = DataIngestion(
            DataIngestionConfig(self.pipeline_config)
        )
        self.ingestion_artifact = ingestion.initiate_data_ingestion()

        logging.info("[PIPELINE] Data ingestion stage completed")

    def _run_data_validation(self) -> None:
        logging.info("[PIPELINE] Data validation stage started")

        validation = DataValidation(
            DataValidationConfig(self.pipeline_config),
            self.ingestion_artifact,
        )
        self.validation_artifact = validation.initiate_data_validation()

        logging.info("[PIPELINE] Data validation stage completed")

    def _run_data_transformation(self) -> None:
        logging.info("[PIPELINE] Data transformation stage started")

        transformation = DataTransformation(
            DataTransformationConfig(self.pipeline_config),
            self.ingestion_artifact,
            self.validation_artifact,
        )
        self.transformation_artifact = (
            transformation.initiate_data_transformation()
        )

        logging.info("[PIPELINE] Data transformation stage completed")

    def _run_model_training(self) -> None:
        logging.info("[PIPELINE] Model training stage started")

        trainer = ModelTrainer(
            ModelTrainerConfig(self.pipeline_config),
            self.ingestion_artifact,
            self.transformation_artifact,
        )
        self.trainer_artifact = trainer.initiate_model_training()

        logging.info("[PIPELINE] Model training stage completed")

    def _run_model_evaluation(self) -> None:
        logging.info("[PIPELINE] Model evaluation stage started")

        evaluation = ModelEvaluation(
            ModelEvaluationConfig(self.pipeline_config),
            self.trainer_artifact,
            self.ingestion_artifact,
            self.transformation_artifact,
        )
        self.evaluation_artifact = evaluation.initiate_model_evaluation()

        logging.info("[PIPELINE] Model evaluation stage completed")

    # ============================================================
    # Cloud Sync (Post-success Only)
    # ============================================================
    def _sync_to_s3(self, local_dir: str, s3_prefix: str) -> None:
        """
        Sync a local directory to S3 with run-level versioning.
        """
        s3_path = (
            f"s3://{TRAINING_BUCKET_NAME}/"
            f"{s3_prefix}/{self.pipeline_config.timestamp}/{self.run_id}"
        )

        logging.info(
            "[PIPELINE S3 SYNC] Started | "
            f"local={local_dir}, destination={s3_path}"
        )

        self.s3_sync.sync_folder_to_s3(
            folder=local_dir,
            aws_bucket_url=s3_path,
        )

        logging.info(
            "[PIPELINE S3 SYNC] Completed | "
            f"destination={s3_path}"
        )

    def _sync_artifacts(self) -> None:
        """
        Sync artifacts and final model to S3.

        This step is executed ONLY after successful evaluation.
        """
        if not self.evaluation_artifact:
            raise RuntimeError(
                "Evaluation artifact missing. "
                "Aborting S3 sync to prevent partial uploads."
            )

        self._sync_to_s3(
            local_dir=self.pipeline_config.artifact_dir,
            s3_prefix="artifacts",
        )

        self._sync_to_s3(
            local_dir="final_model",
            s3_prefix="final_model",
        )

    # ============================================================
    # Entry Point
    # ============================================================
    def run(self) -> None:
        """
        Execute the full training pipeline sequentially.
        """
        try:
            logging.info(
                "[PIPELINE] Execution started | "
                f"run_id={self.run_id}"
            )

            self._run_etl()
            self._run_data_ingestion()
            self._run_data_validation()
            self._run_data_transformation()
            self._run_model_training()
            self._run_model_evaluation()

            # self._sync_artifacts()

            logging.info(
                "[PIPELINE] Execution completed successfully | "
                f"run_id={self.run_id}"
            )

        except Exception as e:
            logging.exception(
                "[PIPELINE] Execution failed | "
                f"run_id={self.run_id}"
            )
            raise CustomerChurnException(e, sys)


# ============================================================
# Script Entry Point
# ============================================================
if __name__ == "__main__":
    TrainingPipeline().run()
