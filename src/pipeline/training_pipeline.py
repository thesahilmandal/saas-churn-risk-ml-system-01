"""
Customer Churn Training Pipeline Orchestrator.

Responsibilities:
- Coordinate execution of all ML pipeline stages
- Enforce execution order and artifact dependencies
- Provide centralized logging and exception handling
- Serve as the single entry point for training workflows

This module intentionally contains NO business logic.
"""

import sys
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
    """

    def __init__(self) -> None:
        try:
            logging.info("[PIPELINE INIT] Initializing training pipeline")

            self.pipeline_config = TrainingPipelineConfig()

            self.etl_artifact: Optional[ETLArtifact] = None
            self.ingestion_artifact: Optional[DataIngestionArtifact] = None
            self.validation_artifact: Optional[DataValidationArtifact] = None
            self.transformation_artifact: Optional[
                DataTransformationArtifact
            ] = None
            self.trainer_artifact: Optional[ModelTrainerArtifact] = None
            self.evaluation_artifact: Optional[ModelEvaluationArtifact] = None

            logging.info("[PIPELINE INIT] Training pipeline initialized")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Pipeline Stages
    # ============================================================
    def run_etl(self) -> ETLArtifact:
        logging.info("[PIPELINE] ETL stage started")

        etl_config = ETLconfig(self.pipeline_config)
        etl = CustomerChurnETL(etl_config)

        artifact = etl.initiate_etl()
        self.etl_artifact = artifact

        logging.info("[PIPELINE] ETL stage completed")
        return artifact

    def run_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("[PIPELINE] Data ingestion stage started")

        ingestion_config = DataIngestionConfig(self.pipeline_config)
        ingestion = DataIngestion(ingestion_config)

        artifact = ingestion.initiate_data_ingestion()
        self.ingestion_artifact = artifact

        logging.info("[PIPELINE] Data ingestion stage completed")
        return artifact

    def run_data_validation(self) -> DataValidationArtifact:
        logging.info("[PIPELINE] Data validation stage started")

        validation_config = DataValidationConfig(self.pipeline_config)
        validation = DataValidation(
            validation_config,
            self.ingestion_artifact,
        )

        artifact = validation.initiate_data_validation()
        self.validation_artifact = artifact

        logging.info("[PIPELINE] Data validation stage completed")
        return artifact

    def run_data_transformation(self) -> DataTransformationArtifact:
        logging.info("[PIPELINE] Data transformation stage started")

        transformation_config = DataTransformationConfig(self.pipeline_config)
        transformation = DataTransformation(
            transformation_config,
            self.ingestion_artifact,
            self.validation_artifact,
        )

        artifact = transformation.initiate_data_transformation()
        self.transformation_artifact = artifact

        logging.info("[PIPELINE] Data transformation stage completed")
        return artifact

    def run_model_training(self) -> ModelTrainerArtifact:
        logging.info("[PIPELINE] Model training stage started")

        trainer_config = ModelTrainerConfig(self.pipeline_config)
        trainer = ModelTrainer(
            trainer_config,
            self.ingestion_artifact,
            self.transformation_artifact,
        )

        artifact = trainer.initiate_model_training()
        self.trainer_artifact = artifact

        logging.info("[PIPELINE] Model training stage completed")
        return artifact

    def run_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info("[PIPELINE] Model evaluation stage started")

        evaluation_config = ModelEvaluationConfig(self.pipeline_config)
        evaluation = ModelEvaluation(
            evaluation_config,
            self.trainer_artifact,
            self.ingestion_artifact,
            self.transformation_artifact,
        )

        artifact = evaluation.initiate_model_evaluation()
        self.evaluation_artifact = artifact

        logging.info("[PIPELINE] Model evaluation stage completed")
        return artifact

    # ============================================================
    # Entry Point
    # ============================================================
    def run(self) -> None:
        """
        Execute the full training pipeline sequentially.
        """
        try:
            logging.info("[PIPELINE] Training pipeline execution started")

            self.run_etl()
            self.run_data_ingestion()
            self.run_data_validation()
            self.run_data_transformation()
            self.run_model_training()
            self.run_model_evaluation()

            logging.info("[PIPELINE] Training pipeline completed successfully")

        except Exception as e:
            logging.exception("[PIPELINE] Training pipeline failed")
            raise CustomerChurnException(e, sys)


# ============================================================
# Script Entry Point
# ============================================================
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
