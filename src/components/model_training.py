"""
Model Training Pipeline.

Responsibilities:
- Train candidate models using pre-built preprocessors
- Perform hyperparameter optimization
- Persist trained model pipelines
- Generate and persist training metadata

NOTE:
This pipeline intentionally excludes model evaluation.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.constants.training_pipeline import TARGET_COLUMN
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import save_object, load_object, write_json_file


class ModelTrainer:
    """
    Handles model training and hyperparameter optimization.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            logging.info("[MODEL TRAINER INIT] Initializing")

            self.config = model_trainer_config
            self.ingestion_artifact = ingestion_artifact
            self.transformation_artifact = transformation_artifact

            os.makedirs(self.config.trained_models_dir, exist_ok=True)

            logging.info("[MODEL TRAINER INIT] Initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def _build_pipeline(preprocessor, model) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    # ============================================================
    # Metadata
    # ============================================================
    def _generate_training_metadata(
        self,
        models_summary: Dict[str, Dict],
        started_at_utc: str,
        completed_at_utc: str,
    ) -> None:
        """
        Generate and persist model training metadata.

        Args:
            models_summary (Dict): Training details for each model
            started_at_utc (str): Pipeline start timestamp
            completed_at_utc (str): Pipeline end timestamp
        """
        metadata = {
            "stage": "model_training",
            "training_strategy": {
                "search_type": "RandomizedSearchCV",
                "cv_strategy": "StratifiedKFold",
                "cv_folds": 5,
                "n_iter": 20,
                "scoring_metric": "f1",
                "random_state": 42,
            },
            "input_artifacts": {
                "train_data_path": self.ingestion_artifact.train_file_path,
                "linear_preprocessor_path": (
                    self.transformation_artifact.linear_preprocessor_file_path
                ),
                "tree_preprocessor_path": (
                    self.transformation_artifact.tree_preprocessor_file_path
                ),
            },
            "models_trained": models_summary,
            "output_artifacts": {
                "trained_models_directory": self.config.trained_models_dir
            },
            "execution_info": {
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "pipeline_version": "v1.0.0",
            },
        }

        write_json_file(
            file_path=self.config.metadata_file_path,
            content=metadata,
        )

        logging.info(
            "[MODEL TRAINER METADATA] Metadata written | "
            f"path={self.config.metadata_file_path}"
        )

    # ============================================================
    # Training Logic
    # ============================================================
    def _train_model(
        self,
        model_name: str,
        model,
        param_grid: Dict,
        preprocessor,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict:
        """
        Train a single model using RandomizedSearchCV.
        """
        logging.info(f"[MODEL TRAINING] Started | model={model_name}")

        pipeline = self._build_pipeline(preprocessor, model)

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                f"model__{k}": v for k, v in param_grid.items()
            },
            n_iter=1,
            scoring="recall",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=2
        )

        search.fit(X, y)

        trained_pipeline = search.best_estimator_

        model_path = os.path.join(
            self.config.trained_models_dir,
            f"{model_name}.pkl",
        )

        save_object(model_path, trained_pipeline)

        logging.info(
            f"[MODEL TRAINING] Completed | model={model_name} | "
            f"saved_at={model_path}"
        )

        return {
            "model_class": model.__class__.__name__,
            "artifact_path": model_path,
            "best_hyperparameters": search.best_params_,
        }

    # ============================================================
    # Pipeline Entry Point
    # ============================================================
    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Execute model training pipeline.
        """
        try:
            logging.info("[MODEL TRAINING PIPELINE] Started")
            started_at_utc = datetime.now(timezone.utc).isoformat()

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in training data"
                )

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            models_summary: Dict[str, Dict] = {}

            for model_name, model in self.config.models.items():
                param_grid = self.config.models_hyperparameter.get(
                    model_name, {}
                )

                preprocessor_path = (
                    self.transformation_artifact.linear_preprocessor_file_path
                    if model_name == "logistic_regression"
                    else self.transformation_artifact.tree_preprocessor_file_path
                )

                preprocessor = load_object(preprocessor_path)

                model_metadata = self._train_model(
                    model_name=model_name,
                    model=model,
                    param_grid=param_grid,
                    preprocessor=preprocessor,
                    X=X_train,
                    y=y_train,
                )

                models_summary[model_name] = model_metadata

            completed_at_utc = datetime.now(timezone.utc).isoformat()

            self._generate_training_metadata(
                models_summary=models_summary,
                started_at_utc=started_at_utc,
                completed_at_utc=completed_at_utc,
            )

            artifact = ModelTrainerArtifact(
                trained_models_dir=self.config.trained_models_dir,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("[MODEL TRAINING PIPELINE] Completed successfully")
            logging.info(f"ModelTrainingArtifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("[MODEL TRAINING PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
