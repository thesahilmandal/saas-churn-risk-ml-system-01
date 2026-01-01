import os
import sys
from datetime import datetime, timezone
from typing import Dict

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file, load_numpy_array_data, save_object


class ModelTrainer:
    """
    Model Training Pipeline (Multi-Model Design).

    Responsibilities:
    - Train multiple candidate models
    - Evaluate each model on validation data
    - Persist all trained models
    - Generate consolidated metrics and metadata
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            logging.info("Initializing ModelTrainer pipeline (multi-model)")

            self.config = model_trainer_config
            self.transformation_artifact = data_transformation_artifact
            self.primary_metric = self.config.primary_metric

            # self.models_dir = os.path.join(
            #     self.config.model_trainer_dir, self.config.trained_model_file_path
            # )

            os.makedirs(self.config.model_trainer_dir, exist_ok=True)

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _compute_metrics(y_true, y_pred, y_proba) -> Dict:
        return {
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

    def _get_models(self) -> Dict[str, object]:
        """
        Instantiate all candidate models.
        """
        return {
            "logistic_regression": LogisticRegression(
                max_iter=10, solver="liblinear"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=5,
                max_depth=10,
                random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=5,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        }

    # ============================================================
    # Run
    # ============================================================

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Model Training pipeline started (multi-model)")

            # ---------------- Load Transformed Data ----------------
            X_train = load_numpy_array_data(self.transformation_artifact.x_train_file_path)
            y_train = load_numpy_array_data(self.transformation_artifact.y_train_file_path)

            X_val = load_numpy_array_data(self.transformation_artifact.x_val_file_path)
            y_val = load_numpy_array_data(self.transformation_artifact.y_val_file_path)

            logging.info(
                f"Loaded data | "
                f"X_train={X_train.shape}, X_val={X_val.shape}"
            )

            models = self._get_models()
            metrics_report = {}

            # ---------------- Train All Models ----------------
            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(X_train, y_train)

                y_val_pred = model.predict(X_val)
                y_val_proba = model.predict_proba(X_val)[:, 1]

                metrics = self._compute_metrics(
                    y_true=y_val,
                    y_pred=y_val_pred,
                    y_proba=y_val_proba
                )

                metrics_report[model_name] = metrics

                model_path = os.path.join(
                    self.config.trained_model_file_path, f"{model_name}.pkl"
                )

                save_object(model_path, model)

                logging.info(
                    f"{model_name} | "
                    f"{self.primary_metric}={metrics[self.primary_metric]:.4f}"
                )

            # ---------------- Persist Metrics Report ----------------
            write_json_file(
                file_path=self.config.training_metrics_file_path,
                content=metrics_report
            )

            # ---------------- Persist Metadata ----------------
            metadata = {
                "training_strategy": "multi_model",
                "primary_metric": self.primary_metric,
                "models_trained": list(models.keys()),
                "trained_at_utc": datetime.now(timezone.utc).isoformat()
            }

            write_json_file(
                file_path=self.config.model_metadata_file_path,
                content=metadata
            )

            artifact = ModelTrainerArtifact(
                trained_models_dir=self.config.trained_model_file_path,
                metrics_report_file_path=self.config.training_metrics_file_path,
                metadata_file_path=self.config.model_metadata_file_path
            )

            logging.info("Model Training completed successfully (multi-model)")
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.error("Model Training pipeline failed")
            raise CustomerChurnException(e, sys)
