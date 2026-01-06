import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    write_json_file,
)


class ModelEvaluation:
    """
    Model Evaluation Pipeline.

    Responsibilities:
    - Evaluate trained candidate models
    - Apply eligibility (gating) rules
    - Perform threshold optimization
    - Select the best model for promotion
    - Generate auditable decision artifacts
    """

    def __init__(
        self,
        evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            logging.info("Initializing ModelEvaluation pipeline")

            self.config = evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.transformation_artifact = data_transformation_artifact

            os.makedirs(self.config.model_evaluation_dir, exist_ok=True)

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Data Loading
    # ============================================================

    def _load_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X_val = load_numpy_array_data(
                self.transformation_artifact.x_val_file_path
            )
            y_val = load_numpy_array_data(
                self.transformation_artifact.y_val_file_path
            )

            return X_val, y_val

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def _load_candidate_models(self) -> Dict[str, object]:
        try:
            models = {}

            for file_name in os.listdir(
                self.model_trainer_artifact.trained_models_dir
            ):
                if file_name.endswith(".pkl"):
                    model_name = file_name.replace(".pkl", "")
                    model_path = os.path.join(
                        self.model_trainer_artifact.trained_models_dir,
                        file_name,
                    )
                    models[model_name] = load_object(model_path)

            if not models:
                raise ValueError("No trained models found for evaluation")

            return models

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Metrics
    # ============================================================

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    # ============================================================
    # Threshold Sweep
    # ============================================================

    def _sweep_thresholds(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[Dict]:
        threshold_results = []

        y_proba = model.predict_proba(X_val)[:, 1]

        for threshold in self.config.thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            metrics = self._compute_metrics(
                y_true=y_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            threshold_results.append(
                {
                    "threshold": threshold,
                    "metrics": metrics,
                }
            )

        return threshold_results

    # ============================================================
    # Gating Rules
    # ============================================================

    def _apply_gates(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        if metrics["roc_auc"] < self.config.min_roc_auc:
            return False, "failed_roc_auc_gate"

        if metrics["precision"] < self.config.min_precision:
            return False, "failed_precision_gate"

        if metrics["recall"] < self.config.min_recall:
            return False, "failed_recall_gate"

        return True, "passed"

    # ============================================================
    # Threshold Selection
    # ============================================================

    def _select_best_threshold(
        self, threshold_results: List[Dict]
    ) -> Dict:
        valid_candidates = []

        for result in threshold_results:
            metrics = result["metrics"]

            if metrics["precision"] >= self.config.min_precision:
                valid_candidates.append(result)

        if not valid_candidates:
            return {}

        primary_metric = self.config.primary_metric

        best_result = max(
            valid_candidates,
            key=lambda x: x["metrics"][primary_metric],
        )

        return best_result

    # ============================================================
    # Model Selection
    # ============================================================

    def _select_best_model(
        self, model_summaries: Dict[str, Dict]
    ) -> Dict:
        if not model_summaries:
            raise ValueError("No eligible models after evaluation")

        primary_metric = self.config.primary_metric

        best_model = max(
            model_summaries.items(),
            key=lambda item: item[1]["metrics"][primary_metric],
        )

        model_name, model_info = best_model

        model_info["model_name"] = model_name
        return model_info

    # ============================================================
    # Pipeline Entry
    # ============================================================

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Model Evaluation pipeline started")

            X_val, y_val = self._load_validation_data()
            models = self._load_candidate_models()

            model_results = {}
            rejected_models = {}

            for model_name, model in models.items():
                logging.info(f"Evaluating model: {model_name}")

                threshold_results = self._sweep_thresholds(
                    model, X_val, y_val
                )

                best_threshold_result = self._select_best_threshold(
                    threshold_results
                )

                if not best_threshold_result:
                    rejected_models[model_name] = "no_valid_threshold"
                    continue

                metrics = best_threshold_result["metrics"]
                is_eligible, reason = self._apply_gates(metrics)

                if not is_eligible:
                    rejected_models[model_name] = reason
                    continue

                model_results[model_name] = {
                    "threshold": best_threshold_result["threshold"],
                    "metrics": metrics,
                }

            selected_model_info = self._select_best_model(model_results)

            decision_report = {
                "selected_model": selected_model_info["model_name"],
                "operating_threshold": selected_model_info["threshold"],
                "primary_metric": self.config.primary_metric,
                "metrics": selected_model_info["metrics"],
                "rejected_models": rejected_models,
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(
                file_path=self.config.model_evaluation_report_file_path,
                content=decision_report,
            )

            artifact = ModelEvaluationArtifact(
                evaluation_report_file_path=self.config.model_evaluation_report_file_path,
                selected_model_name=selected_model_info["model_name"],
                operating_threshold=selected_model_info["threshold"],
            )

            logging.info("Model Evaluation completed successfully")
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("Model Evaluation pipeline failed")
            raise CustomerChurnException(e, sys)
