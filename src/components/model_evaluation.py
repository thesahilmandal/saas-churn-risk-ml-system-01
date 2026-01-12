"""
Model Evaluation Pipeline.

Responsibilities:
- Load validation dataset
- Load trained candidate models
- Perform threshold sweeping and metric computation
- Apply eligibility gates
- Select best model and operating threshold
- Resolve correct preprocessor for selected model
- Generate structured decision report
- Generate audit-ready metadata

NOTE:
This pipeline is strictly evaluation-only.
No training or inference logic is included.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.constants.training_pipeline import TARGET_COLUMN
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import load_object, write_json_file


PIPELINE_VERSION = "v1.0.0"


class ModelEvaluation:
    """
    Handles model evaluation, comparison, and selection.
    """

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            logging.info("[MODEL EVALUATION INIT] Initializing")

            self.config = model_evaluation_config
            self.trainer_artifact = model_trainer_artifact
            self.ingestion_artifact = ingestion_artifact
            self.transformation_artifact = transformation_artifact

            os.makedirs(self.config.model_evaluation_dir, exist_ok=True)

            logging.info("[MODEL EVALUATION INIT] Initialized successfully")

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

    def _sweep_thresholds(
        self,
        model_pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Dict]:
        if not hasattr(model_pipeline, "predict_proba"):
            raise AttributeError("Model does not support predict_proba")

        y_proba = model_pipeline.predict_proba(X)[:, 1]
        results: List[Dict] = []

        for threshold in self.config.thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self._compute_metrics(y, y_pred, y_proba)

            results.append(
                {
                    "threshold": threshold,
                    "metrics": metrics,
                }
            )

        return results

    def _apply_gates(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        if metrics["roc_auc"] < self.config.min_roc_auc:
            return False, "failed_roc_auc_gate"

        if metrics["precision"] < self.config.min_precision:
            return False, "failed_precision_gate"

        if metrics["recall"] < self.config.min_recall:
            return False, "failed_recall_gate"

        return True, "passed"

    def _select_best_threshold(
        self, threshold_results: List[Dict]
    ) -> Dict:
        valid = [
            r for r in threshold_results
            if r["metrics"]["precision"] >= self.config.min_precision
        ]

        if not valid:
            return {}

        return max(
            valid,
            key=lambda r: r["metrics"][self.config.primary_metric],
        )

    def _resolve_preprocessor_path(self, model_name: str) -> str:
        linear_models = {"logistic_regression"}

        if model_name in linear_models:
            return self.transformation_artifact.linear_preprocessor_file_path

        return self.transformation_artifact.tree_preprocessor_file_path

    # ============================================================
    # Artifact Generators
    # ============================================================
    @staticmethod
    def _generate_decision_report(
        selected_model_name: str,
        selected_info: Dict,
        runner_up: Dict,
        rejected_models: Dict[str, str],
        optimization_metric: str,
    ) -> Dict:
        """
        Generate structured decision report with rationale.
        """
        return {
            "selected_model": selected_model_name,
            "selection_reason": (
                f"Best {optimization_metric} among eligible models "
                "after applying evaluation gates"
            ),
            "operating_threshold": selected_info["threshold"],
            "selected_model_metrics": selected_info["metrics"],
            "runner_up_model": runner_up,
            "threshold_analysis": {
                "thresholds_tested": len(selected_info["thresholds_tested"]),
                "optimization_metric": optimization_metric,
            },
            "rejected_models": rejected_models,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _generate_evaluation_metadata(
        models_evaluated: List[str],
        model_trace: Dict[str, Dict],
        validation_snapshot: Dict,
        config: ModelEvaluationConfig,
        start_time: float,
    ) -> Dict:
        """
        Generate production-grade evaluation metadata.
        """
        return {
            "stage": "model_evaluation",
            "pipeline_version": PIPELINE_VERSION,
            "evaluation_config": {
                "primary_metric": config.primary_metric,
                "thresholds": config.thresholds,
                "gates": {
                    "min_roc_auc": config.min_roc_auc,
                    "min_precision": config.min_precision,
                    "min_recall": config.min_recall,
                },
            },
            "data_snapshot": validation_snapshot,
            "models_evaluated": models_evaluated,
            "model_trace": model_trace,
            "execution_info": {
                "duration_seconds": round(
                    time.time() - start_time, 3
                ),
                "evaluated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        }

    # ============================================================
    # Pipeline Entry Point
    # ============================================================
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        start_time = time.time()

        try:
            logging.info("[MODEL EVALUATION PIPELINE] Started")

            val_df = self._read_csv(
                self.ingestion_artifact.val_file_path
            )

            if TARGET_COLUMN not in val_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in validation data"
                )

            X_val = val_df.drop(columns=[TARGET_COLUMN])
            y_val = val_df[TARGET_COLUMN]

            validation_snapshot = {
                "dataset": "validation",
                "rows": len(val_df),
                "churn_rate": round(y_val.mean(), 4),
                "stratified": True,
            }

            model_results: Dict[str, Dict] = {}
            rejected_models: Dict[str, str] = {}
            model_trace: Dict[str, Dict] = {}

            for file_name in os.listdir(
                self.trainer_artifact.trained_models_dir
            ):
                if not file_name.endswith(".pkl"):
                    continue

                model_name = file_name.replace(".pkl", "")
                model_path = os.path.join(
                    self.trainer_artifact.trained_models_dir,
                    file_name,
                )

                model_pipeline = load_object(model_path)

                try:
                    threshold_results = self._sweep_thresholds(
                        model_pipeline, X_val, y_val
                    )
                except AttributeError:
                    rejected_models[model_name] = (
                        "predict_proba_not_supported"
                    )
                    continue

                best_threshold = self._select_best_threshold(
                    threshold_results
                )

                model_trace[model_name] = {
                    "thresholds_tested": len(threshold_results),
                    "best_threshold": (
                        best_threshold.get("threshold")
                        if best_threshold else None
                    ),
                }

                if not best_threshold:
                    rejected_models[model_name] = "no_valid_threshold"
                    continue

                metrics = best_threshold["metrics"]
                eligible, reason = self._apply_gates(metrics)

                model_trace[model_name]["gate_status"] = reason

                if not eligible:
                    rejected_models[model_name] = reason
                    continue

                model_results[model_name] = {
                    "model_path": model_path,
                    "threshold": best_threshold["threshold"],
                    "metrics": metrics,
                    "thresholds_tested": threshold_results,
                }

            if not model_results:
                raise ValueError("No eligible models after evaluation")

            sorted_models = sorted(
                model_results.items(),
                key=lambda x: x[1]["metrics"][self.config.primary_metric],
                reverse=True,
            )

            selected_model_name, selected_info = sorted_models[0]
            runner_up = (
                {
                    "model_name": sorted_models[1][0],
                    "metrics": sorted_models[1][1]["metrics"],
                }
                if len(sorted_models) > 1
                else None
            )

            selected_preprocessor_path = self._resolve_preprocessor_path(
                selected_model_name
            )

            decision_report = self._generate_decision_report(
                selected_model_name=selected_model_name,
                selected_info=selected_info,
                runner_up=runner_up,
                rejected_models=rejected_models,
                optimization_metric=self.config.primary_metric,
            )

            write_json_file(
                self.config.report_file_path,
                decision_report,
            )

            metadata = self._generate_evaluation_metadata(
                models_evaluated=list(model_results.keys()),
                model_trace=model_trace,
                validation_snapshot=validation_snapshot,
                config=self.config,
                start_time=start_time,
            )

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            artifact = ModelEvaluationArtifact(
                report_file_path=self.config.report_file_path,
                selected_trained_model_file_path=selected_info["model_path"],
                selected_preprocessor_file_path=selected_preprocessor_path,
                operating_threshold=selected_info["threshold"],
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("[MODEL EVALUATION PIPELINE] Completed successfully")
            logging.info(f"ModelEvaluationArtifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("[MODEL EVALUATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
