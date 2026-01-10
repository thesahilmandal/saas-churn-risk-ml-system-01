# """
# Model Evaluation Pipeline.

# Responsibilities:
# - Load validation dataset
# - Load trained candidate models
# - Perform threshold sweeping and metric computation
# - Apply eligibility gates
# - Select best model and operating threshold
# - Persist evaluation decision report and metadata

# NOTE:
# This pipeline is strictly evaluation-only.
# No training or inference logic is included.
# """

# import os
# import sys
# from datetime import datetime, timezone
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd

# from sklearn.metrics import (
#     roc_auc_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )

# from src.entity.config_entity import ModelEvaluationConfig
# from src.entity.artifact_entity import (
#     ModelTrainerArtifact,
#     DataIngestionArtifact,
#     DataTransformationArtifact,
#     ModelEvaluationArtifact,
# )
# from src.constants.training_pipeline import TARGET_COLUMN
# from src.exception import CustomerChurnException
# from src.logging import logging
# from src.utils.main_utils import load_object, write_json_file


# class ModelEvaluation:
#     """
#     Handles model evaluation, gating, and selection logic.
#     """

#     def __init__(
#         self,
#         evaluation_config: ModelEvaluationConfig,
#         model_trainer_artifact: ModelTrainerArtifact,
#         data_ingestion_artifact: DataIngestionArtifact,
#         data_transformation_artifact: DataTransformationArtifact,
#     ) -> None:
#         try:
#             logging.info("[MODEL EVALUATION INIT] Initializing pipeline")

#             self.config = evaluation_config
#             self.trainer_artifact = model_trainer_artifact
#             self.ingestion_artifact = data_ingestion_artifact
#             self.transformation_artifact = data_transformation_artifact

#             os.makedirs(self.config.model_evaluation_dir, exist_ok=True)

#             logging.info("[MODEL EVALUATION INIT] Initialized successfully")

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     # ============================================================
#     # Data Loading
#     # ============================================================
#     @staticmethod
#     def _read_csv(file_path: str) -> pd.DataFrame:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#         return pd.read_csv(file_path)

#     def _load_validation_data(
#         self,
#     ) -> Tuple[pd.DataFrame, pd.Series]:
#         """
#         Load validation dataset.
#         """
#         try:
#             logging.info("[MODEL EVALUATION] Loading validation data")

#             val_df = self._read_csv(
#                 self.ingestion_artifact.val_file_path
#             )

#             if TARGET_COLUMN not in val_df.columns:
#                 raise ValueError(
#                     f"Target column '{TARGET_COLUMN}' not found in validation data"
#                 )

#             X_val = val_df.drop(columns=[TARGET_COLUMN])
#             y_val = val_df[TARGET_COLUMN]

#             logging.info(
#                 "[MODEL EVALUATION] Validation data loaded | "
#                 f"rows={len(X_val)}, columns={len(X_val.columns)}"
#             )

#             return X_val, y_val

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     def _load_candidate_models(self) -> Dict[str, object]:
#         """
#         Load all trained model artifacts.
#         """
#         try:
#             logging.info("[MODEL EVALUATION] Loading trained models")

#             models: Dict[str, object] = {}

#             for file_name in os.listdir(
#                 self.trainer_artifact.trained_models_dir
#             ):
#                 if file_name.endswith(".pkl"):
#                     model_name = file_name.replace(".pkl", "")
#                     model_path = os.path.join(
#                         self.trainer_artifact.trained_models_dir,
#                         file_name,
#                     )
#                     models[model_name] = load_object(model_path)

#             if not models:
#                 raise ValueError("No trained models found for evaluation")

#             logging.info(
#                 "[MODEL EVALUATION] Models loaded | count=%d",
#                 len(models),
#             )

#             return models

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     # ============================================================
#     # Metrics & Thresholding
#     # ============================================================
#     @staticmethod
#     def _compute_metrics(
#         y_true: np.ndarray,
#         y_pred: np.ndarray,
#         y_proba: np.ndarray,
#     ) -> Dict[str, float]:
#         """
#         Compute evaluation metrics.
#         """
#         return {
#             "roc_auc": roc_auc_score(y_true, y_proba),
#             "precision": precision_score(
#                 y_true, y_pred, zero_division=0
#             ),
#             "recall": recall_score(
#                 y_true, y_pred, zero_division=0
#             ),
#             "f1_score": f1_score(
#                 y_true, y_pred, zero_division=0
#             ),
#         }

#     def _sweep_thresholds(
#         self,
#         model: object,
#         X_val: pd.DataFrame,
#         y_val: pd.Series,
#     ) -> List[Dict]:
#         """
#         Evaluate model across configured probability thresholds.
#         """
#         threshold_results: List[Dict] = []

#         y_proba = model.predict_proba(X_val)[:, 1]

#         for threshold in self.config.thresholds:
#             y_pred = (y_proba >= threshold).astype(int)

#             metrics = self._compute_metrics(
#                 y_true=y_val,
#                 y_pred=y_pred,
#                 y_proba=y_proba,
#             )

#             threshold_results.append(
#                 {
#                     "threshold": threshold,
#                     "metrics": metrics,
#                 }
#             )

#         return threshold_results

#     # ============================================================
#     # Gating & Selection
#     # ============================================================
#     def _apply_gates(
#         self, metrics: Dict[str, float]
#     ) -> Tuple[bool, str]:
#         """
#         Apply eligibility gates.
#         """
#         if metrics["roc_auc"] < self.config.min_roc_auc:
#             return False, "failed_roc_auc_gate"

#         if metrics["precision"] < self.config.min_precision:
#             return False, "failed_precision_gate"

#         if metrics["recall"] < self.config.min_recall:
#             return False, "failed_recall_gate"

#         return True, "passed"

#     def _select_best_threshold(
#         self, threshold_results: List[Dict]
#     ) -> Dict:
#         """
#         Select best threshold satisfying precision constraint.
#         """
#         valid_candidates = [
#             result
#             for result in threshold_results
#             if result["metrics"]["precision"]
#             >= self.config.min_precision
#         ]

#         if not valid_candidates:
#             return {}

#         primary_metric = self.config.primary_metric

#         return max(
#             valid_candidates,
#             key=lambda x: x["metrics"][primary_metric],
#         )

#     def _select_best_model(
#         self, model_summaries: Dict[str, Dict]
#     ) -> Dict:
#         """
#         Select best model across all eligible candidates.
#         """
#         if not model_summaries:
#             raise ValueError(
#                 "No eligible models after evaluation"
#             )

#         primary_metric = self.config.primary_metric

#         model_name, model_info = max(
#             model_summaries.items(),
#             key=lambda item: item[1]["metrics"][primary_metric],
#         )

#         model_info["model_name"] = model_name
#         return model_info

#     # ============================================================
#     # Metadata
#     # ============================================================
#     def _generate_metadata(
#         self,
#         models_evaluated: List[str],
#         eligible_models_count: int,
#         rejected_models_count: int,
#     ) -> None:
#         """
#         Generate and persist evaluation metadata.
#         """
#         try:
#             metadata = {
#                 "stage": "model_evaluation",
#                 "input_artifacts": {
#                     "validation_data_path": (
#                         self.ingestion_artifact.val_file_path
#                     ),
#                     "trained_models_dir": (
#                         self.trainer_artifact.trained_models_dir
#                     ),
#                 },
#                 "evaluation_strategy": {
#                     "primary_metric": self.config.primary_metric,
#                     "thresholding": {
#                         "type": "fixed_threshold_grid",
#                         "thresholds": self.config.thresholds,
#                     },
#                     "gates": {
#                         "min_roc_auc": self.config.min_roc_auc,
#                         "min_precision": self.config.min_precision,
#                         "min_recall": self.config.min_recall,
#                     },
#                 },
#                 "models_evaluated": models_evaluated,
#                 "evaluation_summary": {
#                     "eligible_models_count": eligible_models_count,
#                     "rejected_models_count": rejected_models_count,
#                 },
#                 "execution_info": {
#                     "evaluated_at_utc": datetime.now(
#                         timezone.utc
#                     ).isoformat(),
#                     "pipeline_version": "v1.0.0",
#                     "executor": "ModelEvaluation",
#                 },
#             }

#             write_json_file(
#                 file_path=self.config.metadata_file_path,
#                 content=metadata,
#             )

#             logging.info(
#                 "[MODEL EVALUATION METADATA] Metadata written | "
#                 f"path={self.config.metadata_file_path}"
#             )

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     # ============================================================
#     # Pipeline Entry Point
#     # ============================================================
#     def initiate_model_evaluation(
#         self,
#     ) -> ModelEvaluationArtifact:
#         """
#         Execute model evaluation pipeline.
#         """
#         try:
#             logging.info("[MODEL EVALUATION PIPELINE] Started")

#             X_val, y_val = self._load_validation_data()
#             models = self._load_candidate_models()

#             model_results: Dict[str, Dict] = {}
#             rejected_models: Dict[str, str] = {}

#             for model_name, model in models.items():
#                 logging.info(
#                     "[MODEL EVALUATION] Evaluating model=%s",
#                     model_name,
#                 )

#                 threshold_results = self._sweep_thresholds(
#                     model, X_val, y_val
#                 )

#                 best_threshold_result = (
#                     self._select_best_threshold(threshold_results)
#                 )

#                 if not best_threshold_result:
#                     rejected_models[model_name] = "no_valid_threshold"
#                     continue

#                 metrics = best_threshold_result["metrics"]
#                 is_eligible, reason = self._apply_gates(metrics)

#                 if not is_eligible:
#                     rejected_models[model_name] = reason
#                     continue

#                 model_results[model_name] = {
#                     "threshold": best_threshold_result["threshold"],
#                     "metrics": metrics,
#                 }

#             selected_model_info = self._select_best_model(
#                 model_results
#             )

#             decision_report = {
#                 "selected_model": selected_model_info["model_name"],
#                 "operating_threshold": selected_model_info["threshold"],
#                 "primary_metric": self.config.primary_metric,
#                 "metrics": selected_model_info["metrics"],
#                 "rejected_models": rejected_models,
#                 "evaluated_at_utc": datetime.now(
#                     timezone.utc
#                 ).isoformat(),
#             }

#             write_json_file(
#                 file_path=self.config.report_file_path,
#                 content=decision_report,
#             )

#             self._generate_metadata(
#                 models_evaluated=list(models.keys()),
#                 eligible_models_count=len(model_results),
#                 rejected_models_count=len(rejected_models),
#             )

#             artifact = ModelEvaluationArtifact(
#                 report_file_path=self.config.report_file_path,
#                 selected_model_name=selected_model_info["model_name"],
#                 operating_threshold=selected_model_info["threshold"],
#                 metadata_file_path=self.config.metadata_file_path,
#             )

#             logging.info(
#                 "[MODEL EVALUATION PIPELINE] Completed successfully"
#             )
#             logging.info(
#                 f"ModelEvaluationArtifact: {artifact}"
#             )

#             return artifact

#         except Exception as e:
#             logging.exception(
#                 "[MODEL EVALUATION PIPELINE] Failed"
#             )
#             raise CustomerChurnException(e, sys)


"""
Model Evaluation Pipeline.

Responsibilities:
- Load validation dataset
- Load trained candidate models
- Perform threshold sweeping and metric computation
- Apply eligibility gates
- Select best model and operating threshold
- Persist evaluation decision report and metadata

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


PIPELINE_VERSION = "v1.1.0"


class ModelEvaluation:
    """
    Handles model evaluation, gating, and selection logic.
    """

    def __init__(
        self,
        evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            logging.info("[MODEL EVALUATION INIT] Initializing pipeline")

            self.config = evaluation_config
            self.trainer_artifact = model_trainer_artifact
            self.ingestion_artifact = data_ingestion_artifact
            self.transformation_artifact = data_transformation_artifact

            os.makedirs(self.config.model_evaluation_dir, exist_ok=True)

            logging.info(
                "[MODEL EVALUATION INIT] Initialized | version=%s",
                PIPELINE_VERSION,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Data Loading
    # ============================================================
    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    def _load_validation_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            logging.info("[MODEL EVALUATION] Loading validation data")

            val_df = self._read_csv(self.ingestion_artifact.val_file_path)

            if TARGET_COLUMN not in val_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in validation data"
                )

            X_val = val_df.drop(columns=[TARGET_COLUMN])
            y_val = val_df[TARGET_COLUMN]

            logging.info(
                "[MODEL EVALUATION] Validation data loaded | rows=%d",
                len(X_val),
            )

            return X_val, y_val

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def _load_candidate_models(self) -> Dict[str, object]:
        try:
            logging.info("[MODEL EVALUATION] Loading trained models")

            models: Dict[str, object] = {}

            for file_name in os.listdir(
                self.trainer_artifact.trained_models_dir
            ):
                if file_name.endswith(".pkl"):
                    model_name = file_name.replace(".pkl", "")
                    model_path = os.path.join(
                        self.trainer_artifact.trained_models_dir,
                        file_name,
                    )
                    models[model_name] = load_object(model_path)

            if not models:
                raise ValueError("No trained models found for evaluation")

            return models

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Metrics & Thresholding
    # ============================================================
    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_true, y_pred, zero_division=0
            ),
        }

    def _sweep_thresholds(
        self,
        model: object,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> List[Dict]:
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "Model does not support predict_proba"
            )

        y_proba = model.predict_proba(X_val)[:, 1]
        threshold_results: List[Dict] = []

        for threshold in self.config.thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            metrics = self._compute_metrics(
                y_true=y_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            threshold_results.append(
                {"threshold": threshold, "metrics": metrics}
            )

        return threshold_results

    # ============================================================
    # Gating & Selection
    # ============================================================
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
        valid_candidates = [
            r for r in threshold_results
            if r["metrics"]["precision"] >= self.config.min_precision
        ]

        if not valid_candidates:
            return {}

        return max(
            valid_candidates,
            key=lambda r: r["metrics"][self.config.primary_metric],
        )

    def _select_best_model(
        self, model_summaries: Dict[str, Dict]
    ) -> Dict:
        if not model_summaries:
            raise ValueError("No eligible models after evaluation")

        model_name, model_info = max(
            model_summaries.items(),
            key=lambda i: i[1]["metrics"][self.config.primary_metric],
        )

        model_info["model_name"] = model_name
        return model_info

    # ============================================================
    # Metadata
    # ============================================================
    def _generate_metadata(
        self,
        models_evaluated: List[str],
        model_trace: Dict[str, Dict],
        start_time: float,
    ) -> None:
        metadata = {
            "stage": "model_evaluation",
            "pipeline_version": PIPELINE_VERSION,
            "input_artifacts": {
                "validation_data_path": self.ingestion_artifact.val_file_path,
                "trained_models_dir": self.trainer_artifact.trained_models_dir,
            },
            "evaluation_strategy": {
                "primary_metric": self.config.primary_metric,
                "thresholds": self.config.thresholds,
                "gates": {
                    "min_roc_auc": self.config.min_roc_auc,
                    "min_precision": self.config.min_precision,
                    "min_recall": self.config.min_recall,
                },
            },
            "models_evaluated": models_evaluated,
            "model_trace": model_trace,
            "execution_info": {
                "duration_seconds": round(time.time() - start_time, 3),
                "evaluated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "executor": "ModelEvaluation",
            },
        }

        write_json_file(
            self.config.metadata_file_path,
            metadata,
        )

        logging.info(
            "[MODEL EVALUATION METADATA] Written | path=%s",
            self.config.metadata_file_path,
        )

    # ============================================================
    # Pipeline Entry Point
    # ============================================================
    def initiate_model_evaluation(
        self,
    ) -> ModelEvaluationArtifact:
        start_time = time.time()

        try:
            logging.info("[MODEL EVALUATION PIPELINE] Started")

            X_val, y_val = self._load_validation_data()
            models = self._load_candidate_models()

            model_results: Dict[str, Dict] = {}
            rejected_models: Dict[str, str] = {}
            model_trace: Dict[str, Dict] = {}

            for model_name, model in models.items():
                logging.info(
                    "[MODEL EVALUATION] Evaluating model=%s",
                    model_name,
                )

                try:
                    threshold_results = self._sweep_thresholds(
                        model, X_val, y_val
                    )
                except AttributeError:
                    rejected_models[model_name] = (
                        "predict_proba_not_supported"
                    )
                    continue

                best_threshold_result = self._select_best_threshold(
                    threshold_results
                )

                if not best_threshold_result:
                    rejected_models[model_name] = "no_valid_threshold"
                    continue

                metrics = best_threshold_result["metrics"]
                eligible, reason = self._apply_gates(metrics)

                model_trace[model_name] = {
                    "thresholds_tested": len(threshold_results),
                    "best_threshold": best_threshold_result["threshold"],
                    "gate_status": reason,
                }

                if not eligible:
                    rejected_models[model_name] = reason
                    continue

                if (
                    self.config.primary_metric == "recall"
                    and metrics["precision"] < 0.5
                ):
                    logging.warning(
                        "[MODEL EVALUATION] Low precision (%.3f) for "
                        "recall-optimized model=%s",
                        metrics["precision"],
                        model_name,
                    )

                model_results[model_name] = {
                    "threshold": best_threshold_result["threshold"],
                    "metrics": metrics,
                }

            selected_model = self._select_best_model(model_results)

            decision_report = {
                "selected_model": selected_model["model_name"],
                "operating_threshold": selected_model["threshold"],
                "primary_metric": self.config.primary_metric,
                "metrics": selected_model["metrics"],
                "rejected_models": rejected_models,
                "evaluated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            write_json_file(
                self.config.report_file_path,
                decision_report,
            )

            self._generate_metadata(
                models_evaluated=list(models.keys()),
                model_trace=model_trace,
                start_time=start_time,
            )

            artifact = ModelEvaluationArtifact(
                report_file_path=self.config.report_file_path,
                selected_model_name=selected_model["model_name"],
                operating_threshold=selected_model["threshold"],
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("[MODEL EVALUATION PIPELINE] Completed successfully")
            logging.info(f"ModelEvaluationArtifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("[MODEL EVALUATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
