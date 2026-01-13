import os
import sys
from typing import Tuple

import pandas as pd

from src.entity.artifact_entity import ModelEvaluationArtifact
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import load_object


class CustomerChurnPredictor:
    """
    Customer Churn Prediction Pipeline.

    Responsibilities:
    - Load selected model, preprocessor, and threshold
    - Validate input data
    - Generate churn probabilities and predictions
    - Persist prediction results
    """

    def __init__(
        self,
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> None:
        try:
            logging.info("[PREDICTION INIT] Initializing predictor")

            self.evaluation_artifact = evaluation_artifact

            (self.model, self.threshold) = self._load_artifacts()

            logging.info("[PREDICTION INIT] Initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Artifact Loading
    # ============================================================
    def _load_artifacts(self) -> Tuple[object, object, float]:
        """
        Load model, preprocessor, and operating threshold.
        """
        try:
            logging.info("[PREDICTION] Loading model artifacts")

            model = load_object(
                self.evaluation_artifact.selected_trained_model_file_path
            )
            threshold = self.evaluation_artifact.operating_threshold

            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    "Loaded model does not support predict_proba"
                )

            return model, threshold

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Validation
    # ============================================================
    @staticmethod
    def _validate_input(input_df: pd.DataFrame) -> None:
        """
        Basic input validation.
        """
        if input_df.empty:
            raise ValueError("Input DataFrame is empty")

    # ============================================================
    # Prediction
    # ============================================================
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn predictions.

        Args:
            input_df (pd.DataFrame): Raw input features

        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        try:
            logging.info("[PREDICTION] Prediction started")

            self._validate_input(input_df)

            # Predict probabilities
            churn_probabilities = (
                self.model.predict_proba(input_df)[:, 1]
            )

            churn_predictions = (
                churn_probabilities >= self.threshold
            ).astype(int)

            output_df = input_df.copy()
            output_df["churn_probability"] = churn_probabilities
            output_df["churn_prediction"] = churn_predictions

            logging.info("[PREDICTION] Prediction completed successfully")

            return output_df

        except Exception as e:
            logging.exception("[PREDICTION] Prediction failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Persistence
    # ============================================================
    @staticmethod
    def save_predictions(
        prediction_df: pd.DataFrame,
        output_file_path: str,
    ) -> None:
        """
        Save predictions to disk.
        """
        try:
            os.makedirs(
                os.path.dirname(output_file_path),
                exist_ok=True,
            )
            prediction_df.to_csv(output_file_path, index=False)

            logging.info(
                "[PREDICTION] Predictions saved | path=%s",
                output_file_path,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)