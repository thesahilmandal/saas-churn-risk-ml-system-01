import os
import sys
import json
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file
from src.constants.training_pipeline import TARGET_COLUMN


class DataTransformation:
    """
    Data Transformation Pipeline.

    Responsibilities:
    - Apply schema-driven feature transformations
    - Handle type casting, imputation, and encoding
    - Produce model-ready numpy arrays
    - Persist fitted preprocessing pipeline
    """

    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> None:
        try:
            logging.info("Initializing DataTransformation pipeline")

            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data Validation failed. Transformation cannot proceed."
                )

            self.config = transformation_config
            self.ingestion_artifact = ingestion_artifact
            self.validation_artifact = validation_artifact

            self.target_column = TARGET_COLUMN

            os.makedirs(self.config.data_transformation_dir, exist_ok=True)

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
    def _encode_target(series: pd.Series) -> np.ndarray:
        return series.map({"Yes": 1, "No": 0}).astype(int).values

    # ============================================================
    # Preprocessor
    # ============================================================

    def _build_preprocessor(
        self,
        X: pd.DataFrame
    ) -> Tuple[ColumnTransformer, list]:
        """
        Build preprocessing pipeline for tree-based models.
        """

        categorical_features = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "SeniorCitizen"
        ]

        numerical_features = [
            "tenure", "MonthlyCharges", "TotalCharges"
        ]

        # Explicit type coercion (schema-driven)
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"], errors="coerce"
        )
        X['SeniorCitizen'] = X['SeniorCitizen'].map({1: "Yes", 0: "No"})

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        all_features = (
            numerical_features +
            categorical_features
        )

        return preprocessor, all_features

    # ============================================================
    # Run
    # ============================================================

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation pipeline started")

            # ---------------- Load Data ----------------
            train_df = self._read_csv(self.ingestion_artifact.train_file_path)
            val_df = self._read_csv(self.ingestion_artifact.val_file_path)
            test_df = self._read_csv(self.ingestion_artifact.test_file_path)

            # ---------------- Split X / y ----------------
            X_train = train_df.drop(columns=[self.target_column])
            y_train = self._encode_target(train_df[self.target_column])

            X_val = val_df.drop(columns=[self.target_column])
            y_val = self._encode_target(val_df[self.target_column])

            X_test = test_df.drop(columns=[self.target_column])
            y_test = self._encode_target(test_df[self.target_column])

            # ---------------- Preprocessing ----------------
            preprocessor, feature_list = self._build_preprocessor(X_train)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_val_transformed = preprocessor.transform(X_val)
            X_test_transformed = preprocessor.transform(X_test)

            # ---------------- Persist Artifacts ----------------
            np.save(self.config.x_train_file_path, X_train_transformed)
            np.save(self.config.x_val_file_path, X_val_transformed)
            np.save(self.config.x_test_file_path, X_test_transformed)

            np.save(self.config.y_train_file_path, y_train)
            np.save(self.config.y_val_file_path, y_val)
            np.save(self.config.y_test_file_path, y_test)

            joblib.dump(
                preprocessor,
                self.config.preprocessor_file_path
            )

            metadata = {
                "target_column": self.target_column,
                "num_features": X_train_transformed.shape[1],
                "imputation": {
                    "numerical": "median",
                    "categorical": "most_frequent"
                },
                "encoding": {
                    "categorical": "OneHotEncoder",
                    "ordinal": "OrdinalEncoder"
                },
                "fitted_on": "train_only",
                "generated_at_utc": datetime.now(timezone.utc).isoformat()
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata
            )

            artifact = DataTransformationArtifact(
                preprocessor_file_path=self.config.preprocessor_file_path,
                x_train_file_path=self.config.x_train_file_path,
                x_val_file_path=self.config.x_val_file_path,
                x_test_file_path=self.config.x_test_file_path,
                y_train_file_path=self.config.y_train_file_path,
                y_val_file_path=self.config.y_val_file_path,
                y_test_file_path=self.config.y_test_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Data Transformation completed successfully")
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.error("Data Transformation pipeline failed")
            raise CustomerChurnException(e, sys)
