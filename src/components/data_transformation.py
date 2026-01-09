"""
Data Transformation Pipeline.

Responsibilities:
- Build model-aware preprocessing pipelines
- Fit preprocessors on training data only (no leakage)
- Persist fitted preprocessors and transformation metadata
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file, save_object
from src.constants.training_pipeline import TARGET_COLUMN


NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


class DataTransformation:
    """
    Handles feature preprocessing and transformation pipeline construction.

    This component:
    - Enforces validation gate before execution
    - Builds separate preprocessors for linear and tree-based models
    - Fits transformations on training data only
    """

    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> None:
        try:
            logging.info("[DATA TRANSFORMATION INIT] Initializing pipeline")

            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data validation failed. Transformation aborted."
                )

            self.config = transformation_config
            self.ingestion_artifact = ingestion_artifact
            self.validation_artifact = validation_artifact

            os.makedirs(
                self.config.data_transformation_dir,
                exist_ok=True
            )

            logging.info(
                "[DATA TRANSFORMATION INIT] Initialized successfully"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        """
        Read CSV file safely.

        Args:
            file_path (str): Path to CSV file

        Returns:
            pd.DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )
        return pd.read_csv(file_path)

    @staticmethod
    def _get_feature_groups(
        X: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        Split features into numerical and categorical groups.

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            Tuple[List[str], List[str]]
        """
        numerical_features = NUMERICAL_FEATURES

        categorical_features = [
            col for col in X.columns
            if col not in numerical_features
        ]

        return numerical_features, categorical_features

    # ============================================================
    # Preprocessor Builders
    # ============================================================
    def _build_linear_preprocessor(
        self,
        X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Build preprocessing pipeline for linear models.

        - Median imputation
        - Standard scaling for numeric features
        - One-hot encoding with drop-first for categoricals
        """
        try:
            num_features, cat_features = self._get_feature_groups(X)

            logging.info(
                "[DATA TRANSFORMATION] Building linear model preprocessor | "
                f"num_features={len(num_features)}, "
                f"cat_features={len(cat_features)}"
            )

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            return ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, num_features),
                    ("cat", categorical_pipeline, cat_features)
                ],
                remainder="drop"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def _build_tree_preprocessor(
        self,
        X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Build preprocessing pipeline for tree-based models.

        - Median imputation for numeric features
        - One-hot encoding without scaling
        """
        try:
            num_features, cat_features = self._get_feature_groups(X)

            logging.info(
                "[DATA TRANSFORMATION] Building tree model preprocessor | "
                f"num_features={len(num_features)}, "
                f"cat_features={len(cat_features)}"
            )

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            return ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, num_features),
                    ("cat", categorical_pipeline, cat_features)
                ],
                remainder="drop"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Metadata
    # ============================================================
    @staticmethod
    def _generate_metadata() -> Dict:
        """
        Generate transformation metadata.
        """
        return {
            "data_transformation_version": "v1.0.0",
            "fitted_on": "X_train_only",
            "numerical_features": NUMERICAL_FEATURES,
            "pipelines": {
                "linear": {
                    "imputation": "median",
                    "scaling": "StandardScaler",
                    "encoding": "OneHotEncoder(drop='first')"
                },
                "tree": {
                    "imputation": "median",
                    "scaling": "none",
                    "encoding": "OneHotEncoder"
                }
            },
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat()
        }

    # ============================================================
    # Pipeline Entry Point
    # ============================================================
    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:
        """
        Execute data transformation pipeline.

        Returns:
            DataTransformationArtifact
        """
        try:
            logging.info("[DATA TRANSFORMATION PIPELINE] Started")

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in training data"
                )

            X_train = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            logging.info(
                "[DATA TRANSFORMATION] Training data loaded | "
                f"Rows={len(X_train)}, Columns={len(X_train.columns)}"
            )

            # -------- Linear model preprocessor --------
            linear_preprocessor = self._build_linear_preprocessor(X_train)
            linear_preprocessor.fit(X_train)

            save_object(
                self.config.lr_preprocessor_file_path,
                linear_preprocessor
            )

            # -------- Tree model preprocessor --------
            tree_preprocessor = self._build_tree_preprocessor(X_train)
            tree_preprocessor.fit(X_train)

            save_object(
                self.config.tree_preprocessor_file_path,
                tree_preprocessor
            )

            # -------- Metadata --------
            metadata = self._generate_metadata()
            write_json_file(
                self.config.metadata_file_path,
                metadata
            )

            artifact = DataTransformationArtifact(
                tree_preprocessor_file_path=(
                    self.config.tree_preprocessor_file_path
                ),
                linear_preprocessor_file_path=(
                    self.config.lr_preprocessor_file_path
                ),
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Completed successfully"
            )
            logging.info(
                f"DataTransformationArtifact: {artifact}"
            )

            return artifact

        except Exception as e:
            logging.exception(
                "[DATA TRANSFORMATION PIPELINE] Failed"
            )
            raise CustomerChurnException(e, sys)
