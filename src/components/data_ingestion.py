"""
Data Ingestion Pipeline for Customer Churn project.

Responsibilities:
- Read cleaned data from MongoDB
- Perform stratified train/validation/test split
- Persist datasets, schema, and metadata
"""

import os
import sys
from datetime import datetime, timezone
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.exception import CustomerChurnException
from src.logging import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants.training_pipeline import TARGET_COLUMN
from src.utils.main_utils import write_json_file

load_dotenv()


class DataIngestion:
    """
    Handles ingestion of cleaned data from MongoDB and
    prepares train/validation/test datasets.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        """
        Initialize Data Ingestion pipeline.

        Args:
            data_ingestion_config (DataIngestionConfig): Ingestion configuration
        """
        try:
            self.config = data_ingestion_config
            self.target_column = TARGET_COLUMN

            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

            logging.info("[DATA INGESTION INIT] Initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def import_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Load cleaned records from MongoDB into a DataFrame.

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        try:
            logging.info("[DATA INGESTION] Reading data from MongoDB")

            with pymongo.MongoClient(self.config.database_url) as client:
                collection = client[
                    self.config.database_name
                ][self.config.collection_name]

                records = list(collection.find())

            if not records:
                raise ValueError("No records found in MongoDB collection")

            df = pd.DataFrame(records)

            drop_columns = [
                "_id",
                "data_source",
                "ingested_at_utc",
                "customerID"
            ]

            df.drop(
                columns=[c for c in drop_columns if c in df.columns],
                inplace=True
            )

            df.drop_duplicates(inplace=True)
            df.replace({"na": np.nan}, inplace=True)

            logging.info(
                "[DATA INGESTION] MongoDB data loaded | "
                f"Rows: {len(df)}, Columns: {len(df.columns)}"
            )

            return df

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/validation/test split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        try:
            if df[self.target_column].nunique() < 2:
                raise ValueError(
                    "Stratified split requires at least two target classes"
                )

            train_df, temp_df = train_test_split(
                df,
                test_size=self.config.train_temp_split_ratio,
                random_state=self.config.random_state,
                stratify=df[self.target_column]
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.config.test_val_split_ratio,
                random_state=self.config.random_state,
                stratify=temp_df[self.target_column]
            )

            logging.info(
                "[DATA INGESTION] Data split completed | "
                f"train={len(train_df)}, "
                f"validation={len(val_df)}, "
                f"test={len(test_df)}"
            )

            return train_df, val_df, test_df

        except Exception as e:
            logging.exception("[DATA INGESTION] Data splitting failed")
            raise CustomerChurnException(e, sys)

    def _generate_schema(self, df: pd.DataFrame) -> Dict:
        """
        Generate schema using training data only
        to prevent data leakage.
        """
        schema: Dict[str, Dict] = {}

        for column in df.columns:
            col_data = df[column]

            schema[column] = {
                "dtype": str(col_data.dtype),
                "nullable": bool(col_data.isna().any()),
                "unique_values": int(col_data.nunique(dropna=True))
            }

            if pd.api.types.is_numeric_dtype(col_data):
                schema[column].update(
                    {
                        "min": float(col_data.min()),
                        "max": float(col_data.max())
                    }
                )

        return schema

    def _generate_metadata(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Generate ingestion metadata.
        """
        total_records = len(train_df) + len(val_df) + len(test_df)

        return {
            "source": "mongodb",
            "split_strategy": "stratified",
            "target_column": self.target_column,
            "actual_split_ratio": {
                "train": round(len(train_df) / total_records, 4),
                "validation": round(len(val_df) / total_records, 4),
                "test": round(len(test_df) / total_records, 4)
            },
            "random_state": self.config.random_state,
            "record_counts": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            },
            "ingested_at_utc": datetime.now(timezone.utc).isoformat()
        }

    # =========================
    # Pipeline Entry Point
    # =========================
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the data ingestion pipeline.
        """
        try:
            logging.info("[DATA INGESTION PIPELINE] Started")

            df = self.import_collection_as_dataframe()

            if self.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in dataset"
                )

            train_df, val_df, test_df = self._split_data(df)

            for path in [
                self.config.train_file_path,
                self.config.val_file_path,
                self.config.test_file_path,
                self.config.schema_file_path,
                self.config.metadata_file_path
            ]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            train_df.to_csv(self.config.train_file_path, index=False)
            val_df.to_csv(self.config.val_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            schema = self._generate_schema(train_df)
            write_json_file(self.config.schema_file_path, schema)

            metadata = self._generate_metadata(train_df, val_df, test_df)
            write_json_file(self.config.metadata_file_path, metadata)

            artifact = DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
                val_file_path=self.config.val_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("[DATA INGESTION PIPELINE] Completed successfully")
            logging.info(f"DataIngestionArtifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("[DATA INGESTION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
