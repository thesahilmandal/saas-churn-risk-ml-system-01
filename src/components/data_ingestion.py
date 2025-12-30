# import os
# import sys
# from datetime import datetime, timezone
# from typing import Dict, Tuple

# import warnings
# warnings.filterwarnings("ignore")

# import pandas as pd
# from sklearn.model_selection import train_test_split

# from src.entity.config_entity import DataIngestionConfig
# from src.entity.artifact_entity import ETLartifact, DataIngestionArtifact
# from src.exception import CustomerChurnException
# from src.logging import logging
# from src.utils.main_utils import write_json_file
# from src.constants.training_pipeline import TARGET_COLUMN


# class DataIngestion:
#     """
#     Data Ingestion pipeline responsible for:
#     - Reading raw ETL output
#     - Stratified train/validation/test split
#     - Persisting ingestion artifacts
#     """

#     def __init__(
#         self,
#         data_ingestion_config: DataIngestionConfig,
#         etl_artifact: ETLartifact
#     ) -> None:
#         try:
#             self.config = data_ingestion_config
#             self.etl_artifact = etl_artifact
#             self.target_column = TARGET_COLUMN

#             # Ensure base ingestion directory exists
#             os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

#             logging.info("DataIngestion initialized successfully")

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     # =========================
#     # Helpers
#     # =========================
#     @staticmethod
#     def _read_csv(file_path: str) -> pd.DataFrame:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#         return pd.read_csv(file_path)

#     def _split_data(
#         self, df: pd.DataFrame
#     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         try:
#             logging.info("Starting stratified data split")

#             train_df, temp_df = train_test_split(
#                 df,
#                 test_size=self.config.train_test_split_ratio,
#                 random_state=self.config.random_state,
#                 stratify=df[self.target_column]
#             )

#             val_df, test_df = train_test_split(
#                 temp_df,
#                 test_size=self.config.train_test_split_ratio,
#                 random_state=self.config.random_state,
#                 stratify=temp_df[self.target_column]
#             )

#             logging.info(
#                 f"Data split completed | "
#                 f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
#             )

#             return train_df, val_df, test_df

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     def _generate_schema(self, df: pd.DataFrame) -> Dict:
#         """
#         Generate extended schema using training data only.
#         """
#         schema = {}

#         for col in df.columns:
#             col_data = df[col]
#             schema[col] = {
#                 "dtype": str(col_data.dtype),
#                 "nullable": bool(col_data.isna().any()),
#                 "unique_values": int(col_data.nunique(dropna=True))
#             }

#             if pd.api.types.is_numeric_dtype(col_data):
#                 schema[col].update(
#                     {
#                         "min": float(col_data.min()),
#                         "max": float(col_data.max())
#                     }
#                 )

#         return schema

#     def _generate_metadata(
#         self,
#         train_df: pd.DataFrame,
#         val_df: pd.DataFrame,
#         test_df: pd.DataFrame
#     ) -> Dict:
#         return {
#             "etl_raw_data_path": self.etl_artifact.raw_data_file_path,
#             "split_strategy": "stratified",
#             "target_column": self.target_column,
#             "split_ratio": {
#                 "train": 1 - self.config.train_test_split_ratio,
#                 "validation": self.config.train_test_split_ratio,
#                 "test": self.config.train_test_split_ratio
#             },
#             "random_state": self.config.random_state,
#             "record_counts": {
#                 "train": len(train_df),
#                 "validation": len(val_df),
#                 "test": len(test_df)
#             },
#             "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat()
#         }

#     # =========================
#     # Pipeline Entry
#     # =========================
#     def initiate_data_ingestion(self) -> DataIngestionArtifact:
#         try:
#             logging.info("Data ingestion pipeline started")

#             df = self._read_csv(self.etl_artifact.raw_data_file_path)

#             if self.target_column not in df.columns:
#                 raise ValueError(
#                     f"Target column '{self.target_column}' not found in dataset"
#                 )

#             train_df, val_df, test_df = self._split_data(df)

#             # Ensure directories exist before writing
#             for path in [
#                 self.config.train_file_path,
#                 self.config.val_file_path,
#                 self.config.test_file_path,
#                 self.config.schema_file_path,
#                 self.config.metadata_file_path,
#             ]:
#                 os.makedirs(os.path.dirname(path), exist_ok=True)

#             train_df.to_csv(self.config.train_file_path, index=False)
#             val_df.to_csv(self.config.val_file_path, index=False)
#             test_df.to_csv(self.config.test_file_path, index=False)

#             schema = self._generate_schema(train_df)
#             write_json_file(
#                 file_path=self.config.schema_file_path,
#                 content=schema
#             )

#             metadata = self._generate_metadata(train_df, val_df, test_df)
#             write_json_file(
#                 file_path=self.config.metadata_file_path,
#                 content=metadata
#             )

#             artifact = DataIngestionArtifact(
#                 train_file_path=self.config.train_file_path,
#                 test_file_path=self.config.test_file_path,
#                 val_file_path=self.config.val_file_path,
#                 schema_file_path=self.config.schema_file_path,
#                 metadata_file_path=self.config.metadata_file_path
#             )

#             logging.info("Data ingestion pipeline completed successfully")
#             logging.info(artifact)
#             return artifact

#         except Exception as e:
#             raise CustomerChurnException(e, sys)


import os
import sys
from datetime import datetime, timezone
from typing import Dict, Tuple

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import ETLartifact, DataIngestionArtifact
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file
from src.constants.training_pipeline import TARGET_COLUMN


class DataIngestion:
    """
    Data Ingestion pipeline.

    Responsibilities:
    - Read immutable raw data from ETL artifacts
    - Perform stratified train/validation/test split
    - Generate schema from training data only
    - Persist ingestion artifacts for downstream stages
    """

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig,
        etl_artifact: ETLartifact
    ) -> None:
        try:
            self.config = data_ingestion_config
            self.etl_artifact = etl_artifact
            self.target_column = TARGET_COLUMN

            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

            logging.info("DataIngestion initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # =========================
    # Helpers
    # =========================
    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_csv(file_path)
            logging.info(f"Raw data loaded successfully | shape={df.shape}")
            return df

        except Exception as e:
            logging.exception("Failed to read raw CSV file")
            raise

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            logging.info(
                "Starting stratified split | "
                f"ratio={self.config.train_test_split_ratio}, "
                f"random_state={self.config.random_state}"
            )

            if df[self.target_column].nunique() < 2:
                raise ValueError(
                    "Stratified split requires at least 2 target classes"
                )

            train_df, temp_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=self.config.random_state,
                stratify=df[self.target_column]
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.config.train_test_split_ratio,
                random_state=self.config.random_state,
                stratify=temp_df[self.target_column]
            )

            logging.info(
                f"Split completed | "
                f"train={len(train_df)}, "
                f"validation={len(val_df)}, "
                f"test={len(test_df)}"
            )

            return train_df, val_df, test_df

        except Exception as e:
            logging.exception("Data splitting failed")
            raise CustomerChurnException(e, sys)

    def _generate_schema(self, df: pd.DataFrame) -> Dict:
        """
        Generate schema using training data only
        to avoid data leakage.
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
        total_records = len(train_df) + len(val_df) + len(test_df)

        return {
            "etl_raw_data_path": self.etl_artifact.raw_data_file_path,
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
            "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat()
        }

    # =========================
    # Pipeline Entry
    # =========================
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data ingestion pipeline started")

            df = self._read_csv(self.etl_artifact.raw_data_file_path)

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

            logging.info("Split datasets persisted successfully")

            schema = self._generate_schema(train_df)
            write_json_file(
                file_path=self.config.schema_file_path,
                content=schema
            )

            metadata = self._generate_metadata(train_df, val_df, test_df)
            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata
            )

            artifact = DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
                val_file_path=self.config.val_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Data ingestion pipeline completed successfully")
            logging.info(f"DataIngestion Artifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("Data ingestion pipeline failed")
            raise CustomerChurnException(e, sys)
