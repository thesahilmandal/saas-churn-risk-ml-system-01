# import os
# import sys
# import json
# import hashlib
# import warnings
# from datetime import datetime, timezone
# from typing import Dict

# import pandas as pd
# import kagglehub
# from dotenv import load_dotenv

# from src.utils.main_utils import write_json_file
# from src.entity.config_entity import ETLconfig, TrainingPipelineConfig
# from src.entity.artifact_entity import ETLartifact
# from src.exception import CustomerChurnException
# from src.logging import logging

# warnings.filterwarnings("ignore")
# load_dotenv()


# class CustomerChurnETL:
#     """
#     End-to-End ETL pipeline for Customer Churn dataset.

#     Responsibilities:
#     - Extract raw dataset from Kaggle
#     - Perform lightweight data cleaning
#     - Generate schema and metadata
#     - Persist raw artifacts for downstream pipelines
#     """

#     def __init__(self, etl_config: ETLconfig):
#         try:
#             logging.info("Initializing CustomerChurnETL pipeline")

#             self.etl_config = etl_config
#             self.data_source = os.getenv("DATA_SOURCE")
#             self.run_id = self._generate_run_id()

#             logging.info(f"ETL initialized with run_id={self.run_id}")

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     @staticmethod
#     def _generate_run_id() -> str:
#         return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

#     @staticmethod
#     def _calculate_checksum(df: pd.DataFrame) -> str:
#         return hashlib.md5(
#             pd.util.hash_pandas_object(df, index=True).values
#         ).hexdigest()

#     # =========================
#     # Extract
#     # =========================
#     def extract(self) -> pd.DataFrame:
#         try:
#             logging.info("Starting data extraction from Kaggle")

#             dataset_path = kagglehub.dataset_download(self.data_source)
#             csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

#             if not csv_files:
#                 raise ValueError("No CSV files found in downloaded dataset")

#             csv_path = os.path.join(dataset_path, csv_files[0])
#             df = pd.read_csv(csv_path)

#             logging.info(
#                 f"Extraction completed | shape={df.shape}"
#             )

#             return df

#         except Exception as e:
#             logging.error("Data extraction failed")
#             raise CustomerChurnException(e, sys)

#     # =========================
#     # Transform
#     # =========================
#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         try:
#             logging.info("Starting data transformation")

#             df = df.copy()
#             df = df.drop_duplicates().reset_index(drop=True)

#             for col in df.select_dtypes(include="object").columns:
#                 df[col] = df[col].astype(str).str.strip()

#             logging.info(
#                 f"Transformation completed | shape={df.shape}"
#             )

#             return df

#         except Exception as e:
#             logging.error("Data transformation failed")
#             raise CustomerChurnException(e, sys)

#     # =========================
#     # Helpers
#     # =========================
#     def _generate_schema(self, df: pd.DataFrame) -> Dict:
#         return {
#             col: {
#                 "dtype": str(df[col].dtype),
#                 "nullable": bool(df[col].isna().sum())
#             }
#             for col in df.columns
#         }

#     def _generate_metadata(self, df: pd.DataFrame, checksum: str) -> Dict:
#         return {
#             "run_id": self.run_id,
#             "data_source": self.data_source,
#             "row_count": len(df),
#             "column_count": len(df.columns),
#             "checksum": checksum,
#             "generated_at_utc": datetime.now(timezone.utc).isoformat()
#         }

#     # =========================
#     # Load
#     # =========================
#     def load(self, df: pd.DataFrame) -> None:
#         try:
#             logging.info("Starting data load phase")
            
#             # Ensure artifact directories exist
#             os.makedirs(
#                 os.path.dirname(self.etl_config.raw_data_file_path),
#                 exist_ok=True
#             )
            
#             df.to_csv(self.etl_config.raw_data_file_path, index=False)

#             schema = self._generate_schema(df)
#             write_json_file(
#                 file_path=self.etl_config.raw_schema_file_path,
#                 content=schema
#             )

#             checksum = self._calculate_checksum(df)
#             metadata = self._generate_metadata(df, checksum)
#             write_json_file(
#                 file_path=self.etl_config.metadata_file_path,
#                 content=metadata
#             )

#             logging.info("Load phase completed successfully")

#         except Exception as e:
#             logging.error("Data load failed")
#             raise CustomerChurnException(e, sys)

#     # =========================
#     # Run
#     # =========================
#     def initiate_etl(self) -> ETLartifact:
#         try:
#             logging.info("ETL pipeline execution started")

#             df = self.extract()
#             df = self.transform(df)
#             self.load(df)

#             artifact = ETLartifact(
#                 raw_data_file_path=self.etl_config.raw_data_file_path,
#                 raw_schema_file_path=self.etl_config.raw_schema_file_path,
#                 metadata_file_path=self.etl_config.metadata_file_path
#             )

#             logging.info("ETL pipeline completed successfully")
#             logging.info(artifact)
#             return artifact

#         except Exception as e:
#             logging.error("ETL pipeline failed")
#             raise CustomerChurnException(e, sys)



import os
import sys
import json
import hashlib
import warnings
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import kagglehub
from dotenv import load_dotenv

from src.utils.main_utils import write_json_file
from src.entity.config_entity import ETLconfig
from src.entity.artifact_entity import ETLartifact
from src.exception import CustomerChurnException
from src.logging import logging

warnings.filterwarnings("ignore")
load_dotenv()


class CustomerChurnETL:
    """
    End-to-End ETL pipeline for the Customer Churn dataset.

    Responsibilities:
    - Extract raw dataset from Kaggle
    - Perform lightweight, non-destructive transformations
    - Generate schema and metadata artifacts
    - Persist immutable raw data for downstream pipelines
    """

    def __init__(self, etl_config: ETLconfig) -> None:
        try:
            logging.info("Initializing CustomerChurnETL pipeline")

            self.etl_config = etl_config
            self.data_source = os.getenv("DATA_SOURCE")

            if not self.data_source:
                raise ValueError("DATA_SOURCE environment variable is not set")

            self.run_id = self._generate_run_id()

            logging.info(f"ETL initialized successfully | run_id={self.run_id}")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # =========================
    # Internal Utilities
    # =========================
    @staticmethod
    def _generate_run_id() -> str:
        """Generate a UTC-based unique run identifier."""
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _calculate_checksum(df: pd.DataFrame) -> str:
        """Generate a deterministic checksum for dataset integrity tracking."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()

    def _generate_schema(self, df: pd.DataFrame) -> Dict:
        """
        Generate lightweight schema information from the dataset.
        Intended for downstream validation and debugging.
        """
        return {
            column: {
                "dtype": str(df[column].dtype),
                "nullable": bool(df[column].isna().any())
            }
            for column in df.columns
        }

    def _generate_metadata(self, df: pd.DataFrame, checksum: str) -> Dict:
        """
        Generate metadata capturing lineage, size, and integrity details.
        """
        return {
            "run_id": self.run_id,
            "data_source": self.data_source,
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "checksum": checksum,
            "generated_at_utc": datetime.now(timezone.utc).isoformat()
        }

    # =========================
    # Extract
    # =========================
    def extract(self) -> pd.DataFrame:
        try:
            logging.info("Extract phase started | source=Kaggle")

            dataset_path = kagglehub.dataset_download(self.data_source)
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

            if not csv_files:
                raise FileNotFoundError("No CSV files found in the Kaggle dataset")

            csv_path = os.path.join(dataset_path, csv_files[0])
            df = pd.read_csv(csv_path)

            logging.info(f"Extract phase completed | shape={df.shape}")

            return df

        except Exception as e:
            logging.exception("Extract phase failed")
            raise CustomerChurnException(e, sys)

    # =========================
    # Transform
    # =========================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Transform phase started")

            df = df.copy()
            df = df.drop_duplicates().reset_index(drop=True)

            object_columns = df.select_dtypes(include="object").columns
            for column in object_columns:
                df[column] = df[column].astype(str).str.strip()

            logging.info(f"Transform phase completed | shape={df.shape}")

            return df

        except Exception as e:
            logging.exception("Transform phase failed")
            raise CustomerChurnException(e, sys)

    # =========================
    # Load
    # =========================
    def load(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Load phase started")

            # Ensure all required directories exist
            for path in [
                self.etl_config.raw_data_file_path,
                self.etl_config.raw_schema_file_path,
                self.etl_config.metadata_file_path
            ]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Persist raw dataset
            df.to_csv(self.etl_config.raw_data_file_path, index=False)

            # Persist schema
            schema = self._generate_schema(df)
            write_json_file(
                file_path=self.etl_config.raw_schema_file_path,
                content=schema
            )

            # Persist metadata
            checksum = self._calculate_checksum(df)
            metadata = self._generate_metadata(df, checksum)
            write_json_file(
                file_path=self.etl_config.metadata_file_path,
                content=metadata
            )

            logging.info("Load phase completed successfully")

        except Exception as e:
            logging.exception("Load phase failed")
            raise CustomerChurnException(e, sys)

    # =========================
    # Orchestration
    # =========================
    def initiate_etl(self) -> ETLartifact:
        try:
            logging.info("ETL pipeline execution started")

            df = self.extract()
            df = self.transform(df)
            self.load(df)

            artifact = ETLartifact(
                raw_data_file_path=self.etl_config.raw_data_file_path,
                raw_schema_file_path=self.etl_config.raw_schema_file_path,
                metadata_file_path=self.etl_config.metadata_file_path
            )

            logging.info("ETL pipeline execution completed successfully")
            logging.info(f"ETL Artifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("ETL pipeline execution failed")
            raise CustomerChurnException(e, sys)
