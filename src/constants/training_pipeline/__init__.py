"""
Centralized configuration constants for the ML training pipeline.

This module defines directory names, file names, environment variables,
and pipeline-wide constants used across different stages.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables once at module import
load_dotenv()


# -------------------------------------------------------------------------
# Global Pipeline Constants
# -------------------------------------------------------------------------

TARGET_COLUMN: str = "Churn"
ARTIFACT_DIR: Path = Path("artifacts")


# -------------------------------------------------------------------------
# ETL Constants
# -------------------------------------------------------------------------

ETL_DIR_NAME: str = "01_etl"
ETL_METADATA_FILE_NAME: str = "etl_metadata.json"


# -------------------------------------------------------------------------
# Data Ingestion Constants
# -------------------------------------------------------------------------

DATA_INGESTION_DIR_NAME: str = "02_data_ingestion"

DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_VAL_FILE_NAME: str = "val.csv"

DATA_INGESTION_SCHEMA_FILE_NAME: str = "ingestion_schema.json"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO: float = 0.30
DATA_INGESTION_TEST_VAL_SPLIT_RATIO: float = 0.50

DATA_INGESTION_RANDOM_STATE: int = 42

# Environment variables (validated at runtime, not import time)
DATA_INGESTION_DATABASE_NAME: str | None = os.getenv("MONGODB_DATABASE")
DATA_INGESTION_COLLECTION_NAME: str | None = os.getenv("MONGODB_COLLECTION")
DATA_INGESTION_MONGODB_URL: str | None = os.getenv("MONGODB_URL")


# -------------------------------------------------------------------------
# Data Validation Constants
# -------------------------------------------------------------------------

DATA_VALIDATION_DIR_NAME: str = "03_data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.json"

DATA_VALIDATION_REFERENCE_SCHEMA: Path = Path("data_schema") / "schema.yaml"


# -------------------------------------------------------------------------
# Data Transformation Constants
# -------------------------------------------------------------------------

DATA_TRANSFORMATION_DIR_NAME: str = "04_data_transformation"

DATA_TRANSFORMATION_LINEAR_PREPROCESSOR_FILE_NAME: str = "lr_preprocessor.pkl"
DATA_TRANSFORMATION_TREE_PREPROCESSOR_FILE_NAME: str = "tree_preprocessor.pkl"

DATA_TRANSFORMATION_METADATA_FILE_NAME: str = "metadata.json"
