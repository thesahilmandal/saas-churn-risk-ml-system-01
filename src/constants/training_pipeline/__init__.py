import os

"""
Defining common constant variable for training pipeline.
"""
TARGET_COLUMN: str = "Churn"
ARTIFACT_DIR: str = "artifacts"
REFERENCE_SCHEMA_FILE_PATH = os.path.join('data_schema', 'schema.yaml')


"""
ETL related constant start with ETL var name
"""
ETL_DIR_NAME: str = "etl"
ETL_RAW_DATA_FILE_NAME: str = "raw.csv"
ETL_RAW_SCHEMA_FILE_NAME: str = "raw_schema.json"
ETL_METADATA_FILE_NAME: str = "etl_metadata.json"

"""
Data Ingestion related constant start with DATA_INGESTION var name
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_VAL_FILE_NAME: str = "val.csv"
DATA_INGESTION_SCHEMA_FILE_NAME: str = "generated_schema.json"
DATA_INGESTION_META_FILE_NAME: str = "meta_data.json"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.15
DATA_INGESTION_RANDOM_STATE: str = 42


"""
Data validation related constant start with DATA_VALIDATION var name
"""
DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_REPORT_FILE_NAME: str = 'report.json'
