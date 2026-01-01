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


"""
Data transformation related constant start with DATA_TRANSFORMATION var name
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_X_TRAIN: str = "x_train.npy"
DATA_TRNSFORMATION_Y_TRAIN: str = "y_train.npy"
DATA_TRANSFORMATION_X_TEST: str = "x_test.npy"
DATA_TRNSFORMATION_Y_TEST: str = "y_test.npy"
DATA_TRANSFORMATION_X_VAL: str = "x_val.npy"
DATA_TRNSFORMATION_Y_VAL: str = "y_val.npy"
DATA_TRANSFORMATION_METADATA: str = "transformation_metadata.json"
DATA_TRNSFORMATION_PREPROCESSOR: str = "preprocessor.pkl"


"""
Model trainer related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_MODEL_FILE_NAME: str = "models"
MODEL_TRAINER_TRAINING_METRICS_FILE_NAME: str = "training_metrics.json"
MODEL_TRAINER_METADATA_FILE_NAME: str = "model_medata.json"
MODEL_TRAINER_PRIMARY_METRIC: str = "roc_auc"