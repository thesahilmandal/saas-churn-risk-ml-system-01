import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

"""
Defining common constant variable for training pipeline.
"""
TARGET_COLUMN: str = "Churn"
ARTIFACT_DIR: str = "artifacts"


"""
ETL related constant start with ETL var name
"""
ETL_DIR_NAME: str = "01_etl"
ETL_METADATA_FILE_NAME: str = "etl_metadata.json"

"""
Data Ingestion related constant start with DATA_INGESTION var name
"""
DATA_INGESTION_DIR_NAME: str = "02_data_ingestion"
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_VAL_FILE_NAME: str = "val.csv"
DATA_INGESTION_SCHEMA_FILE_NAME: str = "ingestion_schema.json"
DATA_INGESTION_META_FILE_NAME: str = "meta_data.json"
DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO: float = 0.3
DATA_INGESTION_TEST_VAL_SPLIT_RATIO: float = 0.5
DATA_INGESTION_RANDOM_STATE: str = 42
DATA_INGESTION_DATABASE_NAME: str = os.getenv("MONGODB_DATABASE")
DATA_INGESTION_COLLECTION_NAME: str = os.getenv("MONGODB_COLLECTION")
DATA_INGESTION_MONGODB_URL: str = os.getenv("MONGODB_URL")


"""
Data validation related constant start with DATA_VALIDATION var name
"""
DATA_VALIDATION_DIR_NAME: str = '03_data_validation'
DATA_VALIDATION_REPORT_FILE_NAME: str = 'report.json'
DATA_VALIDATION_REFERENCE_SCHEMA = os.path.join('data_schema', 'schema.yaml')


"""
Data transformation related constant start with DATA_TRANSFORMATION var name
"""
DATA_TRANSFORMATION_DIR_NAME: str = "04_data_transformation"
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
MODEL_TRAINER_DIR_NAME: str = "05_model_trainer"
MODEL_TRAINER_MODEL_FILE_NAME: str = "models"
MODEL_TRAINER_TRAINING_METRICS_FILE_NAME: str = "training_metrics.json"
MODEL_TRAINER_METADATA_FILE_NAME: str = "model_medata.json"
MODEL_TRAINER_PRIMARY_METRIC: str = "roc_auc"


"""
Model evaluation constant start with MODEL_EVALUATION var name
"""
MODEL_EVALUATION_DIR_NAME = "06_model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME = "model_evaluation_report.json"
MODEL_EVALUATION_PRIMARY_METRIC = "recall"
MODEL_EVALUATION_MIN_ROC_AUC = 0.70
MODEL_EVALUATION_MIN_PRECISION = 0.40
MODEL_EVALUATION_MIN_RECALL = 0.60
MODEL_EVALUATION_THRESHOLDS = [
    round(x, 2) for x in list(np.arange(0.1, 0.91, 0.05))
]