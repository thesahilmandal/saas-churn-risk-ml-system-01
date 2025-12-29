import os
import sys
import json
import certifi
import pandas as pd
import pymongo
import kagglehub
from datetime import datetime, timezone
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import format_ordinal_date

from dotenv import load_dotenv
load_dotenv()


class CustomerChurnETL:
    """
    End-to-end ETL pipeline for Customer Churn data.
    Extracts data from Kaggle, transforms it, and loads it into MongoDB.
    """

    def __init__(self) -> None:
        try:
            self.mongodb_url = os.getenv("MONGODB_URL")
            self.database = os.getenv("MONGODB_DATABASE")
            self.collection = os.getenv("MONGODB_COLLECTION")
            self.data_source = os.getenv("DATA_SOURCE")

            self._validate_config()
            self.ca_file = certifi.where()

            logging.info("CustomerChurnETL initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def _validate_config(self) -> None:
        """Validate required environment variables."""
        if not all([
            self.mongodb_url,
            self.database,
            self.collection,
            self.data_source
        ]):
            raise EnvironmentError(
                "Missing one or more required environment variables: "
                "MONGODB_URL, MONGODB_DATABASE, MONGODB_COLLECTION, DATA_SOURCE"
            )

    # -------------------- EXTRACT --------------------
    def extract_data(self) -> pd.DataFrame:
        try:
            logging.info("Starting data extraction from Kaggle")

            dataset_path = kagglehub.dataset_download(self.data_source)
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

            if not csv_files:
                raise FileNotFoundError(
                    "No CSV files found in the downloaded dataset directory."
                )

            csv_path = os.path.join(dataset_path, csv_files[0])
            df = pd.read_csv(csv_path)

            if df.empty:
                raise ValueError("Extracted CSV file is empty.")

            logging.info(
                f"Data extraction completed successfully | "
                f"rows={len(df)} columns={len(df.columns)}"
            )

            return df

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # -------------------- TRANSFORM --------------------
    def transform_data(self, df: pd.DataFrame) -> List[Dict]:
        try:
            logging.info("Starting data transformation")

            if df.empty:
                raise ValueError("Cannot transform an empty DataFrame.")

            df = df.drop_duplicates().reset_index(drop=True)

            df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip(),
                errors="coerce"
            )

            df.replace({"na": None}, inplace=True)

            df["data_source"] = self.data_source
            df["ingestion_time"] = format_ordinal_date(datetime.utcnow())

            records = json.loads(df.to_json(orient="records"))

            logging.info(
                f"Data transformation completed successfully | "
                f"records_prepared={len(records)}"
            )

            return records

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # -------------------- LOAD --------------------
    def load_data(self, records: List[Dict]) -> int:
        client = None
        try:
            logging.info("Starting data load into MongoDB")

            client = pymongo.MongoClient(
                self.mongodb_url,
                tlsCAFile=self.ca_file
            )
            collection = client[self.database][self.collection]

            # temporary
            collection.delete_many({})

            result = collection.insert_many(records)
            inserted_count = len(result.inserted_ids)

            logging.info(
                f"Data load completed successfully | "
                f"documents_inserted={inserted_count}"
            )

            return inserted_count

        except Exception as e:
            raise CustomerChurnException(e, sys)

        finally:
            if client:
                client.close()

    # -------------------- PIPELINE ORCHESTRATION --------------------
    def run(self) -> int:
        try:
            logging.info("Customer Churn ETL pipeline started")

            df = self.extract_data()
            records = self.transform_data(df)
            inserted_count = self.load_data(records)

            logging.info("Customer Churn ETL pipeline completed successfully")

            return inserted_count

        except Exception as e:
            raise CustomerChurnException(e, sys)


if __name__ == "__main__":
    try:
        etl = CustomerChurnETL()
        etl.run()
    except Exception as e:
        raise CustomerChurnException(e, sys)