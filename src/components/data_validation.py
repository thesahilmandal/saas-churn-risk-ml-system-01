import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any

import yaml

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, write_json_file


class DataValidation:
    """
    Data Validation Pipeline.

    Responsibilities:
    - Validate generated schema against a reference schema
    - Apply severity-aware validation rules
    - Persist validation report and status artifact
    """

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        try:
            logging.info("Initializing DataValidation pipeline")

            self.validation_config = data_validation_config
            self.ingestion_artifact = data_ingestion_artifact

            if not os.path.exists(
                self.validation_config.reference_schema_file_path
            ):
                raise FileNotFoundError(
                    f"Reference schema not found at: "
                    f"{self.validation_config.reference_schema_file_path}"
                )

            logging.info("DataValidation initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _read_yaml(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.exception("Failed to read reference schema YAML")
            raise

    # ============================================================
    # Schema Validation
    # ============================================================
    def _validate_schema(
        self,
        generated_schema: Dict[str, Any],
        reference_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        logging.info("Schema validation started")

        if "columns" not in reference_schema:
            raise KeyError(
                "Invalid reference schema: missing 'columns' key"
            )

        results: Dict[str, Any] = {}
        error_count = 0
        warning_count = 0

        reference_columns = reference_schema["columns"]

        # ---------- Required Columns ----------
        for column, rules in reference_columns.items():
            if column not in generated_schema:
                results[column] = {
                    "status": "error",
                    "message": "Missing required column"
                }
                error_count += 1
                continue

            gen_col = generated_schema[column]
            col_result = {"status": "pass", "details": []}

            # ---------- Data Type Validation ----------
            expected_dtype = rules.get("expected_dtype")
            raw_dtype_allowed = rules.get("raw_dtype_allowed")
            gen_dtype = gen_col.get("dtype")

            if expected_dtype and gen_dtype != expected_dtype:
                if raw_dtype_allowed and gen_dtype == raw_dtype_allowed:
                    col_result["status"] = "warning"
                    col_result["details"].append(
                        f"Raw dtype '{gen_dtype}' differs from expected "
                        f"'{expected_dtype}' but is allowed"
                    )
                    warning_count += 1
                else:
                    col_result["status"] = "error"
                    col_result["details"].append(
                        f"Invalid dtype '{gen_dtype}', expected '{expected_dtype}'"
                    )
                    error_count += 1

            # ---------- Nullability ----------
            if rules.get("nullable") is False and gen_col.get("nullable") is True:
                severity = rules.get("severity", "error")
                col_result["details"].append(
                    "Null values found in non-nullable column"
                )

                if severity == "error":
                    col_result["status"] = "error"
                    error_count += 1
                else:
                    if col_result["status"] != "error":
                        col_result["status"] = "warning"
                    warning_count += 1

            # ---------- Numeric Range ----------
            if "min" in rules and "min" in gen_col:
                if gen_col["min"] < rules["min"]:
                    if col_result["status"] != "error":
                        col_result["status"] = "warning"
                    col_result["details"].append(
                        f"Minimum value {gen_col['min']} below expected {rules['min']}"
                    )
                    warning_count += 1

            if "max" in rules and "max" in gen_col:
                if gen_col["max"] > rules["max"]:
                    if col_result["status"] != "error":
                        col_result["status"] = "warning"
                    col_result["details"].append(
                        f"Maximum value {gen_col['max']} above expected {rules['max']}"
                    )
                    warning_count += 1

            results[column] = col_result

        # ---------- Extra Columns ----------
        for column in generated_schema:
            if column not in reference_columns:
                results[column] = {
                    "status": "warning",
                    "message": "Unexpected column not defined in reference schema"
                }
                warning_count += 1

        logging.info(
            f"Schema validation completed | "
            f"errors={error_count}, warnings={warning_count}"
        )

        return {
            "column_checks": results,
            "summary": {
                "errors": error_count,
                "warnings": warning_count
            }
        }

    # ============================================================
    # Pipeline Entry
    # ============================================================
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation pipeline started")

            generated_schema = read_json_file(
                self.ingestion_artifact.schema_file_path
            )

            reference_schema = self._read_yaml(
                self.validation_config.reference_schema_file_path
            )

            validation_result = self._validate_schema(
                generated_schema, reference_schema
            )

            report = {
                "validation_type": "schema_validation",
                "results": validation_result["column_checks"],
                "summary": validation_result["summary"],
                "validated_at_utc": datetime.now(timezone.utc).isoformat()
            }

            write_json_file(
                file_path=self.validation_config.validation_report_file_path,
                content=report
            )

            validation_status = validation_result["summary"]["errors"] == 0

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                validation_report=self.validation_config.validation_report_file_path
            )

            logging.info(
                f"Data Validation completed | status={validation_status}"
            )
            logging.info(f"DataValidation Artifact: {artifact}")

            return artifact

        except Exception as e:
            logging.exception("Data Validation pipeline failed")
            raise CustomerChurnException(e, sys)
