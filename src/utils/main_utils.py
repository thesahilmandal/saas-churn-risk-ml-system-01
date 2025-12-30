import os
import sys
import yaml, json
import pickle
import numpy as np

from datetime import datetime
from typing import Any, Dict

from src.exception import CustomerChurnException
from src.logging import logging


# ==============================
# Date Utilities
# ==============================

def format_ordinal_date(dt: datetime) -> str:
    """
    Convert datetime to ordinal date format.
    Example: 21st_dec_2024
    """
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    month = dt.strftime("%b").lower()
    year = dt.year
    return f"{day}{suffix}_{month}_{year}"


# ==============================
# Internal Helpers
# ==============================

def _prepare_file_path(file_path: str, replace: bool = True) -> None:
    """
    Prepare file path by removing existing file (if replace=True)
    and creating parent directories.
    """
    try:
        if replace and os.path.exists(file_path):
            logging.info(f"Removing existing file: {file_path}")
            os.remove(file_path)

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# YAML Utilities
# ==============================

def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Read YAML file and return contents as dictionary.
    """
    try:
        logging.info(f"Reading YAML file: {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def write_yaml_file(file_path: str, content: Dict[str, Any], replace: bool = True) -> None:
    """
    Write dictionary content to YAML file.
    """
    try:
        logging.info(f"Writing YAML file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w") as file:
            yaml.dump(content, file, sort_keys=False)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# NumPy Utilities
# ==============================

def save_numpy_array_data(
    file_path: str,
    array: np.ndarray,
    replace: bool = True
) -> None:
    """
    Save NumPy array to disk.
    """
    try:
        logging.info(f"Saving NumPy array: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load NumPy array from disk.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Numpy file not found: {file_path}")

        logging.info(f"Loading NumPy array: {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# Pickle Utilities
# ==============================

def save_object(file_path: str, obj: Any, replace: bool = True) -> None:
    """
    Serialize and save Python object using pickle.
    """
    try:
        logging.info(f"Saving object: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load serialized Python object from disk.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Object file not found: {file_path}")

        logging.info(f"Loading object: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomerChurnException(e, sys)


# ==============================
# JSON Utilities
# ==============================

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read JSON file and return contents as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        logging.info(f"Reading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as e:
        raise CustomerChurnException(e, sys)


def write_json_file(
    file_path: str,
    content: Dict[str, Any],
    replace: bool = True
) -> None:
    """
    Write dictionary content to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        content (Dict[str, Any]): Data to be written.
        replace (bool): Whether to overwrite existing file.
    """
    try:
        logging.info(f"Writing JSON file: {file_path}")
        _prepare_file_path(file_path, replace)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(content, file, indent=4)

    except Exception as e:
        raise CustomerChurnException(e, sys)
