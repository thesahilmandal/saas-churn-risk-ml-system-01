from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Artifact generated after Data Ingestion stage.
    Stores paths to ingested datasets and related metadata.
    """
    train_file_path: str
    test_file_path: str
    val_file_path: str
    schema_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataIngestionArtifact(\n"
            f"  train_file_path  = {self.train_file_path}\n"
            f"  test_file_path     = {self.test_file_path}\n"
            f"  val_file_path      = {self.val_file_path}\n"
            f"  schema_file_path   = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass
class DataValidationArtifact:
    """
    Artifact generated after Data Validation stage.
    Stores validation status and validation report path.
    """
    validation_status: bool
    validation_report: str

    def __str__(self) -> str:
        return (
            "\nDataValidationArtifact(\n"
            f"  validation_status = {self.validation_status}\n"
            f"  validation_report = {self.validation_report}\n"
            ")"
        )


@dataclass
class DataTransformationArtifact:
    x_train_file_path: str
    x_test_file_path: str
    x_val_file_path: str
    y_train_file_path: str
    y_test_file_path: str
    y_val_file_path: str
    metadata_file_path: str
    preprocessor_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataTransformationArtifact(\n"
            f"  x_train_file_path = {self.x_train_file_path}\n"
            f"  x_test_file_path = {self.x_test_file_path}\n"
            f"  x_val_file_path = {self.x_val_file_path}\n"
            f"  y_train_file_path = {self.y_train_file_path}\n"
            f"  y_test_file_path = {self.y_test_file_path}\n"
            f"  y_val_file_path = {self.y_val_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  preprocessor_file_path = {self.preprocessor_file_path}\n"            
            ")"
        )


@dataclass
class ModelTrainerArtifact:
    """
    Artifact generated after Model Training stage (multi-model design).

    Stores:
    - Directory containing all trained models
    - Path to consolidated training metrics report
    - Path to model training metadata
    """
    trained_models_dir: str
    metrics_report_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelTrainerArtifact(\n"
            f"  trained_models_dir        = {self.trained_models_dir}\n"
            f"  metrics_report_file_path  = {self.metrics_report_file_path}\n"
            f"  metadata_file_path        = {self.metadata_file_path}\n"
            ")"
        )


@dataclass
class ModelEvaluationArtifact:
    """
    Artifact generated after Model Evaluation stage.

    Stores:
    - Path to evaluation decision report
    - Selected model name
    - Operating decision threshold for inference
    """

    evaluation_report_file_path: str
    selected_model_name: str
    operating_threshold: float

    def __str__(self) -> str:
        return (
            "\nModelEvaluationArtifact(\n"
            f"  evaluation_report_file_path = {self.evaluation_report_file_path}\n"
            f"  selected_model_name         = {self.selected_model_name}\n"
            f"  operating_threshold         = {self.operating_threshold}\n"
            ")"
        )
