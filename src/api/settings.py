from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """
    API configuration settings.

    This class intentionally includes ONLY API-level configuration.
    It ignores unrelated environment variables (ETL, DB, training).
    """

    # -------------------------
    # Application metadata
    # -------------------------
    PROJECT_NAME: str = "Customer Churn Prediction API"
    API_VERSION: str = "v1"

    # -------------------------
    # Server settings (optional, future-proof)
    # -------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # -------------------------
    # Pydantic v2 configuration
    # -------------------------
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore", 
    )


# Singleton settings instance
settings = Settings()
