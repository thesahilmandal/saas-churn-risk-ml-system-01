"""
Health check API route.

Responsibilities:
- Provide a lightweight endpoint for service health verification
- Support liveness and readiness checks for deployment platforms
- Avoid any dependency on ML models or external systems
"""

from datetime import datetime, timezone
from fastapi import APIRouter

from src.logging import logging
from src.exception import CustomerChurnException

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/",
    summary="Service health check",
    description=(
        "Lightweight health check endpoint used by load balancers, "
        "orchestration systems, and monitoring tools."
    ),
)
def health_check():
    """
    Perform a basic health check.

    This endpoint intentionally performs no I/O, no model loading,
    and no dependency checks. If this endpoint is reachable, the
    service process is considered healthy.
    """
    try:
        logging.info("[API HEALTH] Health check requested")

        response = {
            "status": "ok",
            "service": "customer-churn-api",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        logging.info("[API HEALTH] Health check successful")

        return response

    except Exception as e:
        # This should never happen, but we guard defensively
        logging.error(
            "[API HEALTH] Unexpected error during health check",
            exc_info=True,
        )
        raise CustomerChurnException(e, sys)
