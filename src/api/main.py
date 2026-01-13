"""
FastAPI application entrypoint.

Responsibilities:
- Create and configure the FastAPI application
- Register API routers
- Attach lifespan handlers for model loading
- Define centralized exception handling
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.settings import settings
from src.api.lifespan import lifespan
from src.api.routes import predict, health
from src.logging import logging
from src.exception import CustomerChurnException


def create_app() -> FastAPI:
    """
    Application factory.

    This function creates and configures the FastAPI application.
    Using a factory pattern improves testability and avoids
    side effects at import time.
    """
    logging.info(
        "[API STARTUP] Initializing FastAPI application | "
        f"service={settings.PROJECT_NAME}, version={settings.API_VERSION}"
    )

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.API_VERSION,
        lifespan=lifespan,
    )

    # -------------------------
    # Register routers
    # -------------------------
    app.include_router(health.router)
    app.include_router(predict.router)

    logging.info("[API STARTUP] Routers registered successfully")

    # -------------------------
    # Centralized Exception Handling
    # -------------------------
    @app.exception_handler(CustomerChurnException)
    async def customer_churn_exception_handler(
        request: Request,
        exc: CustomerChurnException,
    ) -> JSONResponse:
        """
        Handle application-specific exceptions.

        All CustomerChurnException instances are treated as
        internal server errors and logged once at this boundary.
        """
        logging.error(
            "[API ERROR] Unhandled CustomerChurnException | "
            f"path={request.url.path}",
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Please try again later."
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """
        Catch-all exception handler.

        This prevents raw tracebacks from leaking to API clients
        and ensures consistent error responses.
        """
        logging.error(
            "[API ERROR] Unhandled exception | "
            f"path={request.url.path}",
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unexpected server error. Please contact support."
            },
        )

    logging.info("[API STARTUP] FastAPI application initialized successfully")

    return app


# -------------------------
# ASGI Entrypoint
# -------------------------
app = create_app()
