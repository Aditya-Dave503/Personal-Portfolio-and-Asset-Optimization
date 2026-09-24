from fastapi import FastAPI

from api.v1.router import api_router


app = FastAPI(
    title="OptiWealth API",
    version="1.0.0"
)


app.include_router(
    api_router,
    prefix="/api/v1"
)


@app.get("/")
def root():
    return {
        "message": "OptiWealth Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }