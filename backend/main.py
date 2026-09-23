from fastapi import FastAPI

app = FastAPI(
    title="OptiWealth API",
    version="1.0.0"
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
