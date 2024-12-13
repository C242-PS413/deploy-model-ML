from fastapi import FastAPI
from app.routes import classify
from app.routes import recommendation  # Import the new recommendation route

app = FastAPI(title="Mood Classification API")

# Existing classify route
app.include_router(classify.router, prefix="/classify", tags=["Classification"])

# New recommendation route
app.include_router(recommendation.router, prefix="/recommendations", tags=["Recommendations"])

@app.get("/")
def root():
    return {"message": "Welcome to Mood Detection API"}
