# app/routes/recommendation.py

from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from app.services.recommendation import recommend_items_by_sub_category  # Import the recommendation service

router = APIRouter()

# Pydantic model to accept input parameters for recommendations (no top_m and top_n)
class RecommendationRequest(BaseModel):
    gender: str
    season: str
    emotion_category: str

# Default values for top_m and top_n in the function itself
@router.post("/")
async def get_recommendations(request: RecommendationRequest):
    # Read the recommendation data
    data = pd.read_csv('https://storage.googleapis.com/clozify-data-ml/recommendation_data.csv')
    
    # Default values for top_m and top_n
    top_m = 3
    top_n = 5
    
    # Call the recommendation function
    recommendations = recommend_items_by_sub_category(
        data,
        gender=request.gender,
        season=request.season,
        emotion_category=request.emotion_category,
        top_m=top_m,
        top_n=top_n
    )
    
    return recommendations
