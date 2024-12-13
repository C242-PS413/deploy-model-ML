# app/services/recommendation.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.config import DATASET_PATH


# Reading the CSV data
data = pd.read_csv(DATASET_PATH)
image_directory = "https://storage.googleapis.com/clozify-data-ml/images/"

def recommend_items_by_sub_category(data, gender, season, emotion_category, top_m=3, top_n=5):
    sub_categories = data['subCategory'].unique()
    
    recommendations = {}

    for category in sub_categories:
        category_data = data[
            (data['gender'] == gender) &
            (data['season'] == season) &
            (data['Emotion_Category'] == emotion_category) & 
            (data['subCategory'] == category)
        ].reset_index(drop=True)
        
        if category_data.empty:
            recommendations[category] = {"recommendations": []}
            continue
        
        embeddings = category_data.loc[:, '0':'2047'].astype(float).values
        similarity_matrix = cosine_similarity(embeddings)
        
        category_recommendations = []
        
        for idx in range(min(len(category_data), top_m)):
            similarities = similarity_matrix[idx]
            similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]
            
            recommendation = {
                'recommendations_item': {
                    'image': f"{image_directory}{category_data.iloc[idx]['image']}",
                    'name': category_data.iloc[idx]['productDisplayName']
                },
                'more_recommended_items': [
                    {
                        'image': f"{image_directory}{category_data.iloc[i]['image']}",
                        'name': category_data.iloc[i]['productDisplayName']
                    } for i in similar_indices
                ]
            }
            category_recommendations.append(recommendation)
        
        recommendations[category] = {"recommendations": category_recommendations}
    
    return recommendations
