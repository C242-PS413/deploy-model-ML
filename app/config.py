import os

MODEL_PATH = os.getenv("MODEL_PATH", "app/my_model.h5")
DATASET_PATH = os.getenv("DATASET_PATH", "https://storage.googleapis.com/clozify-data-ml/recommendation_data.csv")
