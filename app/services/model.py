import tensorflow as tf
from app.config import MODEL_PATH

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

MOOD_CLASSES = ['Anger', 'Contentment', 'Joy', 'Neutral', 'Sadness']

def predict_mood(preprocessed_face):
    # Predict mood
    predictions = model.predict(preprocessed_face)
    predicted_class = MOOD_CLASSES[predictions.argmax()]
    return predicted_class
