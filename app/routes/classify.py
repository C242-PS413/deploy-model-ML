from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.face_detection import preprocess_face
from app.services.model import predict_mood
import os

router = APIRouter()

@router.post("/")
async def classify_image(file: UploadFile = File(...)):
    """
    Endpoint untuk mengklasifikasi mood berdasarkan gambar wajah.
    File gambar akan diproses (preprocessing) dan langsung diklasifikasi.
    """
    try:
        # Save the uploaded file temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        
        # Preprocess the image
        preprocessed = preprocess_face(file_location)
        if preprocessed is None:
            raise HTTPException(status_code=400, detail="No face detected in the image.")
        
        # Extract preprocessed data
        original, gray, face, face_resized = preprocessed

        # Reshape the face to match the model input (1, 48, 48, 1)
        preprocessed_face = face_resized.reshape(1, 48, 48, 1)  # Add batch and channel dimension

        # Predict mood
        result = predict_mood(preprocessed_face)
        return {"predicted_mood": result}
    
    finally:
        # Delete the temporary file
        if os.path.exists(file_location):
            os.remove(file_location)
