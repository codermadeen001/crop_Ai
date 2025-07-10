from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.shortcuts import render
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import io

# Load Maize Validator model (Is this a maize leaf?)
VALIDATOR_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'maize_leaf_validator.h5')
try:
    validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
except Exception as e:
    print(f"Validator model failed to load: {e}")
    validator_model = None

# Load Maize Disease Classifier model
DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'maize_model.h5')
try:
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
except Exception as e:
    print(f"Disease model failed to load: {e}")
    disease_model = None

# Labels for the disease model
DISEASE_LABELS = {
    0: {
        "name": "Maize Blight",
        "cause": "Caused by fungus spreading through wind or water/rain",
        "treatment": [
            "Fungicides: Use TILT 250EC (propiconazole) or MILTHANE SUPER 72WP (mancozeb + metalaxyl)",
            "Apply every 7-10 days when symptoms first appear",
        ],
        "prevention": [
            "Plant resistant varieties like DK 8031 or PHB 30G19",
            "Space plants properly (75cm between rows, 25cm between plants)",
            "Avoid overhead irrigation",
            "Remove and burn infected plant debris after harvest",
            "Rotate with non-cereal crops for 2 seasons"
        ],
    },
    1: {
        "name": "Maize Rust",
        "cause": "Fungus that thrives in cool (15-22°C), humid conditions.",
        "treatment": [
            "Fungicides: Use AMISTAR TOP 325SC or ABSOLUTE 375SC",
            "Spray early at first sign of yellow spots",
            "Repeat after heavy rains"
        ],
        "prevention": [
            "Plant early before rains peak",
            "Avoid excessive nitrogen fertilizer",
            "Remove volunteer maize plants between seasons",
            "Use tolerant varieties like SC 513 or H6213"
        ],
    },
    2: {
        "name": "Leaf Spot",
        "cause": "Soil-borne fungus that splashes onto leaves during rains.",
        "treatment": [
            "Fungicides: Use RIDOMIL GOLD 66WP or LOCKER 720WP",
            "Spray before flowering stage",
            "Ensure coverage of lower leaves"
        ],
        "prevention": [
            "Crop rotation with legumes",
            "Improve field drainage",
            "Burn or bury crop residues",
            "Use certified disease-free seeds"
        ],
    },
    3: {
        "name": "Healthy Maize",
        "cause": "No disease detected",
        "treatment": [
            "Continue good management practices",
            "Monitor field weekly for early signs"
        ],
        "prevention": [
            "Rotate crops annually",
            "Use resistant varieties",
            "Control weeds",
            "Test soil and maintain pH 5.8–7.0"
        ],
    }
}


@api_view(['POST'])
def predict(request):
    """Predict disease if image is a maize leaf"""
    if not disease_model or not validator_model:
        return Response({"success": False, "error": "Model(s) not loaded"}, status=503)

    if 'image' not in request.FILES:
        return Response({
            "success": False,
            "error": "No image file provided",
        }, status=400)

    image_file = request.FILES['image']
    
    try:
        # Read image from memory
        file_bytes = b''.join(chunk for chunk in image_file.chunks())
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img.verify()  # verify image integrity
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')

        # Preprocess for model
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)

        # Step 1: Check if it's a maize leaf
        validator_pred = validator_model.predict(img_array)[0][0]  # sigmoid
        if validator_pred < 0.5:
            return Response({
                "success": False,
                "error": "This image is not a maize leaf. Please upload a valid maize leaf photo."
            }, status=200)

        # Step 2: Predict maize disease
        predictions = disease_model.predict(img_array)
        class_id = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return Response({
            "success": True,
            "prediction": DISEASE_LABELS[class_id]["name"],
            "confidence": confidence,
            "cause": DISEASE_LABELS[class_id]["cause"],
            "treatment": DISEASE_LABELS[class_id]["treatment"],
            "prevention": DISEASE_LABELS[class_id]["prevention"]
        })

    except Exception as e:
        return Response({
            "success": False,
            "error": "Image processing failed",
            "details": str(e)
        }, status=400)


def index(request):
    return render(request, 'cropAi/index.html')






