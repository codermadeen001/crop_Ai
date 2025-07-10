from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.shortcuts import render
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import io

# Load the best model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.h5')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Class labels mapping (must match your training data order)
CLASS_LABELS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy', 'Non_Maize_Leaf']


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
        "cause": "Fungus that thrives in cool (15-22°C), humid conditions. Spreads quickly when dew remains on leaves for long periods. Often comes with late planting.",
        "treatment": [
            "Fungicides: Use AMISTAR TOP 325SC (azoxystrobin + difenoconazole) or ABSOLUTE 375SC (trifloxystrobin + tebuconazole)",
            "Spray early at first sign of yellow spots",
            "Mix with sticker/spreader for better coverage",
            "Repeat after heavy rains wash off fungicide"
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
            "Fungicides: Use RIDOMIL GOLD 66WP (metalaxyl-M + mancozeb) or LOCKER 720WP (mancozeb + cymoxanil)",
            "Begin sprays before flowering stage",
            "Ensure good coverage of lower leaves",
            "Combine with foliar fertilizer for recovery"
        ],
        "prevention": [
            "Practice crop rotation with legumes",
            "Improve field drainage",
            "Burn or bury crop residues deeply",
            "Use certified disease-free seeds"
        ],
       
    },
    3: {
        "name": "Healthy Maize",
        "cause": "No disease detected",
        "treatment": [
            "Continue good management practices",
            "Monitor field weekly for early signs",
            "Maintain balanced fertilizer application"
        ],
        "prevention": [
            "Rotate crops annually",
            "Use resistant varieties",
            "Control weeds that host pests/diseases",
            "Test soil and correct pH (optimal 5.8-7.0)"
        ],
    }
}



@api_view(['POST'])
def predict(request):
    """Prediction endpoint with terminal printing"""
    if not model:
        print("Model not loaded - service unavailable")
        return Response({"success": False, "error": "Model not loaded"}, status=503)

    if 'image' not in request.FILES:
        print("No image file provided in request")
        return Response({
            "success": False,
            "error": "No image file provided"
        }, status=400)

    try:
        # Process image
        image_file = request.FILES['image']
        print(f"Processing image: {image_file.name} ({image_file.size} bytes)")
        
        file_bytes = b''
        for chunk in image_file.chunks():
            file_bytes += chunk
        
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img.verify()
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        
        # Predict
        print("Making prediction...")
        predictions = model.predict(img_array)
        class_id = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_LABELS[class_id]
        
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")

        # Check if maize leaf
        if predicted_class == 'Non_Maize_Leaf':
            print("Rejected - Not a maize leaf")
            return Response({
                "success": False,
                "error": "This image is not a maize leaf. Please upload a valid maize leaf photo.",
                "confidence": confidence
            }, status=200)
        
        # If maize leaf
        print(f"Disease detected: {DISEASE_LABELS[class_id]['name']}")
        return Response({
            "success": True,
            "is_maize_leaf": True,
            "prediction": DISEASE_LABELS[class_id]["name"],
            "confidence": confidence,
            "cause": DISEASE_LABELS[class_id]["cause"],
            "treatment": DISEASE_LABELS[class_id]["treatment"],
            "prevention": DISEASE_LABELS[class_id]["prevention"]
        })
        

        """
          return Response({
            "success": True,
            "prediction": DISEASE_LABELS[class_id]["name"],
            "confidence": confidence,
            "cause": DISEASE_LABELS[class_id]["cause"],
            "treatment": DISEASE_LABELS[class_id]["treatment"],
            "prevention": DISEASE_LABELS[class_id]["prevention"]
        })
        """


    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return Response({
            "success": False,
            "error": "Image processing failed",
            "details": str(e)
        }, status=400)

def index(request):
    return render(request, 'cropAi/index.html')