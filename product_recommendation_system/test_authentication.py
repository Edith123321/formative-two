import os
import sys
import joblib
import numpy as np
import cv2
from src.image_processing import ImageProcessor

def load_model(model_path):
    """Load the trained model"""
    try:
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def authenticate_image(image_path, model, threshold=0.7):
    """Authenticate an image using the trained model"""
    try:
        print(f"\n🔍 Authenticating image: {image_path}")
        
        # Load and process the image
        img_processor = ImageProcessor()
        features_dict = img_processor.extract_features(image_path)
        
        if features_dict is None:
            print("❌ Failed to extract features from image")
            return False, 0.0
        
        # Convert features to array in the correct order
        feature_order = [
            'hist_r_mean', 'hist_r_std', 'hist_g_mean', 'hist_g_std', 
            'hist_b_mean', 'hist_b_std', 'face_detected', 'face_area_ratio',
            'face_aspect_ratio', 'face_symmetry'
        ]
        
        features = []
        for feat in feature_order:
            features.append(features_dict.get(feat, 0.0))  # Use 0.0 as default if feature is missing
        
        # Make prediction
        features_array = np.array([features])
        prediction_proba = model.predict_proba(features_array)
        confidence = prediction_proba[0][1]  # Probability of class 1 (authentic)
        
        is_authentic = confidence >= threshold
        print(f"🔐 Authentication {'SUCCESSFUL' if is_authentic else 'FAILED'}")
        print(f"   Confidence: {confidence:.2f} (Threshold: {threshold})")
        
        return is_authentic, confidence
        
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        return False, 0.0

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--image":
        print("Usage: python test_authentication.py --image path/to/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[2]
    
    # Load the model
    model_path = "models/facial_recognition_model.pkl"  # Update this path if needed
    model = load_model(model_path)
    if model is None:
        print("❌ Could not load the model. Please check the model path.")
        sys.exit(1)
    
    # Perform authentication
    is_authentic, confidence = authenticate_image(image_path, model)
    
    # Print final result
    print("\n" + "="*50)
    print(f"🔍 AUTHENTICATION RESULT")
    print("="*50)
    print(f"File: {image_path}")
    print(f"Status: {'✅ AUTHENTIC' if is_authentic else '❌ UNAUTHORIZED'}")
    print(f"Confidence: {confidence:.2%}")
    print("="*50)