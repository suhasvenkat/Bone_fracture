import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
from PIL import Image
import os


model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")


categories_parts = ["Elbow", "Hand", "Shoulder"]

categories_fracture = ['fractured', 'normal']


def is_xray_image(image_path):
    """
    Simple check if the uploaded image is likely an X-ray image
    Returns True if it appears to be an X-ray, False otherwise
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not load image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic checks for X-ray characteristics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Very basic validation - just check if it's not too bright or too dark
        # and has some contrast
        is_likely_xray = (
            5 < mean_brightness < 250 and
            std_brightness > 5
        )
        
        # Check for face detection (if face is detected, it's likely not an X-ray)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                return False, "Image appears to contain a face, not an X-ray"
        except:
            # If face detection fails, continue with other checks
            pass
        
        if not is_likely_xray:
            return False, "Image doesn't appear to be an X-ray. Please upload a proper X-ray image."
        
        return True, "Valid X-ray image"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def predict(img, model="Parts"):
    # Check if the image is an X-ray
    is_valid, message = is_xray_image(img)
    if not is_valid:
        return f"Not identified: {message}"
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

   
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str