import os
from flask import Flask, request, jsonify
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('verification_model.h5')

# Class labels
class_labels = ['electricity', 'garbage', 'road', 'unknown', 'water logging']

def load_and_preprocess_image_from_url(url):
    """Load and preprocess image from URL."""
    response = requests.get(url)
    img = image.load_img(BytesIO(response.content), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def predict_image_class_from_url(url, threshold=0.7):
    """Predict image class from URL."""
    try:
        img_array = load_and_preprocess_image_from_url(url)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_probability = np.max(predictions)
        
        if predicted_probability >= threshold:
            return {"class": class_labels[predicted_class_index[0]], "confidence": float(predicted_probability)}
        else:
            return {"class": "unknown", "confidence": float(predicted_probability)}
    except Exception as e:
        return {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle prediction requests."""
    try:
        # Get JSON data from the POST request
        data = request.json
        image_url = data.get("url")
        
        if not image_url:
            return jsonify({"error": "No URL provided"}), 400
        
        # Predict the class
        prediction = predict_image_class_from_url(image_url)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT environment variable
    app.run(host='0.0.0.0', port=port)
