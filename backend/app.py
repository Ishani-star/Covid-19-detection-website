import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# --- 1. SETUP ---
app = Flask(__name__)
CORS(app)

# Use absolute path with 'r' prefix to avoid unicode errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')
TARGET_SIZE = (224, 224)

# --- 2. LOAD H5 MODEL & CLASS INDICES ---
try:
    # Use standard Keras loader since the identifier is 'DF' (HDF5)
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ SUCCESS: Keras Model (.h5) loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Could not load .h5 model.")
    print(f"Details: {e}")
    exit(1)

try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    print(f"✅ SUCCESS: Classes loaded: {class_names}")
except Exception as e:
    print(f"❌ ERROR: Could not load {CLASS_INDICES_PATH}")
    exit(1)

# --- 3. PREDICTION ROUTE ---
@app.route('/analyze', methods=['POST'])
def analyze_xray():
    if 'xray_image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['xray_image']
    try:
        # Preprocess
        img = Image.open(io.BytesIO(file.read())).convert('RGB').resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction_probs = model.predict(img_array)[0]
        predicted_index = int(np.argmax(prediction_probs))
        
        return jsonify({
            "prediction": class_names[predicted_index],
            "confidence": round(float(prediction_probs[predicted_index]) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)