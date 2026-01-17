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

MODEL_PATH = 'backend/final_tiny_model.tflite'
CLASS_INDICES_PATH = 'backend/class_indices.json'
TARGET_SIZE = (224, 224)

# --- 2. LOAD TFLITE INTERPRETER ---
try:
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ TFLite Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load TFLite model from {MODEL_PATH}")
    print(e)
    exit(1)

# Load Class Indices
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    print(f"✅ Class indices loaded: {class_names}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load class indices from {CLASS_INDICES_PATH}")
    print(e)
    exit(1)

# --- 3. PREDICTION ROUTE ---
@app.route('/analyze', methods=['POST'])
def analyze_xray():
    if 'xray_image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['xray_image']
    
    try:
        # Preprocess Image
        img = Image.open(io.BytesIO(file.read())).convert('RGB').resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # TFLite Inference Step
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get result
        prediction_probs = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = np.argmax(prediction_probs)
        predicted_label = class_names[predicted_index]
        confidence = float(prediction_probs[predicted_index])

        # Confidence threshold logic
        if predicted_label == "COVID" and confidence < 0.85:
            predicted_label = "Normal"

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "all_probabilities": {
                class_names[i]: float(prediction_probs[i]) for i in range(len(class_names))
            }
        })

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)