from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


model = tf.keras.models.load_model("cnn_model.h5")
inv_map = {0: "Real Image", 1: "AI Generated Image"}
IMG_SIZE = (128, 128)

app = Flask(__name__)

def preprocess_image(file):
    """Read and preprocess uploaded image"""
    try:
        file.seek(0)  # reset pointer
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        return np.expand_dims(img, axis=0)   # (1,128,128,3)

    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        img = preprocess_image(file)
        preds = model.predict(img)
        print(preds)
        label = np.argmax(preds, axis=1)[0]
        confidence = preds[0].tolist()
        prediction = inv_map[label]
        
        return jsonify({"label": prediction, "confidence": confidence})
        
    return render_template("index.html", label=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)