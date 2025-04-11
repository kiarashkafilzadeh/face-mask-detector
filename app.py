from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("mask_model.h5")
labels = ["No Mask", "Mask"]

html_form = """
    <h1>Upload a Face Image</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <input type="submit" value="Predict">
    </form>
    <div class="footer">
        <p>Made by <strong>Amirhossein Kafilzadeh</strong></p>
    </div>
    <style>
        .footer {
            position: absolute;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px 0;
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            color: #555;
        }
        .footer p {
            margin: 0;
            color: #333;
        }
        .footer strong {
            color: #2196F3;
        }
    </style>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file.stream)
        image = image.resize((28, 28)).convert("RGB")
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        class_id = np.argmax(pred)
        result = labels[class_id]
        return html_form + f"<h2>Prediction: {result}</h2>"

    return html_form

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
