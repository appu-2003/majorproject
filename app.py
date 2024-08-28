from flask import Flask, request, jsonify
from PIL import Image
import pickle
import numpy as np
import io

app = Flask(__name__)
@app.route('/',methods=['GET'])
def hello_world():
    return "hello world"
    

# Load your pickled model
with open(r"D:\detection_flask_app\.venv\efficient_model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define your class names
class_names = {
    0: "VeryMildDemented",
    1: "NonDemented",
    2: "ModerateDemented",
    3: "MildDemented",
    # Add all other classes
}

def preprocess_image(image):
    # Preprocess the image (resize, normalize, etc.) to match the input requirements of your model
    image = image.resize((224, 224))  # Example: resizing to 224x224 pixels
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Convert the uploaded file into an image
    image = Image.open(file.stream).convert('RGB')
    processed_image = preprocess_image(image)

    # Make a prediction using the loaded model
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)  # Get the index of the highest confidence class
    predicted_disease = class_names.get(predicted_index, "Unknown Disease")

    return jsonify({"predicted_disease": predicted_disease})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    