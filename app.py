from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from functools import wraps
import pickle
import os
import io
from PIL import Image
import numpy as np
import base64
import tensorflow as tf 


app = Flask(__name__)

# --- Basic Auth ---
USERNAME = 'admin'
PASSWORD = 'password'

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    message = {'message': "Authentication required."}
    resp = jsonify(message)
    resp.status_code = 401
    return resp

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# --- Load model ---
# Load a pretrained model
model = tf.keras.models.load_model("fashion_mnist_model.keras")

# Preprocess image (adjust depending on your model)
def preprocess_image(image):
    image = image.convert('L')            # Convert to grayscale (1 channel)
    image = image.resize((28, 28))        # Resize to 28x28
    image = np.array(image) / 255.0       # Normalize
    image = image.reshape(1, 28, 28, 1)    # Reshape to (1, 28, 28, 1)
    return image


# --- Predict endpoint ---
@app.route('/', methods=['GET'])
def upload_form():
    return '''
        <html>
            <body>
                <h2>Upload Image for Prediction</h2>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    Username: <input type="text" name="username"><br>
                    Password: <input type="password" name="password"><br>
                    <input type="file" name="image"><br><br>
                    <input type="submit" value="Predict">
                </form>
            </body>
        </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form.get('username')
    password = request.form.get('password')

    if not check_auth(username, password):
        return jsonify({'error': 'Unauthorized'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)

    # Class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
