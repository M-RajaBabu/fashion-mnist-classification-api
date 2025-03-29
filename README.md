# Fashion MNIST Classification API

## Overview
This project implements an image classification model using **EfficientNetB0** to classify images from the **Fashion MNIST dataset**. The model is deployed as a **Flask API**, allowing users to upload images and receive predictions.A deep learning-based Flask API for Fashion MNIST image classification using EfficientNetB0. The project includes model training, evaluation, and an authentication-protected endpoint for predicting fashion items from uploaded images.

## Features
- **Deep Learning Model:** Uses EfficientNetB0 for high-accuracy classification.
- **Preprocessing Pipeline:** Normalization, augmentation, and dataset handling.
- **Flask API:** Exposes an endpoint for image classification.
- **Authentication:** Basic username-password authentication.

## Dataset
The dataset used is **Fashion MNIST**, which consists of grayscale images (28x28) belonging to 10 different classes:
- `T-shirt/top`
- `Trouser`
- `Pullover`
- `Dress`
- `Coat`
- `Sandal`
- `Shirt`
- `Sneaker`
- `Bag`
- `Ankle boot`

## Project Structure
```
├── app.py                 # Flask API for image classification
├── train.py               # Model training script
├── preprocess.py          # Data preprocessing and augmentation
├── fashion_mnist_model.keras  # Trained model
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
```

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/M-RajaBabu/fashion-mnist-classification-api.git
cd fashion-mnist-classification-api
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you want to train the model from scratch:
```bash
python train.py
```

### 4. Run the Flask API
```bash
python app.py
```
The API will be available at `http://127.0.0.1:5000/`

## Usage
### 1. Upload an Image for Prediction
You can access the API in two ways:

#### **Via Web Interface**
- Open `http://127.0.0.1:5000/` in a browser.
- Enter credentials and upload an image.

#### **Via cURL**
```bash
curl -X POST -F "image=@path/to/image.png" -F "username=admin" -F "password=password" http://127.0.0.1:5000/predict
```

#### **Via Python**
```python
import requests
files = {'image': open('test_image.png', 'rb')}
data = {'username': 'admin', 'password': 'password'}
response = requests.post('http://127.0.0.1:5000/predict', files=files, data=data)
print(response.json())
```

## Authentication
The API requires authentication. Use:
- **Username:** `admin`
- **Password:** `password`

## Requirements
- Python 3.x
- TensorFlow
- Flask
- NumPy
- Pillow
- Scikit-learn
- Matplotlib

## License
This project is open-source and available under the MIT License.

## Author
**Raja Babu Meena**



