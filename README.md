# fashion-mnist-classification-api
A deep learning-based Flask API for Fashion MNIST image classification using EfficientNetB0. The project includes model training, evaluation, and an authentication-protected endpoint for predicting fashion items from uploaded images.
# Fashion MNIST Image Classification API

This repository contains an end-to-end **Fashion MNIST** image classification system. It includes:
- A deep learning model built using **EfficientNetB0**
- A **Flask API** for image classification
- Preprocessing and dataset handling using **TensorFlow**

## 📌 Project Overview
This project classifies images from the **Fashion MNIST dataset** into 10 categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## 🛠 Installation
Clone the repository and install the dependencies:
```bash
# Clone the repository
git clone https://github.com/your-username/fashion-mnist-api.git
cd fashion-mnist-api

# Install dependencies
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model
To train the model, run:
```bash
python train.py
```
This script:
- Loads the Fashion MNIST dataset
- Preprocesses and augments the data
- Trains an **EfficientNetB0** model
- Saves the trained model as `fashion_mnist_model.keras`

## 🚀 Running the Flask API
Start the API server using:
```bash
python app.py
```

### API Endpoints
#### 1️⃣ Upload & Predict
**Endpoint:** `POST /predict`

**Authentication:** Basic Auth (`admin:password`)

**Request:** Upload an image (`.png`, `.jpg`, `.jpeg`)
```bash
curl -X POST -F "username=admin" -F "password=password" -F "image=@sample.png" http://127.0.0.1:5000/predict
```

**Response:**
```json
{
  "predicted_class": "Sneaker"
}
```

## 📂 File Structure
```
📂 fashion-mnist-api
│── app.py            # Flask API for image classification
│── train.py          # Model training script
│── preprocess.py     # Data preprocessing
│── requirements.txt  # Dependencies
│── fashion_mnist_model.keras  # Trained model (generated after training)
│── README.md         # Documentation
```

## 📢 Contributing
Feel free to open issues and contribute to improving this project. 🚀

## 📜 License
This project is licensed under the **MIT License**.

