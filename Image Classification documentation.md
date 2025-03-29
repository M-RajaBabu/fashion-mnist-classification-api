# Image Classification Model Documentation

## Project Overview
This project involves building and deploying an image classification model using deep learning techniques. The model is trained on the Fashion MNIST dataset and deployed as a REST API using Flask.

## Dataset
### Fashion MNIST
- The dataset consists of grayscale images (28x28 pixels) belonging to 10 different classes:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- Training set: 60,000 images
- Test set: 10,000 images

## Data Preprocessing
### Steps:
1. **Data Loading**
   - The dataset is loaded using TensorFlow's `fashion_mnist` module.
2. **Exploratory Data Analysis (EDA)**
   - Printed dataset shapes and unique classes.
   - Visualized sample images using Matplotlib.
3. **Normalization & Reshaping**
   - Normalized pixel values to range [0,1].
   - Reshaped images to (28,28,1) to match CNN input requirements.
4. **Data Splitting**
   - Split training data into training (90%) and validation (10%) sets.
5. **Data Augmentation**
   - Applied transformations such as rotation, width/height shift, and zoom.
6. **TensorFlow Dataset Conversion**
   - Converted dataset into `tf.data.Dataset` for better performance.

## Model Training
### Model Architecture:
- Used **EfficientNetB0** (without pre-trained weights).
- Added **GlobalAveragePooling2D** and **Dense layers** for classification.
- Output layer with **softmax activation** for multi-class classification.

### Training Details:
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Callbacks:**
  - Early stopping (monitoring validation loss, patience=3)
  - Learning rate scheduling (ReduceLROnPlateau, factor=0.5, patience=2)
- **Epochs:** 15
- **Evaluation:**
  - Achieved test accuracy is printed after evaluation.
- **Model Saving:**
  - The trained model is saved in `fashion_mnist_model.keras` format.

## Model Deployment
The model is deployed as a REST API using Flask.
### API Endpoints
1. **GET /**
   - Returns an HTML form to upload an image for classification.
   
2. **POST /predict**
   - Accepts an image file and performs classification.
   - Requires authentication via username and password.
   - Returns the predicted class of the image.
   
### Authentication
- Basic authentication is used.
- Default credentials:
  - Username: `admin`
  - Password: `password`

### Running the API Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Access the API at `http://127.0.0.1:5000/`

## Repository Structure
- `preprocess.py`: Data loading, preprocessing, augmentation.
- `train.py`: Model training and evaluation.
- `app.py`: API for model inference.
- `requirements.txt`: Dependencies.
- `README.md`: Instructions to run the project.
- `fashion_mnist_model.keras`: Saved model file.

## Future Enhancements
- Improve authentication mechanisms.
- Deploy on a cloud platform like AWS/GCP/Azure.
- Implement logging and monitoring for API requests.

