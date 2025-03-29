import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# EDA - Check shapes and class distribution
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)
print("Unique classes:", np.unique(train_labels))

# Display sample images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Preprocessing: Normalize pixel values (0-1) and reshape
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape to (28, 28, 1) to match CNN input
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# Split training data into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Convert datasets to tf.data for better performance
def create_tf_dataset(images, labels, batch_size=32, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(10000).map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_tf_dataset(train_images, train_labels, training=True)
val_ds = create_tf_dataset(val_images, val_labels)
test_ds = create_tf_dataset(test_images, test_labels)

print("Train/Val/Test sizes:", len(train_ds), len(val_ds), len(test_ds))