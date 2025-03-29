# Build model using EfficientNet
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from preprocess import train_ds, val_ds, test_ds

input_layer = Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(input_layer)  # To make it RGB
x = EfficientNetB0(include_top=False, input_tensor=x, weights=None)
x = GlobalAveragePooling2D()(x.output)
x = Dense(128, activation='relu')(x)
out = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=out)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=15,
                    callbacks=[early_stop, lr_schedule])

# Evaluate model
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save("fashion_mnist_model.keras")


