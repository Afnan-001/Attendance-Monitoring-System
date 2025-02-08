import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

# Load the CSV file
csv_path = r"C:\Users\Arth\OneDrive\Desktop\CameraSnaps\UpdatedRatings1.csv" # Update with your actual path
data = pd.read_csv(csv_path)

# Parameters
IMAGE_SIZE = (224, 224)  # Resizingh all images to 224x224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Function to preprocess and load images
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=IMAGE_SIZE)  # Load and    resize image
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = preprocess_input(img_array)  # Preprocess for ResNet50
        images.append(img_array)
    return np.array(images)

# Prepare data
image_paths = data['Photo'].tolist()
ratings = data['rating'].values

# Split data into train, validation, and test sets
train_paths, temp_paths, train_ratings, temp_ratings = train_test_split(
    image_paths, ratings, test_size=0.3, random_state=42
)
val_paths, test_paths, val_ratings, test_ratings = train_test_split(
    temp_paths, temp_ratings, test_size=0.5, random_state=42
)

# Preprocess images
X_train = preprocess_images(train_paths)
X_val = preprocess_images(val_paths)
X_test = preprocess_images(test_paths)

y_train = np.array(train_ratings)
y_val = np.array(val_ratings)
y_test = np.array(test_ratings)

# Build the model using a pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduce feature maps to a single vector
    Dense(256, activation='relu'),  # Add a dense layer
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss=MeanSquaredError(),
              metrics=['mae'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the model
model.save('image_rating_model2.h5')
print("Model saved as 'image_rating_model2.h5'")
