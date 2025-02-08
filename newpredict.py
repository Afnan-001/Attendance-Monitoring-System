import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('image_rating_model2.h5') # Replace with your model's path

# Define image size and the path to your images
IMAGE_SIZE = (224, 224)
PHOTO_FOLDER_PATH = r"C:\Users\Arth\OneDrive\Desktop\CameraSnaps\Sumit2\Ashutoshfaces\Ashutosh_faces\Ashutosh_faces"  # Update with your folder path

# Function to preprocess and load images
def preprocess_images_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
            image_paths.append(os.path.join(folder_path, filename))
    
    images = []
    for path in image_paths:
        img = load_img(path, target_size=IMAGE_SIZE)  # Load and resize image
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = preprocess_input(img_array)  # Preprocess for ResNet50
        images.append(img_array)
    
    return np.array(images), image_paths

# Load and preprocess images from the folder
X_images, image_paths = preprocess_images_from_folder(PHOTO_FOLDER_PATH)

# Predict ratings using the trained model
predictions = model.predict(X_images)

# Clip predictions to be between 0 and 7 and convert them to integers
predictions = np.clip(np.round(predictions.flatten()), 0, 7).astype(int)

# Create a DataFrame with image paths and predicted ratings
results_df = pd.DataFrame({
    'Image Path': image_paths,
    'Predicted Rating': predictions
})

# Save the results to a CSV file
results_df.to_csv('predicted_ratings_from.csv', index=False)

print("Predicted ratings for images saved to 'predicted_ratings_from_folder.csv'")
