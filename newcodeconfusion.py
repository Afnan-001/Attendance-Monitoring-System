import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Parameters
CSV_PATH = r"C:\Users\Arth\OneDrive\Desktop\CameraSnaps\FinalCombinedRatingsCpy.csv"  # Original CSV file path
MODEL_PATH = r"C:\Users\Arth\OneDrive\Desktop\CameraSnaps\image_rating_model.h5"  # Model file path
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
KFOLDS = 10

# Load data
data = pd.read_csv(CSV_PATH)
image_paths = data['Photo'].tolist()
ratings = data['rating'].values

# Function to preprocess images
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=IMAGE_SIZE)  # Load and resize image
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = preprocess_input(img_array)  # Preprocess for ResNet50
        images.append(img_array)
    return np.array(images)

# Load the pre-trained model
model = load_model(MODEL_PATH)

# Initialize KFold cross-validation
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

# Perform K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"\nProcessing Fold {fold + 1}/{KFOLDS}")
    
    # Split data into training and validation sets
    train_paths = [image_paths[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    y_train = ratings[train_idx]
    y_val = ratings[val_idx]
    
    # Preprocess images
    X_val = preprocess_images(val_paths)
    
    # Predict on the validation set
    y_val_pred = model.predict(X_val, batch_size=BATCH_SIZE)
    y_val_pred_rounded = np.rint(y_val_pred).astype(int)  # Round predictions to nearest integer
    
    # Update the original dataset with predicted ratings for the current fold
    for i, idx in enumerate(val_idx):
        data.loc[idx, 'rating'] = y_val_pred_rounded[i, 0]
    
    # Compute confusion matrix for the current fold
    conf_matrix = confusion_matrix(y_val, y_val_pred_rounded)
    accuracy = accuracy_score(y_val, y_val_pred_rounded)
    
    print(f"Confusion Matrix for Fold {fold + 1}:\n{conf_matrix}")
    print(f"Accuracy for Fold {fold + 1}: {accuracy:.4f}")
    
    # Save the updated dataset after each fold
    UPDATED_CSV_PATH = r"C:\Users\Arth\OneDrive\Desktop\CameraSnaps\UpdatedRatings1.csv"
    data.to_csv(UPDATED_CSV_PATH, index=False)
    print(f"Updated dataset saved after Fold {fold + 1} to {UPDATED_CSV_PATH}")

# Final Dataset Updated
print("\nAll folds processed. The final dataset with updated ratings has been saved.")
