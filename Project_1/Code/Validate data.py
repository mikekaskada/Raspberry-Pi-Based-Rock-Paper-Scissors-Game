import os
import shutil
import numpy as np
from keras.models import load_model
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Directory containing the validation images
validation_dir = 'validation_data'

# Create directories if they do not exist within the validation_data directory
output_dirs = ["paper", "scissors", "rock"]
for dir_name in output_dirs:
    os.makedirs(os.path.join(validation_dir, dir_name), exist_ok=True)

# List all PNG files in the validation directory
all_images = [f for f in os.listdir(validation_dir) if f.endswith('.png')]

for image_file in all_images:
    image_path = os.path.join(validation_dir, image_file)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predict the image class
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")

    # Move the image to the corresponding folder
    output_dir = os.path.join(validation_dir, output_dirs[index])
    shutil.move(image_path, os.path.join(output_dir, image_file))

    print(f"Moved image {image_file} to {output_dir}")

print("All images processed and moved successfully!")
