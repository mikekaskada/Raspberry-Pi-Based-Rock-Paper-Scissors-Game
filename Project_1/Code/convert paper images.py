import os
from PIL import Image

# Define input and output directories
input_dir = "paper"
output_dir = "Paper_New"

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Desired size after cropping
desired_width = 224
desired_height = 200

# Amount to crop from left and right
left_crop = 12
right_crop = 64

# Loop through input directory
for file_name in os.listdir(input_dir):
    # Check if file has PNG extension
    if file_name.endswith(".png"):
        # Open image
        image_path = os.path.join(input_dir, file_name)
        img = Image.open(image_path)

        # Calculate cropping dimensions
        width, height = img.size
        left = left_crop
        top = 0
        right = width - right_crop
        bottom = height

        # Crop image
        img_cropped = img.crop((left, top, right, bottom))

        # Resize image to desired dimensions
        img_resized = img_cropped.resize((desired_width, desired_height))

        # Save resized image to output directory
        output_path = os.path.join(output_dir, file_name)
        img_resized.save(output_path)

print("Cropping and resizing complete.")
