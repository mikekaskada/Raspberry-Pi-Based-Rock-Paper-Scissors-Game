import os
from PIL import Image

# Define source directories
source_dirs = ["paper", "scissors", "rock"]

# Define destination directories
base_rotated_dir = "rotated_images"
base_final_dir = "final_images"

# Create base destination directories if they don't exist
os.makedirs(base_rotated_dir, exist_ok=True)
os.makedirs(base_final_dir, exist_ok=True)

# Define size, rotation angles, and flip operations
resize_size = (224, 224)
angles = [0, 90, 180, 270]
flip_operations = ['horizontal', 'vertical', 'both']

# Function to perform flipping operations
def flip_image(image, operation):
    if operation == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif operation == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif operation == 'both':
        return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

# Process each source directory
for source_dir in source_dirs:
    # Create specific directories for each source
    rotated_dir = os.path.join(base_rotated_dir, source_dir)
    final_dir = os.path.join(base_final_dir, source_dir)
    
    os.makedirs(rotated_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    # Rotate and resize images, then save them in the rotated_dir
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".png"):
            image_path = os.path.join(source_dir, file_name)
            image = Image.open(image_path)
            resized_image = image.resize(resize_size)
            for angle in angles:
                rotated_image = resized_image.rotate(angle, expand=True)
                rotated_file_name = f"{os.path.splitext(file_name)[0]}_rotated_{angle}.png"
                rotated_image.save(os.path.join(rotated_dir, rotated_file_name))

    # Flip the rotated images and save them in the final_dir
    for file_name in os.listdir(rotated_dir):
        if file_name.endswith(".png"):
            image_path = os.path.join(rotated_dir, file_name)
            image = Image.open(image_path)
            # Save original rotated image to final_dir
            image.save(os.path.join(final_dir, file_name))
            # Save flipped images to final_dir
            for operation in flip_operations:
                flipped_image = flip_image(image, operation)
                flipped_file_name = f"{os.path.splitext(file_name)[0]}_flipped_{operation}.png"
                flipped_image.save(os.path.join(final_dir, flipped_file_name))

print("Process completed. Images have been resized, rotated, flipped, and saved to their respective final directories.")
