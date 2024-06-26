import os
import random
import shutil

def copy_and_rename_images(source_dir, target_dir, num_images, prefix):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # List all PNG files in the source directory
    all_images = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    # Randomly select the specified number of images
    selected_images = random.sample(all_images, num_images)

    # Copy and rename the selected images to the target directory
    for i, image in enumerate(selected_images):
        new_file_name = f"{prefix}_{i+1:04d}.png"
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, new_file_name)
        shutil.copy(source_path, target_path)

# Directories
root_folder = "final_images"
paper_dir = os.path.join(root_folder, 'paper')
scissors_dir = os.path.join(root_folder, 'scissors')
rock_dir = os.path.join(root_folder, 'rock')
validation_dir = 'validation_data'

# Number of images to copy from each directory
num_images_to_copy = 5000

# Copy and rename images
copy_and_rename_images(paper_dir, validation_dir, num_images_to_copy, 'paper')
copy_and_rename_images(scissors_dir, validation_dir, num_images_to_copy, 'scissors')
copy_and_rename_images(rock_dir, validation_dir, num_images_to_copy, 'rock')

print("Images copied and renamed successfully!")
