import os
import random
import shutil

# Define the directories
source_dir = "final_images"
target_dir = "Training_Data"
subdirs = ["paper", "scissors", "rock"]

# Ensure the target directories exist
for subdir in subdirs:
    os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)

# Function to move and rename random files
def move_and_rename_random_files(source_subdir, target_subdir, file_count):
    source_path = os.path.join(source_dir, source_subdir)
    target_path = os.path.join(target_dir, source_subdir)
    
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Source directory {source_path} does not exist.")
        return
    
    # Get list of all PNG files in the source directory
    files = [f for f in os.listdir(source_path) if f.endswith('.png')]
    
    # Check if there are enough files to move
    if len(files) < file_count:
        print(f"Not enough files in {source_path} to move. Found {len(files)} files.")
        return
    
    # Select random files
    selected_files = random.sample(files, file_count)
    
    # Move and rename files to target directory
    for count, file in enumerate(selected_files, start=1):
        new_name = f"{source_subdir}_{count:04d}.png"
        shutil.move(os.path.join(source_path, file), os.path.join(target_path, new_name))

# Move and rename 1500 random files from each subdirectory
for subdir in subdirs:
    move_and_rename_random_files(subdir, subdir, 8000)

print("Random selection, moving, and renaming of files is complete.")
