#Data Cleaning For Yoga-82 dataset.
import os
from PIL import Image  

# Define the base directory for all pose categories
dataset_path = r"C:\Users\Rimjhim\Desktop\Yoga Pose Detection and Correction"

# Check if the directory exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The directory '{dataset_path}' does not exist. Please check the path.")

# Function to clean corrupted images
def clean_data(path):
    """
    Iterates through each category and removes corrupted image files.
    """
    removed_files = 0  # Counter for removed files
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if not os.path.isdir(category_path):  # Skip non-directory entries
            continue

        print(f"Processing category: {category}")
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if not os.path.isfile(file_path):  # Skip invalid entries
                continue

            try:
                with Image.open(file_path) as img:  # Open the image to verify
                    img.verify()  # Verify that the image file is valid
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)
                removed_files += 1

    print(f"Data cleaning completed. Removed {removed_files} corrupted files.")

# Clean the data
clean_data(dataset_path)
