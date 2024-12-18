#Preparation of Training Data that would be used in GUI

import os
import glob
import cv2
import numpy as np

# Set the dimensions for the image (already consistent as 90x140)
rezimgw = 90
rezimgh = 140

# Initialize the digit characters for classification (0-9)
char_digits = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]  # ASCII codes for 0-9

# Dataset folder path
dataset_folder = "dataset"

# Initialize containers for storing flattened images and classifications
flattenedimg = np.empty((0, rezimgw * rezimgh), dtype=np.float32)  # Empty array for flattened images
intClassification = []  #List for storing digit labels

# Counter for processed images
total_images = 0

print("Starting image classification process...\n")

# Iterate over each folder corresponding to digits 0-9
for i, char in enumerate(char_digits):
    digit_folder = os.path.join(dataset_folder, str(i))

    if not os.path.exists(digit_folder):
        print(f"Warning: Folder '{digit_folder}' does not exist.")
        continue

    # Retrieve all .jpwg files in the folder
    image_files = glob.glob(os.path.join(digit_folder, '*.jpg'))
    for image_file in image_files:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        if image is None:
            print(f"Error: Unable to read image '{image_file}'")
            continue

        # Ensure the image size is 90x140
        if image.shape != (rezimgh, rezimgw):
            print(f"Skipping '{image_file}': Size {image.shape} does not match 90x140.")
            continue

        # Threshold the image (binarization)
        _, img_thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

        # Flatten the image and append it to the dataset
        flatimg = img_thresh.reshape(1, rezimgw * rezimgh).astype(np.float32)
        flattenedimg = np.append(flattenedimg, flatimg, axis=0)

        # Append the corresponding label (digit) to the classification list
        intClassification.append(char)
        total_images += 1

        print(f"Image '{image_file}' classified as '{chr(char)}'.")

# Convert the classification list to a numpy array and reshape
fltClassification = np.array(intClassification, dtype=np.float32)
finalClassification = fltClassification.reshape(fltClassification.size, 1)

# Save the training data to text files
np.savetxt("classification.txt", finalClassification)
np.savetxt("flatcharimages.txt", flattenedimg)

print("\n--- Classification Complete ---")
print(f"Total Images Processed: {total_images}")
print("Training data saved to 'classification.txt' and 'flatcharimages.txt'.")
