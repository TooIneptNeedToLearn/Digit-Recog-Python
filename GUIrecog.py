import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageGrab

# Load the KNN model
rezimgw = 90  # Match with main.py
rezimgh = 140  # Match with main.py

# Load classification and flattened image data
try:
    charClassification = np.loadtxt("classification.txt", np.float32)
    flatCharImages = np.loadtxt("flatcharimages.txt", np.float32)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model files: {e}")
    exit()

charClassification = charClassification.reshape(charClassification.size, 1)

# Train the KNN model
knn = cv2.ml.KNearest_create() #Create KNN Model
knn.train(flatCharImages, cv2.ml.ROW_SAMPLE, charClassification) #Train using flatcharimages with the


# Function to preprocess the canvas image for KNN
def process_canvas_image(canvas):
    # Save canvas content to an image
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    
    # Grab the canvas area
    img = ImageGrab.grab(bbox=(x, y, x1, y1)).convert('L')  # Grayscale
    img = np.array(img)
    
    # Threshold the image to binary (invert colors)
    _, img_thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the digits
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        messagebox.showinfo("Recognition Result", "No digits detected!")
        return

    # Sort contours from left to right (for multiple digits)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    recognized_digits = []

    # Process each contour and classify
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        
        # Ignore small contours (noise)
        if w < 10 or h < 10:
            continue

        # Extract ROI (Region of Interest)
        roi = img_thresh[y:y + h, x:x + w]

        # Resize ROI to match training dimensions
        try:
            roi_resized = cv2.resize(roi, (rezimgw, rezimgh))  # Resize ROI to 90x140
            roi_flattened = roi_resized.reshape((1, rezimgw * rezimgh)).astype(np.float32)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resize ROI: {e}")
            return

        # Flatten the image and convert to float32
        roi_flattened = roi_resized.reshape((1, rezimgw * rezimgh)).astype(np.float32)
        
        # Check dimensions
        if roi_flattened.shape[1] != flatCharImages.shape[1]:
            messagebox.showerror("Error", "ROI dimensions do not match training data dimensions.")
            return

        # Recognize digit using KNN
        _, result, _, _ = knn.findNearest(roi_flattened, k=1)  # k=1 for nearest neighbor
        recognized_digit = chr(int(result[0][0]))  # Convert ASCII code to character
        recognized_digits.append(recognized_digit)

    # Join recognized digits and display the result
    result_string = "".join(recognized_digits)
    messagebox.showinfo("Recognition Result", f"Recognized Digits: {result_string}")


# Function to clear the canvas
def clear_canvas(canvas):
    canvas.delete("all")


# GUI Setup
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Create a canvas for drawing
canvas = tk.Canvas(root, bg="white", width=300, height=300)
canvas.pack()

# Drawing functionality
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)

canvas.bind("<B1-Motion>", paint)

# Buttons
btn_recognize = tk.Button(root, text="Recognize Digits", command=lambda: process_canvas_image(canvas))
btn_recognize.pack(pady=5)

btn_clear = tk.Button(root, text="Clear Canvas", command=lambda: clear_canvas(canvas))
btn_clear.pack(pady=5)

root.mainloop()
