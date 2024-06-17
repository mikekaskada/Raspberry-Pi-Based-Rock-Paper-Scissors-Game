import cv2
import numpy as np
from picamera2 import Picamera2
import os
from datetime import datetime

cv2.startWindowThread()

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (800, 600)}))
picam2.start()

# Create directories if they don't exist
os.makedirs("paper", exist_ok=True)
os.makedirs("scissors", exist_ok=True)
os.makedirs("rock", exist_ok=True)

# Function to capture image within the square and resize
def capture_image(im, x, y, label, square_size=300, target_size=224):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.png"
    square = im[y:y+square_size, x:x+square_size]
    resized_square = cv2.resize(square, (target_size, target_size))
    cv2.imwrite(os.path.join(label, filename), resized_square)

# Define the size of the square
square_size = 300

# Main loop
while True:
    im = picam2.capture_array()

    # Get the center coordinates for the square
    height, width, _ = im.shape
    center_x, center_y = width // 2, height // 2
    top_left_x = center_x - square_size // 2
    top_left_y = center_y - square_size // 2

    # Draw the square in the center
    cv2.rectangle(im, (top_left_x, top_left_y), (top_left_x + square_size, top_left_y + square_size), (0, 255, 0), 2)

    # Flip the image horizontally 
    horizontal_flip = cv2.flip(im, 1)
    # vertical_flip = cv2.flip(im, 0)

    # Display the original and flipped images
    # cv2.imshow("Original Camera", im)
    cv2.imshow("Horizontal Flip", horizontal_flip)
    # cv2.imshow("Vertical Flip", vertical_flip)

    key = cv2.waitKey(1)

    if key in [ord('p'), ord('P')]:  # 'P' or 'p' to capture image for paper
        capture_image(im, top_left_x, top_left_y, "paper", square_size)
    elif key in [ord('s'), ord('S')]:  # 'S' or 's' to capture image for scissors
        capture_image(im, top_left_x, top_left_y, "scissors", square_size)
    elif key in [ord('r'), ord('R')]:  # 'R' or 'r' to capture image for rock
        capture_image(im, top_left_x, top_left_y, "rock", square_size)
    elif key == 27:  # Escape key to exit
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
