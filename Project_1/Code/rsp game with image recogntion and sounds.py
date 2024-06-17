import cv2
import numpy as np
from picamera2 import Picamera2
import time
from keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pygame
import os
from datetime import datetime

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load the sounds
countdown_sound = pygame.mixer.Sound("countdown.wav")
rock_sound = pygame.mixer.Sound("rock.wav")
scissors_sound = pygame.mixer.Sound("scissors.wav")
paper_sound = pygame.mixer.Sound("paper.wav")
computer_wins_sound = pygame.mixer.Sound("computer_wins.wav")
human_wins_sound = pygame.mixer.Sound("human_wins.wav")
tie_sound = pygame.mixer.Sound("tie.wav")

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (500, 500)}))
picam2.start()

# Load the Keras model
model = load_model("keras_model.h5", compile=False)

# Load the labels and extract gesture names
class_names = [line.strip().split()[1] for line in open("labels.txt", "r").readlines()]

# Define the size of the square
square_size = 300

# Variables to keep track of the score and rounds
human_score = 0
computer_score = 0
round_count = 0
total_rounds = 5

# Ensure directories for storing incorrect predictions exist
for class_name in class_names:
    if not os.path.exists(class_name):
        os.makedirs(class_name)

# Function to capture image within the square and resize
def capture_image(im, x, y, square_size=300, target_size=224):
    square = im[y:y+square_size, x:x+square_size]
    resized_square = cv2.resize(square, (target_size, target_size))
    # Convert the image to RGB if it has an alpha channel
    if resized_square.shape[2] == 4:
        resized_square = cv2.cvtColor(resized_square, cv2.COLOR_BGRA2BGR)
    return resized_square

# Function to predict the class of an image
def predict_image(image):
    image = np.asarray(image, dtype=np.float32)
    image = image.reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Function to display a message box with the result
def show_result(result):
    messagebox.showinfo("Result", result)

# Function to play the appropriate sound for the given choice
def play_choice_sound(choice):
    if choice == 'rock':
        rock_sound.play()
    elif choice == 'scissors':
        scissors_sound.play()
    elif choice == 'paper':
        paper_sound.play()

# Function to capture and predict the gesture
def capture_and_predict():
    global human_choice, round_count, human_score, computer_score, original_choice, im
    if round_count >= total_rounds:
        final_result = f"Game Over! Final Score:\nHuman: {human_score}\nComputer: {computer_score}"
        show_result(final_result)
        root.quit()
        return

    round_count += 1
    print(f"Round {round_count} - Get ready to make your move!")
    pygame.mixer.Sound.play(countdown_sound)  # Play countdown sound
    for i in range(3, 0, -1):
        print(f"{i}...")
        countdown_label.config(text=f"{i}...")
        root.update()
        time.sleep(1)

    # Capture an image and predict the class
    im = picam2.capture_array()
    image = capture_image(im, top_left_x, top_left_y, square_size)
    original_choice, confidence = predict_image(image)
    human_choice = original_choice  # Initialize human_choice with original prediction
    print(f"Predicted move: {human_choice} with confidence {confidence * 100:.2f}%")
    play_choice_sound(human_choice)  # Play human choice sound

    # Display the predicted choice and allow the user to correct it
    prediction_label.config(text=f"Predicted: {human_choice}")
    correction_label.config(text="Press 'p', 's', 'r' to correct, 'space' to confirm")
    root.update()

    # Save image for future use if prediction is corrected
    return im, image

# Function to handle key presses for correction and confirmation
def handle_key_press(event):
    global human_choice, human_score, computer_score, round_count, original_choice, im
    if event.char in ['p', 'P']:
        human_choice = 'paper'
        prediction_label.config(text=f"Corrected: {human_choice}")
    elif event.char in ['s', 'S']:
        human_choice = 'scissors'
        prediction_label.config(text=f"Corrected: {human_choice}")
    elif event.char in ['r', 'R']:
        human_choice = 'rock'
        prediction_label.config(text=f"Corrected: {human_choice}")
    elif event.char == ' ':
        if human_choice != original_choice:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{human_choice}_{timestamp}.png"
            filepath = os.path.join(human_choice, filename)
            cv2.imwrite(filepath, im)

        computer_choice = np.random.choice(['paper', 'scissors', 'rock'])
        print(f"Computer move: {computer_choice}")
        play_choice_sound(computer_choice)  # Play computer choice sound
        time.sleep(1)

        # Debug: Print choices
        print(f"Human choice: {human_choice}, Computer choice: {computer_choice}")

        # Determine the winner
        if human_choice == computer_choice:
            result = "It's a tie!"
            pygame.mixer.Sound.play(tie_sound)  # Play tie sound
        elif (human_choice == 'rock' and computer_choice == 'scissors') or \
             (human_choice == 'scissors' and computer_choice == 'paper') or \
             (human_choice == 'paper' and computer_choice == 'rock'):
            result = "Human wins!"
            human_score += 1
            pygame.mixer.Sound.play(human_wins_sound)  # Play human wins sound
        else:
            result = "Computer wins!"
            computer_score += 1
            pygame.mixer.Sound.play(computer_wins_sound)  # Play computer wins sound

        # Debug: Print results and scores
        print(f"Round result: {result}")
        print(f"Current Score - Human: {human_score}, Computer: {computer_score}")

        result_label.config(text=f"Human: {human_choice} | Computer: {computer_choice} | {result}")
        score_label.config(text=f"Round: {round_count}/{total_rounds} | Human: {human_score} | Computer: {computer_score}")
        show_result(result)
        if round_count < total_rounds:
            im, image = capture_and_predict()
        else:
            final_result = f"Game Over! Final Score:\nHuman: {human_score}\nComputer: {computer_score}"
            show_result(final_result)
            root.quit()

# Initialize GUI
root = tk.Tk()
root.title("Rock-Paper-Scissors Game")

# Set up GUI elements
countdown_label = tk.Label(root, text="Press 'c' to start the countdown", font=("Helvetica", 16))
countdown_label.pack()

prediction_label = tk.Label(root, text="", font=("Helvetica", 16))
prediction_label.pack()

correction_label = tk.Label(root, text="", font=("Helvetica", 12))
correction_label.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

score_label = tk.Label(root, text=f"Round: {round_count}/{total_rounds} | Human: {human_score} | Computer: {computer_score}", font=("Helvetica", 16))
score_label.pack()

# Bind key press events
root.bind('<KeyPress>', handle_key_press)

# Main loop to update camera feed
def update_frame():
    global top_left_x, top_left_y
    im = picam2.capture_array()
    height, width, _ = im.shape
    center_x, center_y = width // 2, height // 2
    top_left_x = center_x - square_size // 2
    top_left_y = center_y - square_size // 2

    # Draw the square in the center
    cv2.rectangle(im, (top_left_x, top_left_y), (top_left_x + square_size, top_left_y + square_size), (0, 255, 0), 2)
    horizontal_flip = cv2.flip(im, 1)
    
    # Convert the image to RGB and then to a format Tkinter can use
    horizontal_flip = cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(horizontal_flip)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.config(image=imgtk)
    camera_label.after(10, update_frame)

camera_label = tk.Label(root)
camera_label.pack()

start_button = tk.Button(root, text="Start Countdown", command=capture_and_predict, font=("Helvetica", 16))
start_button.pack()

update_frame()
root.mainloop()
