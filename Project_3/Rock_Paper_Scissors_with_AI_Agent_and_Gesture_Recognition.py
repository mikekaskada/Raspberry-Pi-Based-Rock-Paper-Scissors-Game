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
import random
import csv
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Function to load sounds with error handling
def load_sound(filename):
    try:
        return pygame.mixer.Sound(filename)
    except pygame.error as e:
        print(f"Error loading sound: {filename}, {e}")
        return None

# Load the sounds
countdown_sound = load_sound("countdown.wav")
rock_sound = load_sound("rock.wav")
scissors_sound = load_sound("scissors.wav")
paper_sound = load_sound("paper.wav")
computer_wins_sound = load_sound("computer_wins.wav")
human_wins_sound = load_sound("human_wins.wav")
tie_sound = load_sound("tie.wav")

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
tie_score = 0
round_count = 0
total_rounds = 20  # Adjusted the number of rounds

# Advanced DQN Agent and Game Logic
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.value_fc = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        advantage_mean = advantage.mean(dim=-1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention_fc = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        attention_weights = torch.softmax(self.attention_fc(out), dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)
        out = self.fc(context_vector)
        return out

class EnsembleModel(nn.Module):
    def __init__(self, dqn, lstm):
        super(EnsembleModel, self).__init__()
        self.dqn = dqn
        self.lstm = lstm

    def forward(self, x_dqn, x_lstm):
        dqn_out = self.dqn(x_dqn)
        lstm_out = self.lstm(x_lstm)
        return dqn_out + lstm_out

class AdvancedDQNAgent:
    def __init__(self, state_size, action_size, pre_trained_path=None):
        self.state_size = state_size + 1 + 3
        self.action_size = action_size
        self.actions = ['R', 'P', 'S']
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        self.model = DQNetwork(self.state_size, self.action_size)
        self.target_model = DQNetwork(self.state_size, self.action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_move = None
        self.game_history = []
        self.strategy_switch_threshold = 0.15

        if pre_trained_path and os.path.exists(pre_trained_path):
            try:
                self.load(pre_trained_path)
            except RuntimeError as e:
                print(f"Error loading pre-trained model: {e}. Training a new model.")

        self.move_frequencies, self.transition_counts = load_and_preprocess_history()
        
        self.lstm_model = LSTMNetwork(input_dim=3, hidden_dim=60, output_dim=3, num_layers=2)
        self.lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.lstm_criterion = nn.CrossEntropyLoss()

        self.ensemble_model = EnsembleModel(self.model, self.lstm_model)
        self.ensemble_optimizer = optim.Adam(self.ensemble_model.parameters(), lr=0.001)

        self.historical_data = load_game_history()
        self.kmeans, self.cluster_labels = cluster_opponent_strategies(self.historical_data)
        self.current_opponent_cluster = None

        self.win_streak = 0
        self.lose_streak = 0
        self.tie_streak = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self, performance_metric):
        if performance_metric > 0.2:
            self.epsilon *= 0.99
        else:
            self.epsilon *= 1.01
        self.epsilon = max(self.epsilon_min, min(self.epsilon, 1.0))

    def adjust_learning_rate(self, performance_metric):
        if performance_metric > 0.2:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.05
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.learning_rate / 10)
            param_group['lr'] = min(param_group['lr'], self.learning_rate * 10)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state)
                next_action = torch.argmax(self.model(next_state_tensor)).item()
                target = reward + self.gamma * self.target_model(next_state_tensor)[next_action].item()
            state_tensor = torch.FloatTensor(state)
            target_f = self.model(state_tensor)
            target_f[action] = target
            target_f = target_f.unsqueeze(0)
            self.optimizer.zero_grad()
            outputs = self.model(state_tensor.unsqueeze(0))
            loss = self.criterion(outputs, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def get_state(self, my_move, opponent_move, player_moves, computer_moves):
        if my_move is None:
            my_move = 'R'  # Default to 'rock' to avoid crashes
        
        state = [0] * 6
        if my_move == "R":
            state[0] = 1
        elif my_move == "P":
            state[1] = 1
        elif my_move == "S":
            state[2] = 1
        if opponent_move == "R":
            state[3] = 1
        elif opponent_move == "P":
            state[4] = 1
        elif opponent_move == "S":
            state[5] = 1

        opp_freq = []
        for a in self.actions:
            try:
                opp_freq.append(self.transition_counts[move_to_name(my_move)][move_to_name(a)])
            except KeyError:
                opp_freq.append(0)

        opp_freq_total = sum(opp_freq) + 1
        opp_freq_normalized = [f / opp_freq_total for f in opp_freq]
        state.extend(opp_freq_normalized)
        
        recent_length = 30
        recent_player_moves = [player_moves[-i] for i in range(1, min(len(player_moves), recent_length) + 1)]
        recent_computer_moves = [computer_moves[-i] for i in range(1, min(len(computer_moves), recent_length) + 1)]
        move_counts = [recent_player_moves.count('rock'), recent_player_moves.count('paper'), recent_player_moves.count('scissors'),
                       recent_computer_moves.count('rock'), recent_computer_moves.count('paper'), recent_computer_moves.count('scissors')]
        state.extend(move_counts)
        
        state.append(0)

        if self.current_opponent_cluster is not None:
            state.append(self.current_opponent_cluster)
        else:
            state.append(-1)

        state.extend([self.win_streak, self.lose_streak, self.tie_streak])
        
        return state

    def pre_train_from_history(self):
        historical_data = []
        for move1 in self.actions:
            for move2 in self.actions:
                move1_name = move_to_name(move1)
                move2_name = move_to_name(move2)
                count = self.transition_counts[move1_name][move2_name]
                if count > 0:
                    historical_data.extend([(move1, move2)] * count)

        if not historical_data:
            print("No historical data available for pre-training.")
            return

        for _ in range (10000):
            state = random.choice(historical_data)
            action = self.actions.index(state[1])
            reward = determine_reward(state[0], state[1])
            next_state = random.choice(historical_data)
            self.remember(self.get_state(state[0], state[1], [], []), action, reward, self.get_state(next_state[0], next_state[1], [], []), False)
        if len(self.memory) > 64:
            self.replay(64)

    def detect_pattern_with_lstm(self, player_moves, sequence_length=20):
        if len(player_moves) < sequence_length:
            return None
        
        def move_to_one_hot(move):
            if move == 'R':
                return [1, 0, 0]
            elif move == 'P':
                return [0, 1, 0]
            elif move == 'S':
                return [0, 0, 1]
            else:
                raise ValueError(f"Invalid move: {move}")

        input_sequence = [move_to_one_hot(move) for move in player_moves[-sequence_length:]]
        input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0)

        output = self.lstm_model(input_sequence)
        predicted_move_idx = torch.argmax(output, dim=1).item()

        return self.actions[predicted_move_idx]

    def compute_performance_metric(self):
        if len(self.game_history) < 100:
            return 0.5
        wins = sum(1 for result in self.game_history[-100:] if result == 'computer')
        return wins / 100.0

    def dynamic_strategy(self, state, player_moves, computer_moves):
        performance_metric = self.compute_performance_metric()
        self.adjust_learning_rate(performance_metric)
        if performance_metric > self.strategy_switch_threshold:
            return self.act(state, player_moves, computer_moves)
        else:
            return self.act(state, player_moves, computer_moves)

    def classify_opponent(self, player_moves):
        if len(player_moves) < 10:
            return 'random'
        move_counts = [player_moves.count('R'), player_moves.count('P'), player_moves.count('S')]
        max_move = max(move_counts)
        if max_move / len(player_moves) > 0.6:
            return 'aggressive'
        elif max_move / len(player_moves) < 0.4:
            return 'defensive'
        else:
            return 'random'

    def adaptive_strategy(self, state, player_moves, computer_moves):
        opponent_type = self.classify_opponent(player_moves)
        context = {'aggressive': 0, 'defensive': 1, 'random': 2}
        state[-1] = context[opponent_type]
        return self.dynamic_strategy(state, player_moves, computer_moves)

    def search_pattern_in_history(self, last_ten_player_moves, last_ten_computer_moves):
        for i in range(len(self.historical_data) - 12):
            if (self.historical_data[i][1:13] == last_ten_player_moves and
                self.historical_data[i][13:25] == last_ten_computer_moves):
                next_move = self.historical_data[i+12][1]
                return next_move
        return None

    def real_time_learning(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.replay(1)

    def act(self, state, player_moves, computer_moves):
        if len(player_moves) >= 12 and len(computer_moves) >= 12:
            pattern_move = self.search_pattern_in_history(player_moves[-12:], computer_moves[-12:])
            if pattern_move:
                return self.actions.index(pattern_move)

        self.update_epsilon(self.compute_performance_metric())
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))

        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def update_streaks(self, result):
        if result == 'player' or result == 'tie':
            self.win_streak += 1
            self.lose_streak = 0
            self.tie_streak = 0
        elif result == 'computer':
            self.win_streak = 0
            self.lose_streak += 1
            self.tie_streak = 0

        if self.win_streak >= 2:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.95)
        elif self.lose_streak >= 3:
            self.epsilon = min(1.0, self.epsilon * 1.1)

def load_game_history():
    game_data = []
    directory = "Games"
    if not os.path.exists(directory):
        return game_data
    
    for filename in os.listdir(directory):
        if filename.startswith("game_history") and filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            with open(filepath, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                game_data.extend(list(reader))
    return game_data

def cluster_opponent_strategies(historical_data, n_clusters=5):
    move_patterns = []
    for game in historical_data:
        player_moves = [move_to_name(move) for move in game[1:] if move_to_name(move) is not None]
        move_pattern = [player_moves.count('rock'), player_moves.count('paper'), player_moves.count('scissors')]
        
        win_streak, lose_streak, tie_streak = 0, 0, 0
        for i in range(1, len(player_moves)):
            result = determine_winner(player_moves[i-1], player_moves[i])
            if result == 'player' or result == 'tie':
                win_streak += 1
                lose_streak = 0
                tie_streak = 0
            elif result == 'computer':
                win_streak = 0
                lose_streak += 1
                tie_streak = 0

        move_pattern.extend([win_streak, lose_streak, tie_streak])
        move_patterns.append(move_pattern)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(move_patterns)
    cluster_labels = kmeans.labels_
    return kmeans, cluster_labels

def load_and_preprocess_history(directory="Games"):
    game_data = []
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return defaultdict(int), defaultdict(lambda: defaultdict(int))
    
    for filename in os.listdir(directory):
        if filename.startswith("game_history") and filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            print(f"Loading file: {filepath}")
            with open(filepath, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if len(row) == 4:
                        game_data.append(row)
                    else:
                        print(f"Skipping malformed row in {filepath}: {row}")
    
    if not game_data:
        print("No game data found in the directory.")
        return defaultdict(int), defaultdict(lambda: defaultdict(int))
    
    move_frequencies = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    for round_num, player_move, computer_move, result in game_data:
        move_frequencies[player_move] += 1
        if round_num != '1':
            previous_move = game_data[int(round_num) - 2][1]
            transition_counts[previous_move][player_move] += 1

    print("Historical game data successfully loaded and processed.")
    return move_frequencies, transition_counts

# Function to play the appropriate sound for the given choice
def play_choice_sound(choice):
    if choice == 'rock':
        rock_sound.play()
    elif choice == 'scissors':
        scissors_sound.play()
    elif choice == 'paper':
        paper_sound.play()
        
# Function to play the appropriate sound based on the result
def play_result_sound(result):
    if result == "player":
        human_wins_sound.play()
    elif result == "computer":
        computer_wins_sound.play()
    else:
        tie_sound.play()

# Function to capture image within the square and resize
def capture_image(im, x, y, square_size=300, target_size=224):
    square = im[y:y+square_size, x:x+square_size]
    resized_square = cv2.resize(square, (target_size, target_size))
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

def capture_and_predict():
    global human_choice, round_count, human_score, computer_score, tie_score, original_choice, im
    round_count += 1
    print(f"Round {round_count} - Get ready to make your move!")
    countdown_sound.play()
    for i in range(3, 0, -1):
        print(f"{i}...")
        countdown_label.config(text=f"{i}...")
        root.update()
        time.sleep(1)

    im = picam2.capture_array()
    image = capture_image(im, top_left_x, top_left_y, square_size)
    human_choice, confidence = predict_image(image)
    print(f"Predicted move: {human_choice} with confidence {confidence * 100:.2f}%")

    return human_choice

def move_to_name(move):
    valid_moves = {'r': 'rock', 'p': 'paper', 's': 'scissors', 'R': 'rock', 'P': 'paper', 'S': 'scissors',
                   'rock': 'rock', 'paper': 'paper', 'scissors': 'scissors'}
    if move in valid_moves:
        return valid_moves[move]
    return None  # Return None for invalid entries

def determine_winner(player_move, computer_move):
    if player_move == computer_move:
        return 'tie'
    elif (player_move == 'rock' and computer_move == 'scissors') or \
         (player_move == 'paper' and computer_move == 'rock') or \
         (player_move == 'scissors' and computer_move == 'paper'):
        return 'player'
    else:
        return 'computer'

def determine_reward(player_move, computer_move):
    if player_move == computer_move:
        return 0  # Tie
    elif (player_move == 'rock' and computer_move == 'scissors') or \
         (player_move == 'paper' and computer_move == 'rock') or \
         (player_move == 'scissors' and computer_move == 'paper'):
        return 1  # Player wins (assuming agent is the computer)
    else:
        return -1  # Computer wins (assuming agent is the computer)

# Function to ask the player if they want to continue
def ask_to_continue():
    return messagebox.askyesno("Continue", "Do you want to play again?")

class RockPaperScissorsGame:
    def __init__(self, goal_score=total_rounds):
        self.goal_score = goal_score
        self.pre_trained_path = "Games/dqn_model.pth"
        self.reset_game()
        self.game_data = load_game_history()

    def reset_game(self):
        self.player_moves = []
        self.computer_moves = []
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0
        self.move_number = 1
        self.recent_states = deque(maxlen=60)
        self.game_history = []
        self.agent = AdvancedDQNAgent(state_size=16, action_size=3, pre_trained_path=self.pre_trained_path)
        self.agent.pre_train_from_history()

    def handle_move(self, player_move):
        player_name = move_to_name(player_move)
        if self.move_number == 1:
            computer_name = random.choice(['rock', 'paper', 'scissors'])
        else:
            if self.agent.last_move is None:
                self.agent.last_move = 'R'
            last_state = self.recent_states[-1] if self.recent_states else [0] * self.agent.state_size
            computer_move_idx = self.agent.adaptive_strategy(
                self.agent.get_state(self.agent.last_move, player_move, self.player_moves, self.computer_moves),
                self.player_moves,
                self.computer_moves
            )
            computer_name = move_to_name(self.agent.actions[computer_move_idx].lower())

        play_choice_sound(player_name)
        time.sleep(1)
        play_choice_sound(computer_name)
        time.sleep(1)

        result = determine_winner(player_name, computer_name)
        self.update_scores(result)
        self.update_moves(player_name, computer_name)
        self.update_agent(player_name, computer_name, result)

        self.agent.game_history.append(result)
        self.game_history.append([self.move_number, player_name, computer_name, result])

        if result == 'player':
            play_result_sound("player")
        elif result == 'computer':
            play_result_sound("computer")
        else:
            play_result_sound("tie")

        self.agent.update_streaks(result)
        
        return computer_name, result
        

    def update_scores(self, result):
        if result == 'player':
            self.player_score += 1
        elif result == 'computer':
            self.computer_score += 1
        else:
            self.ties += 1

    def update_moves(self, player_name, computer_name):
        self.player_moves.append(player_name)
        self.computer_moves.append(computer_name)
        self.recent_states.append((name_to_move(player_name).upper(), name_to_move(computer_name).upper()))
        self.move_number += 1
        self.agent.last_move = name_to_move(computer_name).upper()

    def update_agent(self, player_name, computer_name, result):
        if len(self.recent_states) > 1:
            previous_state = self.recent_states[-2]
            next_state = self.recent_states[-1]
            if result == 'computer':
                reward = 1.0
            elif result == 'player':
                reward = -1.0
            else:
                reward = 0.0
            self.agent.real_time_learning(
                self.agent.get_state(previous_state[0], previous_state[1], self.player_moves, self.computer_moves),
                self.agent.actions.index(previous_state[1]),
                reward,
                self.agent.get_state(next_state[0], next_state[1], self.player_moves, self.computer_moves),
                False
            )

    def end_game(self):
        if self.player_score == self.goal_score:
            result_label.config(text="Congratulations! You won the game!")
            play_result_sound("player")
        else:
            result_label.config(text="Computer wins the game! Better luck next time!")
            play_result_sound("computer")
        
        save_game_history(self.game_history)
        self.agent.save(self.pre_trained_path)

def name_to_move(name):
    return {'rock': 'r', 'paper': 'p', 'scissors': 's'}[name]

# Function to display a message box with the result
def show_result(result):
    messagebox.showinfo("Result", result)

# Function to restart the game
def restart_game():
    global human_score, computer_score, tie_score, round_count, game_instance
    human_score = 0
    computer_score = 0
    tie_score = 0
    round_count = 0
    game_instance.reset_game()
    update_game()

def save_game_history(game_history):
    os.makedirs("Games", exist_ok=True)
    filename = f"Games/game_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round Number", "Player's Choice", "Computer's Choice", "Result"])
        writer.writerows(game_history)
    messagebox.showinfo("Game Over", f"Game history saved to {filename}")

# Function to handle game logic and update the GUI
def update_game():
    global human_choice, computer_choice, human_score, computer_score, tie_score, round_count
    if round_count >= total_rounds:
        final_result = f"Game Over! Final Score:\nHuman: {human_score}\nComputer: {computer_score}\nTies: {tie_score}"
        show_result(final_result)
        if not ask_to_continue():
            root.quit()
        else:
            restart_game()
        return

    human_choice = capture_and_predict()
    computer_choice, result = game_instance.handle_move(human_choice)

    if result == 'tie':
        tie_score += 1
    elif result == 'player':
        human_score += 1
    else:
        computer_score += 1
        
    play_result_sound(result)

    print(f"Round result: {result}")
    print(f"Current Score - Human: {human_score}, Computer: {computer_score}, Ties: {tie_score}")

    result_label.config(text=f"Human: {human_choice} | Computer: {computer_choice} | {result}")
    score_label.config(text=f"Round: {round_count}/{total_rounds} | Human: {human_score} | Computer: {computer_score} | Ties: {tie_score}")
    root.update()
    
    if round_count < total_rounds:
        time.sleep(2)
        update_game()
    else:
        final_result = f"Game Over! Final Score:\nHuman: {human_score}\nComputer: {computer_score}\nTies: {tie_score}"
        show_result(final_result)
        save_game_history(game_instance.game_history)  # Ensure game history is saved
        game_instance.agent.save(game_instance.pre_trained_path)  # Ensure model is saved
        if not ask_to_continue():
            root.quit()
        else:
            restart_game()

# Initialize GUI
root = tk.Tk()
root.title("Rock-Paper-Scissors Game")

# Set up GUI elements
countdown_label = tk.Label(root, text="Press 'Start' to begin the game", font=("Helvetica", 16))
countdown_label.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

score_label = tk.Label(root, text=f"Round: {round_count}/{total_rounds} | Human: {human_score} | Computer: {computer_score} | Ties: {tie_score}", font=("Helvetica", 16))
score_label.pack()

# Main loop to update camera feed
def update_frame():
    global top_left_x, top_left_y
    im = picam2.capture_array()
    height, width, _ = im.shape
    center_x, center_y = width // 2, height // 2
    top_left_x = center_x - square_size // 2
    top_left_y = center_y - square_size // 2

    cv2.rectangle(im, (top_left_x, top_left_y), (top_left_x + square_size, top_left_y + square_size), (0, 255, 0), 2)
    horizontal_flip = cv2.flip(im, 1)
    
    horizontal_flip = cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(horizontal_flip)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.config(image=imgtk)
    camera_label.after(10, update_frame)

camera_label = tk.Label(root)
camera_label.pack()

start_button = tk.Button(root, text="Start", command=update_game, font=("Helvetica", 16))
start_button.pack()

game_instance = RockPaperScissorsGame()

update_frame()
root.mainloop()