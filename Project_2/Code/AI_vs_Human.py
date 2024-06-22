import tkinter as tk
from tkinter import messagebox
import random
import pygame
import time
import os
import csv
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
from sklearn.cluster import KMeans

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Function to load and preprocess historical game data
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
                next(reader)  # Skip header
                for row in reader:
                    if len(row) == 4:  # Ensure the row has exactly 4 elements
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
        if round_num != '1':  # Skip the first round of each game for transition counts
            previous_move = game_data[int(round_num) - 2][1]
            transition_counts[previous_move][player_move] += 1

    print("Historical game data successfully loaded and processed.")
    return move_frequencies, transition_counts

def cluster_opponent_strategies(historical_data, n_clusters=7):
    # Extract move patterns and streaks from historical data
    move_patterns = []
    for game in historical_data:
        player_moves = [move_to_name(move) for move in game[1:] if move_to_name(move) is not None]  # Filter invalid moves
        move_pattern = [player_moves.count('rock'), player_moves.count('paper'), player_moves.count('scissors')]
        
        # Calculate streaks for this game
        win_streak, lose_streak, tie_streak = 0, 0, 0
        for i in range(1, len(player_moves)):
            result = determine_winner(player_moves[i-1], player_moves[i])
            if result == 'player' or result == 'tie' :
                win_streak += 1
                lose_streak = 0
                tie_streak = 0
            elif result == 'computer':
                win_streak = 0
                lose_streak += 1
                tie_streak = 0

        move_pattern.extend([win_streak, lose_streak, tie_streak])
        move_patterns.append(move_pattern)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(move_patterns)
    cluster_labels = kmeans.labels_
    return kmeans, cluster_labels

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        # Increased units for better learning capacity
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Separate streams for state value and advantage
        self.value_fc = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(128, output_dim)
        
        # Adjusted dropout rate to prevent overfitting
        self.dropout = nn.Dropout(0.6)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Combine value and advantage streams
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
        
        # Apply attention mechanism
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
        self.state_size = state_size + 1 + 3  # Update to include the opponent cluster information and streaks
        self.action_size = action_size
        self.actions = ['R', 'P', 'S']
        self.gamma = 0.98  # Discount factor for future rewards
        self.epsilon = 0.90  # Initial exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 0.0005  # Learning rate for the optimizer
        self.memory = deque(maxlen=20000)  # Increased memory size for experience replay
        self.model = DQNetwork(self.state_size, self.action_size)
        self.target_model = DQNetwork(self.state_size, self.action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_move = None
        self.game_history = []
        self.strategy_switch_threshold = 0.35  # Threshold for switching strategy

        # Load pre-trained model if path is provided
        if pre_trained_path and os.path.exists(pre_trained_path):
            try:
                self.load(pre_trained_path)
            except RuntimeError as e:
                print(f"Error loading pre-trained model: {e}. Training a new model.")

        # Load and preprocess historical data
        self.move_frequencies, self.transition_counts = load_and_preprocess_history()
        
        # LSTM model for pattern detection with attention mechanism
        self.lstm_model = LSTMNetwork(input_dim=3, hidden_dim=60, output_dim=3, num_layers=2)
        self.lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.lstm_criterion = nn.CrossEntropyLoss()

        # Ensemble model combining DQN and LSTM
        self.ensemble_model = EnsembleModel(self.model, self.lstm_model)
        self.ensemble_optimizer = optim.Adam(self.ensemble_model.parameters(), lr=0.001)

        self.historical_data = load_game_history()
        
        # Perform clustering on historical data
        self.kmeans, self.cluster_labels = cluster_opponent_strategies(self.historical_data)
        self.current_opponent_cluster = None

        # Initialize streak tracking
        self.win_streak = 0
        self.lose_streak = 0
        self.tie_streak = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self, performance_metric):
        # Adjust epsilon based on performance (e.g., win rate)
        if performance_metric > 0.2:
            self.epsilon *= 0.99
        else:
            self.epsilon *= 1.01
        self.epsilon = max(self.epsilon_min, min(self.epsilon, 1.0))

    def adjust_learning_rate(self, performance_metric):
        # Adjust learning rate based on performance
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
            print("Warning: my_move is None in get_state")
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

        # Add opponent modeling features from historical data
        opp_freq = []
        for a in self.actions:
            try:
                opp_freq.append(self.transition_counts[move_to_name(my_move)][move_to_name(a)])
            except KeyError as e:
                print(f"KeyError: {e}, my_move: {my_move}, action: {a}")
                opp_freq.append(0)

        opp_freq_total = sum(opp_freq) + 1  # Avoid division by zero
        opp_freq_normalized = [f / opp_freq_total for f in opp_freq]
        state.extend(opp_freq_normalized)
        
        # Add move frequencies in recent rounds (e.g., last 10 moves)
        recent_length = 30
        recent_player_moves = [player_moves[-i] for i in range(1, min(len(player_moves), recent_length) + 1)]
        recent_computer_moves = [computer_moves[-i] for i in range(1, min(len(computer_moves), recent_length) + 1)]
        move_counts = [recent_player_moves.count('rock'), recent_player_moves.count('paper'), recent_player_moves.count('scissors'),
                       recent_computer_moves.count('rock'), recent_computer_moves.count('paper'), recent_computer_moves.count('scissors')]
        state.extend(move_counts)
        
        # Add context features (opponent type)
        state.append(0)  # Placeholder for opponent type

        # Add opponent cluster information
        if self.current_opponent_cluster is not None:
            state.append(self.current_opponent_cluster)
        else:
            state.append(-1)  # Unknown cluster

        # Add streaks information
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

        for _ in range (10000):  # Adjust the number of pre-training iterations as needed
            state = random.choice(historical_data)
            action = self.actions.index(state[1])
            reward = determine_reward(state[0], state[1])
            next_state = random.choice(historical_data)
            self.remember(self.get_state(state[0], state[1], [], []), action, reward, self.get_state(next_state[0], next_state[1], [], []), False)
        if len(self.memory) > 64:
            self.replay(64)

    def detect_pattern_with_lstm(self, player_moves, sequence_length=20):
        # Ensure there are enough moves to analyze
        if len(player_moves) < sequence_length:
            return None
        
        # Convert moves to one-hot encoded vectors for the LSTM input
        def move_to_one_hot(move):
            if move == 'R':
                return [1, 0, 0]
            elif move == 'P':
                return [0, 1, 0]
            elif move == 'S':
                return [0, 0, 1]
            else:
                raise ValueError(f"Invalid move: {move}")

        # Prepare the input sequence for the LSTM
        input_sequence = [move_to_one_hot(move) for move in player_moves[-sequence_length:]]
        input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0)  # Add batch dimension

        # Forward pass through the LSTM model
        output = self.lstm_model(input_sequence)
        predicted_move_idx = torch.argmax(output, dim=1).item()

        # Return the predicted move
        return self.actions[predicted_move_idx]

    def compute_performance_metric(self):
        # Example: Compute win rate over the last 100 games
        if len(self.game_history) < 100:
            return 0.5  # Default value if insufficient history
        wins = sum(1 for result in self.game_history[-100:] if result == 'computer')
        return wins / 100.0

    def dynamic_strategy(self, state, player_moves, computer_moves):
        performance_metric = self.compute_performance_metric()
        self.adjust_learning_rate(performance_metric)  # Adjust learning rate based on performance
        if performance_metric > self.strategy_switch_threshold:
            # Use pattern detection strategy
            return self.act(state, player_moves, computer_moves)
        else:
            # Use DQN strategy
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
        state[-1] = context[opponent_type]  # Update the context feature in the state
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
        self.replay(1)  # Real-time learning with single experience replay

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
        if result == 'player' or result == 'tie':  # Counting ties as part of player's wins
            self.win_streak += 1
            self.lose_streak = 0
            self.tie_streak = 0
        elif result == 'computer':
            self.win_streak = 0
            self.lose_streak += 1
            self.tie_streak = 0

        # Example thresholds for changing strategy
        if self.win_streak >= 3:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.85)  # Become more exploitative
        elif self.lose_streak >= 2:
            self.epsilon = min(1.0, self.epsilon * 1.2)  # Become more explorative

# Function to load all game history files
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
                next(reader)  # Skip header
                game_data.extend(list(reader))
    return game_data

class RockPaperScissorsGame:
    def __init__(self, goal_score=25):
        self.goal_score = goal_score
        self.pre_trained_path = "Games/dqn_model.pth"
        self.reset_game()
        self.game_data = load_game_history()  # Load game history

    def reset_game(self):
        self.player_moves = []
        self.computer_moves = []
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0
        self.move_number = 1
        self.recent_states = deque(maxlen=60)
        self.game_history = []  # This will store the history of the current game
        self.agent = AdvancedDQNAgent(state_size=16, action_size=3, pre_trained_path=None)  # State size increased to 16
        self.agent.pre_train_from_history()

    def handle_move(self, player_move):
        player_name = move_to_name(player_move)
        if self.move_number == 1:
            computer_name = weighted_first_move()
        else:
            pattern_move = dynamic_pattern_detection(self.player_moves, max_length=25)
            if pattern_move:
                computer_name = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}[pattern_move]
            else:
                if self.agent.last_move is None:
                    self.agent.last_move = 'R'  # Ensure last_move is initialized
                last_state = self.recent_states[-1] if self.recent_states else [0] * self.agent.state_size
                computer_move_idx = self.agent.adaptive_strategy(self.agent.get_state(self.agent.last_move, name_to_move(player_name).upper(), self.player_moves, self.computer_moves), self.player_moves, self.computer_moves)
                computer_name = move_to_name(self.agent.actions[computer_move_idx].lower())

        play_sound(f"{player_name}.wav")
        time.sleep(1)  # Add delay to ensure the sounds are played sequentially
        play_sound(f"{computer_name}.wav")
        time.sleep(1)  # Add delay to ensure the sounds are played sequentially

        result = determine_winner(player_name, computer_name)
        self.update_scores(result)
        self.update_moves(player_name, computer_name)
        self.update_agent(player_name, computer_name, result)

        # Update game history for performance metric calculation
        self.agent.game_history.append(result)

        # Log the move to the game history
        self.game_history.append([self.move_number, player_name, computer_name, result])

        if result == 'player':
            play_sound("human_wins.wav")
        elif result == 'computer':
            play_sound("computer_wins.wav")
        else:
            play_sound("tie.wav")

        # Update streaks
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
            play_sound("human_wins.wav")
        else:
            result_label.config(text="Computer wins the game! Better luck next time!")
            play_sound("computer_wins.wav")
        
        save_game_history(self.game_history)
        self.agent.save(self.pre_trained_path)  # Save the model at the end of the game
        for btn in buttons:
            btn.config(state='disabled')

def play_sound(file):
    if sound_enabled:
        def _play():
            try:
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.5)
            except pygame.error as e:
                print(f"Error playing sound: {e}")
        threading.Thread(target=_play).start()

def move_to_name(move):
    valid_moves = {'r': 'rock', 'p': 'paper', 's': 'scissors', 'R': 'rock', 'P': 'paper', 'S': 'scissors',
                   'rock': 'rock', 'paper': 'paper', 'scissors': 'scissors'}
    if move in valid_moves:
        return valid_moves[move]
    return None  # Return None for invalid entries

def name_to_move(name):
    return {'rock': 'r', 'paper': 'p', 'scissors': 's'}[name]

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

def weighted_first_move():
    moves = ['rock', 'paper', 'scissors']
    probabilities = [0.26, 0.4, 0.34]
    return random.choices(moves, probabilities)[0]

def detect_pattern(player_moves, length=70):
    if len(player_moves) < length:
        return None
    suffix = tuple(player_moves[-length:])
    counts = defaultdict(int)
    for i in range(len(player_moves) - length):
        if tuple(player_moves[i:i+length]) == suffix:
            next_move = player_moves[i+length]
            counts[next_move] += 1
    if counts:
        return max(counts, key=counts.get)
    return None

def dynamic_pattern_detection(player_moves, max_length=70):
    for length in range(2, max_length + 1):
        pattern = detect_pattern(player_moves, length=length)
        if pattern:
            return pattern
    return None

def save_game_history(game_history):
    os.makedirs("Games", exist_ok=True)
    filename = f"Games/game_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round Number", "Player's Choice", "Computer's Choice", "Result"])
        writer.writerows(game_history)
    messagebox.showinfo("Game Over", f"Game history saved to {filename}")

def toggle_sound():
    global sound_enabled
    sound_enabled = not sound_enabled
    sound_button.config(text="Sound: On" if sound_enabled else "Sound: Off")

def restart_game():
    global game_instance
    game_instance.reset_game()
    for btn in buttons:
        btn.config(state='normal')
    round_label.config(text=f"Round: {game_instance.move_number}")
    score_label.config(text=f"Score - You: {game_instance.player_score}, Computer: {game_instance.computer_score}, Ties: {game_instance.ties}")
    result_label.config(text="")

def handle_move_gui(player_move):
    computer_name, result = game_instance.handle_move(player_move)
    result_label.config(text=f"Computer chose: {computer_name}\nResult: {'You win!' if result == 'player' else 'Computer wins!' if result == 'computer' else 'It is a tie!'}")
    round_label.config(text=f"Round: {game_instance.move_number}")
    score_label.config(text=f"Score - You: {game_instance.player_score}, Computer: {game_instance.computer_score}, Ties: {game_instance.ties}")
    if game_instance.player_score == game_instance.goal_score or game_instance.computer_score == game_instance.goal_score:
        game_instance.end_game()

# GUI setup
root = tk.Tk()
root.title("Rock-Paper-Scissors")
root.geometry("800x600")

font_large = ("Helvetica", 18)
font_medium = ("Helvetica", 14)

sound_enabled = True
game_instance = RockPaperScissorsGame()  # Start with pre-trained model if available

tk.Label(root, text="Choose your move:", font=font_large).pack(pady=10)

buttons = []
for move in ['r', 'p', 's']:
    btn = tk.Button(root, text=move_to_name(move).capitalize(), font=font_medium, command=lambda m=move: handle_move_gui(m))
    btn.pack(pady=10)
    buttons.append(btn)

round_label = tk.Label(root, text=f"Round: {game_instance.move_number}", font=font_large)
round_label.pack(pady=10)
score_label = tk.Label(root, text=f"Score - You: {game_instance.player_score}, Computer: {game_instance.computer_score}, Ties: {game_instance.ties}", font=font_large)
score_label.pack(pady=10)

sound_button = tk.Button(root, text="Sound: On", command=toggle_sound, font=font_medium)
sound_button.pack(pady=10)

restart_button = tk.Button(root, text="Restart Game", command=restart_game, font=font_medium)
restart_button.pack(pady=10)

result_label = tk.Label(root, text="", font=font_medium)
result_label.pack(side="bottom", pady=20)

root.mainloop()

