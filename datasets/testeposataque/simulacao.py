import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gym
from gym import spaces
import random
from collections import deque, defaultdict
from glob import glob
import pickle
from Training import RealTimeEnvironment

# Load the dataset
file_path1 = glob(f"./day*.csv")
file_path2 = glob(f"./poison*.csv")

file_path1 = sorted(file_path1, key=lambda x: int(x[5:].split('.')[0]))
file_path2 = sorted(file_path2, key=lambda x: int(x[8:].split('.')[0]))

data1 = [pd.read_csv(x) for x in file_path1]
data2 = [pd.read_csv(x) for x in file_path2]

def preprocess_data(df):
    # Convert 'Weapon Detected' to binary (1 for 'Yes', 0 for 'No')
    #df['Weapon Detected'] = df['Weapon Detected'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Encode 'Timestamp' using LabelEncoder
    label_encoder = LabelEncoder()
    df['Hour'] = label_encoder.fit_transform(df['Hour'])
    
    week_label = LabelEncoder()
    df['Weekday/Weekend'] = week_label.fit_transform(df['Weekday/Weekend'])

    # Extract state and action data
    state_data = df[['Hour', 'Presence Detector', 'Face Recognition', 'Weapon Recognition']].values
    actions = df['Action'].values
    
    return state_data, actions

# Preprocess the dataset
#state_data_N, actions_N = preprocess_data(data1)
state_data_N = []
actions_N = []
for d in data1:
    a, b = preprocess_data(d)
    state_data_N.append(a)
    actions_N.append(b)

# Preprocess the dataset
#state_data_P, actions_P = preprocess_data(data2)
state_data_P = []
actions_P = []
for d in data2:
    a, b = preprocess_data(d)
    state_data_P.append(a)
    actions_P.append(b)

# Initialize the environment
env = RealTimeEnvironment(state_data_N, actions_N, state_data_P, actions_P)
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Reconvertendo para defaultdict
q_table = defaultdict(lambda: [0, 0], q_table)

state_discretizer = (10, 2, 2, 2)

# Helper function to discretize continuous states
def discretize_state(state):
    """Convert continuous state into a discrete tuple."""
    return tuple(int(state[i] // (1 / state_discretizer[i])) for i in range(len(state)))

state = env.reset_N()  # Ou `env.reset_P()` dependendo do cenário
state = discretize_state(state)
done = False
env.index = 0
# Metrics for tracking performance
reward_history = deque(maxlen=144)
action_accuracy = deque(maxlen=144)


poison = False
avg_reward = []
suc_rate = []
epsilon = []
index = 0
episode = 0
episode_reward = 0
while not done:
    # Escolher a melhor ação (exploração)
    action = np.argmax(q_table[tuple(state)])
    
    # Realizar a ação no ambiente
    next_state, reward, done, _ = env.step_N(action)  # Use step_P para outro cenário
    next_state = discretize_state(next_state)

    print(f"Estado: {state}, Ação: {action}, Recompensa: {reward}")
    state = next_state
    episode_reward += reward
    #action_accuracy.append(1 if reward == 1 else 0)  # Track if action was correct
    action_accuracy.append(1 if reward > 0 else 0)  # Track if action was correct 
    reward_history.append(episode_reward)
    index += 1
    #print(index)
    
    if index % 143 == 0:
        average_reward = np.mean(reward_history)
        success_rate = np.mean(action_accuracy)
        avg_reward.append(average_reward)
        suc_rate.append(success_rate)
        print(f"Episode {episode}: Avg Reward = {average_reward:.2f}, Success Rate = {success_rate:.2f}")
        episode += 1
        state = env.reset_N() if poison == False else env.reset_P()
        episode_reward = 0
    if episode == 1000:
        break    