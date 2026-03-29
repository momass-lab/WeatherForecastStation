import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import os

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Minimal gym-compatible fallback classes (accepts all kwargs from Box / Discrete)
    class _BoxSpace:
        def __init__(self, low=None, high=None, shape=None, dtype=np.float32, **kwargs):
            if shape is None and low is not None:
                self.shape = np.array(low).shape
            else:
                self.shape = shape
            self.low = low
            self.high = high
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _DiscreteSpace:
        def __init__(self, n, **kwargs):
            self.n = n
            self.shape = (1,)

        def sample(self):
            return random.randint(0, self.n - 1)

    class gym:
        class Env:
            def __init__(self):
                pass
        class spaces:
            Box = _BoxSpace
            Discrete = _DiscreteSpace

def get_synthetic_weather_data(samples=2000):
    """
    Generates synthetic historical weather data.
    In a real scenario, you would load this from a CSV file (e.g., OpenWeatherMap, NOAA).
    """
    np.random.seed(42)
    # Sine wave for temp variations (approx daily cycles)
    time = np.linspace(0, samples/24 * 2 * np.pi, samples)
    temp = np.sin(time) * 10 + 20 + np.random.normal(0, 1, samples)
    humidity = np.clip(np.random.normal(60, 10, samples), 0, 100)
    pressure = np.random.normal(1013, 5, samples)
    
    df = pd.DataFrame({'temp': temp, 'humidity': humidity, 'pressure': pressure})
    df['target_temp_3h'] = df['temp'].shift(-3)
    df.dropna(inplace=True)
    return df

class WeatherEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Weather Forecasting.
    """
    def __init__(self, df, episode_length=24):
        super(WeatherEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.episode_length = episode_length
        self.current_step = 0
        self.end_step = 0
        
        # State: [temp, humidity, pressure]
        # Bounding the state inputs for standard atmospheric conditions
        self.observation_space = gym.spaces.Box(
            low=np.array([-50.0, 0.0, 800.0], dtype=np.float32),
            high=np.array([50.0, 100.0, 1200.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Predict Temperature Offset
        # Discrete space of 41 actions -> representing [-10.0, -9.5, ... , 0.0, ..., 9.5, 10.0] degrees
        # This simplifies DQN deployment since continuous actions require more complex setups (like DDPG)
        self.num_actions = 41
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.action_mapping = np.linspace(-10.0, 10.0, self.num_actions)
        
    def reset(self, seed=None, options=None):
        # Pick a random starting point in the time-series data
        max_start = len(self.df) - self.episode_length - 1
        self.current_step = random.randint(0, max_start)
        self.end_step = self.current_step + self.episode_length
        return self._get_obs(), {}
        
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # Optional: Min-max scaling hints (important for Neural Network stability, particularly TinyML)
        # temp (-20 to 50), humidity (0 to 100), pressure (900 to 1100)
        norm_temp = (row['temp'] - 15.0) / 35.0
        norm_hum = (row['humidity'] - 50.0) / 50.0
        norm_press = (row['pressure'] - 1013.0) / 50.0
        return np.array([norm_temp, norm_hum, norm_press], dtype=np.float32)
        
    def step(self, action_idx):
        row = self.df.iloc[self.current_step]
        actual_temp = row['temp']
        target_temp_3h = row['target_temp_3h']
        
        # Get chosen temp change from discrete action class
        predicted_offset = self.action_mapping[action_idx]
        predicted_temp = actual_temp + predicted_offset
        
        # Calculate Reward = -|PredictedTemp - ActualTemp|
        error = abs(predicted_temp - target_temp_3h)
        reward = -error
        
        self.current_step += 1
        done = self.current_step >= self.end_step
        
        # Pass info dict containing debugging info
        info = {
            'actual_temp': actual_temp,
            'target_temp_3h': target_temp_3h,
            'predicted_temp': predicted_temp,
            'error': error
        }
        
        return self._get_obs(), reward, done, False, info


def build_dqn_model(state_dim, num_actions):
    """
    Builds a very small Neural Network suitable for ESP32 and TFLite Micro constraints.
    Structure: 2 hidden layers with 16 nodes each.
    """
    model = models.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_actions, activation='linear') # Linear activation for Q-Values
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

class DQNAgent:
    """
    A foundational Deep Q-Network Agent to learn the weather mappings.
    """
    def __init__(self, state_dim, num_actions):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.model = build_dqn_model(state_dim, num_actions)
        self.target_model = build_dqn_model(state_dim, num_actions)
        self.update_target_model()
        
        self.memory = []
        self.max_memory = 2000
        self.gamma = 0.90
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        # Using functional form instead of model.predict for faster eager execution in basic train loops
        q_values = self.model(state[np.newaxis, :], training=False).numpy()
        return np.argmax(q_values[0])
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Predict Q-values of next states using target network
        next_q_values = self.target_model(next_states, training=False).numpy()
        max_next_q = np.amax(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Get current Q-values to update
        target_f = self.model(states, training=False).numpy()
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
            
        # Fast batch fit
        self.model.train_on_batch(states, target_f)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def convert_to_tflite(keras_model, tflite_path="weather_rl_model.tflite"):
    """
    Exports the trained Sequential Keras model to TensorFlow Lite and applies TinyML optimization.
    """
    print(f"\n[*] Exporting model to {tflite_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Quantize and optimize for embedded devices (reduces size dramatically)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[+] Saved TFLite model: {tflite_path} ({len(tflite_model)} bytes)")
    return tflite_path

def convert_tflite_to_cpp(tflite_path, cpp_path="model_data.h"):
    """
    Converts a .tflite binary file into a C/C++ header array so it can be flashed onto an ESP32.
    This replaces the need for the linux 'xxd' command, making the script OS-independent.
    """
    print(f"\n[*] Converting {tflite_path} to C++ header {cpp_path}...")
    with open(tflite_path, "rb") as f:
        tflite_content = f.read()
        
    hex_array = [f"0x{b:02x}" for b in tflite_content]
    
    with open(cpp_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        # Provide the required alignment macro for microcontrollers
        f.write("alignas(8) const unsigned char g_weather_model[] = {\n    ")
        
        for i, hex_val in enumerate(hex_array):
            f.write(hex_val + ", ")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
                
        f.write("\n};\n")
        f.write(f"const int g_weather_model_len = {len(hex_array)};\n\n")
        f.write("#endif // MODEL_DATA_H\n")
    print(f"[+] Saved C++ header to {cpp_path}")


def main():
    print("--- TinyML Weather RL Agent Pre-Training Pipeline ---")
    
    # Load the real dataset generated by weather_api.py
    csv_path = "weather_history.csv"
    if not os.path.exists(csv_path):
        print(f"[!] Error: {csv_path} not found! Please run weather_api.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} historical weather data points.")
    
    # Initialize Environment & Agent
    env = WeatherEnv(df, episode_length=24) # Evaluate over 24-hour shifting windows
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(state_dim, num_actions)
    
    # Setting the exact number of episodes here (usually 1000+)
    episodes = 1000 
    print(f"[*] Starting Training for {episodes} episodes...")
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                if e % 10 == 0:
                    agent.update_target_model()
                    
                if (e + 1) % 100 == 0:
                    print(f"Episode: {e+1:04d}/{episodes} | Total Reward (Negative Error): {total_reward:06.2f} | Epsilon: {agent.epsilon:.2f}")
                break
                
        # Train agent off replay buffer
        for _ in range(5):
            agent.replay()
            
    print("\n[*] Training Complete!")
    
    # --- Deliverable Conversion Pipeline ---
    tflite_file = "weather_rl_model.tflite"
    cpp_header_file = "model_data.h"
    
    convert_to_tflite(agent.model, tflite_file)
    convert_tflite_to_cpp(tflite_file, cpp_header_file)
    
    print("\n=======================================================")
    print("Steps for ESP32 Deployment:")
    print("=======================================================")
    print("1. Include 'model_data.h' in your Arduino/ESP-IDF project.")
    print("2. Initialize tfLiteMicro with the 'g_weather_model' array.")
    print("3. Read [temp, humidity, pressure] from external sensors.")
    print("4." + " "*3 + "CRITICAL NOTE: Normalize inputs identically to Python before inferencing.")
    print("   float norm_temp = (temp - 15.0) / 35.0;")
    print("   float norm_hum = (humidity - 50.0) / 50.0;")
    print("   float norm_press = (pressure - 1013.0) / 50.0;")
    print("5. Load the normalized array into the interpreter tensor.")
    print("6. Invoke interpreter. Find the index 'i' of the max output value.")
    print("7. Decode index 'i' to get your temperature offset in degrees Celsius:")
    print("   float offset = -10.0 + (i * 0.5);")

if __name__ == "__main__":
    main()
