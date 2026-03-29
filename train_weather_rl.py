"""
train_weather_rl.py - Pre-train a DQN agent for 3-hour temperature forecasting.

Reads:  weather_history.csv  (produced by weather_api.py)
Writes: weather_rl_model.tflite  (quantized TFLite model)
        model_data.h             (C++ header array for ESP32)

Usage:
    python train_weather_rl.py
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Gymnasium fallback (avoids requiring gymnasium as a hard dependency)
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
except ImportError:
    class _BoxSpace:
        def __init__(self, low=None, high=None, shape=None, dtype=np.float32, **kw):
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
        def __init__(self, n, **kw):
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


# ---------------------------------------------------------------------------
# Custom RL Environment
# ---------------------------------------------------------------------------
class WeatherEnv(gym.Env):
    """
    Gym-style environment for weather temperature forecasting.

    State:   Normalized [temp, humidity, pressure]
    Action:  41 discrete bins  ->  temperature offset [-10 .. +10] C
    Reward:  -abs(predicted_temp - actual_temp_3h)
    """

    # Normalization constants — MUST be identical in predict.py and ESP32 sketch
    TEMP_MEAN  = 15.0;   TEMP_STD  = 35.0
    HUM_MEAN   = 50.0;   HUM_STD   = 50.0
    PRESS_MEAN = 1013.0; PRESS_STD = 50.0

    NUM_ACTIONS = 41

    def __init__(self, df, episode_length=24):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.episode_length = episode_length
        self.current_step = 0
        self.end_step = 0

        self.observation_space = gym.spaces.Box(
            low=np.array([-50.0, 0.0, 800.0], dtype=np.float32),
            high=np.array([50.0, 100.0, 1200.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)
        self.action_mapping = np.linspace(-10.0, 10.0, self.NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        max_start = len(self.df) - self.episode_length - 1
        if max_start < 0:
            raise ValueError(
                f"Dataset too small ({len(self.df)} rows) for episode_length={self.episode_length}. "
                f"Need at least {self.episode_length + 1} rows."
            )
        self.current_step = random.randint(0, max_start)
        self.end_step = self.current_step + self.episode_length
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            (row["temp"]     - self.TEMP_MEAN)  / self.TEMP_STD,
            (row["humidity"] - self.HUM_MEAN)   / self.HUM_STD,
            (row["pressure"] - self.PRESS_MEAN) / self.PRESS_STD,
        ], dtype=np.float32)

    def step(self, action_idx):
        row = self.df.iloc[self.current_step]
        actual_temp    = row["temp"]
        target_temp_3h = row["target_temp_3h"]

        predicted_offset = self.action_mapping[action_idx]
        predicted_temp   = actual_temp + predicted_offset

        error  = abs(predicted_temp - target_temp_3h)
        reward = -error

        self.current_step += 1
        done = self.current_step >= self.end_step

        info = {
            "actual_temp":    actual_temp,
            "target_temp_3h": target_temp_3h,
            "predicted_temp": predicted_temp,
            "error":          error,
        }
        return self._get_obs(), reward, done, False, info


# ---------------------------------------------------------------------------
# DQN Model & Agent
# ---------------------------------------------------------------------------
def build_dqn_model(state_dim, num_actions):
    """
    Tiny neural network for ESP32: 2 hidden layers x 16 units.
    Total ~1,000 parameters -> fits in a few KB of SRAM.
    """
    model = models.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_actions, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(self, state_dim, num_actions):
        self.state_dim   = state_dim
        self.num_actions = num_actions
        self.model        = build_dqn_model(state_dim, num_actions)
        self.target_model = build_dqn_model(state_dim, num_actions)
        self.update_target_model()

        self.memory        = []
        self.max_memory    = 2000
        self.gamma         = 0.90
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995
        self.batch_size    = 32

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model(state[np.newaxis, :], training=False).numpy()
        return int(np.argmax(q_values[0]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch   = random.sample(self.memory, self.batch_size)
        states      = np.array([m[0] for m in minibatch])
        actions     = np.array([m[1] for m in minibatch])
        rewards     = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones       = np.array([m[4] for m in minibatch], dtype=np.float32)

        next_q  = self.target_model(next_states, training=False).numpy()
        max_q   = np.amax(next_q, axis=1)
        targets = rewards + self.gamma * max_q * (1.0 - dones)

        target_f = self.model(states, training=False).numpy()
        for i, a in enumerate(actions):
            target_f[i][a] = targets[i]

        self.model.train_on_batch(states, target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ---------------------------------------------------------------------------
# TFLite Export
# ---------------------------------------------------------------------------
def convert_to_tflite(keras_model, tflite_path="weather_rl_model.tflite"):
    """Export Keras model to quantized TFLite format."""
    print(f"\n[*] Exporting model to {tflite_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"[+] Saved TFLite model: {tflite_path} ({len(tflite_bytes):,} bytes)")
    return tflite_path


def convert_tflite_to_cpp(tflite_path, cpp_path="model_data.h"):
    """Convert .tflite binary into a C++ header array for ESP32 flashing."""
    print(f"\n[*] Converting {tflite_path} -> {cpp_path}...")
    with open(tflite_path, "rb") as f:
        data = f.read()

    hex_vals = [f"0x{b:02x}" for b in data]

    with open(cpp_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("alignas(8) const unsigned char g_weather_model[] = {\n    ")
        for i, hv in enumerate(hex_vals):
            f.write(hv + ", ")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
        f.write("\n};\n")
        f.write(f"const unsigned int g_weather_model_len = {len(hex_vals)};\n\n")
        f.write("#endif // MODEL_DATA_H\n")

    print(f"[+] Saved C++ header: {cpp_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("--- TinyML Weather RL Agent Pre-Training Pipeline ---\n")

    csv_path = "weather_history.csv"
    if not os.path.exists(csv_path):
        print(f"[!] Error: {csv_path} not found.")
        print("    Run: python weather_api.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required_cols = {"temp", "humidity", "pressure", "target_temp_3h"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[!] Error: CSV is missing columns: {missing}")
        sys.exit(1)

    print(f"[*] Loaded {len(df)} hourly weather records from {csv_path}")

    # Validate dataset size
    episode_length = 24
    if len(df) < episode_length + 1:
        print(f"[!] Error: Need at least {episode_length + 1} rows, got {len(df)}.")
        sys.exit(1)

    env = WeatherEnv(df, episode_length=episode_length)
    state_dim   = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(state_dim, num_actions)

    episodes = 1000
    print(f"[*] Training for {episodes} episodes (episode_length={episode_length})...\n")

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

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
                    avg_err = -total_reward / episode_length
                    print(f"  Episode {e+1:4d}/{episodes} | "
                          f"Avg Error: {avg_err:5.2f} C | "
                          f"Epsilon: {agent.epsilon:.3f}")
                break

        for _ in range(5):
            agent.replay()

    print("\n[*] Training complete!")

    # Export
    tflite_file = "weather_rl_model.tflite"
    cpp_file    = "model_data.h"

    convert_to_tflite(agent.model, tflite_file)
    convert_tflite_to_cpp(tflite_file, cpp_file)

    print("\n" + "=" * 55)
    print("  Next steps for ESP32 deployment:")
    print("=" * 55)
    print("  1. Copy model_data.h into esp32/ folder")
    print("  2. Open weather_station.ino in Arduino IDE")
    print("  3. Flash to ESP32 and open Serial Monitor (115200)")
    print("=" * 55)


if __name__ == "__main__":
    main()
