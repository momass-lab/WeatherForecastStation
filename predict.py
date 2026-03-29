"""
predict.py - Live inference using the pre-trained TFLite weather RL model.

Usage:
    python predict.py

Environment Variables:
    WEATHER_API_KEY   (optional) - Visual Crossing API key for live data
    WEATHER_LOCATION  (optional) - City name, defaults to "Prague"

Requires:
    - weather_rl_model.tflite  (produced by train_weather_rl.py)
    - weather_history.csv      (produced by weather_api.py, used as fallback)
"""
import os
import sys
import urllib.request
import urllib.error
import json
import csv
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCATION   = os.getenv("WEATHER_LOCATION", "Prague")
API_KEY    = os.getenv("WEATHER_API_KEY")
MODEL_PATH = "weather_rl_model.tflite"
CSV_PATH   = "weather_history.csv"

# Must match train_weather_rl.py normalization constants exactly!
TEMP_MEAN  = 15.0;   TEMP_STD  = 35.0
HUM_MEAN   = 50.0;   HUM_STD   = 50.0
PRESS_MEAN = 1013.0; PRESS_STD = 50.0

# Action mapping: 41 discrete bins -> temperature offset [-10 ... +10 C]
NUM_ACTIONS    = 41
ACTION_OFFSETS = np.linspace(-10.0, 10.0, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Step 1 - Fetch the current conditions
# ---------------------------------------------------------------------------
def fetch_current_conditions():
    """Pulls the most recent hourly observation from Visual Crossing.
    Returns None if the API key is missing or call fails."""
    if not API_KEY:
        return None

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/"
        f"rest/services/timeline/{LOCATION}/today"
        f"?unitGroup=metric&contentType=json&key={API_KEY}"
    )
    try:
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read())
        today = data["days"][0]
        hours = sorted(today.get("hours", []), key=lambda h: h["datetime"])
        if not hours:
            raise ValueError("No hourly data returned from API.")
        latest = hours[-1]
        return {
            "datetime": f"{today['datetime']} {latest['datetime']}",
            "temp":     float(latest["temp"]),
            "humidity": float(latest["humidity"]),
            "pressure": float(latest["pressure"]),
        }
    except urllib.error.HTTPError as e:
        print(f"[!] API HTTP Error {e.code}. Falling back to CSV.")
        return None
    except Exception as ex:
        print(f"[!] API error: {ex}. Falling back to CSV.")
        return None


def fallback_from_csv():
    """Reads the last row of the pre-built CSV as a fallback."""
    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: Neither API nor {CSV_PATH} available.")
        print("    Run: python weather_api.py first.")
        sys.exit(1)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[!] Error: {CSV_PATH} is empty!")
        sys.exit(1)

    last = rows[-1]
    result = {
        "datetime": last["datetime"],
        "temp":     float(last["temp"]),
        "humidity": float(last["humidity"]),
        "pressure": float(last["pressure"]),
    }
    if "target_temp_3h" in last and last["target_temp_3h"]:
        result["target_temp_3h"] = float(last["target_temp_3h"])
    return result


# ---------------------------------------------------------------------------
# Step 2 - Normalize exactly as the training env did
# ---------------------------------------------------------------------------
def normalize(temp, humidity, pressure):
    return np.array([
        (temp     - TEMP_MEAN)  / TEMP_STD,
        (humidity - HUM_MEAN)   / HUM_STD,
        (pressure - PRESS_MEAN) / PRESS_STD,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 3 - Run TFLite inference
# ---------------------------------------------------------------------------
def run_inference(obs: np.ndarray):
    """Returns (best_action_index, q_value)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()

    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    interp.set_tensor(inp_det["index"], obs.reshape(1, -1))
    interp.invoke()

    q_values = interp.get_tensor(out_det["index"])[0]
    best_action = int(np.argmax(q_values))
    return best_action, float(q_values[best_action])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        print("    Run: python train_weather_rl.py first.")
        sys.exit(1)

    # 1. Fetch current conditions (API first, CSV fallback)
    conditions = fetch_current_conditions()
    if conditions is None:
        if not API_KEY:
            print("[*] No API key set. Using CSV fallback data.")
        conditions = fallback_from_csv()

    sep = "-" * 55
    print("\n" + sep)
    print("  ESP32 Weather Forecast Station -- Live Inference")
    print(sep)
    print(f"  Location  : {LOCATION}")
    print(f"  Timestamp : {conditions['datetime']}")
    print(f"  Temp      : {conditions['temp']:.1f} C")
    print(f"  Humidity  : {conditions['humidity']:.1f} %")
    print(f"  Pressure  : {conditions['pressure']:.1f} hPa")
    print(sep)

    # 2. Normalize
    obs = normalize(conditions["temp"], conditions["humidity"], conditions["pressure"])

    # 3. Infer
    best_action, best_q = run_inference(obs)
    offset         = ACTION_OFFSETS[best_action]
    predicted_temp = conditions["temp"] + offset

    print(f"  Agent Q-value  : {best_q:.4f}  (action #{best_action})")
    print(f"  Predicted temp in 3 h: {predicted_temp:.1f} C  "
          f"(offset {offset:+.1f} C from current)")

    # 4. Compare against known target if available
    if "target_temp_3h" in conditions:
        actual = conditions["target_temp_3h"]
        error  = abs(predicted_temp - actual)
        print(f"  Actual   temp in 3 h: {actual:.1f} C")
        print(f"  Absolute error      : {error:.2f} C")

    print(sep + "\n")


if __name__ == "__main__":
    main()
