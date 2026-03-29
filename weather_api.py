"""
weather_api.py - Fetch hourly weather data from Visual Crossing and export as CSV.

Environment Variables:
    WEATHER_API_KEY   (required) - Your Visual Crossing API key
    WEATHER_LOCATION  (optional) - City name, defaults to "Prague"

Output:
    weather_history.csv  - Hourly records with temp, humidity, pressure, target_temp_3h
"""
import os
import sys
import urllib.request
import urllib.error
import json
import csv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCATION = os.getenv("WEATHER_LOCATION", "Prague")
API_KEY = os.getenv("WEATHER_API_KEY")

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
OUTPUT_CSV = "weather_history.csv"


def fetch_and_save_data():
    """Fetch 15-day hourly forecast, compute 3h target, save to CSV."""

    if not API_KEY:
        print("[!] Error: WEATHER_API_KEY environment variable is not set.")
        print("    PowerShell : $env:WEATHER_API_KEY='your_key_here'")
        print("    Linux/Mac  : export WEATHER_API_KEY='your_key_here'")
        print("    Or copy .env.example to .env and fill in your key.")
        sys.exit(1)

    request_url = (
        f"{BASE_URL}{LOCATION}"
        f"?unitGroup=metric&contentType=json&key={API_KEY}"
    )

    try:
        print(f"[*] Fetching weather data for {LOCATION}...")
        with urllib.request.urlopen(request_url) as response:
            weather_data = json.loads(response.read())

        print(f"[+] Data retrieved for: {weather_data['resolvedAddress']}")

        # Extract hourly records
        hourly_records = []
        for day in weather_data["days"]:
            for hour in day.get("hours", []):
                hourly_records.append({
                    "datetime": f"{day['datetime']} {hour['datetime']}",
                    "temp": hour["temp"],
                    "humidity": hour["humidity"],
                    "pressure": hour["pressure"],
                })

        if len(hourly_records) < 4:
            print("[!] Error: Not enough hourly data returned (need at least 4 hours).")
            sys.exit(1)

        # Compute target_temp_3h (temperature 3 hours in the future)
        for i in range(len(hourly_records) - 3):
            hourly_records[i]["target_temp_3h"] = hourly_records[i + 3]["temp"]

        # Drop last 3 (no future target available)
        hourly_records = hourly_records[:-3]

        # Write CSV
        fieldnames = ["datetime", "temp", "humidity", "pressure", "target_temp_3h"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hourly_records)

        print(f"[+] Saved {len(hourly_records)} hourly records to {OUTPUT_CSV}")
        return OUTPUT_CSV

    except urllib.error.HTTPError as e:
        print(f"[!] HTTP Error {e.code}: {e.read().decode('utf-8', errors='replace')}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[!] URL Error: {e.reason}")
        sys.exit(1)


if __name__ == "__main__":
    fetch_and_save_data()