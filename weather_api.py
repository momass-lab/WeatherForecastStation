import urllib.request
import json
import csv
import urllib.error

# Define the weather API parameters
location = "Prague"
unit_group = "metric"
content_type = "json"
api_key = "8LNTSZ4T5336JZB5SASEPJH4L"

# We request the default 15-day forecast to get hourly predictions because the historical API limit was reached
endpoint = ""
base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

# Construct the full URL
request_url = f"{base_url}{location}?unitGroup={unit_group}&contentType={content_type}&key={api_key}"

def fetch_and_save_data():
    try:
        print(f"Fetching historical weather data for {location}...")
        # Make the request
        with urllib.request.urlopen(request_url) as response:
            raw_data = response.read()
            weather_data = json.loads(raw_data)
            
            print(f"Data retrieved successfully for: {weather_data['resolvedAddress']}")
            
            # Extract hourly data
            hourly_records = []
            for day in weather_data['days']:
                if 'hours' in day:
                    for hour in day['hours']:
                        hourly_records.append({
                            'datetime': f"{day['datetime']} {hour['datetime']}",
                            'temp': hour['temp'],
                            'humidity': hour['humidity'],
                            'pressure': hour['pressure']
                        })
            
            # We need to add 'target_temp_3h' by looking 3 hours ahead
            for i in range(len(hourly_records) - 3):
                hourly_records[i]['target_temp_3h'] = hourly_records[i + 3]['temp']
                
            # Drop the last 3 records since they don't have a 3-hour target
            hourly_records = hourly_records[:-3]
            
            # Save to CSV
            output_csv = "weather_history.csv"
            with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['datetime', 'temp', 'humidity', 'pressure', 'target_temp_3h'])
                writer.writeheader()
                writer.writerows(hourly_records)
                
            print(f"Saved {len(hourly_records)} hourly records to {output_csv}")
            return output_csv

    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code)
        print(e.read())
    except urllib.error.URLError as e:
        print("URL Error:", e.reason)

if __name__ == "__main__":
    fetch_and_save_data()