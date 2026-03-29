/*
 * weather_station.ino
 * ESP32 TinyML Weather Forecast Station
 *
 * Hardware required:
 *   - ESP32 Dev Module (any variant)
 *   - BME280 sensor (I2C: SDA=GPIO21, SCL=GPIO22)
 *
 * Arduino Libraries (install via Library Manager):
 *   - TensorFlowLite_ESP32      by TensorFlow Authors
 *   - Adafruit BME280 Library   by Adafruit
 *   - Adafruit Unified Sensor   by Adafruit
 *
 * Workflow:
 *   1. Run  python weather_api.py        → weather_history.csv
 *   2. Run  python train_weather_rl.py   → model_data.h
 *   3. Copy model_data.h into this sketch folder
 *   4. Flash to ESP32 and open Serial Monitor (115200 baud)
 */

#include "model_data.h"

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// ── TFLite Micro setup ──────────────────────────────────────────
const int kTensorArenaSize = 8 * 1024;   // 8 KB is plenty for this small model
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model*          model       = nullptr;
tflite::MicroInterpreter*     interpreter = nullptr;
TfLiteTensor*                 input       = nullptr;
TfLiteTensor*                 output      = nullptr;

// ── BME280 sensor ────────────────────────────────────────────────
Adafruit_BME280 bme;          // I2C

// ── Action mapping (must match train_weather_rl.py exactly) ─────
const int   NUM_ACTIONS   = 41;
const float ACTION_MIN    = -10.0f;   // °C offset
const float ACTION_STEP   =   0.5f;  // step between bins

// ── Normalization constants (must match train_weather_rl.py) ────
const float TEMP_MEAN    = 15.0f;   const float TEMP_STD    = 35.0f;
const float HUM_MEAN     = 50.0f;   const float HUM_STD     = 50.0f;
const float PRESS_MEAN   = 1013.0f; const float PRESS_STD   = 50.0f;

// Interval between predictions (ms)
const unsigned long PREDICT_INTERVAL_MS = 60000UL;   // every 60 s
unsigned long last_predict_ms = 0;

// ────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=== ESP32 TinyML Weather Forecast Station ===");

  // ── 1. Initialise BME280 ─────────────────────────────────────
  if (!bme.begin(0x76)) {
    Serial.println("[ERROR] BME280 not found! Check wiring: SDA=21, SCL=22");
    while (true) delay(1000);   // halt
  }
  Serial.println("[OK] BME280 sensor initialised.");

  // ── 2. Load TFLite model ─────────────────────────────────────
  model = tflite::GetModel(g_weather_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] TFLite schema version mismatch!");
    while (true) delay(1000);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() failed!");
    while (true) delay(1000);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("[OK] Model loaded. Input shape: [1, %d]  Output shape: [1, %d]\n",
                input->dims->data[1], output->dims->data[1]);
  Serial.println("─────────────────────────────────────────────\n");
}

// ────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();
  if (now - last_predict_ms < PREDICT_INTERVAL_MS) return;
  last_predict_ms = now;

  // ── 3. Read sensor ───────────────────────────────────────────
  float temp     = bme.readTemperature();         // °C
  float humidity = bme.readHumidity();            // %
  float pressure = bme.readPressure() / 100.0f;  // Pa → hPa

  Serial.println("─────────────────────────────────────────────");
  Serial.printf("  Current reading:\n");
  Serial.printf("    Temp     : %.1f °C\n",  temp);
  Serial.printf("    Humidity : %.1f %%\n",  humidity);
  Serial.printf("    Pressure : %.1f hPa\n", pressure);

  // ── 4. Normalise (MUST match Python training constants!) ─────
  input->data.f[0] = (temp     - TEMP_MEAN)  / TEMP_STD;
  input->data.f[1] = (humidity - HUM_MEAN)   / HUM_STD;
  input->data.f[2] = (pressure - PRESS_MEAN) / PRESS_STD;

  // ── 5. Run inference ─────────────────────────────────────────
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Inference failed!");
    return;
  }

  // ── 6. Decode argmax → temperature offset ───────────────────
  int   best_action = 0;
  float best_q      = output->data.f[0];
  for (int i = 1; i < NUM_ACTIONS; i++) {
    if (output->data.f[i] > best_q) {
      best_q      = output->data.f[i];
      best_action = i;
    }
  }

  float offset         = ACTION_MIN + (best_action * ACTION_STEP);
  float predicted_temp = temp + offset;

  Serial.printf("\n  🌡  3-hour forecast:\n");
  Serial.printf("    Predicted temp : %.1f °C  (offset %+.1f °C)\n",
                predicted_temp, offset);
  Serial.printf("    Agent Q-value  : %.4f  (action #%d)\n", best_q, best_action);
  Serial.println("─────────────────────────────────────────────\n");
}
