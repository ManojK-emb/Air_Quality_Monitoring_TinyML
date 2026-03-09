/******************** LIBRARIES ********************/
#include <ArduinoIoTCloud.h>
#include <Arduino_ConnectionHandler.h>
#include <DHT.h>
#include <LiquidCrystal.h>

// ===== TinyML / TensorFlow Lite =====
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "air_quality_model.h"   // Your tflite model as C array

/******************** LCD ********************/
LiquidCrystal lcd(2, 15, 19, 18, 5, 4);

/******************** PINS ********************/
#define PM25PIN   26
#define MQ2_PIN   34
#define MQ135_PIN 35
#define BUZZER    25
#define DHTPIN    14
#define DHTTYPE   DHT11

/******************** OBJECTS ********************/
DHT dht(DHTPIN, DHTTYPE);

/******************** WIFI & IOT ********************/
const char DEVICE_LOGIN_NAME[] = "98b62c20-9753-475c-ad47-44ac83595926";
const char SSID[] = "projectiot";        
const char PASS[] = "123456789";
const char DEVICE_KEY[] = "3WTqWvyVuTVv6ETq2uk#KxC5O";

/******************** CLOUD VARIABLES ********************/
String status;          // AI Result
String temp_hum;
float dust;
String mq2;
String mq135;

/******************** ADC CONSTANTS ********************/
#define ADC_MAX 4095.0
#define RL 10.0

float MQ2_R0   = 9.8;
float MQ135_R0 = 3.6;

/******************** THRESHOLDS ********************/
#define TEMP_THRESHOLD 35
#define HUM_THRESHOLD  80
#define DUST_THRESHOLD 20

/******************** DUST VARIABLES ********************/
unsigned long lowPM = 0;
unsigned long starttime;
unsigned long sampletime_ms = 5000;
bool lastPM = HIGH;
unsigned long pmLowStart = 0;

/******************** TINYML VARIABLES ********************/
constexpr int kTensorArenaSize = 2 * 1024; // 2KB, adjust if needed
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model_tflite;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
tflite::AllOpsResolver resolver;

/******************** CLOUD INIT ********************/
void initProperties() {
  ArduinoCloud.setBoardId(DEVICE_LOGIN_NAME);
  ArduinoCloud.setSecretDeviceKey(DEVICE_KEY);

  ArduinoCloud.addProperty(status, READWRITE);
  ArduinoCloud.addProperty(temp_hum, READWRITE);
  ArduinoCloud.addProperty(dust, READWRITE);
  ArduinoCloud.addProperty(mq2, READWRITE);
  ArduinoCloud.addProperty(mq135, READWRITE);
}

WiFiConnectionHandler ArduinoIoTPreferredConnection(SSID, PASS);

/******************** SETUP ********************/
void setup() {
  Serial.begin(115200);

  analogSetWidth(12);
  analogSetAttenuation(ADC_11db);

  dht.begin();
  lcd.begin(16, 2);

  pinMode(PM25PIN, INPUT);
  pinMode(BUZZER, OUTPUT);

  initProperties();
  ArduinoCloud.begin(ArduinoIoTPreferredConnection);
  ArduinoCloud.printDebugInfo();

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("AIR MONITORING");
  lcd.setCursor(0,1);
  lcd.print(" SYSTEM");
  delay(3000);

  starttime = millis();

  // ===== TinyML Setup =====
  model_tflite = tflite::GetModel(air_quality_model_tflite);
  if (model_tflite->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while(1);
  }

  interpreter = new tflite::MicroInterpreter(model_tflite, resolver, tensor_arena, kTensorArenaSize);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while(1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
}

/******************** GAS RESISTANCE ********************/
float getRs(int adc) {
  if (adc == 0) adc = 1;
  return RL * (ADC_MAX - adc) / adc;
}

/******************** DUST CALC ********************/
float calculateDust(unsigned long lowpulse) {
  float ratio = (lowpulse / 1000000.0) / (sampletime_ms / 1000.0) * 100.0;
  float d = (0.001915 * ratio * ratio + 0.09522 * ratio - 0.04884) * 10;
  return d < 0 ? 0 : d;
}

/******************** AI FUNCTION (TinyML) ********************/
String airQualityAI(float temp, float hum, float dust,
                    float LPG, float CO2, float NH3, float Smoke) {
  
  // Fill input tensor
  input->data.f[0] = temp;
  input->data.f[1] = hum;
  input->data.f[2] = dust;
  input->data.f[3] = LPG;
  input->data.f[4] = CO2;
  input->data.f[5] = NH3;
  input->data.f[6] = Smoke;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return "ERROR";
  }

  // Find max output index
  int maxIndex = 0;
  float maxVal = output->data.f[0];
  for (int i = 1; i < 4; i++) {
    if (output->data.f[i] > maxVal) {
      maxVal = output->data.f[i];
      maxIndex = i;
    }
  }

  // Map index to string
  switch(maxIndex) {
    case 0: return "AIR QUALITY GOOD";
    case 1: return "MODERATE AIR";
    case 2: return "UNHEALTHY AIR";
    case 3: return "DANGEROUS AIR!";
    default: return "UNKNOWN";
  }
}

/******************** LOOP ********************/
void loop() {
  ArduinoCloud.update();

  bool state = digitalRead(PM25PIN);
  if (state == LOW && lastPM == HIGH) pmLowStart = micros();
  if (state == HIGH && lastPM == LOW) lowPM += micros() - pmLowStart;
  lastPM = state;

  if (millis() - starttime >= sampletime_ms) {

    float temp = dht.readTemperature();
    float hum  = dht.readHumidity();
    dust = calculateDust(lowPM);

    temp_hum = "T:" + String(temp,1) + " H:" + String(hum,1);

    int mq2_adc   = analogRead(MQ2_PIN);
    int mq135_adc = analogRead(MQ135_PIN);

    float mq2_ratio   = getRs(mq2_adc) / MQ2_R0;
    float mq135_ratio = getRs(mq135_adc) / MQ135_R0;

    float LPG_ppm   = 574.25 * pow(mq2_ratio, -2.222);
    float Smoke_ppm = 300.0  * pow(mq2_ratio, -1.5);

    float CO2_ppm = 116.6 * pow(mq135_ratio, -2.769);
    float NH3_ppm = 102.2 * pow(mq135_ratio, -2.473);

    mq2 = "LPG:" + String(LPG_ppm,0);
    mq135 = "CO2:" + String(CO2_ppm,0);

    // 🔥 AI RESULT via TinyML
    status = airQualityAI(
      temp, hum, dust,
      LPG_ppm, CO2_ppm, NH3_ppm, Smoke_ppm
    );

    // 🔔 Buzzer controlled by AI
    digitalWrite(BUZZER, status.startsWith("DANGEROUS") ? HIGH : LOW);

    // ================= LCD DISPLAY =================
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("LPG:");
    lcd.print((int)LPG_ppm);
    lcd.print(" CO2:");
    lcd.print((int)CO2_ppm);

    lcd.setCursor(0,1);
    lcd.print("T:");
    lcd.print((int)temp);
    lcd.print(" H:");
    lcd.print((int)hum);
    lcd.print(" D:");
    lcd.print((int)dust);

    delay(3000);   // Show gases

    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("AI STATUS:");
    lcd.setCursor(0,1);
    lcd.print(status.substring(0,16));

    delay(3000);   // Show AI result

    lowPM = 0;
    starttime = millis();
  }
}
