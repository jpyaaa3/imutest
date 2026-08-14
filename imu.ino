/*
 * Single SparkFun BNO080/BNO085 reader for Teensy
 *
 * Hardware:
 *   one sensor on Wire, SDA 17 / SCL 16, I2C address 0x4B
 *
 * Serial protocol (115200 baud, newline terminated):
 *   MAG ON       use the magnetometer-backed Rotation Vector
 *   MAG OFF      use the gyro/accelerometer Game Rotation Vector
 *   IMU ON       wake the IMU and resume reports
 *   IMU OFF      put the IMU to sleep
 *   MAG?         print the current mode
 *   STATUS       print connection and mode status
 *   HELP
 *
 * Data line consumed by new/map.py:
 *   IMU1,time_ms,qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg,mag_enabled
 */

#include <Arduino.h>
#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"

// ------------------------- User configuration -------------------------
#define IMU_BUS Wire
#define IMU_SDA_PIN 17
#define IMU_SCL_PIN 16
#define IMU_ADDRESS 0x4B

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint16_t REPORT_PERIOD_MS = 50;  // 20 Hz
constexpr uint32_t PRINT_PERIOD_MS = 20;
constexpr uint32_t STARTUP_DELAY_MS = 1000;
// ----------------------------------------------------------------------

BNO080 imu;

struct ImuState {
  float qw = 1.0f;
  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  bool initialized = false;
  bool has_sample = false;
};

ImuState state;
bool magnetometer_enabled = true;
bool imu_enabled = true;
String serial_command;

static float clampUnit(float value) {
  if (value < -1.0f) return -1.0f;
  if (value > 1.0f) return 1.0f;
  return value;
}

static float radiansToDegrees(float radians) {
  return radians * 57.29577951308232f;
}

static void quaternionToEuler(
    float qw, float qx, float qy, float qz,
    float &roll, float &pitch, float &yaw) {
  const float sinr_cosp = 2.0f * (qw * qx + qy * qz);
  const float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
  roll = radiansToDegrees(atan2f(sinr_cosp, cosr_cosp));

  const float sinp = 2.0f * (qw * qy - qz * qx);
  pitch = radiansToDegrees(asinf(clampUnit(sinp)));

  const float siny_cosp = 2.0f * (qw * qz + qx * qy);
  const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
  yaw = radiansToDegrees(atan2f(siny_cosp, cosy_cosp));
}

static void configureReports() {
  // Rotation Vector uses the magnetometer; Game Rotation Vector does not.
  if (magnetometer_enabled) {
    imu.setFeatureCommand(SENSOR_REPORTID_GAME_ROTATION_VECTOR, 0);
    imu.enableRotationVector(REPORT_PERIOD_MS);
  } else {
    imu.setFeatureCommand(SENSOR_REPORTID_ROTATION_VECTOR, 0);
    imu.enableGameRotationVector(REPORT_PERIOD_MS);
  }
}

static void setMagnetometerEnabled(bool enabled) {
  magnetometer_enabled = enabled;
  if (state.initialized) configureReports();
  Serial.print("MAGNETOMETER,");
  Serial.println(magnetometer_enabled ? "ON" : "OFF");
}

static void setImuEnabled(bool enabled) {
  imu_enabled = enabled;
  if (!imu_enabled) state.has_sample = false;
  if (state.initialized) {
    if (imu_enabled) {
      imu.modeOn();
      configureReports();
    } else {
      imu.modeSleep();
    }
  }
  Serial.print("IMU_POWER,");
  Serial.println(imu_enabled ? "ON" : "OFF");
}

static void printStatus() {
  Serial.print("STATUS,IMU1=");
  Serial.print(state.initialized ? "OK" : "FAIL");
  Serial.print(",MAG=");
  Serial.print(magnetometer_enabled ? "ON" : "OFF");
  Serial.print(",IMU=");
  Serial.println(imu_enabled ? "ON" : "OFF");
}

static void handleCommand(String command) {
  command.trim();
  command.toUpperCase();
  if (command.length() == 0) return;

  if (command == "MAG ON" || command == "MAG=ON" || command == "MAG 1") {
    setMagnetometerEnabled(true);
  } else if (command == "MAG OFF" || command == "MAG=OFF" || command == "MAG 0") {
    setMagnetometerEnabled(false);
  } else if (command == "IMU ON" || command == "IMU=ON") {
    setImuEnabled(true);
  } else if (command == "IMU OFF" || command == "IMU=OFF") {
    setImuEnabled(false);
  } else if (command == "MAG?" || command == "STATUS") {
    printStatus();
  } else if (command == "HELP") {
    Serial.println("COMMANDS,MAG ON|MAG OFF|IMU ON|IMU OFF|MAG?|STATUS|HELP");
  } else {
    Serial.print("ERROR,UNKNOWN_COMMAND,");
    Serial.println(command);
  }
}

static void pollSerialCommands() {
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\n' || character == '\r') {
      handleCommand(serial_command);
      serial_command = "";
    } else if (serial_command.length() < 48) {
      serial_command += character;
    }
  }
}

static void updateSensor() {
  if (!imu_enabled || !state.initialized || !imu.dataAvailable()) return;
  state.qw = imu.getQuatReal();
  state.qx = imu.getQuatI();
  state.qy = imu.getQuatJ();
  state.qz = imu.getQuatK();
  state.has_sample = true;
}

static void printSensorLine() {
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;
  quaternionToEuler(state.qw, state.qx, state.qy, state.qz, roll, pitch, yaw);

  Serial.print("IMU1,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(state.qw, 6);
  Serial.print(",");
  Serial.print(state.qx, 6);
  Serial.print(",");
  Serial.print(state.qy, 6);
  Serial.print(",");
  Serial.print(state.qz, 6);
  Serial.print(",");
  Serial.print(roll, 3);
  Serial.print(",");
  Serial.print(pitch, 3);
  Serial.print(",");
  Serial.print(yaw, 3);
  Serial.print(",");
  Serial.println(magnetometer_enabled ? 1 : 0);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 4000) {
  }

  Serial.println("SPARKFUN_BNO080,READY");
  Serial.println("PINOUT,IMU1=Wire/SDA17/SCL16/ADDR0x4B");

  IMU_BUS.begin();
  IMU_BUS.setSCL(IMU_SCL_PIN);
  IMU_BUS.setSDA(IMU_SDA_PIN);
  IMU_BUS.setClock(400000);

  delay(STARTUP_DELAY_MS);

  Serial.print("INIT,IMU1,");
  state.initialized = imu.begin(IMU_ADDRESS, IMU_BUS);
  Serial.println(state.initialized ? "OK" : "FAIL");
  if (state.initialized) configureReports();

  printStatus();
  Serial.println("DATA_FORMAT,IMU#,time_ms,qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg,mag_enabled");
}

void loop() {
  pollSerialCommands();
  updateSensor();

  static uint32_t last_print = 0;
  const uint32_t now = millis();
  if (state.has_sample && now - last_print >= PRINT_PERIOD_MS) {
    printSensorLine();
    last_print = now;
  }
}
