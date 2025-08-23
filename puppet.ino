// #include <Wire.h>
// #include <QuickPID.h>
// #include <driver/twai.h>

// // Pin definitions
// #define TX_GPIO_NUM 33
// #define RX_GPIO_NUM 32
// const uint8_t motorPins[4][2] = {{21, 19}, {5, 18}, {12, 13}, {26, 25}}; 
// const uint8_t encoderPins[4][2] = {{22, 23}, {4, 2}, {14, 27}, {34, 35}};

// const uint8_t motorChannels[4][2] = {
//   {0, 1}, // Channels for Motor 0
//   {2, 3}, // Channels for Motor 1
//   {4, 5}, // Channels for Motor 2
//   {6, 7}  // Channels for Motor 3
// };

// // For Serial Debug
// #define SERIAL_DEBUG true

// // Motor and encoder constants
// #define NUM_MOTORS 4
// #define GEAR_RATIO 21.3
// #define ENCODER_PPR 11
// #define COUNTS_PER_REV (GEAR_RATIO * ENCODER_PPR)
// #define UPDATE_INTERVAL 100 // ms
// #define MAX_RPM 280.0

// // Message
// #define DIR_FORWARD 0xA1
// #define DIR_BACKWARD 0xA2
// #define DIR_LEFT 0xA3
// #define DIR_RIGHT 0xA4
// #define DIR_STOP 0xAF

// // Global variables
// const uint32_t PWM_FREQ = 31000;
// const uint8_t PWM_RESOLUTION = 10; // 10-bit
// const uint16_t PWM_MAX = (1 << PWM_RESOLUTION) - 1;

// volatile int32_t encoderCounts[4] = {0, 0, 0, 0};
// long lastEncoderCounts[4] = {0, 0, 0, 0};
// unsigned long lastUpdateTime = 0;

// float currentRPM[NUM_MOTORS] = {0.0, 0.0, 0.0, 0.0};
// float targetRPM[NUM_MOTORS] = {0.0, 0.0, 0.0, 0.0};
// float dutyCycle[NUM_MOTORS] = {0.0, 0.0, 0.0, 0.0};
// unsigned long now = 0;

// // PID constants
// const float KP = 1.59f;
// const float KI = 11.787f;
// const float KD = 0.0f;
// QuickPID myPID[NUM_MOTORS] = {
//   QuickPID(&currentRPM[0], &dutyCycle[0], &targetRPM[0], KP, KI, KD, QuickPID::Action::direct),
//   QuickPID(&currentRPM[1], &dutyCycle[1], &targetRPM[1], KP, KI, KD, QuickPID::Action::direct),
//   QuickPID(&currentRPM[2], &dutyCycle[2], &targetRPM[2], KP, KI, KD, QuickPID::Action::direct),
//   QuickPID(&currentRPM[3], &dutyCycle[3], &targetRPM[3], KP, KI, KD, QuickPID::Action::direct),
// };

// // ================ TWAI/CAN Configuration ================ //
// const twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
// twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

// // SỬA LỖI 1: Ép kiểu GPIO sang gpio_num_t
// const twai_general_config_t g_config = {
//     .mode = TWAI_MODE_NORMAL,
//     .tx_io = (gpio_num_t)TX_GPIO_NUM, // Ép kiểu
//     .rx_io = (gpio_num_t)RX_GPIO_NUM, // Ép kiểu
//     .clkout_io = TWAI_IO_UNUSED,
//     .bus_off_io = TWAI_IO_UNUSED,
//     .tx_queue_len = 5,
//     .rx_queue_len = 5,
//     .alerts_enabled = TWAI_ALERT_NONE,
//     .clkout_divider = 0,
//     .intr_flags = ESP_INTR_FLAG_LEVEL1
// };

// // ================ KHAI BÁO NGUYÊN MẪU HÀM ================ //
// // SỬA LỖI 2: Khai báo nguyên mẫu các hàm trước khi sử dụng
// void measureMotorSpeeds();
// void pidCompute();
// void runMotors();
// void processCANControl();
// void sendCANRPM();
// void setupTWAI();
// // ========================================================= //

// // Encoder ISRs
// void IRAM_ATTR encoderISR0() {
//   if (digitalRead(encoderPins[0][1]) == HIGH) encoderCounts[0]--;
//   else encoderCounts[0]++;
// }

// void IRAM_ATTR encoderISR1() {
//   if (digitalRead(encoderPins[1][1]) == HIGH) encoderCounts[1]--;
//   else encoderCounts[1]++;
// }

// void IRAM_ATTR encoderISR2() {
//   if (digitalRead(encoderPins[2][1]) == HIGH) encoderCounts[2]++;
//   else encoderCounts[2]--;
// }

// void IRAM_ATTR encoderISR3() {
//   if (digitalRead(encoderPins[3][1]) == HIGH) encoderCounts[3]++;
//   else encoderCounts[3]--;
// }

// // Cấu hình bộ lọc chỉ nhận ID 0x21
// void setupTWAI() {
//   f_config.acceptance_code = (0x21 << 21) | 0x1FFFFF;
//   f_config.acceptance_mask = ~(0x7FF << 21);
//   f_config.single_filter = true;

//   // Cài đặt driver TWAI
//   if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
//     if(SERIAL_DEBUG) Serial.println("Failed to install TWAI driver");
//     while(1);
//   }

//   // Khởi động driver TWAI
//   if (twai_start() != ESP_OK) {
//     if(SERIAL_DEBUG) Serial.println("Failed to start TWAI");
//     while(1);
//   }

//   if(SERIAL_DEBUG) Serial.println("TWAI started successfully");
// }

// void setup() {
//   if(SERIAL_DEBUG) Serial.begin(115200);

//   // Initialize motor pins
//   for (uint8_t i = 0; i < NUM_MOTORS; i++) {
//     pinMode(motorPins[i][0], OUTPUT);
//     pinMode(motorPins[i][1], OUTPUT);
//   }

//   // Initialize encoder pins
//   for (uint8_t i = 0; i < NUM_MOTORS; i++) {
//     pinMode(encoderPins[i][0], INPUT_PULLUP);
//     pinMode(encoderPins[i][1], INPUT_PULLUP);
//   }

//   // Attach interrupts on rising edge
//   attachInterrupt(digitalPinToInterrupt(encoderPins[0][0]), encoderISR0, RISING);
//   attachInterrupt(digitalPinToInterrupt(encoderPins[1][0]), encoderISR1, RISING);
//   attachInterrupt(digitalPinToInterrupt(encoderPins[2][0]), encoderISR2, RISING);
//   attachInterrupt(digitalPinToInterrupt(encoderPins[3][0]), encoderISR3, RISING);

//   // Initialize PWM
//   for (uint8_t i = 0; i < NUM_MOTORS; i++) {
//     for (uint8_t j = 0; j < 2; j++) {
//       uint8_t channel = motorChannels[i][j];
//       ledcSetup(channel, PWM_FREQ, PWM_RESOLUTION);
//       ledcAttachPin(motorPins[i][j], channel);
//     }
//   }

//   // Initialize PID
//   for (uint8_t i = 0; i < NUM_MOTORS; i++) {
//     myPID[i].SetOutputLimits(-PWM_MAX, PWM_MAX);
//     myPID[i].SetMode(myPID[i].Control::automatic);
//     myPID[i].SetAntiWindupMode(myPID[i].iAwMode::iAwCondition);
//     myPID[i].SetSampleTimeUs(100000);
//     myPID[i].SetTunings(KP, KI, KD);
//     delay(50);
//   }

//   // Khởi tạo TWAI
//   setupTWAI();
// }

// void loop() {
//   processCANControl();
//   measureMotorSpeeds();
//   pidCompute();
//   runMotors();
//   sendCANRPM(); 
  
//   if(SERIAL_DEBUG) {
//     Serial.printf("%.2f,%.2f,%.2f,%.2f\n",currentRPM[0], currentRPM[1], currentRPM[2], currentRPM[3]);
//   }
// }

// // ================ ĐỊNH NGHĨA CÁC HÀM ================ //
// void measureMotorSpeeds() {
//   now = millis();
//   if (now - lastUpdateTime >= UPDATE_INTERVAL) {
//     for(uint8_t i = 0; i < NUM_MOTORS; i++) {
//       long currentCounts = encoderCounts[i];
//       long deltaCounts = currentCounts - lastEncoderCounts[i];
//       lastEncoderCounts[i] = currentCounts;
//       currentRPM[i] = (deltaCounts * (60000.0 / UPDATE_INTERVAL)) / COUNTS_PER_REV;
//     }
//     lastUpdateTime = now;
//   }
// }

// void pidCompute() {
//   for (uint8_t i = 0; i < NUM_MOTORS; i++) {
//     myPID[i].Compute();
//   }
// }

// void runMotors() {
//   for(uint8_t i = 0; i < NUM_MOTORS; i++) {
//     dutyCycle[i] = constrain(dutyCycle[i], -PWM_MAX, PWM_MAX);
//     if (dutyCycle[i] > 0) {
//       ledcWrite(motorChannels[i][0], PWM_MAX);
//       ledcWrite(motorChannels[i][1], PWM_MAX - (int)dutyCycle[i]);
//     } else if (dutyCycle[i] < 0) {
//       ledcWrite(motorChannels[i][0], PWM_MAX - (int)(-dutyCycle[i]));
//       ledcWrite(motorChannels[i][1], PWM_MAX);
//     } else {
//       ledcWrite(motorChannels[i][0], PWM_MAX);
//       ledcWrite(motorChannels[i][1], PWM_MAX);
//     }
//   }
// }

// void processCANControl() {
//   twai_message_t message;
  
//   if (twai_receive(&message, 0) == ESP_OK) {
//     if (message.identifier == 0x21 && message.data_length_code == 8) {
//       for (int i = 0; i < NUM_MOTORS; i++) {
//         int16_t rawRPM = (int16_t)((message.data[i * 2] << 8) | message.data[i * 2 + 1]);
//         targetRPM[i] = constrain(rawRPM, -MAX_RPM, MAX_RPM);
//       }

//       if (SERIAL_DEBUG) {
//         Serial.print("Target RPMs: ");
//         for (int i = 0; i < NUM_MOTORS; i++) {
//           Serial.print(targetRPM[i]);
//           Serial.print(" ");
//         }
//         Serial.println();
//       }
//     }
//   }
// }

// void sendCANRPM() {
//   static unsigned long lastSendTime = 0;
//   if (millis() - lastSendTime < 50) return;
//   lastSendTime = millis();

//   twai_message_t message;
//   message.identifier = 0x13;
//   message.data_length_code = 8;
//   message.flags = TWAI_MSG_FLAG_NONE;
  
//   for (int i = 0; i < NUM_MOTORS; i++) {
//     int16_t rpm_i = (int16_t)(constrain(currentRPM[i], -MAX_RPM, MAX_RPM)) * 100;
//     message.data[2 * i]     = (rpm_i >> 8) & 0xFF;
//     message.data[2 * i + 1] = rpm_i & 0xFF;
//   }

//   if (twai_transmit(&message, pdMS_TO_TICKS(10)) != ESP_OK) {
//     if(SERIAL_DEBUG) Serial.println("Failed to send RPM data");
//   }
// }



#include <Wire.h>
#include <QuickPID.h>
#include <driver/twai.h>
#include <esp_system.h>

// Pin definitions
#define TX_GPIO_NUM 33
#define RX_GPIO_NUM 32
const uint8_t motorPins[4][2] = {{21, 19}, {5, 18}, {12, 13}, {26, 25}}; 
const uint8_t encoderPins[4][2] = {{22, 23}, {4, 2}, {14, 27}, {34, 35}};

const uint8_t motorChannels[4][2] = {
  {0, 1}, {2, 3}, {4, 5}, {6, 7}
};

// For Serial Debug
#define SERIAL_DEBUG true

// Motor and encoder constants
#define NUM_MOTORS 4
#define GEAR_RATIO 21.3
#define ENCODER_PPR 11
#define COUNTS_PER_REV (GEAR_RATIO * ENCODER_PPR)
#define UPDATE_INTERVAL 100
#define MAX_RPM 280.0

// Global variables
const uint32_t PWM_FREQ = 31000;
const uint8_t PWM_RESOLUTION = 10;
const uint16_t PWM_MAX = (1 << PWM_RESOLUTION) - 1;

volatile int32_t encoderCounts[4] = {0};
long lastEncoderCounts[4] = {0};
unsigned long lastUpdateTime = 0;

float currentRPM[NUM_MOTORS] = {0.0};
float targetRPM[NUM_MOTORS] = {0.0};
float dutyCycle[NUM_MOTORS] = {0.0};
unsigned long now = 0;

// PID constants
const float KP = 1.59f;
const float KI = 11.787f;
const float KD = 0.0f;
QuickPID myPID[NUM_MOTORS] = {
  QuickPID(&currentRPM[0], &dutyCycle[0], &targetRPM[0], KP, KI, KD, QuickPID::Action::direct),
  QuickPID(&currentRPM[1], &dutyCycle[1], &targetRPM[1], KP, KI, KD, QuickPID::Action::direct),
  QuickPID(&currentRPM[2], &dutyCycle[2], &targetRPM[2], KP, KI, KD, QuickPID::Action::direct),
  QuickPID(&currentRPM[3], &dutyCycle[3], &targetRPM[3], KP, KI, KD, QuickPID::Action::direct),
};

// CAN Configuration
const twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

const twai_general_config_t g_config = {
    .mode = TWAI_MODE_NORMAL,
    .tx_io = (gpio_num_t)TX_GPIO_NUM,
    .rx_io = (gpio_num_t)RX_GPIO_NUM,
    .clkout_io = TWAI_IO_UNUSED,
    .bus_off_io = TWAI_IO_UNUSED,
    .tx_queue_len = 5,
    .rx_queue_len = 5,
    .alerts_enabled = TWAI_ALERT_NONE,
    .clkout_divider = 0,
    .intr_flags = ESP_INTR_FLAG_LEVEL1
};

// Function prototypes
void measureMotorSpeeds();
void pidCompute();
void runMotors();
void processCANControl();
void sendCANRPM();
void setupTWAI();
void processUSBReset(); // Hàm mới xử lý reset qua USB

// Encoder ISRs
void IRAM_ATTR encoderISR0() {
  digitalRead(encoderPins[0][1]) ? encoderCounts[0]-- : encoderCounts[0]++;
}

void IRAM_ATTR encoderISR1() {
  digitalRead(encoderPins[1][1]) ? encoderCounts[1]-- : encoderCounts[1]++;
}

void IRAM_ATTR encoderISR2() {
  digitalRead(encoderPins[2][1]) ? encoderCounts[2]++ : encoderCounts[2]--;
}

void IRAM_ATTR encoderISR3() {
  digitalRead(encoderPins[3][1]) ? encoderCounts[3]++ : encoderCounts[3]--;
}

void setupTWAI() {
  f_config.acceptance_code = (0x21 << 21) | 0x1FFFFF;
  f_config.acceptance_mask = ~(0x7FF << 21);
  f_config.single_filter = true;

  if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
    if(SERIAL_DEBUG) Serial.println("CAN driver install failed");
    while(1);
  }

  if (twai_start() != ESP_OK) {
    if(SERIAL_DEBUG) Serial.println("CAN start failed");
    while(1);
  }

  if(SERIAL_DEBUG) Serial.println("CAN started");
}

void setup() {
  // Khởi tạo Serial USB (tốc độ cao hơn nếu cần)
  Serial.begin(115200);
  
  // Chờ kết nối Serial (tối đa 5s)
  if(SERIAL_DEBUG) {
    unsigned long start = millis();
    while (!Serial && millis() - start < 5000) {
      delay(10);
    }
    Serial.println("ESP32 Motor Controller Started");
  }

  // Initialize motor pins
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    pinMode(motorPins[i][0], OUTPUT);
    pinMode(motorPins[i][1], OUTPUT);
  }

  // Initialize encoder pins
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    pinMode(encoderPins[i][0], INPUT_PULLUP);
    pinMode(encoderPins[i][1], INPUT_PULLUP);
  }

  // Attach interrupts on rising edge
  attachInterrupt(digitalPinToInterrupt(encoderPins[0][0]), encoderISR0, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPins[1][0]), encoderISR1, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPins[2][0]), encoderISR2, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPins[3][0]), encoderISR3, RISING);

  // Initialize PWM
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    for (uint8_t j = 0; j < 2; j++) {
      uint8_t channel = motorChannels[i][j];
      ledcSetup(channel, PWM_FREQ, PWM_RESOLUTION);
      ledcAttachPin(motorPins[i][j], channel);
    }
  }

  // Initialize PID
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    myPID[i].SetOutputLimits(-PWM_MAX, PWM_MAX);
    myPID[i].SetMode(myPID[i].Control::automatic);
    myPID[i].SetAntiWindupMode(myPID[i].iAwMode::iAwCondition);
    myPID[i].SetSampleTimeUs(100000);
    myPID[i].SetTunings(KP, KI, KD);
    delay(50);
  }

  // Khởi tạo TWAI
  setupTWAI();
}

void loop() {
  processUSBReset(); // Xử lý lệnh reset từ USB Serial
  processCANControl();
  measureMotorSpeeds();
  pidCompute();
  runMotors();
  sendCANRPM();
  
  if(SERIAL_DEBUG) {
    Serial.printf("RPM: %.2f,%.2f,%.2f,%.2f\n", currentRPM[0], currentRPM[1], currentRPM[2], currentRPM[3]);
  }
}

// Hàm xử lý lệnh reset từ USB Serial
void processUSBReset() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "RESET") {
      if (SERIAL_DEBUG) {
        Serial.println("RESET command received via USB");
      }
      
      // Dừng tất cả động cơ
      for (int i = 0; i < NUM_MOTORS; i++) {
        targetRPM[i] = 0;
        dutyCycle[i] = 0;
      }
      runMotors();
      delay(100);
      
      // Reset ESP32
      esp_restart();
    }
  }
}


// ================ ĐỊNH NGHĨA CÁC HÀM ================ //
void measureMotorSpeeds() {
  now = millis();
  if (now - lastUpdateTime >= UPDATE_INTERVAL) {
    for(uint8_t i = 0; i < NUM_MOTORS; i++) {
      long currentCounts = encoderCounts[i];
      long deltaCounts = currentCounts - lastEncoderCounts[i];
      lastEncoderCounts[i] = currentCounts;
      currentRPM[i] = (deltaCounts * (60000.0 / UPDATE_INTERVAL)) / COUNTS_PER_REV;
    }
    lastUpdateTime = now;
  }
}

void pidCompute() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    myPID[i].Compute();
  }
}

void runMotors() {
  for(uint8_t i = 0; i < NUM_MOTORS; i++) {
    dutyCycle[i] = constrain(dutyCycle[i], -PWM_MAX, PWM_MAX);
    if (dutyCycle[i] > 0) {
      ledcWrite(motorChannels[i][0], PWM_MAX);
      ledcWrite(motorChannels[i][1], PWM_MAX - (int)dutyCycle[i]);
    } else if (dutyCycle[i] < 0) {
      ledcWrite(motorChannels[i][0], PWM_MAX - (int)(-dutyCycle[i]));
      ledcWrite(motorChannels[i][1], PWM_MAX);
    } else {
      ledcWrite(motorChannels[i][0], PWM_MAX);
      ledcWrite(motorChannels[i][1], PWM_MAX);
    }
  }
}

void processCANControl() {
  twai_message_t message;
  
  if (twai_receive(&message, 0) == ESP_OK) {
    if (message.identifier == 0x21 && message.data_length_code == 8) {
      for (int i = 0; i < NUM_MOTORS; i++) {
        int16_t rawRPM = (int16_t)((message.data[i * 2] << 8) | message.data[i * 2 + 1]);
        targetRPM[i] = constrain(rawRPM, -MAX_RPM, MAX_RPM);
      }

      if (SERIAL_DEBUG) {
        Serial.print("Target RPMs: ");
        for (int i = 0; i < NUM_MOTORS; i++) {
          Serial.print(targetRPM[i]);
          Serial.print(" ");
        }
        Serial.println();
      }
    }
  }
}

void sendCANRPM() {
  static unsigned long lastSendTime = 0;
  if (millis() - lastSendTime < 50) return;
  lastSendTime = millis();

  twai_message_t message;
  message.identifier = 0x13;
  message.data_length_code = 8;
  message.flags = TWAI_MSG_FLAG_NONE;
  
  for (int i = 0; i < NUM_MOTORS; i++) {
    int16_t rpm_i = (int16_t)(constrain(currentRPM[i], -MAX_RPM, MAX_RPM)) * 100;
    message.data[2 * i]     = (rpm_i >> 8) & 0xFF;
    message.data[2 * i + 1] = rpm_i & 0xFF;
  }

  if (twai_transmit(&message, pdMS_TO_TICKS(10)) != ESP_OK) {
    if(SERIAL_DEBUG) Serial.println("Failed to send RPM data");
  }
}


// Hàm mới: Xử lý lệnh reset từ UART
// void processUARTReset() {
//   if (Serial2.available() > 0) {
//     String command = Serial2.readStringUntil('\n');
//     command.trim();
    
//     if (command == "RESET") {
//       if (SERIAL_DEBUG) {
//         Serial.println("Received RESET command. Restarting ESP32...");
//       }
      
//       // Dừng tất cả động cơ trước khi reset
//       for (int i = 0; i < NUM_MOTORS; i++) {
//         targetRPM[i] = 0;
//         dutyCycle[i] = 0;
//       }
//       runMotors();
      
//       delay(100);
      
//       // Reset ESP32
//       esp_restart();
//     }
//   }
// }
