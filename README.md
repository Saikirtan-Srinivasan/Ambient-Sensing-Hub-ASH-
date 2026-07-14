# ASH — Ambient Sensing Hub

> Occupancy Detection using Thermal Camera | Silicon Labs Edge Intelligence Challenge 2026

ASH is an offline, edge-AI system that uses a low-resolution thermal camera to count the number of people in a room and automatically adjust appliances like fans and lights accordingly — with zero cloud dependency.

---

## How to Reproduce

### Prerequisites

**Hardware required:**
- Silicon Labs SiWx917 microcontroller board
- MLX90640 32×24 thermal IR sensor
- IR transmitter module
- USB cable for flashing

**Software required:**
- [Simplicity Studio 5](https://www.silabs.com/developers/simplicity-studio)
- WiSeConnect 3 (WC3) SDK — installable via Simplicity Studio package manager
- Python 3.8+
- Node.js (optional, for tooling)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-repo/ASH.git
cd ASH
```

Install Python dependencies:

```bash
pip install numpy tensorflow matplotlib adafruit-circuitpython-mlx90640
```

---

### Step 2 — Hardware Wiring

Connect the **MLX90640 thermal sensor** to the **SiWx917** board via I2C:

| MLX90640 Pin | SiWx917 Pin |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | I2C SDA |
| SCL | I2C SCL |

Connect the **IR transmitter module**:

| IR Module Pin | SiWx917 Pin |
|---|---|
| VCC | 3.3V |
| GND | GND |
| DATA/IN | GPIO (e.g. P15) |

> Refer to the wiring diagram in `/docs/wiring_diagram.png` for exact pin numbers.

---

### Step 3 — Set Up Simplicity Studio

1. Download and install **Simplicity Studio 5** from the Silicon Labs website
2. Open Simplicity Studio → go to **Install** → search for **WiSeConnect 3 SDK** and install it
3. Connect your SiWx917 board to your PC via USB
4. In Simplicity Studio, the board should be auto-detected under **Connected Devices**

---

### Step 4 — Import and Configure the Firmware Project

1. In Simplicity Studio, go to **File → Import → Existing Project**
2. Navigate to the cloned repo's `/firmware` folder and import it
3. Open `config.h` and verify the following are set correctly:

```c
#define I2C_SDA_PIN     XX   // Match your wiring
#define I2C_SCL_PIN     XX
#define IR_TX_PIN       XX   // GPIO pin for IR transmitter
```

4. Open `ir_codes.h` and confirm your appliance IR command codes are present (fan speed codes, light on/off codes). If not, record them using the IR receiver script:

```bash
python scripts/record_ir_codes.py
```

---

### Step 5 — Add the TFLite Model

The pre-trained model is included in the repo at `model/ash_occupancy_model.tflite`.

It is embedded into the firmware as a C header array at:

```c
#include "model/ash_model_data.h"
```

If you want to use your own retrained model, replace `ash_model_data.h` by converting your `.tflite` file:

```bash
python scripts/convert_model_to_header.py model/your_model.tflite
```

---

### Step 6 — Build and Flash the Firmware

1. In Simplicity Studio, click the **Build** button (hammer icon) to compile the project
2. Resolve any dependency or path errors (ensure WC3 SDK is correctly linked)
3. Click **Flash** (or go to **Run → Debug**) to upload the firmware to the SiWx917 board
4. Open the **Serial Console** in Simplicity Studio to monitor live logs:
   - Baud rate: `115200`
   - Port: auto-detected USB port

---

### Step 7 — Power On and Test

1. Power the board via USB or an external 3.3V supply
2. Point the MLX90640 sensor toward the room
3. The system begins capturing thermal frames immediately — no configuration needed

Watch the serial console for output like:

```
[ASH] Frame captured: 32x24
[ASH] Occupancy detected: 2
[ASH] Sending IR command: FAN_MEDIUM
```

4. Verify that the target appliance (fan/light) responds to the IR signal

---

### Step 8 — Validate All Occupancy Classes

Test each scenario manually to confirm correct behavior:

| Scenario | Expected Log Output | Appliance State |
|---|---|---|
| Empty room | `Occupancy detected: 0` | OFF |
| 1 person in frame | `Occupancy detected: 1` | LOW |
| 2 people in frame | `Occupancy detected: 2` | MEDIUM |
| 3+ people in frame | `Occupancy detected: 3` | HIGH |

---

### Optional — Retrain the Model

If you want to collect your own thermal data and retrain the CNN:

**1. Collect thermal frames:**

```bash
python scripts/collect_data.py --label 0 --samples 200   # Empty room
python scripts/collect_data.py --label 1 --samples 200   # 1 person
python scripts/collect_data.py --label 2 --samples 200   # 2 people
python scripts/collect_data.py --label 3 --samples 200   # 3+ people
```

**2. Upload to Edge Impulse:**

- Go to [edgeimpulse.com](https://edgeimpulse.com) and create a new project
- Upload the collected dataset under **Data Acquisition**
- Design an impulse: raw thermal input → CNN classifier
- Train and validate the model

**3. Export and deploy:**

- Export as **TensorFlow Lite (quantized)** from Edge Impulse
- Replace `model/ash_occupancy_model.tflite` with the new file
- Regenerate the header: `python scripts/convert_model_to_header.py model/ash_occupancy_model.tflite`
- Rebuild and reflash the firmware (repeat Steps 6–7)

---

## Project Structure

```
ASH/
├── firmware/           # SiWx917 firmware source (C)
│   ├── config.h        # Pin and system configuration
│   ├── ir_codes.h      # IR command codes for appliances
│   └── main.c          # Main pipeline logic
├── model/
│   ├── ash_occupancy_model.tflite
│   └── ash_model_data.h
├── scripts/
│   ├── collect_data.py         # Thermal data collection
│   ├── record_ir_codes.py      # IR code recorder
│   └── convert_model_to_header.py
├── docs/
│   └── wiring_diagram.png
└── README.md
```

---

## System Pipeline

```
Thermal Sensor → Preprocess → CNN Inference → Decision Logic → IR Actuation
  (32×24 grid)   (normalize)   (TFLite)       (0→OFF, 1→LOW,   (fan / lights)
                                               2→MED, 3→HIGH)
```

---

## Hardware & Software

| Component | Details |
|---|---|
| Microcontroller | Silicon Labs SiWx917 |
| Thermal Sensor | MLX90640 (32×24 IR array) |
| Actuation | IR transmitter module |
| ML Framework | TensorFlow Lite (quantized CNN) |
| Training Platform | Edge Impulse |
| Firmware SDK | WiSeConnect 3 / Simplicity Studio |
| Scripting | Python 3.8+ |

---

## About

Team ASH — electronics and embedded systems engineering students passionate about TinyML, edge AI, and building practical solutions that work without the cloud.

**Links:** [GitHub](#) | [LinkedIn](#)

---

*ASH — Ambient Sensing Hub | Silicon Labs Edge Intelligence Challenge 2026*

