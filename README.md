# Adaptive Vision-Based Posture Monitoring with Bio-Inspired Fatigue Modeling 👤💻

<img width="900" alt="demo" src="https://github.com/user-attachments/assets/2616b45a-a209-4d79-9f13-0a397d42a8a4" />

---

## 📌 Overview

This project is a real-time posture monitoring system built using **MediaPipe Pose** and a standard webcam.

Unlike basic posture apps that only classify posture as *good* or *bad*, this system models **cumulative fatigue** using a biologically inspired approach.

It combines:

- Real-time skeletal landmark detection  
- Orientation-aware posture analysis  
- Personalized calibration  
- Multi-angle deviation modeling  
- Cumulative fatigue accumulation  
- Adaptive alert timing  
- Probabilistic break recommendation  
- (Optional) Recovery and eye-strain modeling  

The system behaves like a **fatigue-aware ergonomic assistant**, not just a posture checker.

---

# 🧠 Core Idea

Most posture monitoring systems use fixed thresholds:

> If angle > threshold → Bad posture

However, real fatigue does not behave in a binary manner.

Fatigue:

- Builds gradually over time  
- Depends on deviation intensity  
- Decreases during rest  
- Reduces tolerance as it accumulates  

This project models posture strain as a **time-dependent fatigue state variable** rather than a simple threshold event.

---

# ⚙️ System Workflow

## 1️⃣ Landmark Extraction

MediaPipe Pose extracts 33 body landmarks in real time using your webcam.

---

## 2️⃣ Orientation Detection

The system automatically detects:

- **Front View**
- **Side View**

Different posture models are applied depending on the detected orientation.

---

## 3️⃣ Personalized Calibration

During the first few seconds:

- Your natural neutral posture is recorded  
- Baseline angles are computed  
- All future deviations are compared to your own reference  

This avoids using global fixed thresholds.

---

## 4️⃣ Multi-Angle Severity Modeling

Posture deviation is computed as:
D = |θ - θ_ref|


Severity is normalized:
Severity = D / T_angle


For side view, a **2-out-of-3 rule** is used across multiple angles to ensure robustness.

A moving average filter reduces landmark noise.

---

## 5️⃣ Cumulative Fatigue Model

Fatigue accumulates over time:
F(t + Δt) = F(t) + (S_baseline + k × Severity) Δt


- Neutral posture causes slow fatigue accumulation  
- Poor posture accelerates fatigue  
- Fatigue is bounded between 0 and 100  

---

## 6️⃣ Adaptive Alert Window

As fatigue increases, posture tolerance decreases:
T_limit = T_max - (F / 100)(T_max - T_min)


- Fresh user → more tolerance  
- Fatigued user → stricter posture enforcement  

---

## 7️⃣ Break Recommendation (Sigmoid Model)
P_break = 1 / (1 + e^(-a(F - F_critical)))


When fatigue crosses a high threshold:

- The system transitions from posture correction  
- To break recommendation  

---

## 8️⃣ Recovery Model (Extended Component)

During detected breaks:
F(t + Δt) = F(t) e^(-β Δt)


Fatigue gradually decays, modeling physiological recovery.

---

## 9️⃣ Eye-Strain Model (EAR – Extended Component)

Eye Aspect Ratio (EAR) can be used to estimate visual fatigue and integrate it into the fatigue model.

This enables multi-modal fatigue estimation.

---

# 📊 On-Screen Output

The system displays:

- Orientation (Front / Side)  
- Posture status  
- Fatigue score (0–100)  
- Adaptive tolerance timer  
- Break probability  
- Real-time visual overlays  

---

# 🧪 Experimental Logging

The system can log:

- Time  
- Severity  
- Fatigue  
- Break probability  
- Orientation  
- Posture status  

Logs are saved as CSV files for plotting and analysis.

---

# 🚀 How To Run This Project

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/posture-monitor
cd posture-monitor
```
## Step 2: Create a Virtual Environment - (use python 3.10.x)

python -m venv venv

## Step 3: Activate it:
```bash
venv\Scripts\activate
```bash

Install requirements.txt
```bash
pip install -r requirements.txt
```

## Step 4: Run the Application

To use your webcam:
```bash
python app.py
```

To use a video file:
```bash
python app.py --video path/to/video.mp4
```
## 📦 Requirements

- Python 3.8+

- Webcam

- OpenCV

- MediaPipe

- NumPy

- (Optional) Matplotlib / Pandas for analysis

## 📌 Why This Project Is Different

This system:

- Models fatigue dynamically

- Adapts posture tolerance over time

- Uses personalized calibration

- Integrates ergonomic research principles

- Requires no wearable sensors

- It bridges computer vision + biomechanics + fatigue modeling.

## 🛠 Possible Improvements

- EMG-based physiological validation

- Long-term user adaptation

- Machine learning parameter tuning

- Clinical or workplace validation studies

## 🙌 Acknowledgment

Built using MediaPipe Pose
 for real-time skeletal tracking.

## 📜 License

This project is licensed under the MIT License.
