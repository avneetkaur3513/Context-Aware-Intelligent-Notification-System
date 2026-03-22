# Context-Aware Network-Based Notification System

## Problem Statement
Most applications send notifications without considering **user context** (driving, studying, sleeping) or **network conditions** (weak/strong connectivity). This can cause:
- bad timing
- notification fatigue
- interruptions during critical activities
- wasted network usage when connectivity is weak

This project builds a **smart notification decision system** that classifies each notification into one of:
- `send_now`
- `delay`
- `suppress`

Based on:
- `time_of_day`
- `activity`
- `network_strength`
- `notification_type`
- `priority`
- `previous_response`

---

## Why This Project Matters
In real-world mobile and distributed systems, network and context awareness improves user experience by:
- reducing unnecessary interruptions
- prioritizing critical alerts
- adapting delivery when connectivity is limited

---

## How Networking Concepts Are Used
This project incorporates networking concepts via `network_strength`:
- **Weak network** can cause delayed delivery or batching to reduce retries and wasted transmissions.
- **Strong network** supports immediate delivery for actionable notifications.
- Decision-making mimics application-layer behavior that adapts to link quality / connectivity conditions.

---

## How AI Is Integrated
For non-obvious cases, the system uses a **machine learning classifier** trained on a labeled dataset to predict:
- `send_now`
- `delay`
- `suppress`

The AI model helps handle complex combinations of context + network where rule-writing becomes difficult.

---

## Hybrid Architecture: Rules + ML
This project uses a **hybrid decision system**:

### 1) Rule-Based Logic (Overrides)
Used for obvious and critical cases:
- emergency + high priority → `send_now`
- driving + low priority → `suppress`
- sleeping + low/medium priority → `delay`

### 2) Machine Learning Model (Non-obvious Cases)
If no rule matches, the ML model predicts the action using contextual features.

This is realistic for production systems: rules provide safety/interpretability, ML provides flexibility.

---

## Features
- Synthetic dataset generation (800–1000 rows)
- RandomForest baseline classifier
- Hybrid decision inference (rules first, ML second)
- Streamlit dashboard:
  - input form
  - final decision (highlighted)
  - confidence score (when ML is used)
  - explanation text
  - dataset distribution chart

---

## Dataset Description
File: `data/notifications_dataset.csv`

Columns:
- `time_of_day`: morning, afternoon, evening, night
- `activity`: driving, studying, working, free_time, sleeping
- `network_strength`: weak, medium, strong
- `notification_type`: social, work, promotional, reminder, emergency
- `priority`: low, medium, high
- `previous_response`: opened, ignored, delayed
- `action`: send_now, delay, suppress

Dataset logic includes realistic tendencies like:
- emergency + high → mostly send_now
- driving + low → mostly suppress
- sleeping + non-urgent → mostly delay
- weak network shifts non-urgent notifications toward delay/suppress

---

## Model Training Process
Script: `train_model.py`

Steps:
1. Load dataset
2. Train/test split
3. One-hot encode categorical features using `OneHotEncoder`
4. Train `RandomForestClassifier`
5. Print:
   - accuracy
   - classification report
   - confusion matrix
6. Save artifacts:
   - `model/notification_model.pkl`
   - `model/preprocessor.pkl`

---

## Streamlit Dashboard Overview
Script: `app.py`

Dashboard includes:
- context inputs in a form
- rule-based override checks
- ML prediction + confidence score (when applicable)
- explanation for the decision
- dataset preview and class distribution plot

---

## Setup & Installation

### 1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Generate dataset
```bash
python generate_data.py
```
This creates:
- `data/notifications_dataset.csv`

### Step 2: Train model
```bash
python train_model.py
```
This creates:
- `model/notification_model.pkl`
- `model/preprocessor.pkl`

### Step 3: Run Streamlit app
```bash
streamlit run app.py
```

---

## Example Predictions
- driving + social + low priority → likely `suppress`
- sleeping + reminder + medium priority → likely `delay`
- working + work + high priority → likely `send_now`

Note: predictions vary slightly because the dataset is synthetic and includes some realistic noise.
<img width="1081" height="786" alt="4eb54ec7d17ebfb42168632b74d684695f940012128ca1cd607bb015" src="https://github.com/user-attachments/assets/3bba4394-6cc9-46f9-8016-7b393e3c77f4" />

---

## Limitations
- Synthetic dataset (not collected from real users/devices)
- Network is simplified to weak/medium/strong (no latency/jitter/packet loss)
- Model is a baseline RandomForest (no deep personalization yet)

---

## Future Improvements
- Add more networking metrics: latency, jitter, packet loss, signal strength (RSSI)
- Add user-level personalization and feedback loops (online learning)
- Add notification batching and retry scheduling strategies
- Add temporal features (weekday/weekend, hour-of-day)
- Add model interpretability (feature importances or SHAP)
