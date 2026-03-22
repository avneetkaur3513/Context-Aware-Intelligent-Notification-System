import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "notification_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

FEATURE_COLUMNS = [
    "time_of_day",
    "activity",
    "network_strength",
    "notification_type",
    "priority",
    "previous_response",
]

VALID_ACTIONS = ["send_now", "delay", "suppress"]


def apply_rule_based_override(input_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns:
        (action, reason) if a rule overrides; otherwise (None, None).
    """
    time_of_day = input_dict.get("time_of_day")
    activity = input_dict.get("activity")
    network_strength = input_dict.get("network_strength")
    notification_type = input_dict.get("notification_type")
    priority = input_dict.get("priority")
    previous_response = input_dict.get("previous_response")

    # Rule 1: emergency + high priority -> send_now
    if notification_type == "emergency" and priority == "high":
        return "send_now", "Sent now because the notification is urgent and should not be missed."

    # Rule 2: driving + low priority -> suppress
    if activity == "driving" and priority == "low":
        return "suppress", "Suppressed because the user is driving and the notification is low priority."

    # Rule 3: sleeping + low or medium priority -> delay (non-urgent)
    if activity == "sleeping" and priority in ["low", "medium"] and notification_type != "emergency":
        return "delay", "Delayed because the user is sleeping and the notification is non-urgent."

    # Optional realistic rule: weak network + promotional -> delay or suppress (choose delay here)
    if network_strength == "weak" and notification_type == "promotional" and priority != "high":
        return "delay", "Delayed due to weak network and low-urgency promotional content."

    # No override
    return None, None


def load_model_and_preprocessor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Train the model first by running train_model.py "
            "after generating the dataset with generate_data.py."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def _to_feature_frame(input_dict: Dict[str, Any]) -> pd.DataFrame:
    row = {col: input_dict.get(col) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_notification_action(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict using rule-based overrides first, else ML model.

    Returns a dict with:
      - action
      - source: "rules" or "ml"
      - confidence (float or None)
      - explanation
    """
    # 1) Rule-based overrides (obvious cases)
    rule_action, rule_reason = apply_rule_based_override(input_dict)
    if rule_action is not None:
        return {
            "action": rule_action,
            "source": "rules",
            "confidence": None,
            "explanation": rule_reason,
        }

    # 2) ML prediction for non-obvious cases
    model, preprocessor = load_model_and_preprocessor()
    X = _to_feature_frame(input_dict)

    X_transformed = preprocessor.transform(X)
    pred = model.predict(X_transformed)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_transformed)[0]
        classes = list(model.classes_)
        if pred in classes:
            confidence = float(proba[classes.index(pred)])

    explanation = generate_explanation(input_dict, pred, rule_applied=False)

    return {
        "action": pred,
        "source": "ml",
        "confidence": confidence,
        "explanation": explanation,
    }


def generate_explanation(
    input_dict: Dict[str, Any],
    predicted_action: str,
    rule_applied: bool = False
) -> str:
    """
    Explanation rules:
    - If suppressed because driving + low priority:
      “Suppressed because the user is driving and the notification is low priority.”
    - If delayed because sleeping:
      “Delayed because the user is sleeping and the notification is non-urgent.”
    - If sent because emergency/high priority:
      “Sent now because the notification is urgent and should not be missed.”
    - Otherwise dynamic explanation based on model prediction.
    """
    activity = input_dict.get("activity")
    notification_type = input_dict.get("notification_type")
    priority = input_dict.get("priority")
    network_strength = input_dict.get("network_strength")
    previous_response = input_dict.get("previous_response")
    time_of_day = input_dict.get("time_of_day")

    # If the rule engine was used, those reasons are already handled in apply_rule_based_override().
    # But keep fallback logic here for completeness.
    if notification_type == "emergency" and priority == "high" and predicted_action == "send_now":
        return "Sent now because the notification is urgent and should not be missed."
    if activity == "driving" and priority == "low" and predicted_action == "suppress":
        return "Suppressed because the user is driving and the notification is low priority."
    if activity == "sleeping" and priority in ["low", "medium"] and predicted_action == "delay":
        return "Delayed because the user is sleeping and the notification is non-urgent."

    # Dynamic explanation (interview-friendly, short)
    parts = []
    parts.append(f"Predicted '{predicted_action}' based on context and network conditions.")

    # Add 1-2 key drivers (human-readable heuristic explanation)
    if predicted_action == "delay":
        if network_strength == "weak":
            parts.append("Weak network suggests delaying non-critical delivery.")
        if activity in ["studying", "working", "sleeping"]:
            parts.append(f"User activity ('{activity}') suggests reducing interruption.")
    elif predicted_action == "suppress":
        if previous_response == "ignored":
            parts.append("User previously ignored similar notifications, so suppression reduces fatigue.")
        if notification_type == "promotional":
            parts.append("Promotional notifications are often low value in busy contexts.")
    else:  # send_now
        if priority == "high":
            parts.append("High priority increases the chance the user should see it immediately.")
        if notification_type in ["work", "reminder"] and activity in ["working", "free_time"]:
            parts.append("Context indicates the user is likely able to act on it now.")

    # Add small detail for realism
    parts.append(f"(time={time_of_day}, activity={activity}, network={network_strength})")
    return " ".join(parts)


def plot_class_distribution(df: pd.DataFrame):
    """
    Returns a matplotlib figure showing action distribution.
    """
    if "action" not in df.columns:
        raise ValueError("DataFrame must include an 'action' column for plotting.")

    counts = df["action"].value_counts().reindex(["send_now", "delay", "suppress"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#2E86AB", "#F6C85F", "#C70039"])
    ax.set_title("Notification Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig