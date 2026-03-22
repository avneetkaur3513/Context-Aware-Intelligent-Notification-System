import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Context-Aware Dataset Generator
# -----------------------------
# Goal: Create realistic synthetic data for a hybrid (rules + ML) project.
# The "action" labels are generated using probabilistic logic based on:
# - time_of_day
# - activity
# - network_strength
# - notification_type
# - priority
# - previous_response
#
# We intentionally introduce some noise so the ML model has something to learn.


TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
ACTIVITY = ["driving", "studying", "working", "free_time", "sleeping"]
NETWORK = ["weak", "medium", "strong"]
NOTIF_TYPE = ["social", "work", "promotional", "reminder", "emergency"]
PRIORITY = ["low", "medium", "high"]
PREV_RESPONSE = ["opened", "ignored", "delayed"]

ACTIONS = ["send_now", "delay", "suppress"]


@dataclass
class Weights:
    send_now: float
    delay: float
    suppress: float

    def normalize(self) -> "Weights":
        s = self.send_now + self.delay + self.suppress
        return Weights(self.send_now / s, self.delay / s, self.suppress / s)

    def to_list(self) -> List[float]:
        w = self.normalize()
        return [w.send_now, w.delay, w.suppress]


def _base_action_weights() -> Weights:
    # Neutral baseline (slightly favor delay in ambiguous cases)
    return Weights(send_now=0.34, delay=0.38, suppress=0.28)


def _apply_realistic_bias(
    time_of_day: str,
    activity: str,
    network_strength: str,
    notification_type: str,
    priority: str,
    previous_response: str,
) -> Weights:
    w = _base_action_weights()

    # 1) Emergency / High priority bias
    if notification_type == "emergency":
        if priority == "high":
            w.send_now += 1.7
            w.delay -= 0.2
            w.suppress -= 0.3
        elif priority == "medium":
            w.send_now += 0.9
            w.delay += 0.2
        else:
            w.send_now += 0.5
            w.delay += 0.3

    # 2) Driving context: low priority should mostly suppress
    if activity == "driving":
        if priority == "low":
            w.suppress += 1.4
            w.send_now -= 0.2
        elif priority == "medium":
            w.delay += 0.5
            w.suppress += 0.3
        else:  # high priority while driving
            w.send_now += 0.4
            w.delay += 0.2

    # 3) Sleeping context: non-urgent should mostly delay
    if activity == "sleeping":
        if notification_type in ["promotional", "social", "reminder"] and priority in ["low", "medium"]:
            w.delay += 1.4
            w.send_now -= 0.25
        elif notification_type == "work" and priority == "high":
            # Sometimes still delay; sometimes send (depends on realistic settings)
            w.delay += 0.6
            w.send_now += 0.2
        elif notification_type == "emergency":
            w.send_now += 0.7

    # 4) Studying: reduce noise, lean delay for non-urgent
    if activity == "studying":
        if notification_type in ["social", "promotional"] and priority != "high":
            w.delay += 0.8
            w.suppress += 0.2
        if notification_type == "reminder" and priority in ["medium", "high"]:
            w.send_now += 0.4

    # 5) Working: work notifications high priority should send
    if activity == "working":
        if notification_type == "work" and priority in ["medium", "high"]:
            w.send_now += 0.9
        if notification_type == "promotional":
            w.suppress += 0.8

    # 6) Free time: social medium priority may send
    if activity == "free_time":
        if notification_type == "social" and priority == "medium":
            w.send_now += 0.7
        if notification_type == "promotional":
            w.delay += 0.2
            w.suppress += 0.3

    # 7) Time-of-day influence
    if time_of_day == "night":
        # At night, discourage send_now unless urgent
        if notification_type != "emergency" and priority != "high":
            w.delay += 0.6
            w.suppress += 0.2

    # 8) Network strength effect: weak network -> delay for non-urgent
    if network_strength == "weak":
        if notification_type in ["promotional", "social"] and priority in ["low", "medium"]:
            w.delay += 0.8
            w.suppress += 0.2
        if notification_type == "work" and priority == "high":
            # Might still send now, but with some delay tendency due to connectivity
            w.delay += 0.3
        if notification_type == "emergency":
            # Emergency should mostly send_now even on weak network
            w.send_now += 0.4

    if network_strength == "strong":
        if notification_type in ["reminder", "work"] and priority in ["medium", "high"]:
            w.send_now += 0.2

    # 9) Previous response: if ignored often, suppress/delay more
    if previous_response == "ignored":
        if notification_type in ["promotional", "social"] and priority != "high":
            w.suppress += 0.8
        else:
            w.delay += 0.2
    elif previous_response == "delayed":
        w.delay += 0.2
    elif previous_response == "opened":
        w.send_now += 0.2

    # Clip negative weights (can happen due to subtracts)
    w.send_now = max(w.send_now, 0.01)
    w.delay = max(w.delay, 0.01)
    w.suppress = max(w.suppress, 0.01)
    return w


def _sample_action(weights: Weights) -> str:
    probs = weights.to_list()
    return random.choices(ACTIONS, weights=probs, k=1)[0]


def generate_dataset(num_rows: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    rows: List[Dict[str, str]] = []
    for _ in range(num_rows):
        time_of_day = random.choice(TIME_OF_DAY)
        activity = random.choice(ACTIVITY)
        network_strength = random.choice(NETWORK)
        notification_type = random.choice(NOTIF_TYPE)
        priority = random.choice(PRIORITY)
        previous_response = random.choice(PREV_RESPONSE)

        weights = _apply_realistic_bias(
            time_of_day=time_of_day,
            activity=activity,
            network_strength=network_strength,
            notification_type=notification_type,
            priority=priority,
            previous_response=previous_response,
        )
        action = _sample_action(weights)

        rows.append(
            {
                "time_of_day": time_of_day,
                "activity": activity,
                "network_strength": network_strength,
                "notification_type": notification_type,
                "priority": priority,
                "previous_response": previous_response,
                "action": action,
            }
        )

    df = pd.DataFrame(rows)

    # Small post-processing sanity nudges:
    # Ensure emergency+high is "mostly send_now"
    mask = (df["notification_type"] == "emergency") & (df["priority"] == "high")
    if mask.sum() > 0:
        flip_candidates = df[mask & (df["action"] != "send_now")].sample(
            frac=0.6, random_state=seed
        )
        df.loc[flip_candidates.index, "action"] = "send_now"

    # Ensure driving+low is "mostly suppress"
    mask = (df["activity"] == "driving") & (df["priority"] == "low")
    if mask.sum() > 0:
        flip_candidates = df[mask & (df["action"] != "suppress")].sample(
            frac=0.6, random_state=seed
        )
        df.loc[flip_candidates.index, "action"] = "suppress"

    # Ensure sleeping+non-urgent is "mostly delay"
    mask = (df["activity"] == "sleeping") & (df["notification_type"].isin(["social", "promotional", "reminder"])) & (
        df["priority"].isin(["low", "medium"])
    )
    if mask.sum() > 0:
        flip_candidates = df[mask & (df["action"] != "delay")].sample(
            frac=0.6, random_state=seed
        )
        df.loc[flip_candidates.index, "action"] = "delay"

    return df


def main() -> None:
    # Generate 800-1000 rows as required
    num_rows = random.randint(800, 1000)

    df = generate_dataset(num_rows=num_rows, seed=42)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "notifications_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"[OK] Generated dataset with {len(df)} rows -> {out_path}")
    print("\nClass distribution:")
    print(df["action"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()