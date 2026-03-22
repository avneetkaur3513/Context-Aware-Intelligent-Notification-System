import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = os.path.join("data", "notifications_dataset.csv")
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
TARGET_COLUMN = "action"


def build_pipeline(random_state: int = 42) -> Pipeline:
    categorical_features = FEATURE_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=random_state,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Run generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Typical split for assignment/demo purposes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n====================")
    print("Model Evaluation")
    print("====================")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["send_now", "delay", "suppress"]))

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Also store preprocessor separately (useful for explanations / interview clarity)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print(f"\n[OK] Saved model -> {MODEL_PATH}")
    print(f"[OK] Saved preprocessor -> {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()