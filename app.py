import os
import sys
import subprocess

import pandas as pd
import streamlit as st

from utils import predict_notification_action, plot_class_distribution

DATA_PATH = os.path.join("data", "notifications_dataset.csv")
MODEL_PATH = os.path.join("model", "notification_model.pkl")
PREPROCESSOR_PATH = os.path.join("model", "preprocessor.pkl")


def inject_custom_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(255, 166, 158, 0.22), transparent 35%),
        radial-gradient(circle at 90% 15%, rgba(125, 211, 252, 0.22), transparent 32%),
        radial-gradient(circle at 80% 85%, rgba(134, 239, 172, 0.18), transparent 32%),
        linear-gradient(150deg, #f8fafc 0%, #eef2ff 52%, #f0fdfa 100%);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.2px;
}

.hero-wrap {
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    margin: 0.4rem 0 1.1rem 0;
    color: #0f172a;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.72));
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(4px);
}

.hero-title {
    margin: 0;
    font-size: 1.95rem;
    line-height: 1.2;
}

.hero-subtitle {
    margin: 0.35rem 0 0 0;
    color: #334155;
    font-size: 1.02rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(140px, 1fr));
    gap: 0.65rem;
    margin: 0.55rem 0 1rem 0;
}

.metric-card {
    border-radius: 14px;
    padding: 0.9rem;
    color: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.35);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}

.metric-card .label {
    margin: 0;
    font-size: 0.8rem;
    opacity: 0.9;
}

.metric-card .value {
    margin: 0.2rem 0 0 0;
    font-size: 1.25rem;
    font-weight: 700;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 14px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

div[data-testid="stForm"] {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 0.9rem 0.95rem;
    background: rgba(255, 255, 255, 0.78);
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(110deg, #0ea5e9, #14b8a6);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    box-shadow: 0 8px 18px rgba(20, 184, 166, 0.28);
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 20px rgba(14, 165, 233, 0.35);
}

@media (max-width: 860px) {
    .hero-title {
        font-size: 1.5rem;
    }

    .metric-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header_cards(df: pd.DataFrame) -> None:
    total_samples = len(df) if not df.empty else 0
    unique_contexts = df.drop(columns=["action"]).drop_duplicates().shape[0] if not df.empty else 0
    classes = df["action"].nunique() if not df.empty else 3

    st.markdown(
        f"""
<div class="hero-wrap">
  <h1 class="hero-title">Context-Aware Network-Based Notification System</h1>
  <p class="hero-subtitle">Hybrid AI dashboard that blends network awareness with user context for smarter notification timing.</p>
</div>

<div class="metric-grid">
  <div class="metric-card" style="background: linear-gradient(135deg, #fde68a, #fca5a5);">
    <p class="label">Dataset Size</p>
    <p class="value">{total_samples}</p>
  </div>
  <div class="metric-card" style="background: linear-gradient(135deg, #93c5fd, #67e8f9);">
    <p class="label">Unique Context States</p>
    <p class="value">{unique_contexts}</p>
  </div>
  <div class="metric-card" style="background: linear-gradient(135deg, #a7f3d0, #86efac);">
    <p class="label">Decision Classes</p>
    <p class="value">{classes}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def run_script(script_name: str) -> subprocess.CompletedProcess:
    """
    Run a python script using the same interpreter that runs Streamlit.
    This avoids issues with venv vs system python.
    """
    return subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_artifacts() -> None:
    """
    Ensures dataset + model artifacts exist.
    Runs generate_data.py / train_model.py only if needed.

    Uses st.session_state so Streamlit reruns don't retrigger training.
    """
    if st.session_state.get("artifacts_ready", False):
        return

    st.sidebar.header("Setup Status")

    # 1) Ensure dataset exists
    if not os.path.exists(DATA_PATH):
        st.sidebar.info("Dataset not found. Generating synthetic dataset...")
        result = run_script("generate_data.py")
        if result.returncode != 0:
            st.sidebar.error("Failed to generate dataset.")
            st.sidebar.code(result.stderr or result.stdout)
            # Stop app because prediction won't work without data
            st.stop()
        st.sidebar.success("Dataset generated successfully.")

    # 2) Ensure model artifacts exist
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        st.sidebar.info("Model not found. Training ML model...")
        result = run_script("train_model.py")
        if result.returncode != 0:
            st.sidebar.error("Failed to train model.")
            st.sidebar.code(result.stderr or result.stdout)
            st.stop()
        st.sidebar.success("Model trained and saved successfully.")

    st.session_state["artifacts_ready"] = True
    st.sidebar.success("All required files are ready.")


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def main():
    st.set_page_config(page_title="Context-Aware Notifications", layout="wide")

    # Auto-setup (generate data / train model if missing)
    ensure_artifacts()

    inject_custom_styles()

    df = load_dataset()
    render_header_cards(df)

    st.markdown(
        """
<div class="glass-panel">
<strong>Overview:</strong><br>
This project demonstrates a <strong>hybrid decision system</strong> that uses <strong>user context</strong> and <strong>network strength</strong>
to decide whether a notification should be <strong>sent immediately</strong>, <strong>delayed</strong>, or <strong>suppressed</strong>.
<br><br>
- <strong>Rule engine</strong> handles obvious safety/urgency cases (e.g., emergency + high priority).<br>
- <strong>Machine Learning model</strong> handles remaining non-obvious cases using a trained classifier.
</div>
""",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.05, 1.0], gap="large")

    with col_left:
        st.subheader("Input Context (User + Network)")

        with st.form("prediction_form"):
            time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
            activity = st.selectbox("Current Activity", ["driving", "studying", "working", "free_time", "sleeping"])
            network_strength = st.selectbox("Network Strength", ["weak", "medium", "strong"])
            notification_type = st.selectbox(
                "Notification Type", ["social", "work", "promotional", "reminder", "emergency"]
            )
            priority = st.selectbox("Priority", ["low", "medium", "high"])
            previous_response = st.selectbox("Previous Response", ["opened", "ignored", "delayed"])

            submitted = st.form_submit_button("Predict Best Action")

        if submitted:
            input_dict = {
                "time_of_day": time_of_day,
                "activity": activity,
                "network_strength": network_strength,
                "notification_type": notification_type,
                "priority": priority,
                "previous_response": previous_response,
            }

            try:
                result = predict_notification_action(input_dict)
                action = result["action"]
                source = result["source"]
                confidence = result["confidence"]
                explanation = result["explanation"]

                st.markdown("---")
                st.subheader("Prediction Result")

                if action == "send_now":
                    st.success(f"Final Action: **{action}**")
                elif action == "delay":
                    st.warning(f"Final Action: **{action}**")
                else:
                    st.error(f"Final Action: **{action}**")

                st.write(f"Decision Source: **{source.upper()}**")

                if confidence is not None:
                    st.write(f"Model Confidence: **{confidence:.2f}**")
                else:
                    st.write("Model Confidence: **N/A (rule-based override)**")

                st.info(explanation)

            except Exception as e:
                st.error("Could not generate prediction.")
                st.code(str(e))

        st.markdown("---")
        st.subheader("Sample Use Cases (Quick Cards)")
        st.markdown(
            """
    - **Case 1:** driving + social + low priority -> typically **suppress**  
    - **Case 2:** sleeping + reminder + medium priority -> typically **delay**  
    - **Case 3:** working + work + high priority -> typically **send_now**  
"""
        )

    with col_right:
        st.subheader("Dataset Insights")

        if df.empty:
            st.warning("Dataset is empty or missing.")
        else:
            st.write("Preview:")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("### Class Distribution")
            try:
                fig = plot_class_distribution(df)
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error("Could not plot class distribution.")
                st.code(str(e))

        st.markdown("---")
        st.subheader("How it works (Hybrid Architecture)")
        st.markdown(
            """
1. **Rule-based overrides:** Safety + urgency rules are applied first.  
2. **ML model:** If no rule triggers, an ML classifier predicts one of:
   - `send_now`
   - `delay`
   - `suppress`
"""
        )


if __name__ == "__main__":
    main()