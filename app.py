import os

import pandas as pd
import streamlit as st

from utils import predict_notification_action, plot_class_distribution

DATA_PATH = os.path.join("data", "notifications_dataset.csv")


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def main():
    st.set_page_config(page_title="Context-Aware Notifications", layout="wide")

    st.title("Context-Aware Network-Based Notification System")
    st.caption("AI + Networking Demo Project")

    st.markdown(
        """
**Overview:**  
This project demonstrates a **hybrid decision system** that uses **user context** and **network strength**
to decide whether a notification should be **sent immediately**, **delayed**, or **suppressed**.

- **Rule engine** handles obvious safety/urgency cases (e.g., *emergency + high priority*).
- **Machine Learning model** handles the remaining non-obvious cases using a trained classifier.
"""
    )

    df = load_dataset()

    col_left, col_right = st.columns([1.05, 1.0], gap="large")

    with col_left:
        st.subheader("Input Context (User + Network)")

        with st.form("prediction_form"):
            time_of_day = st.selectbox("time_of_day", ["morning", "afternoon", "evening", "night"])
            activity = st.selectbox("activity", ["driving", "studying", "working", "free_time", "sleeping"])
            network_strength = st.selectbox("network_strength", ["weak", "medium", "strong"])
            notification_type = st.selectbox(
                "notification_type", ["social", "work", "promotional", "reminder", "emergency"]
            )
            priority = st.selectbox("priority", ["low", "medium", "high"])
            previous_response = st.selectbox("previous_response", ["opened", "ignored", "delayed"])

            submitted = st.form_submit_button("Predict Notification Action")

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

                # Highlighted action box
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
                st.error("Could not generate prediction. Ensure you have generated data and trained the model.")
                st.code(str(e))

        st.markdown("---")
        st.subheader("Sample Use Cases (Quick Cards)")
        st.markdown(
            """
- **Case 1:** driving + social + low priority → typically **suppress**  
- **Case 2:** sleeping + reminder + medium priority → typically **delay**  
- **Case 3:** working + work + high priority → typically **send_now**  
"""
        )

    with col_right:
        st.subheader("Dataset Insights")

        if df.empty:
            st.warning(
                "Dataset not found yet. Run `python generate_data.py` to create it, then `python train_model.py`."
            )
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

This makes the solution both **interpretable** (rules) and **adaptive** (ML).
"""
        )


if __name__ == "__main__":
    main()