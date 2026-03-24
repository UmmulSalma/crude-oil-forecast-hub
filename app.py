import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Crude Oil Forecast Hub",
    page_icon="🛢️",
    layout="wide"
)

@st.cache_resource
def load_tide_artifacts():
    tide_model = load_model("models/tide_model.keras", compile=False)
    tide_scaler = joblib.load("models/tide_scaler.pkl")
    tide_meta = joblib.load("models/tide_meta.pkl")
    return tide_model, tide_scaler, tide_meta

@st.cache_resource
def load_hybrid_artifacts():
    bilstm_model = load_model("models/bilstm_model.keras", compile=False)
    svr_model = joblib.load("models/svr_model.pkl")
    scaler_features = joblib.load("models/scaler_features.pkl")
    scaler_target = joblib.load("models/scaler_target.pkl")
    hybrid_meta = joblib.load("models/hybrid_meta.pkl")
    return bilstm_model, svr_model, scaler_features, scaler_target, hybrid_meta

def predict_tide(last_12_values):
    tide_model, tide_scaler, tide_meta = load_tide_artifacts()

    arr = np.array(last_12_values, dtype=float).reshape(-1, 1)
    scaled = tide_scaler.transform(arr).flatten().reshape(1, tide_meta["LOOKBACK"])

    pred_scaled = tide_model.predict(scaled, verbose=0)
    pred = tide_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return float(pred[0, 0])

def predict_hybrid(input_df):
    bilstm_model, svr_model, scaler_features, scaler_target, hybrid_meta = load_hybrid_artifacts()

    feature_columns = hybrid_meta["feature_columns"]
    time_steps = hybrid_meta["TIME_STEPS"]

    df_input = input_df[feature_columns].copy()
    scaled_features = scaler_features.transform(df_input.values)
    X_input = scaled_features.reshape(1, time_steps, len(feature_columns))

    bilstm_pred_scaled = bilstm_model.predict(X_input, verbose=0)
    svr_input = np.hstack([X_input[:, -1, :], bilstm_pred_scaled])
    correction = svr_model.predict(svr_input).reshape(-1, 1)

    hybrid_scaled = bilstm_pred_scaled + correction
    pred = scaler_target.inverse_transform(hybrid_scaled)

    return float(pred[0, 0])

st.title("🛢️ Crude Oil Forecast Hub")
st.caption("Choose a forecasting model and enter recent values to predict the next production value.")

model_choice = st.segmented_control(
    "Select forecasting model",
    ["TiDe", "Bi-LSTM + SVR Hybrid"],
    default="Bi-LSTM + SVR Hybrid"
)

st.divider()

if model_choice == "TiDe":
    st.subheader("TiDe Forecast")
    st.write("Enter the last 12 production values from oldest to newest.")

    default_tide = pd.DataFrame({
        "Production": [0.0] * 12
    })

    with st.form("tide_form"):
        tide_input = st.data_editor(
            default_tide,
            hide_index=True,
            num_rows="fixed",
            use_container_width=True
        )
        submit_tide = st.form_submit_button("Predict next production")

    if submit_tide:
        try:
            values = tide_input["Production"].astype(float).tolist()

            if len(values) != 12:
                st.error("TiDe requires exactly 12 production values.")
            else:
                prediction = predict_tide(values)
                st.success("Prediction complete.")
                st.metric("Next predicted production", f"{prediction:,.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif model_choice == "Bi-LSTM + SVR Hybrid":
    st.subheader("Bi-LSTM + SVR Hybrid Forecast")
    st.write("Enter the last 3 rows in time order from oldest to newest.")

    default_hybrid = pd.DataFrame({
        "Production": [0.0, 0.0, 0.0],
        "Lag1": [0.0, 0.0, 0.0],
        "Lag2": [0.0, 0.0, 0.0],
        "oil price": [0.0, 0.0, 0.0]
    })

    with st.form("hybrid_form"):
        hybrid_input = st.data_editor(
            default_hybrid,
            hide_index=True,
            num_rows="fixed",
            use_container_width=True
        )
        submit_hybrid = st.form_submit_button("Predict next production")

    if submit_hybrid:
        try:
            required_cols = ["Production", "Lag1", "Lag2", "oil price"]

            for col in required_cols:
                hybrid_input[col] = hybrid_input[col].astype(float)

            if hybrid_input.shape[0] != 3:
                st.error("Hybrid model requires exactly 3 rows.")
            else:
                prediction = predict_hybrid(hybrid_input[required_cols])
                st.success("Prediction complete.")
                st.metric("Next predicted production", f"{prediction:,.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.info("Tip: enter values from oldest to newest.")