import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
from tensorflow.keras.models import load_model

from darts import TimeSeries
from darts.models import TiDEModel

# Needed because your Bi-LSTM .keras model may contain Lambda layers
keras.config.enable_unsafe_deserialization()

st.set_page_config(
    page_title="Crude Oil Forecast Hub",
    page_icon="🛢️",
    layout="wide"
)

# -----------------------------
# MODEL SETTINGS
# -----------------------------
TIDE_MIN_ROWS = 8
HYBRID_MIN_ROWS = 3
TIDE_COV_COLS = ["Lag1", "Lag2", "oil price"]
ALL_COLS = ["Production", "Lag1", "Lag2", "oil price"]


# -----------------------------
# LOADERS
# -----------------------------
@st.cache_resource
def load_tide_svr_artifacts():
    tide_model = TiDEModel.load("models/tide_final.pt", map_location="cpu")
    svr_model = joblib.load("models/tide_svr_corrector.pkl")
    target_scaler = joblib.load("models/tide_target_scaler.pkl")
    cov_scaler = joblib.load("models/tide_cov_scaler.pkl")
    return tide_model, svr_model, target_scaler, cov_scaler


@st.cache_resource
def load_hybrid_artifacts():
    bilstm_model = load_model(
        "models/bilstm_model.keras",
        compile=False,
        safe_mode=False
    )
    svr_model = joblib.load("models/svr_model.pkl")
    scaler_features = joblib.load("models/scaler_features.pkl")
    scaler_target = joblib.load("models/scaler_target.pkl")
    hybrid_meta = joblib.load("models/hybrid_meta.pkl")
    return bilstm_model, svr_model, scaler_features, scaler_target, hybrid_meta


# -----------------------------
# HELPERS
# -----------------------------
def build_tide_history_series(history_df: pd.DataFrame):
    time_index = pd.date_range("2000-01-01", periods=len(history_df), freq="MS")

    target_series = TimeSeries.from_times_and_values(
        times=time_index,
        values=history_df["Production"].astype(float).to_numpy(),
        columns=["Production"]
    )

    cov_series = TimeSeries.from_times_and_values(
        times=time_index,
        values=history_df[TIDE_COV_COLS].astype(float).to_numpy(),
        columns=TIDE_COV_COLS
    )

    return target_series, cov_series


def predict_tide_svr(history_df: pd.DataFrame, next_oil_price: float):
    tide_model, svr_model, target_scaler, cov_scaler = load_tide_svr_artifacts()

    target_series, cov_series = build_tide_history_series(history_df)

    target_scaled = target_scaler.transform(target_series)
    cov_scaled = cov_scaler.transform(cov_series)

    # Base TiDE forecast
    tide_pred_scaled = tide_model.predict(
        n=1,
        series=target_scaled,
        past_covariates=cov_scaled
    )

    tide_pred_series = target_scaler.inverse_transform(tide_pred_scaled)
    tide_pred_value = float(tide_pred_series.values().flatten()[0])

    # Next-step features for SVR correction
    next_lag1 = float(history_df["Production"].iloc[-1])
    next_lag2 = float(history_df["Production"].iloc[-2])

    svr_features = np.array([[
        next_lag1,
        next_lag2,
        float(next_oil_price),
        tide_pred_value
    ]])

    correction = float(svr_model.predict(svr_features)[0])
    final_pred = tide_pred_value + correction

    return tide_pred_value, correction, final_pred


def predict_bilstm_hybrid(input_df: pd.DataFrame):
    bilstm_model, svr_model, scaler_features, scaler_target, hybrid_meta = load_hybrid_artifacts()

    feature_columns = hybrid_meta["feature_columns"]
    time_steps = int(hybrid_meta["TIME_STEPS"])

    df_input = input_df[feature_columns].copy()
    scaled_features = scaler_features.transform(df_input.values)
    X_input = scaled_features.reshape(1, time_steps, len(feature_columns))

    bilstm_pred_scaled = bilstm_model.predict(X_input, verbose=0)
    svr_input = np.hstack([X_input[:, -1, :], bilstm_pred_scaled])
    correction = svr_model.predict(svr_input).reshape(-1, 1)

    hybrid_scaled = bilstm_pred_scaled + correction
    pred = scaler_target.inverse_transform(hybrid_scaled)

    return float(pred[0, 0])


# -----------------------------
# UI
# -----------------------------
st.title("🛢️ Crude Oil Forecast Hub")
st.caption("Choose a model and enter recent values to predict the next production value.")

model_choice = st.segmented_control(
    "Select forecasting model",
    ["TiDE + SVR", "Bi-LSTM + SVR Hybrid"],
    default="TiDE + SVR"
)

st.divider()

if model_choice == "TiDE + SVR":
    st.subheader("TiDE + SVR Forecast")
    st.write(
        "You must enter at least 8 rows. You can add more if you want. "
        "The model will use the most recent 8 rows."
    )

    tide_row_count = st.number_input(
        "How many rows do you want to enter?",
        min_value=TIDE_MIN_ROWS,
        value=TIDE_MIN_ROWS,
        step=1,
        key="tide_row_count"
    )

    with st.form("tide_svr_form"):
        tide_rows = []

        for i in range(int(tide_row_count)):
            st.markdown(f"**Row {i + 1}**")
            c1, c2, c3, c4 = st.columns(4)

            production = c1.number_input(
                f"Production {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"tide_prod_{i}"
            )
            lag1 = c2.number_input(
                f"Lag1 {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"tide_lag1_{i}"
            )
            lag2 = c3.number_input(
                f"Lag2 {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"tide_lag2_{i}"
            )
            oil_price = c4.number_input(
                f"Oil price {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"tide_oil_{i}"
            )

            tide_rows.append([production, lag1, lag2, oil_price])

        next_oil_price = st.number_input(
            "Expected next oil price",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="next_oil_price"
        )

        submit_tide = st.form_submit_button("Predict next production")

    if submit_tide:
        try:
            tide_input = pd.DataFrame(tide_rows, columns=ALL_COLS)

            for col in ALL_COLS:
                tide_input[col] = tide_input[col].astype(float)

            # Use only the most recent 8 rows
            tide_input = tide_input.tail(TIDE_MIN_ROWS).reset_index(drop=True)

            tide_base, svr_correction, final_prediction = predict_tide_svr(
                tide_input,
                next_oil_price
            )

            st.success("Prediction complete.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Base TiDE forecast", f"{tide_base:,.4f}")
            c2.metric("SVR correction", f"{svr_correction:,.4f}")
            c3.metric("Final TiDE + SVR forecast", f"{final_prediction:,.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif model_choice == "Bi-LSTM + SVR Hybrid":
    st.subheader("Bi-LSTM + SVR Hybrid Forecast")
    st.write(
        "You must enter at least 3 rows. You can add more if you want. "
        "The model will use the most recent 3 rows."
    )

    hybrid_row_count = st.number_input(
        "How many rows do you want to enter?",
        min_value=HYBRID_MIN_ROWS,
        value=HYBRID_MIN_ROWS,
        step=1,
        key="hybrid_row_count"
    )

    with st.form("hybrid_form"):
        hybrid_rows = []

        for i in range(int(hybrid_row_count)):
            st.markdown(f"**Row {i + 1}**")
            c1, c2, c3, c4 = st.columns(4)

            production = c1.number_input(
                f"Production {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"hybrid_prod_{i}"
            )
            lag1 = c2.number_input(
                f"Lag1 {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"hybrid_lag1_{i}"
            )
            lag2 = c3.number_input(
                f"Lag2 {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"hybrid_lag2_{i}"
            )
            oil_price = c4.number_input(
                f"Oil price {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"hybrid_oil_{i}"
            )

            hybrid_rows.append([production, lag1, lag2, oil_price])

        submit_hybrid = st.form_submit_button("Predict next production")

    if submit_hybrid:
        try:
            hybrid_input = pd.DataFrame(hybrid_rows, columns=ALL_COLS)

            for col in ALL_COLS:
                hybrid_input[col] = hybrid_input[col].astype(float)

            # Use only the most recent 3 rows
            hybrid_input = hybrid_input.tail(HYBRID_MIN_ROWS).reset_index(drop=True)

            prediction = predict_bilstm_hybrid(hybrid_input)

            st.success("Prediction complete.")
            st.metric("Next predicted production", f"{prediction:,.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.info("Tip: enter rows from oldest to newest. If you enter extra rows, the app uses the most recent ones.")
