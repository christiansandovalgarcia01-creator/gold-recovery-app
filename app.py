import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TRAIN = os.path.join(BASE_DIR, "data", "gold_recovery_train.csv")
PATH_TEST  = os.path.join(BASE_DIR, "data", "gold_recovery_test.csv")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Recovery Predictor", layout="centered")
st.title("Gold Recovery — sMAPE Project")
st.caption("Predice rougher.output.recovery y final.output.recovery. Si subes objetivos, calculo sMAPE y la métrica 25/75.")

@st.cache_resource
def load_assets():
    m1 = joblib.load("model_rough.pkl")
    m2 = joblib.load("model_final.pkl")
    with open("feature_list.json", "r") as f:
        cols = json.load(f)
    return m1, m2, cols

model_rough, model_final, X_cols = load_assets()

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    out = np.zeros_like(denom, dtype=float)
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out) * 100)

st.markdown(
    "Sube un CSV con **las mismas columnas** que el set de prueba (sin targets) para predecir. "
    "Si además incluye las columnas objetivo, reporto sMAPE y la métrica final 25/75."
)

file = st.file_uploader("Archivo CSV", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.write("Dimensiones:", df.shape)

    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas para predecir: {missing[:15]}{' ...' if len(missing) > 15 else ''}")
    else:
        # Select features in the required order
        X = df[X_cols].copy()
        # Ensure numeric dtype where possible
        X = X.apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            st.warning("Hay valores no numéricos o NaNs en las columnas de entrada; serán tratados como NaN por el modelo.")
        # Predict and clip to [0, 100]
        pred_rough = np.clip(model_rough.predict(X), 0, 100)
        pred_final = np.clip(model_final.predict(X), 0, 100)
        pred = pd.DataFrame({
            "rougher.output.recovery": pred_rough,
            "final.output.recovery": pred_final
        })
        st.subheader("Predicciones")
        st.dataframe(pred.head(20))
        st.download_button(
            "Descargar predicciones (CSV)",
            pred.to_csv(index=False),
            "predicciones.csv",
            "text/csv"
        )

        tgt_cols = ["rougher.output.recovery", "final.output.recovery"]
        if all(c in df.columns for c in tgt_cols):
            s1 = smape(df["rougher.output.recovery"], pred["rougher.output.recovery"])
            s2 = smape(df["final.output.recovery"], pred["final.output.recovery"])
            sW = 0.25 * s1 + 0.75 * s2
            st.subheader("Métricas (con objetivos)")
            st.write(f"sMAPE(rougher) = {s1:.3f}")
            st.write(f"sMAPE(final)   = {s2:.3f}")
            st.write(f"Métrica final 25/75 = {sW:.3f}")

        st.subheader("Distribuciones de predicciones")
        fig1, ax1 = plt.subplots()
        pred["rougher.output.recovery"].hist(bins=40, ax=ax1)
        ax1.set_title("Pred: rougher")
        fig1.tight_layout()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        pred["final.output.recovery"].hist(bins=40, ax=ax2)
        ax2.set_title("Pred: final")
        fig2.tight_layout()
        st.pyplot(fig2)
else:
    st.info("Sube un archivo CSV para comenzar.")