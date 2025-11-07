# app.py
# Streamlit app — Gold Recovery (usa SOLO el modelo FINAL)
# Requisitos: streamlit, pandas, numpy, joblib, matplotlib, scikit-learn (para objetos serializados), json

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Configuración de página
# -----------------------------
st.set_page_config(page_title="Gold Recovery — sMAPE Project", layout="centered")
st.title("Gold Recovery — sMAPE Project")
st.caption(
    "Predice **final.output.recovery** con el modelo final. "
    "Si tu CSV incluye la columna objetivo `final.output.recovery`, calcularé sMAPE(final)."
)

# -----------------------------
# Utilidades
# -----------------------------
def smape(y_true, y_pred, eps=1e-9):
    """Calcula sMAPE (en %) con protección ante división por cero."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def _load_first_that_exists(paths):
    """Carga con joblib el primer archivo existente en la lista `paths`."""
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"No encontré ninguno de: {paths}")

@st.cache_resource
def load_model():
    # Acepta modelo comprimido (.pkl.gz) o sin comprimir (.pkl)
    return _load_first_that_exists(["model_final.pkl.gz", "model_final.pkl"])

@st.cache_resource
def load_feature_list():
    """
    Devuelve la lista de columnas de entrada esperadas (y su orden).
    Busca en raíz 'feature_list.json' o dentro de la carpeta 'feature_list/'.
    Si no existe, devuelve None y se infiere de las columnas del CSV subido.
    """
    candidates = [
        "feature_list.json",
        os.path.join("feature_list", "feature_list.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

# Cargar artefactos
try:
    model_final = load_model()
except Exception as e:
    st.error(f"No pude cargar el modelo final: {e}")
    st.stop()

X_cols = load_feature_list()

# -----------------------------
# UI: Entrada de CSV
# -----------------------------
st.subheader("Archivo CSV")
file = st.file_uploader(
    "Sube un CSV con las **mismas columnas de entrada** usadas en el set de prueba.",
    type=["csv"]
)

if file is None:
    st.info("Sube un archivo CSV para comenzar.")
    st.stop()

# -----------------------------
# Lectura y validación del CSV
# -----------------------------
try:
    df = pd.read_csv(file, index_col=None, parse_dates=False)
except Exception as e:
    st.error(f"No pude leer el CSV: {e}")
    st.stop()

st.write("**Dimensiones del archivo:**", df.shape)

# Determinar columnas de entrada:
# - Si tenemos feature_list.json, lo usamos.
# - Si no, usamos todas las columnas excepto targets (si vienen en el CSV).
target_cols = ["final.output.recovery"]
if X_cols is None:
    X_cols = [c for c in df.columns if c not in target_cols]

# Validación de columnas faltantes para predecir
missing = [c for c in X_cols if c not in df.columns]
if missing:
    preview = ", ".join(missing[:15]) + (" ..." if len(missing) > 15 else "")
    st.error(f"Faltan columnas para predecir: {preview}")
    st.stop()

# Selección y preparación de X
X = df[X_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce")  # asegurar numérico

if X.isna().any().any():
    st.warning(
        "Hay valores no numéricos o NaN en las columnas de entrada; "
        "el modelo los tratará como NaN según corresponda."
    )

# -----------------------------
# Predicción
# -----------------------------
try:
    pred_final = np.clip(model_final.predict(X), 0, 100)
except Exception as e:
    st.error(f"Error al predecir con el modelo final: {e}")
    st.stop()

pred = pd.DataFrame(
    {"final.output.recovery": pred_final},
    index=df.index
)

st.subheader("Predicciones (primeras 20 filas)")
st.dataframe(pred.head(20))

st.download_button(
    "Descargar predicciones (CSV)",
    pred.to_csv(index=False),
    "predicciones_final.csv",
    "text/csv",
)

# -----------------------------
# Métricas (si hay objetivo real)
# -----------------------------
if target_cols[0] in df.columns:
    try:
        s2 = smape(df["final.output.recovery"], pred["final.output.recovery"])
        st.subheader("Métricas (con objetivos)")
        st.write(f"sMAPE(final) = **{s2:.3f}%**")
    except Exception as e:
        st.warning(f"No pude calcular sMAPE(final): {e}")
else:
    st.info("Tu CSV no incluye la columna objetivo `final.output.recovery`; se omite el sMAPE.")

# -----------------------------
# Distribución de predicciones
# -----------------------------
st.subheader("Distribución de predicciones — final")
fig, ax = plt.subplots()
pd.Series(pred_final).hist(bins=40, ax=ax)
ax.set_xlabel("final.output.recovery (predicho)")
ax.set_ylabel("Frecuencia")
ax.set_title("Histograma de predicciones (final)")
fig.tight_layout()
st.pyplot(fig)
