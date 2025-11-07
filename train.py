# train.py — rápido y estable (con limpieza de objetivos)
import os, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# --- Rutas ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PATH_TRAIN = os.path.join(BASE_DIR, "data", "gold_recovery_train.csv")
PATH_TEST  = os.path.join(BASE_DIR, "data", "gold_recovery_test.csv")

# --- Configuración de velocidad ---
SEED       = 42
FAST_MODE  = True      # pon False para entreno completo
NROWS_FAST = 8000      # filas en modo rápido

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1.0
    return (100.0 / len(y_true)) * np.sum(np.abs(y_pred - y_true) / denom)

print("Cargando datos...")

# Leemos test para fijar el set de features permitido
test = pd.read_csv(PATH_TEST, nrows=(NROWS_FAST if FAST_MODE else None))
feat_cols = [c for c in test.columns if c != "date"]

targets = ["rougher.output.recovery", "final.output.recovery"]
usecols = ["date"] + feat_cols + targets

train = pd.read_csv(
    PATH_TRAIN,
    usecols=usecols,
    nrows=(NROWS_FAST if FAST_MODE else None)
)

# Orden y tipos ligeros
train = train.sort_values("date").reset_index(drop=True)
X = train[feat_cols].astype("float32")

# --- Limpieza robusta ---
# Sustituir inf por NaN en features
X = X.replace([np.inf, -np.inf], np.nan)

# Objetivos como float32
y_rough = train["rougher.output.recovery"].astype("float32")
y_final = train["final.output.recovery"].astype("float32")

# Clip opcional de objetivos a [0, 100] (según negocio)
y_rough = y_rough.clip(lower=0, upper=100)
y_final = y_final.clip(lower=0, upper=100)

# Filtramos filas con NaN en objetivos (requisito de sklearn)
mask = y_rough.notna() & y_final.notna()
n_before = len(train)
X = X.loc[mask].reset_index(drop=True)
y_rough = y_rough.loc[mask].reset_index(drop=True)
y_final = y_final.loc[mask].reset_index(drop=True)
print(f"Filtrado objetivos: {n_before} → {len(X)} filas (sin NaN en y)")

# --- Split rápido (holdout) ---
X_tr, X_va, y_r_tr, y_r_va = train_test_split(
    X, y_rough, test_size=0.2, random_state=SEED
)
# Usamos el MISMO split de índices para el objetivo final
X_tr2, X_va2, y_f_tr, y_f_va = train_test_split(
    X, y_final, test_size=0.2, random_state=SEED
)

def make_hgb():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=6,
        learning_rate=0.08,
        max_iter=(200 if not FAST_MODE else 120),
        l2_regularization=1.0,
        random_state=SEED
    )

print("Entrenando modelos...")
m_rough = make_hgb().fit(X_tr,  y_r_tr)
m_final = make_hgb().fit(X_tr2, y_f_tr)

# --- Validación ---
pred_r = np.clip(m_rough.predict(X_va),  0, 100)
pred_f = np.clip(m_final.predict(X_va2), 0, 100)
s1 = smape(y_r_va, pred_r)
s2 = smape(y_f_va, pred_f)
s2575 = 0.25 * s1 + 0.75 * s2
print(f"Validación — sMAPE(rougher) = {s1:.3f}")
print(f"Validación — sMAPE(final)   = {s2:.3f}")
print(f"Métrica 25/75               = {s2575:.3f}")

# --- Guardado comprimido ---
print("Guardando modelos y lista de features...")
joblib.dump(m_rough, "model_rough.pkl.gz", compress=5)
joblib.dump(m_final, "model_final.pkl.gz", compress=5)
with open("feature_list.json", "w", encoding="utf-8") as f:
    json.dump(feat_cols, f, ensure_ascii=False, indent=2)

print("OK: model_rough.pkl.gz, model_final.pkl.gz, feature_list.json")
