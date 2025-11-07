# train.py — entrena, selecciona el mejor modelo por sMAPE final y guarda artefactos
import pandas as pd, numpy as np, json, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

RANDOM_STATE = 42

# Rutas a tus CSV (colócalos en la carpeta data/)
PATH_TRAIN = "data/gold_recovery_train.csv"
PATH_TEST  = "data/gold_recovery_test.csv"

print("Cargando datos...")
train = pd.read_csv(PATH_TRAIN, index_col="date", parse_dates=True).sort_index()
test  = pd.read_csv(PATH_TEST,  index_col="date", parse_dates=True).sort_index()

targets = ["rougher.output.recovery", "final.output.recovery"]

# Columnas comunes train ∩ test
X_cols = sorted([c for c in set(train.columns) & set(test.columns) if c not in targets])

# Filtros de consistencia física (si existen esas columnas)
def sum_if_exists(df, like):
    cols = df.filter(like=like).filter(regex="(_au|_ag|_pb|_sol)$").columns
    return df[cols].sum(axis=1) if len(cols) else pd.Series(index=df.index, dtype=float)

tot_feed  = sum_if_exists(train, "rougher.input")
tot_rough = sum_if_exists(train, "rougher.output")
tot_final = sum_if_exists(train, "final.output")

def valid01(s): 
    return (~s.isna()) & (s >= 0) & (s <= 100)

mask = pd.Series(True, index=train.index)
for s in [tot_feed, tot_rough, tot_final]:
    if s.notna().any():
        mask &= valid01(s)

train = train.loc[mask].copy()
train = train.dropna(subset=targets)
train[targets] = train[targets].clip(0, 100)

X_train = train[X_cols].copy()
y_train = train[targets].copy()
X_test  = test[X_cols].copy()

# Preprocesamiento: imputación mediana + escalado
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
pre = ColumnTransformer([("num", num_pipe, X_cols)], remainder="drop")

# sMAPE y sMAPE final (25/75)
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    out = np.zeros_like(denom, float)
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out) * 100)

def final_smape(y_true_df, y_pred_df):
    s1 = smape(y_true_df["rougher.output.recovery"], y_pred_df["rougher.output.recovery"])
    s2 = smape(y_true_df["final.output.recovery"],    y_pred_df["final.output.recovery"])
    return 0.25 * s1 + 0.75 * s2

# Modelos candidatos
candidates = {
    "linreg": Pipeline([("pre", pre), ("est", LinearRegression())]),
    "gbr":    Pipeline([("pre", pre), ("est", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
    "rf":     Pipeline([("pre", pre), ("est", RandomForestRegressor(
                        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]),
}

# CV temporal y selección del mejor por sMAPE final
tscv = TimeSeriesSplit(n_splits=5)

def eval_pipe(pipe, X, y):
    scores = []
    for tr, va in tscv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        keep_tr = ~y_tr.isna().any(axis=1)
        X_tr, y_tr = X_tr.loc[keep_tr], y_tr.loc[keep_tr]

        m1 = Pipeline(pipe.steps); m2 = Pipeline(pipe.steps)
        m1.fit(X_tr, y_tr["rougher.output.recovery"])
        m2.fit(X_tr, y_tr["final.output.recovery"])

        p1 = np.clip(m1.predict(X_va), 0, 100)
        p2 = np.clip(m2.predict(X_va), 0, 100)
        yhat = pd.DataFrame({
            "rougher.output.recovery": p1,
            "final.output.recovery":   p2
        }, index=y_va.index)
        scores.append(final_smape(y_va, yhat))
    return float(np.mean(scores))

best_name, best_score = None, 1e9
for name, pipe in candidates.items():
    s = eval_pipe(pipe, X_train, y_train)
    print(f"{name}: sMAPE_final CV = {s:.3f}")
    if s < best_score:
        best_name, best_score = name, s

print(f"Mejor modelo: {best_name}  con sMAPE_final CV = {best_score:.3f}")
best_pipe = candidates[best_name]

# Entrenamiento final por objetivo y guardado de artefactos
model_rough = Pipeline(best_pipe.steps).fit(X_train, y_train["rougher.output.recovery"])
model_final = Pipeline(best_pipe.steps).fit(X_train, y_train["final.output.recovery"])

joblib.dump(model_rough, "model_rough.pkl")
joblib.dump(model_final, "model_final.pkl")
json.dump(X_cols, open("feature_list.json", "w"))

print("Guardado: model_rough.pkl, model_final.pkl, feature_list.json")
