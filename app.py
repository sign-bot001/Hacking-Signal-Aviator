# app.py — Signal Alert AVTR (Pro Max: XGBoost + Stacking + Conformal)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# XGBoost (optionnel si installation OK)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ------------------ CONFIG & THEME ------------------
st.set_page_config(page_title="Signal Alert AVTR — Pro Max", layout="wide")
TZ = pytz.timezone("Asia/Seoul")
st.markdown("""
<style>
body {background:#050608;}
.stApp {color:#d5f5ff;}
.title {font-family:'Courier New',monospace;color:#7afcff;font-size:28px;}
.small {color:#9fd;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">⚡ Signal Alert AVTR — 60 prédictions minute par minute (KST)</div>', unsafe_allow_html=True)
st.caption("1 bouton. Zéro saisie. Ensemble ML (RF+ET+GBR+XGBoost) + Conformal prediction pour la confiance.")

# ------------------ HISTORIQUE INTÉGRÉ ------------------
DEFAULT_HISTORY = [
    1.3,1.23,1.56,2.25,1.15,13.09,20.91,2.05,10.17,3.82,
    1.00,1.46,1.40,1.73,1.17,1.00,26.60,8.60,1.27,1.46,
    1.36,1.76,3.61,2.74,1.47,3.70,1.05
]

def history_df(values):
    now = datetime.now(TZ)
    ts = [now - timedelta(minutes=(len(values)-1-i)) for i in range(len(values))]
    return pd.DataFrame({"timestamp": ts, "multiplier": values})

BASE_DF = history_df(DEFAULT_HISTORY)

# ------------------ FEATURES ------------------
def add_time_features(df):
    t = df["timestamp"]
    df["minute"] = t.dt.minute
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    # encodage cyclique
    df["min_sin"]  = np.sin(2*np.pi*df["minute"]/60.0)
    df["min_cos"]  = np.cos(2*np.pi*df["minute"]/60.0)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    return df

def add_lags_and_rolls(df, lags=30):
    s = df["multiplier"]
    for k in range(1, lags+1):
        df[f"lag_{k}"] = s.shift(k)
    # rolling (shift pour éviter la fuite)
    df["roll_mean_5"]  = s.rolling(5).mean().shift(1)
    df["roll_std_5"]   = s.rolling(5).std().shift(1).fillna(0)
    df["roll_mean_15"] = s.rolling(15).mean().shift(1)
    df["roll_std_15"]  = s.rolling(15).std().shift(1).fillna(0)
    df["roll_mean_30"] = s.rolling(30).mean().shift(1)
    df["roll_std_30"]  = s.rolling(30).std().shift(1).fillna(0)
    # dynamiques
    df["mom_3"] = s / s.shift(3) - 1
    df["pct_1"] = s.pct_change(1).shift(1).fillna(0)
    df["pct_3"] = s.pct_change(3).shift(1).fillna(0)
    df["vol_15"] = df["roll_std_15"] / (df["roll_mean_15"] + 1e-9)
    return df

def build_feature_matrix(df, lags=30):
    df2 = df.copy()
    df2 = add_time_features(df2)
    df2 = add_lags_and_rolls(df2, lags=lags)
    df2 = df2.dropna().reset_index(drop=True)
    y = df2["multiplier"].values
    X = df2.drop(columns=["multiplier","timestamp"]).values
    return X, y, df2

# ------------------ MODÈLES (STACKING + CONFORMAL) ------------------
def _base_models(n_estimators=300):
    models = {
        "rf": RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=1),
        "et": ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=2),
        "gbr": GradientBoostingRegressor(random_state=3),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.05,
            tree_method="hist",
            random_state=4,
            n_jobs=-1,
        )
    return models

def _fit_oof_stack(X, y, models):
    tscv = TimeSeriesSplit(n_splits=5)
    meta_X = np.zeros((len(X), len(models)))
    order = list(models.keys())
    for tr, va in tscv.split(X):
        X_tr, X_va = X[tr], X[va]
        y_tr = y[tr]
        for j, name in enumerate(order):
            m = models[name]
            m.fit(X_tr, y_tr)
            meta_X[va, j] = m.predict(X_va)
    # fit full
    for name in order:
        models[name].fit(X, y)
    meta = Ridge(alpha=1.0).fit(meta_X, y)
    return models, meta, meta_X

def _stack_predict(models, meta, X):
    order = list(models.keys())
    P = np.column_stack([models[name].predict(X) for name in order])
    return meta.predict(P), P

def _conformal_interval(y_true_oof, y_pred_oof, alpha=0.10):
    # nonconformité = |erreur|, seuil quantile 1 - alpha
    resid = np.abs(y_true_oof - y_pred_oof)
    q = np.quantile(resid, 1 - alpha)
    return float(q)

@st.cache_resource(show_spinner=False)
def fit_models_with_conformal(df, lags=30, n_estimators=300, alpha=0.10):
    X, y, df2 = build_feature_matrix(df, lags=lags)
    if len(X) < 10:
        raise ValueError("Historique insuffisant pour entraîner un modèle.")
    models = _base_models(n_estimators=n_estimators)
    models, meta, oof = _fit_oof_stack(X, y, models)
    y_oof_pred = meta.predict(oof)
    q = _conformal_interval(y_true_oof=y, y_pred_oof=y_oof_pred, alpha=alpha)
    # métriques internes
    mae = mean_absolute_error(y, y_oof_pred)
    rmse = mean_squared_error(y, y_oof_pred, squared=False)
    return models, meta, df2, q, mae, rmse

def _confidence_from_interval_width(p, width, recent_vol):
    # largeur relative vs amplitude prédite + volatilité récente
    denom = max(abs(p), 1.0) + 0.5*max(recent_vol, 1e-6)
    rel = width / denom
    conf = 100 * (1 - np.clip(rel, 0, 1))
    return float(np.clip(conf, 0, 100))

def forecast_next_60(df, lags, models, meta, q):
    work = df.copy()
    # stats récentes pour la volatilité (normalisation de la confiance)
    recent_vol = float(pd.Series(work["multiplier"]).tail(15).std() or 0.0)

    preds, confs, times = [], [], []
    last_ts = work["timestamp"].iloc[-1]

    for _ in range(60):
        X_all, _, _ = build_feature_matrix(work, lags=lags)
        x_last = X_all[-1:].copy()
        p, base_preds = _stack_predict(models, meta, x_last)
        p = float(p[0])
        width = 2.0 * q  # intervalle conforme symétrique [p-q, p+q]
        c = _confidence_from_interval_width(p, width, recent_vol)

        next_ts = last_ts + timedelta(minutes=1)
        # on réinjecte la prédiction comme observation suivante
        work = pd.concat([work, pd.DataFrame({"timestamp":[next_ts], "multiplier":[p]})], ignore_index=True)

        preds.append(p); confs.append(c); times.append(next_ts)
        last_ts = next_ts

    return pd.DataFrame({
        "timestamp_kst": times,
        "predicted_multiplier": np.round(preds, 4),
        "confidence_0_100": np.round(confs, 1),
    })

# ------------------ UI (1 BOUTON) ------------------
st.sidebar.header("Paramètres (optionnels)")
LAGS = st.sidebar.slider("Mémoire (lags)", 20, 60, 30)
N_EST = st.sidebar.slider("Arbres par modèle", 200, 600, 300)
ALPHA = 0.10  # 90% de couverture ciblée

run = st.button("🚀 Charger et prédire l’heure suivante (KST)")

if run:
    try:
        models, meta, df2, q, mae, rmse = fit_models_with_conformal(BASE_DF, lags=LAGS, n_estimators=N_EST, alpha=ALPHA)
        out = forecast_next_60(BASE_DF, LAGS, models, meta, q)

        st.markdown("### Prédictions minute par minute (60 prochaines minutes)")
        st.dataframe(out)

        st.download_button(
            "Télécharger les prédictions (CSV)",
            out.to_csv(index=False),
            file_name="signal_alert_avtr_predictions_60min.csv"
        )

        used = "RF + ET + GBR" + (" + XGBoost" if HAS_XGB else "")
        st.caption(f"Modèles: {used} • Conformal α={ALPHA:.2f} → seuil |erreur|={q:.4f} • Qualité interne (OOF): MAE={mae:.3f} | RMSE={rmse:.3f}")
        if not HAS_XGB:
            st.caption("XGBoost non détecté (fallback automatique). Vérifie requirements/deploy si tu souhaites l'activer.")
    except Exception as e:
        st.error(f"Impossible de générer les prédictions : {e}")

