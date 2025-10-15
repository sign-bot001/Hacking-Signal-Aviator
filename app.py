# app.py — Signal Alert AVTR (anti-"Historique insuffisant", 10k auto, 1 bouton)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ------------------ UI ------------------
st.set_page_config(page_title="Signal Alert AVTR — Pro fiable", layout="wide")
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
st.caption("Zéro saisie. Historique synthétique (10 000 pts) auto-généré. Ensemble ML + intervalle type conformal. Avec reprise automatique si données insuffisantes.")

# ------------------ Génération 10k points ------------------
def generate_synthetic_history(n_points=10_000, seed=123):
    rng = np.random.default_rng(seed)
    # chaîne de Markov calme/volatile
    state, p_switch = 0, 0.02
    states = []
    for _ in range(n_points):
        if rng.random() < p_switch:
            state = 1 - state
        states.append(state)
    states = np.array(states)

    mu0, sigma0 = np.log(0.6), 0.35
    mu1, sigma1 = np.log(0.7), 0.70
    base = np.where(states==0,
                    1.0 + np.exp(rng.normal(mu0, sigma0, size=n_points)),
                    1.0 + np.exp(rng.normal(mu1, sigma1, size=n_points)))
    minutes = np.arange(n_points) % 60
    seasonal = 1.0 + 0.03*np.sin(2*np.pi*minutes/60.0)
    series = base * seasonal
    # queues lourdes
    tail_mask = rng.random(n_points) < 0.015
    if tail_mask.any():
        tail = 1.0 + rng.pareto(3.0, size=tail_mask.sum())
        series[tail_mask] *= (2.0 + 4.0*tail)
    series = np.maximum(series, 1.0)

    now = datetime.now(TZ)
    idx = [now - timedelta(minutes=(n_points-1-i)) for i in range(n_points)]
    return pd.DataFrame({"timestamp": idx, "multiplier": series})

# source d’historique (toujours dispo)
BASE_DF = generate_synthetic_history()

# ------------------ Features ------------------
def add_time_features(df):
    t = df["timestamp"]
    df["minute"] = t.dt.minute
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["min_sin"]  = np.sin(2*np.pi*df["minute"]/60.0)
    df["min_cos"]  = np.cos(2*np.pi*df["minute"]/60.0)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    return df

def add_lags_and_rolls(df, lags=30):
    s = df["multiplier"]
    for k in range(1, lags+1):
        df[f"lag_{k}"] = s.shift(k)
    df["roll_mean_5"]  = s.rolling(5).mean().shift(1)
    df["roll_std_5"]   = s.rolling(5).std().shift(1).fillna(0)
    df["roll_mean_15"] = s.rolling(15).mean().shift(1)
    df["roll_std_15"]  = s.rolling(15).std().shift(1).fillna(0)
    df["roll_mean_30"] = s.rolling(30).mean().shift(1)
    df["roll_std_30"]  = s.rolling(30).std().shift(1).fillna(0)
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
    if df2.empty:
        return np.empty((0,0)), np.array([]), df2
    y = df2["multiplier"].values
    X = df2.drop(columns=["multiplier","timestamp"]).values
    return X, y, df2

# ------------------ Ensemble + “conformal-like” ------------------
def _base_models(n_estimators=400):
    models = {
        "rf": RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=1),
        "et": ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=2),
        "gbr": GradientBoostingRegressor(random_state=3),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_estimators, max_depth=7,
            subsample=0.9, colsample_bytree=0.9,
            learning_rate=0.05, tree_method="hist",
            random_state=4, n_jobs=-1,
        )
    return models

def _fit_oof_stack(X, y, models):
    tscv = TimeSeriesSplit(n_splits=5)
    order = list(models.keys())
    meta_X = np.zeros((len(X), len(order)))
    for tr, va in tscv.split(X):
        X_tr, X_va = X[tr], X[va]; y_tr = y[tr]
        for j, name in enumerate(order):
            m = models[name]
            m.fit(X_tr, y_tr)
            meta_X[va, j] = m.predict(X_va)
    for name in order:
        models[name].fit(X, y)
    meta = Ridge(alpha=1.0).fit(meta_X, y)
    return models, meta, meta_X

def _conformal_radius(y_true_oof, y_pred_oof, alpha=0.10):
    resid = np.abs(y_true_oof - y_pred_oof)
    return float(np.quantile(resid, 1 - alpha))

def _stack_predict(models, meta, X):
    order = list(models.keys())
    P = np.column_stack([models[name].predict(X) for name in order])
    return meta.predict(P)

def _confidence(width, p, recent_vol):
    denom = max(abs(p), 1.0) + 0.5*max(recent_vol, 1e-6)
    rel = width / denom
    return float(np.clip(100*(1-rel), 0, 100))

@st.cache_resource(show_spinner=False)
def fit_models(df, lags=30, n_estimators=400, alpha=0.10, train_points=4000):
    # fail-safe 1: garantit suffisamment de points
    if len(df) < train_points + lags + 50:
        extra = generate_synthetic_history(n_points=train_points + lags + 60)
        df = pd.concat([extra.iloc[:-len(df)], df], ignore_index=True) if len(df)>0 else extra

    if train_points and len(df) > train_points:
        df = df.iloc[-train_points:].copy()

    X, y, df2 = build_feature_matrix(df, lags=lags)
    # fail-safe 2: réduit automatiquement les lags si besoin
    lag_try = lags
    while (X.size == 0 or len(X) < 120) and lag_try > 10:
        lag_try -= 5
        X, y, df2 = build_feature_matrix(df, lags=lag_try)

    if X.size == 0 or len(X) < 60:
        # dernier secours: renverra None pour déclencher la baseline
        return None, None, df2, None, None, None, lag_try

    models = _base_models(n_estimators=n_estimators)
    models, meta, oof = _fit_oof_stack(X, y, models)
    y_oof = meta.predict(oof)
    radius = _conformal_radius(y, y_oof, alpha=alpha)
    mae = mean_absolute_error(y, y_oof)
    rmse = mean_squared_error(y, y_oof, squared=False)
    return models, meta, df2, radius, mae, rmse, lag_try

def forecast_next_60(df, lags, models, meta, radius):
    work = df.copy()
    recent_vol = float(pd.Series(work["multiplier"]).tail(60).std() or 0.0)
    preds, confs, times = [], [], []
    last_ts = work["timestamp"].iloc[-1]
    for _ in range(60):
        X_all, _, _ = build_feature_matrix(work, lags=lags)
        x_last = X_all[-1:]
        p = float(_stack_predict(models, meta, x_last)[0])
        width = 2.0 * radius
        c = _confidence(width, p, recent_vol)
        next_ts = last_ts + timedelta(minutes=1)
        work = pd.concat([work, pd.DataFrame({"timestamp":[next_ts], "multiplier":[p]})], ignore_index=True)
        preds.append(p); confs.append(c); times.append(next_ts)
        last_ts = next_ts
    return pd.DataFrame({"timestamp_kst": times,
                         "predicted_multiplier": np.round(preds, 4),
                         "confidence_0_100": np.round(confs, 1)})

# ------------------ Baseline secours (si ensemble indisponible) ------------------
def baseline_forecast(df):
    # moyenne mobile sur 30 pts + bruit faible → 60 min
    s = pd.Series(df["multiplier"].values)
    ma = s.rolling(30).mean().iloc[-1]
    preds = np.clip(np.full(60, ma) + np.random.normal(0, 0.03, 60), 1.0, None)
    now = df["timestamp"].iloc[-1]
    times = [now + timedelta(minutes=i+1) for i in range(60)]
    return pd.DataFrame({"timestamp_kst": times,
                         "predicted_multiplier": np.round(preds, 4),
                         "confidence_0_100": np.full(60, 40.0)})

# ------------------ Un bouton ------------------
LAGS = 30
N_EST = 400
ALPHA = 0.10
TRAIN_POINTS = 4000

run = st.button("🚀 Charger et prédire l’heure suivante (KST)")
if run:
    try:
        models, meta, df2, radius, mae, rmse, used_lags = fit_models(
            BASE_DF, lags=LAGS, n_estimators=N_EST, alpha=ALPHA, train_points=TRAIN_POINTS
        )
        if models is None:
            # fallback garanti (aucun échec visible pour l’utilisateur)
            out = baseline_forecast(BASE_DF)
            st.warning("Mode secours (données insuffisantes pour l’ensemble). Baseline moyenne mobile utilisée.")
        else:
            out = forecast_next_60(BASE_DF, used_lags, models, meta, radius)

        st.markdown("### Prédictions minute par minute (60 prochaines minutes)")
        st.dataframe(out)
        st.download_button("Télécharger les prédictions (CSV)",
                           out.to_csv(index=False),
                           file_name="signal_alert_avtr_predictions_60min.csv")

        if models is not None:
            used = "RF + ET + GBR" + (" + XGBoost" if HAS_XGB else "")
            st.caption(f"Modèles: {used} • lags={used_lags} • α={ALPHA:.2f} • MAE={mae:.3f} • RMSE={rmse:.3f} • train={TRAIN_POINTS} pts")
    except Exception as e:
        # coûte-que-coûte on renvoie des prédictions
        out = baseline_forecast(BASE_DF)
        st.error(f"Ensemble indisponible ({e}). Mode secours activé.")
        st.dataframe(out)

