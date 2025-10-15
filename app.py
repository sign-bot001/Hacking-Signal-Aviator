# app.py — Signal Alert AVTR (Pro Max, 10k points synthétiques, 1 bouton)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# XGBoost (optionnel, auto-fallback si non dispo)
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
st.caption("Zéro saisie. Ensemble ML (RF+ET+GBR+XGBoost) + Conformal-like pour la confiance. Historique synthétique 10 000 points (queues lourdes, régimes).")

# ------------------ GÉNÉRATION D'UN HISTORIQUE SYNTHÉTIQUE (10k) ------------------
def generate_synthetic_history(n_points=10_000, seed=42):
    """
    Série minute par minute réaliste pour jeux à multiplicateur:
    - Base lognormale (valeurs >= 1.0) pour la majorité
    - Pics rares via multiplicateur Pareto (queues lourdes)
    - Deux régimes (calme/volatile) avec chaîne de Markov
    - Légère saisonnalité intra-heure
    """
    rng = np.random.default_rng(seed)
    # Régimes: 0 = calme, 1 = volatile
    p_switch = 0.02
    state = 0
    states = []
    for _ in range(n_points):
        if rng.random() < p_switch:
            state = 1 - state
        states.append(state)
    states = np.array(states)

    # paramètres par régime (lognormal sur log(multiplicateur-1))
    # on produit m = 1 + exp(N(mu, sigma)) pour garder m>=1
    mu0, sigma0 = np.log(0.6), 0.35   # calme
    mu1, sigma1 = np.log(0.7), 0.70   # volatile (beaucoup plus dispersé)

    base = np.where(
        states == 0,
        1.0 + np.exp(rng.normal(mu0, sigma0, size=n_points)),
        1.0 + np.exp(rng.normal(mu1, sigma1, size=n_points))
    )

    # Effet saisonnier léger intra-heure (sinus minute)
    minutes = np.arange(n_points) % 60
    seasonal = 1.0 + 0.03*np.sin(2*np.pi*minutes/60.0)  # +/-3%
    series = base * seasonal

    # Pics rares (queues lourdes) via Pareto (alpha>2)
    tail_prob = 0.015  # ~1.5% de grosses sorties
    tail_mask = rng.random(n_points) < tail_prob
    if tail_mask.any():
        # pareto(alpha) + 1 => heavy tail, scale 2x à 6x
        alpha = 3.0
        tail = 1.0 + rng.pareto(alpha, size=tail_mask.sum())
        # ecrase les série aux instants tail_mask en multipliant
        series[tail_mask] *= (2.0 + 4.0*tail)  # pics 5x..30x+ occasionnels

    # Clamp minimal (>=1.0)
    series = np.maximum(series, 1.0)

    # timestamps (finissant "maintenant" en KST)
    now = datetime.now(TZ)
    idx = [now - timedelta(minutes=(n_points-1-i)) for i in range(n_points)]
    df = pd.DataFrame({"timestamp": idx, "multiplier": series})
    return df

BASE_DF = generate_synthetic_history(n_points=10_000, seed=123)

# ------------------ FEATURES ------------------
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
    y = df2["multiplier"].values
    X = df2.drop(columns=["multiplier","timestamp"]).values
    return X, y, df2

# ------------------ MODÈLES (STACKING + INTERVALLE TYPE CONFORMAL) ------------------
def _base_models(n_estimators=400):
    models = {
        "rf": RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=1),
        "et": ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=2),
        "gbr": GradientBoostingRegressor(random_state=3),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=7,
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
    order = list(models.keys())
    meta_X = np.zeros((len(X), len(order)))
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

def _conformal_radius(y_true_oof, y_pred_oof, alpha=0.10):
    resid = np.abs(y_true_oof - y_pred_oof)
    return float(np.quantile(resid, 1 - alpha))  # rayon d'intervalle

@st.cache_resource(show_spinner=False)
def fit_models_with_radius(df, lags=30, n_estimators=400, alpha=0.10):
    X, y, df2 = build_feature_matrix(df, lags=lags)
    models = _base_models(n_estimators=n_estimators)
    models, meta, oof = _fit_oof_stack(X, y, models)
    y_oof = meta.predict(oof)
    radius = _conformal_radius(y, y_oof, alpha=alpha)
    mae = mean_absolute_error(y, y_oof)
    rmse = mean_squared_error(y, y_oof, squared=False)
    return models, meta, df2, radius, mae, rmse

def _confidence_from_width(p, width, recent_vol):
    denom = max(abs(p), 1.0) + 0.5*max(recent_vol, 1e-6)
    rel = width / denom
    conf = 100 * (1 - np.clip(rel, 0, 1))
    return float(np.clip(conf, 0, 100))

def forecast_next_60(df, lags, models, meta, radius):
    work = df.copy()
    recent_vol = float(pd.Series(work["multiplier"]).tail(60).std() or 0.0)
    preds, confs, times = [], [], []
    last_ts = work["timestamp"].iloc[-1]
    for _ in range(60):
        X_all, _, _ = build_feature_matrix(work, lags=lags)
        x_last = X_all[-1:]
        p, _ = _stack_predict(models, meta, x_last)
        p = float(p[0])
        width = 2.0 * radius  # intervalle [p - r, p + r]
        c = _confidence_from_width(p, width, recent_vol)
        next_ts = last_ts + timedelta(minutes=1)
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
N_EST = st.sidebar.slider("Arbres par modèle", 300, 800, 400)
ALPHA = 0.10  # ~90% couverture

run = st.button("🚀 Charger et prédire l’heure suivante (KST)")

if run:
    try:
        models, meta, df2, radius, mae, rmse = fit_models_with_radius(
            BASE_DF, lags=LAGS, n_estimators=N_EST, alpha=ALPHA
        )
        out = forecast_next_60(BASE_DF, LAGS, models, meta, radius)
        st.markdown("### Prédictions minute par minute (60 prochaines minutes)")
        st.dataframe(out)
        st.download_button(
            "Télécharger les prédictions (CSV)",
            out.to_csv(index=False),
            file_name="signal_alert_avtr_predictions_60min.csv"
        )
        used = "RF + ET + GBR" + (" + XGBoost" if HAS_XGB else "")
        st.caption(f"Modèles: {used} • Intervalle (α={ALPHA:.2f}) → rayon={radius:.4f} • Qualité interne (OOF): MAE={mae:.3f} | RMSE={rmse:.3f}")
        if not HAS_XGB:
            st.caption("XGBoost non détecté (fallback automatique).")
    except Exception as e:
        st.error(f"Impossible de générer les prédictions : {e}")

