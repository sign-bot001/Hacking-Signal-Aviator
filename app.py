# app.py simplifié et robuste
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Signal Alert AVTR — Simple", layout="wide")
TZ = pytz.timezone("Asia/Seoul")

st.markdown('<h2 style="color:#7afcff">⚡ Signal Alert AVTR — Prédictions minute/minute</h2>', unsafe_allow_html=True)

# --- Historique synthétique rapide ---
def generate_history(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    vals = 1.0 + np.exp(rng.normal(np.log(0.7), 0.5, n))
    vals = np.clip(vals, 1.0, 50.0)
    now = datetime.now(TZ)
    ts = [now - timedelta(minutes=(n-1-i)) for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "multiplier": vals})

df = generate_history()

# --- Features simples ---
def make_features(df, lags=20):
    d = df.copy()
    for k in range(1, lags+1):
        d[f"lag_{k}"] = d["multiplier"].shift(k)
    d = d.dropna().reset_index(drop=True)
    X = d.drop(columns=["timestamp","multiplier"]).values
    y = d["multiplier"].values
    return X, y, d

# --- Entraînement + prédiction ---
if st.button("🚀 Charger et prédire l’heure suivante (KST)"):
    X, y, d = make_features(df)
    if len(X) < 100:
        st.error("Pas assez de données pour entraîner.")
    else:
        model = ExtraTreesRegressor(n_estimators=150, random_state=42)
        model.fit(X, y)
        y_hat = model.predict(X)
        mae = mean_absolute_error(y, y_hat)
        rmse = mean_squared_error(y, y_hat, squared=False)

        # Prédire les 60 prochaines minutes
        preds = []
        last_row = d.iloc[-1:].copy()
        last_ts = last_row["timestamp"].iloc[0]
        last_feats = X[-1:].copy()
        for i in range(60):
            p = model.predict(last_feats)[0]
            next_ts = last_ts + timedelta(minutes=1)
            preds.append((next_ts, round(p,4)))
            # décaler pour prédire itérativement
            last_feats = np.roll(last_feats, shift=-1, axis=1)
            last_feats[0, -1] = p
            last_ts = next_ts

        out = pd.DataFrame(preds, columns=["timestamp_kst","predicted_multiplier"])
        st.success(f"Modèle entraîné (MAE={mae:.3f}, RMSE={rmse:.3f})")
        st.dataframe(out)

