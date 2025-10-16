# app.py — Signal Alert AVTR (session_state + tableau cliquable + fiche)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz, time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------- CONFIG UI --------------------
st.set_page_config(page_title="Signal Alert AVTR — Prédictions minute/minute", layout="wide")
TZ = pytz.timezone("Asia/Seoul")
st.markdown("""
<style>
body{background:#050608}
.stApp{color:#cfeef8}
.title{font-family:'Courier New',monospace;color:#7afcff;font-size:28px}
.card{
  background: linear-gradient(135deg, #0b0f14 0%, #121821 100%);
  border:1px solid #1f2a36; border-radius:16px; padding:18px; box-shadow:0 10px 25px rgba(0,0,0,.35)
}
.k{color:#7afcff; font-family:'Courier New',monospace;}
.v{color:#e8ffe5; font-weight:700}
.badge{display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700;
  background:#0e2233; border:1px solid #274b63; color:#9fe3ff; margin-left:8px}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">⚡ Signal Alert AVTR — 60 prédictions (KST)</div>', unsafe_allow_html=True)
st.caption("Clique « Charger et prédire », puis coche une ligne pour afficher la fiche détaillée. (Heure HH:MM, cotes à 2 décimales)")

# -------------------- DATA SYNTHÉTIQUE --------------------
@st.cache_data(show_spinner=False)
def generate_history(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    p_switch, state = 0.02, 0
    states=[]
    for _ in range(n):
        if rng.random()<p_switch: state = 1-state
        states.append(state)
    states=np.array(states)
    mu0,s0 = np.log(0.6),0.35
    mu1,s1 = np.log(0.7),0.70
    base = np.where(states==0, 1.0+np.exp(rng.normal(mu0,s0,n)),
                              1.0+np.exp(rng.normal(mu1,s1,n)))
    minutes = np.arange(n)%60
    series = base*(1.0+0.03*np.sin(2*np.pi*minutes/60.0))
    mask = rng.random(n)<0.015
    if mask.any():
        series[mask] *= (2.0 + 4.0*(1.0 + rng.pareto(3.0, mask.sum())))
    series = np.maximum(series, 1.0)
    now = datetime.now(TZ)
    ts = [now - timedelta(minutes=(n-1-i)) for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "multiplier": series})

BASE = generate_history()

# -------------------- FEATURES + MODEL --------------------
def build_features(df, lags=30):
    df2 = df.copy()
    t = df2["timestamp"]
    df2["min_sin"]  = np.sin(2*np.pi*t.dt.minute/60.0)
    df2["min_cos"]  = np.cos(2*np.pi*t.dt.minute/60.0)
    df2["hour_sin"] = np.sin(2*np.pi*t.dt.hour/24.0)
    df2["hour_cos"] = np.cos(2*np.pi*t.dt.hour/24.0)
    s = df2["multiplier"]
    for k in range(1, lags+1):
        df2[f"lag_{k}"] = s.shift(k)
    df2["roll_mean_15"] = s.rolling(15).mean().shift(1)
    df2["roll_std_15"]  = s.rolling(15).std().shift(1).fillna(0)
    df2 = df2.dropna().reset_index(drop=True)
    y = df2["multiplier"].values
    X = df2.drop(columns=["multiplier","timestamp"]).values
    return X, y, df2

def fit_fast_ensemble(X, y, n_trees=150, n_models=5):
    models=[]; seeds=[1,2,3,4,5][:n_models]
    for sd in seeds:
        m = ExtraTreesRegressor(n_estimators=n_trees, random_state=sd, n_jobs=-1)
        m.fit(X,y); models.append(m)
    return models

def predict_ensemble(models, X):
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1), preds.std(axis=1)

def conf_from_std(std_val, ref):
    rel = std_val / max(ref, 1.0)
    return float(np.clip(100*(1-rel), 0, 100))

def forecast_next_60(df, models, lags=30):
    work = df.copy()
    last_ts = work["timestamp"].iloc[-1]
    rows=[]
    for _ in range(60):
        X_all, _, _ = build_features(work, lags=lags)
        x_last = X_all[-1:]
        p_mean, p_std = predict_ensemble(models, x_last)
        p = float(p_mean[0])  # aucune limite max
        c = conf_from_std(float(p_std[0]), ref=work["multiplier"].tail(60).std() or 1.0)
        next_ts = last_ts + timedelta(minutes=1)
        rows.append((next_ts, p, c))
        work = pd.concat([work, pd.DataFrame({"timestamp":[next_ts], "multiplier":[p]})], ignore_index=True)
        last_ts = next_ts

    out = pd.DataFrame(rows, columns=["timestamp_kst","predicted_multiplier","confidence_0_100"])
    out["time_kst"] = out["timestamp_kst"].dt.tz_convert(TZ).dt.strftime("%H:%M")
    out["predicted_multiplier"] = out["predicted_multiplier"].round(2)
    out["confidence_0_100"] = out["confidence_0_100"].round(1)
    return out

# -------------------- BOUTON : CALCUL & STOCKAGE EN SESSION --------------------
if st.button("🚀 Charger et prédire l’heure suivante (KST)", type="primary"):
    t0=time.time()
    df_train = BASE.iloc[-3000:].copy()
    X, y, _ = build_features(df_train, lags=30)
    models = fit_fast_ensemble(X, y, n_trees=150, n_models=5)
    y_hat, _ = predict_ensemble(models, X)
    st.session_state["metrics"] = {
        "mae": float(mean_absolute_error(y, y_hat)),
        "rmse": float(mean_squared_error(y, y_hat, squared=False)),
        "elapsed": time.time()-t0
    }
    st.session_state["preds_df"] = forecast_next_60(df_train, models, lags=30)

# -------------------- RENDU TABLE + FICHE (TOUJOURS EN DEHORS DU BOUTON) --------------------
preds_df = st.session_state.get("preds_df", None)

if preds_df is not None and isinstance(preds_df, pd.DataFrame) and len(preds_df) > 0:
    # ajoute colonne de sélection si absente
    if "selected" not in preds_df.columns:
        preds_df = preds_df.copy()
        preds_df.insert(0, "selected", False)
        st.session_state["preds_df"] = preds_df  # sauvegarde

    st.markdown("### Prédictions (prochaine heure, KST)")
    edited = st.data_editor(
        preds_df[["selected","time_kst","predicted_multiplier","confidence_0_100"]],
        use_container_width=True,
        hide_index=True,
        key="preds_table",
        column_config={
            "selected": st.column_config.CheckboxColumn("Sélection", help="Coche une ligne pour afficher la fiche"),
            "time_kst": "Heure (KST)",
            "predicted_multiplier": st.column_config.NumberColumn("Cote prédite (x)", format="%.2f"),
            "confidence_0_100": st.column_config.NumberColumn("Confiance (0–100)", format="%.1f"),
        },
        disabled=["time_kst","predicted_multiplier","confidence_0_100"],
    )

    # met à jour la sélection en session
    st.session_state["preds_df"]["selected"] = edited["selected"].values

    # ligne cochée ?
    sel_rows = st.session_state["preds_df"][st.session_state["preds_df"]["selected"]]
    if sel_rows.shape[0] == 0:
        st.info("Coche une ligne pour afficher la fiche détaillée.")
    else:
        i = int(sel_rows.index[0])
        row = st.session_state["preds_df"].iloc[i]
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <div class="k">Signal Alert AVTR</div>
            <span class="badge">KST</span>
          </div>
          <div style="display:grid;grid-template-columns: 180px 1fr; row-gap:8px">
            <div class="k">Heure :</div><div class="v">{row['time_kst']}</div>
            <div class="k">Cote prédite :</div><div class="v">{row['predicted_multiplier']:.2f}×</div>
            <div class="k">Confiance :</div><div class="v">{row['confidence_0_100']:.1f} / 100</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # métriques
    m = st.session_state.get("metrics", {})
    if m:
        st.caption(f"Modèle: 5×ExtraTrees(150) • Train=3000 pts • MAE={m['mae']:.3f} • RMSE={m['rmse']:.3f} • Génération {m['elapsed']:.1f}s")

else:
    st.info("Clique le bouton ci-dessus pour générer les 60 prédictions.")
