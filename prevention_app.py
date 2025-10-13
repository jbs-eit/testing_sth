# app.py
# -------------------------------------------------------------
# Economic Simulation Prototype (Streamlit) — Model Selection only
#   - Model Estimation tab REMOVED from UI (recoverable via flag)
#   - Pre-defined variables with pre-trained models (seeded on startup)
#   - Diagnostics per model (metrics, calibration, coefficients/importances)
#   - Multiple interventions, simulation run, JSON export of decisions
# -------------------------------------------------------------
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# === Feature flag: set to True to re-enable the Model Estimation tab ===
SHOW_MODEL_ESTIMATION = False

st.set_page_config(
    page_title="Economic Simulation Prototype",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_LIKE = {"int64", "int32", "float64", "float32", "int16", "float16"}

# Minimal default configuration so UI always has a pipeline to work with
def default_config() -> Dict[str, Any]:
    return {
        "meta": {"name": "Demo Scenario"},
        "pipeline": {
            "catalog": {},          # pre-trained catalog metadata (filled later)
            "selection": {},        # chosen model per target
            "interventions": [],    # list of interventions
            "simulation": {         # moved from sidebar into Interventions tab
                "population_size": 69_000_000,
                "random_seed": 42,
                "num_simulations": 100,
            },
        },
    }

# --------------------------- Synthetic data & toy risk ---------------------------
def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def risk_heart_attack(prob_inputs: pd.DataFrame) -> np.ndarray:
    bmi = prob_inputs["bmi"].to_numpy()
    age = prob_inputs["age"].to_numpy()
    a = -4.2; b1 = 0.50; b2 = 0.65
    z = a + b1 * ((bmi - 25.0) / 5.0) + b2 * ((age - 50.0) / 10.0)
    return logistic(z)

def generate_synthetic_population(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 90, size=n)
    base_log_income = rng.normal(loc=10.5, scale=0.6, size=n)
    age_effect = (ages - 45) / 45 * 0.15
    log_income = base_log_income + age_effect
    employed = rng.random(size=n) < (0.7 - 0.002 * np.clip(ages - 40, 0, None))
    employment_status = np.where(employed, "employed", "other")
    bmi = rng.normal(loc=25 + 0.03 * (ages - 45), scale=4.0, size=n)
    bmi = np.clip(bmi, 15, 60)
    df = pd.DataFrame({
        "age": ages.astype(int),
        "log_income": log_income.astype(float),
        "employment_status": employment_status.astype(str),
        "bmi": bmi.astype(float),
    })
    df["is_employed"] = (df["employment_status"] == "employed").astype(int)
    df["prob_hattack_true"] = risk_heart_attack(df)
    df["hattack_ever_w10"] = (rng.random(size=n) < df["prob_hattack_true"]).astype(int)
    return df

# --------------------------- Helpers ---------------------------
def available_variables(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        if c in ("prob_hattack_true",):
            continue
        vals = df[c].dropna().unique()
        if len(vals) <= 2 and set(vals).issubset({0,1}):
            out[c] = "binary"
        elif str(df[c].dtype) in NUMERIC_LIKE:
            out[c] = "numeric"
        else:
            out[c] = "categorical"
    return out

def build_design_matrix(df: pd.DataFrame, target: str, spec: Dict[str, Any]) -> pd.DataFrame:
    base = [b for b in spec.get("base_features", []) if b != target and b in df.columns]
    X = pd.DataFrame(index=df.index)
    for b in base: X[b] = df[b]
    for b in spec.get("log", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            X[f"log_{b}"] = np.log1p(np.clip(df[b].to_numpy(), a_min=0, a_max=None))
    for b in spec.get("square", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            X[f"{b}_sq"] = np.square(df[b])
    for (u,v) in spec.get("interactions", []):
        if u in df.columns and v in df.columns and (str(df[u].dtype) in NUMERIC_LIKE) and (str(df[v].dtype) in NUMERIC_LIKE):
            X[f"{u}*{v}"] = df[u].to_numpy() * df[v].to_numpy()
    X = pd.get_dummies(X, drop_first=True).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return X

def _range_to_tuple(val: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    try:
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return fallback

def apply_intervention(df: pd.DataFrame, inv: Dict[str, Any]) -> pd.DataFrame:
    new_df = df.copy()
    target_var = inv.get("target")
    if target_var not in new_df.columns: return new_df
    iv_type = inv.get("type", "percentage_decrease"); amount = float(inv.get("amount", 0.0))
    filters = inv.get("filters", {}) or {}
    mask = pd.Series(True, index=new_df.index)
    if "age" in filters:
        a_min,a_max = _range_to_tuple(filters["age"], (new_df["age"].min(), new_df["age"].max()))
        mask &= (new_df["age"]>=a_min) & (new_df["age"]<=a_max)
    if "bmi" in filters and "bmi" in new_df:
        b_min,b_max = _range_to_tuple(filters["bmi"], (new_df["bmi"].min(), new_df["bmi"].max()))
        mask &= (new_df["bmi"]>=b_min) & (new_df["bmi"]<=b_max)
    if iv_type == "percentage_decrease":
        new_df.loc[mask, target_var] = new_df.loc[mask, target_var] * (1.0 - amount)
    elif iv_type == "absolute_change":
        new_df.loc[mask, target_var] = new_df.loc[mask, target_var] + amount
    return new_df

# --------------------------- Defaults & seeding ---------------------------
def apply_defaults_to_selection():
    # Choose default models: XGBoost where available else GLM
    trained = st.session_state.get("trained_models", {})
    chosen = {}
    for tgt, pack in trained.items():
        names = list(pack.get("models", {}).keys())
        pick = None
        for pref in ["XGBoost Regressor", "XGBoost Classifier", "Logistic Regression", "Linear Regression"]:
            if pref in names: pick = pref; break
        pick = pick or (names[0] if names else None)
        if pick: chosen[tgt] = pick
    st.session_state["chosen_models"] = chosen

def apply_defaults_to_interventions(df: pd.DataFrame):
    st.session_state["interventions"] = [
        {"target": "bmi", "type": "percentage_decrease", "amount": 0.2, "filters": {"age":[40,60], "bmi":[30,100]}},
        {"target": "bmi", "type": "percentage_decrease", "amount": 0.1, "filters": {"age":[60,90], "bmi":[28,100]}},
    ]

def seed_pretrained_models():
    """Create a tiny pre-trained 'model repo' on synthetic data so selection works without estimation."""
    if st.session_state.get("trained_models"):
        return
    # Ensure data
    if "training_data" not in st.session_state:
        st.session_state["training_data"] = generate_synthetic_population(
            st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42)
        )
    df = st.session_state["training_data"]
    presets = []

    algos_reg = ["linear_regression"]; 
    if XGB_AVAILABLE: algos_reg.append("xgboost_regressor")
    presets.append({
        "target": "bmi",
        "target_type": "continuous",
        "spec": {"base_features": ["age","log_income","is_employed"], "log":["log_income"], "square":["age"], "interactions":[("age","log_income")]},
        "algorithms": algos_reg,
        "train_test_split": {"test_size":0.2, "random_state":42, "stratify": None},
    })
    algos_cls = ["logistic_regression"];
    if XGB_AVAILABLE: algos_cls.append("xgboost_classifier")
    presets.append({
        "target": "hattack_ever_w10",
        "target_type": "binary",
        "spec": {"base_features": ["age","bmi","is_employed"], "log":[], "square":["age"], "interactions":[("age","bmi")]},
        "algorithms": algos_cls,
        "train_test_split": {"test_size":0.2, "random_state":42, "stratify":"hattack_ever_w10"},
    })

    st.session_state["trained_models"] = {}
    st.session_state["trained_models_meta"] = {}

    for p in presets:
        target = p["target"]; ttype = p["target_type"]; spec = p["spec"]; algos = p["algorithms"]; tts = p["train_test_split"]
        y = df[target].astype(int) if ttype=="binary" else df[target]
        X = build_design_matrix(df, target, spec)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=tts["test_size"], random_state=tts["random_state"],
                                           stratify=y if (ttype=="binary" and tts.get("stratify")) else None)
        pack = {"type": ttype, "feature_spec": spec, "train_test_split": tts, "models": {}, "columns": list(X.columns)}
        # Linear
        if ttype=="continuous" and "linear_regression" in algos:
            est = LinearRegression().fit(Xtr,ytr); yhat = est.predict(Xte)
            pack["models"]["Linear Regression"] = {
                "estimator": est,
                "metrics": {"RMSE": float(np.sqrt(mean_squared_error(yte, yhat))), "MAE": float(mean_absolute_error(yte, yhat)), "R2": float(r2_score(yte, yhat))},
                "pred_true": {"y_true": yte.to_numpy(), "y_pred": yhat},
            }
        # XGB reg
        if ttype=="continuous" and "xgboost_regressor" in algos and XGB_AVAILABLE:
            est = XGBRegressor(n_estimators=250,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=tts["random_state"],reg_lambda=1.0,n_jobs=4).fit(Xtr,ytr)
            yhat = est.predict(Xte)
            pack["models"]["XGBoost Regressor"] = {
                "estimator": est,
                "metrics": {"RMSE": float(np.sqrt(mean_squared_error(yte, yhat))), "MAE": float(mean_absolute_error(yte, yhat)), "R2": float(r2_score(yte, yhat))},
                "pred_true": {"y_true": yte.to_numpy(), "y_pred": yhat},
            }
        # Logit
        if ttype=="binary" and "logistic_regression" in algos:
            est = LogisticRegression(max_iter=1000).fit(Xtr,ytr)
            prob = est.predict_proba(Xte)[:,1]
            pt,pp = calibration_curve(yte, prob, n_bins=12, strategy="uniform")
            pack["models"]["Logistic Regression"] = {
                "estimator": est,
                "metrics": {"RMSE": float(np.sqrt(mean_squared_error(yte, prob))), "MAE": float(mean_absolute_error(yte, prob)), "LogLoss": float(log_loss(yte, prob))},
                "pred_true": {"y_true": yte.to_numpy(), "y_pred": prob},
                "calibration": {"prob_true": pt, "prob_pred": pp},
            }
        # XGB cls
        if ttype=="binary" and "xgboost_classifier" in algos and XGB_AVAILABLE:
            est = XGBClassifier(n_estimators=350,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=tts["random_state"],reg_lambda=1.0,n_jobs=4,eval_metric="logloss").fit(Xtr,ytr)
            prob = est.predict_proba(Xte)[:,1]
            pt,pp = calibration_curve(yte, prob, n_bins=12, strategy="uniform")
            pack["models"]["XGBoost Classifier"] = {
                "estimator": est,
                "metrics": {"RMSE": float(np.sqrt(mean_squared_error(yte, prob))), "MAE": float(mean_absolute_error(yte, prob)), "LogLoss": float(log_loss(yte, prob))},
                "pred_true": {"y_true": yte.to_numpy(), "y_pred": prob},
                "calibration": {"prob_true": pt, "prob_pred": pp},
            }
        st.session_state["trained_models"][target] = pack
        st.session_state["trained_models_meta"][target] = {
            "type": pack["type"],
            "feature_spec": pack["feature_spec"],
            "train_test_split": pack["train_test_split"],
            "columns": pack["columns"],
            "models": {k: {"metrics": v["metrics"]} for k,v in pack["models"].items()},
        }

# --------------------------- Sidebar & JSON ---------------------------

def build_pipeline_json() -> Dict[str, Any]:
    scenario_name = st.session_state.get("scenario_name", "Demo Scenario")
    selection = st.session_state.get("chosen_models", {})
    interventions = st.session_state.get("interventions", [])
    catalog_meta = st.session_state.get("catalog_meta", {})
    sim = {
        "population_size": int(st.session_state.get("sim_population", 69_000_000)),
        "random_seed": int(st.session_state.get("sim_seed", 42)),
        "num_simulations": int(st.session_state.get("sim_runs", 100)),
    }

    return {
        "meta": {"name": scenario_name},
        "pipeline": {
            "catalog": catalog_meta,
            "selection": selection,
            "interventions": interventions,
            "simulation": sim,
        },
    }
def sidebar_controls():
    st.sidebar.header("⚙️ Scenario & Data")
    st.sidebar.text_input("Scenario name", key="scenario_name", value=st.session_state.get("scenario_name", "Demo Scenario"))
    st.sidebar.caption("Synthetic Training/Simulation Settings")
    st.sidebar.number_input("Population size", min_value=2_000, max_value=300_000, value=30_000, step=1_000, key="pop_n")
    st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1, key="seed")
    st.sidebar.divider()
    st.sidebar.caption("Download pipeline configuration (decisions so far)")
    st.sidebar.download_button("💾 Download pipeline JSON", data=json.dumps(build_pipeline_json(), indent=2), file_name="pipeline_config.json", mime="application/json", use_container_width=True)

# --------------------------- Model Estimation UI (kept for recovery) ---------------------------
# NOTE: This code block is NOT used unless SHOW_MODEL_ESTIMATION=True.
# It is kept here (commented) so you can easily recover the page if needed.
# """
# def model_estimation_tab():
#     # (Disabled) Original UI allowed recording multiple training specs and training all at once.
#     # To recover: set SHOW_MODEL_ESTIMATION=True at the top, and add the tab back in main().
#     pass
# """

# --------------------------- Model Selection ---------------------------
def model_selection_tab():
    top_cols = st.columns([1,4,1])
    with top_cols[1]:
        if st.button("Use default configuration", key="btn_default_selection"):
            apply_defaults_to_selection()
            st.success("Default selections applied (prefers XGBoost where available).")

    st.subheader("🧩 Model selection")
    trained = st.session_state.get("trained_models", {})
    if not trained:
        st.info("Seeding pre-trained models...")
        seed_pretrained_models()
        trained = st.session_state.get("trained_models", {})

    variables = list(trained.keys())
    var = st.selectbox("Select a variable to inspect & choose model", options=variables, index=0, key="sel_target")
    pack = trained[var]; mtype = pack["type"]; models_dict = pack["models"]

    st.markdown("**Error metrics comparison**")
    rows = [{"model": name, **info["metrics"]} for name, info in models_dict.items()]
    met_df = pd.DataFrame(rows)

    if mtype == "continuous":
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(alt.Chart(met_df).mark_bar().encode(x=alt.X("model:N"), y=alt.Y("RMSE:Q"), tooltip=["model", alt.Tooltip("RMSE:Q", format=".4f")]).properties(height=250), use_container_width=True)
        with col2:
            st.altair_chart(alt.Chart(met_df).mark_bar().encode(x=alt.X("model:N"), y=alt.Y("MAE:Q"), tooltip=["model", alt.Tooltip("MAE:Q", format=".4f")]).properties(height=250), use_container_width=True)
        st.markdown("**Predicted vs Actual (test set)**")
        for name, info in models_dict.items():
            d = pd.DataFrame({"y_true": info["pred_true"]["y_true"], "y_pred": info["pred_true"]["y_pred"]})
            st.altair_chart(alt.Chart(d).mark_circle(opacity=0.4).encode(x=alt.X("y_true:Q", title="Actual"), y=alt.Y("y_pred:Q", title="Predicted"), tooltip=[alt.Tooltip("y_true:Q", format=".2f"), alt.Tooltip("y_pred:Q", format=".2f")]).properties(height=250, title=name), use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(alt.Chart(met_df).mark_bar().encode(x=alt.X("model:N"), y=alt.Y("RMSE:Q", title="RMSE (prob vs 0/1)"), tooltip=["model", alt.Tooltip("RMSE:Q", format=".4f")]).properties(height=250), use_container_width=True)
        with col2:
            st.altair_chart(alt.Chart(met_df).mark_bar().encode(x=alt.X("model:N"), y=alt.Y("MAE:Q", title="MAE (prob vs 0/1)"), tooltip=["model", alt.Tooltip("MAE:Q", format=".4f")]).properties(height=250), use_container_width=True)
        st.markdown("**Calibration curves**")
        layers = []
        for name, info in models_dict.items():
            cal = info.get("calibration")
            if cal is None: continue
            layers.append(pd.DataFrame({"prob_pred": cal["prob_pred"], "prob_true": cal["prob_true"], "model": name}))
        if layers:
            dd = pd.concat(layers, ignore_index=True)
            diag = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line().encode(x="x:Q", y="y:Q")
            chart = diag + alt.Chart(dd).mark_line(point=True).encode(x=alt.X("prob_pred:Q", title="Predicted probability (binned)"), y=alt.Y("prob_true:Q", title="Empirical frequency"), color=alt.Color("model:N"))
            st.altair_chart(chart.properties(height=300), use_container_width=True)

    st.divider()
    st.markdown("**Select model for simulation**")
    model_names = list(models_dict.keys())
    chosen = st.radio("Choose one", options=model_names, index=0, key="sel_chosen_model")
    st.session_state.setdefault("chosen_models", {})
    st.session_state["chosen_models"][var] = chosen
    st.success(f"Selected: **{chosen}** for `{var}`")

    st.markdown("**Model details**")
    info = models_dict[chosen]; cols = pack["columns"]
    if chosen in ("Linear Regression", "Logistic Regression"):
        est = info["estimator"]; coef = getattr(est, "coef_", None); intercept = getattr(est, "intercept_", None)
        if coef is not None:
            if getattr(coef, "ndim", 1) > 1: coef = coef.ravel()
            df_coef = pd.DataFrame({"feature": cols, "beta": coef})
            if chosen == "Logistic Regression":
                df_coef["odds_ratio"] = np.exp(df_coef["beta"])
                st.dataframe(df_coef, use_container_width=True, hide_index=True)
                st.caption(f"Intercept (log-odds): {float(intercept):+.4f} • OR: {float(np.exp(intercept)):.4f}")
            else:
                st.dataframe(df_coef, use_container_width=True, hide_index=True)
                st.caption(f"Intercept: {float(intercept):+.4f}")
        else:
            st.info("No coefficients available for this estimator.")
    else:
        est = info["estimator"]
        try:
            fi = est.feature_importances_
            st.dataframe(pd.DataFrame({"feature": cols, "importance": fi}).sort_values("importance", ascending=False), use_container_width=True, hide_index=True)
        except Exception:
            st.info("Feature importances not available for this estimator.")

    st.markdown("### Selections recorded")
    all_sel = st.session_state.get("chosen_models", {})
    if all_sel:
        rows = [{"target": t, "chosen_model": m} for t,m in all_sel.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Update export mirror
    pipe = st.session_state.get("config", {"pipeline": {}}).get("pipeline", {})
    pipe["selection"] = st.session_state.get("chosen_models", {})
    pipe["trained"] = st.session_state.get("trained_models_meta", {})
    st.session_state.setdefault("config", {"pipeline": pipe})
    st.session_state["config"]["pipeline"] = pipe

# --------------------------- Interventions ---------------------------

def interventions_tab():
    top_cols = st.columns([1, 4, 1])
    with top_cols[1]:
        if st.button("Use default configuration", key="btn_default_interventions"):
            # Default: one BMI intervention -20% for BMI 30–200; set simulation defaults
            st.session_state["interventions"] = [
                {"target": "bmi", "type": "percentage_decrease", "amount": 0.2, "filters": {"bmi": [30, 200]}}
            ]
            st.session_state["sim_population"] = 69_000_000
            st.session_state["sim_seed"] = 42
            st.session_state["sim_runs"] = 100
            st.success("Default intervention and simulation settings applied.")

    st.subheader("🧪 Interventions & Simulation settings")
    st.caption("Add multiple interventions. For this demo, filters support BMI (and age if needed). Simulation settings live here.")

    # --- Simulation Settings ---
    st.session_state.setdefault("sim_population", 69_000_000)
    st.session_state.setdefault("sim_seed", 42)
    st.session_state.setdefault("sim_runs", 100)
    with st.container(border=True):
        colA, colB, colC = st.columns(3)
        with colA:
            st.session_state["sim_population"] = st.number_input("Population size", min_value=1_000, max_value=200_000_000, value=int(st.session_state["sim_population"]), step=1000, help="Aggregated simulation; large values will not create microdata.")
        with colB:
            st.session_state["sim_seed"] = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=int(st.session_state["sim_seed"]), step=1)
        with colC:
            st.session_state["sim_runs"] = st.number_input("Monte Carlo simulations", min_value=1, max_value=10_000, value=int(st.session_state["sim_runs"]), step=1)

    # --- Interventions ---
    st.session_state.setdefault("interventions", [])
    # Ensure default intervention exists on first open
    if not st.session_state["interventions"]:
        st.session_state["interventions"] = [
            {"target": "bmi", "type": "percentage_decrease", "amount": 0.2, "filters": {"bmi": [30, 200]}}
        ]

    to_delete = []
    for idx, inv in enumerate(st.session_state["interventions"]):
        with st.container(border=True):
            st.markdown(f"**Intervention {idx+1}**")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                inv["target"] = st.selectbox("Target variable", options=["bmi","log_income"], index=0 if inv.get("target","bmi")=="bmi" else 1, key=f"inv_target_{idx}")
            with col2:
                inv["type"] = st.selectbox("Type", options=["percentage_decrease","absolute_change"], index=["percentage_decrease","absolute_change"].index(inv.get("type","percentage_decrease")), key=f"inv_type_{idx}")
            with col3:
                inv["amount"] = st.number_input("Amount", value=float(inv.get("amount", 0.2)), step=0.01, key=f"inv_amount_{idx}")
            st.markdown("**Filters**")
            f = inv.get("filters", {})
            a_min, a_max = _range_to_tuple(f.get("age"), (18, 90))
            b_min, b_max = _range_to_tuple(f.get("bmi"), (15, 200))
            age_min, age_max = st.slider("Age range", min_value=0, max_value=100, value=(int(a_min), int(a_max)), key=f"inv_age_{idx}")
            bmi_min, bmi_max = st.slider("BMI range", min_value=10, max_value=200, value=(int(b_min), int(b_max)), key=f"inv_bmi_{idx}")
            inv["filters"] = {"age": [age_min, age_max], "bmi": [bmi_min, bmi_max]}
            if st.button("Remove", key=f"inv_remove_{idx}"):
                to_delete.append(idx)
        st.write("")

    for i in sorted(to_delete, reverse=True):
        del st.session_state["interventions"][i]

    if st.button("➕ Add intervention", key="btn_add_intervention"):
        st.session_state["interventions"].append({"target": "bmi", "type": "percentage_decrease", "amount": 0.1, "filters": {"bmi":[30,200]}})

    # Update export mirror
    cfg = st.session_state.get("config", default_config())
    pipe = cfg["pipeline"]
    pipe["interventions"] = st.session_state.get("interventions", [])
    pipe["simulation"] = {
        "population_size": int(st.session_state.get("sim_population", 69_000_000)),
        "random_seed": int(st.session_state.get("sim_seed", 42)),
        "num_simulations": int(st.session_state.get("sim_runs", 100)),
    }
    st.session_state["config"]["pipeline"] = pipe

# --------------------------- Run & Results ---------------------------
def run_and_visualize():
    st.subheader("🚀 Run Simulation")
    pop_n = int(st.session_state.get("pop_n", 30_000)); seed = int(st.session_state.get("seed", 42))
    c1,c2 = st.columns([1,1])
    with c1:
        st.caption("Generate baseline, apply interventions, and use your selected models.")
        go = st.button("Run Simulation", type="primary", use_container_width=True)
    with c2:
        st.caption("Export results or share a quick snapshot.")
        export_name = st.text_input("Export name", value="demo_run")
    if not go:
        st.info("Choose models in **Model selection**, configure **Interventions**, then click **Run Simulation**.")
        return

    df_base = generate_synthetic_population(pop_n, seed=seed)

    # Selected models
    chosen = st.session_state.get("chosen_models", {})
    trained = st.session_state.get("trained_models", {})
    # Predict BMI if selected
    if "bmi" in chosen and "bmi" in trained:
        pack = trained["bmi"]; model_name = chosen["bmi"]; est = pack["models"][model_name]["estimator"]
        Xb = build_design_matrix(df_base, "bmi", pack["feature_spec"])
        for col in pack["columns"]:
            if col not in Xb.columns: Xb[col] = 0.0
        Xb = Xb[pack["columns"]]
        df_base["bmi"] = est.predict(Xb)

    # Apply interventions
    df_post = df_base.copy()
    for inv in st.session_state.get("interventions", []):
        df_post = apply_intervention(df_post, inv)

    # Predict heart-attack probabilities if selected; else fallback
    if "hattack_ever_w10" in chosen and "hattack_ever_w10" in trained:
        pack_h = trained["hattack_ever_w10"]; model_name_h = chosen["hattack_ever_w10"]; est_h = pack_h["models"][model_name_h]["estimator"]
        Xh0 = build_design_matrix(df_base, "hattack_ever_w10", pack_h["feature_spec"])
        for col in pack_h["columns"]:
            if col not in Xh0.columns: Xh0[col] = 0.0
        Xh0 = Xh0[pack_h["columns"]]
        if pack_h["type"] == "binary":
            df_base["prob_hattack"] = est_h.predict_proba(Xh0)[:,1] if hasattr(est_h,"predict_proba") else est_h.predict(Xh0)
        else:
            df_base["prob_hattack"] = np.clip(est_h.predict(Xh0), 0, 1)
        Xh1 = build_design_matrix(df_post, "hattack_ever_w10", pack_h["feature_spec"])
        for col in pack_h["columns"]:
            if col not in Xh1.columns: Xh1[col] = 0.0
        Xh1 = Xh1[pack_h["columns"]]
        if pack_h["type"] == "binary":
            df_post["prob_hattack"] = est_h.predict_proba(Xh1)[:,1] if hasattr(est_h,"predict_proba") else est_h.predict(Xh1)
        else:
            df_post["prob_hattack"] = np.clip(est_h.predict(Xh1), 0, 1)
    else:
        df_base["prob_hattack"] = risk_heart_attack(df_base)
        df_post["prob_hattack"] = risk_heart_attack(df_post)

    rng = np.random.default_rng(seed + 123)
    df_base["hattack_event"] = (rng.random(size=len(df_base)) < df_base["prob_hattack"]).astype(int)
    df_post["hattack_event"] = (rng.random(size=len(df_post)) < df_post["prob_hattack"]).astype(int)

    # Metrics
    prev_base = float(df_base["hattack_event"].mean()); prev_post = float(df_post["hattack_event"].mean())
    delta_prev_abs = prev_post - prev_base
    delta_prev_rel = (prev_post / prev_base - 1.0) if prev_base > 0 else math.nan
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Population", f"{len(df_base):,}")
    m2.metric("Mean BMI (baseline)", f"{df_base['bmi'].mean():.2f}", delta=f"{(df_post['bmi'].mean()-df_base['bmi'].mean()):+.2f}")
    m3.metric("Heart attack prevalence (baseline)", f"{prev_base*100:.2f}%", delta=f"{delta_prev_abs*100:+.2f}%")
    m4.metric("Relative change in prevalence", f"{delta_prev_rel*100:+.2f}%")

    st.divider()
    col_left, col_right = st.columns([1.2,1])
    with col_left:
        st.markdown("**BMI distribution (baseline vs. post-intervention)**")
        tidy_bmi = pd.concat([df_base[["bmi"]].assign(state="Baseline"), df_post[["bmi"]].assign(state="Post")], ignore_index=True)
        chart_bmi = alt.Chart(tidy_bmi).transform_bin(["bmi_bin"], field="bmi", bin=alt.Bin(maxbins=40)).mark_bar(opacity=0.6).encode(
            x=alt.X("bmi_bin:Q", title="BMI (binned)"), y=alt.Y("count()", stack=None, title="Count"), color=alt.Color("state:N"), tooltip=["state","count()"]
        ).properties(height=300)
        st.altair_chart(chart_bmi, use_container_width=True)
    with col_right:
        st.markdown("**Prevalence of heart attack (baseline vs. post)**")
        prev_df = pd.DataFrame({"scenario":["Baseline","Post"], "prevalence":[prev_base*100.0, prev_post*100.0]})
        chart_prev = alt.Chart(prev_df).mark_bar().encode(x=alt.X("scenario:N"), y=alt.Y("prevalence:Q", title="Prevalence (%)"), tooltip=["scenario", alt.Tooltip("prevalence:Q", format=".2f")]).properties(height=300)
        st.altair_chart(chart_prev, use_container_width=True)

    st.divider()
    st.markdown("**Breakdown by age group**")
    def _breakdown(df: pd.DataFrame) -> pd.DataFrame:
        bins=[18,30,40,50,60,70,90]; labels=["18–29","30–39","40–49","50–59","60–69","70–89"]
        g = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
        out = df.assign(age_group=g).groupby("age_group", as_index=False).agg(n=("age","size"), bmi_mean=("bmi","mean"), prev=("hattack_event","mean"))
        out["prev"] = out["prev"]*100.0; return out
    merged = _breakdown(df_base).merge(_breakdown(df_post), on="age_group", suffixes=("_base","_post"))
    c = alt.Chart(merged).transform_fold(["prev_base","prev_post"], as_=["scenario","value"]).mark_line(point=True).encode(
        x=alt.X("age_group:N", title="Age group"), y=alt.Y("value:Q", title="Prevalence (%)", scale=alt.Scale(zero=False)), color=alt.Color("scenario:N", title=""), tooltip=["age_group:N", alt.Tooltip("value:Q", format=".2f"), "scenario:N"]
    ).properties(height=300)
    st.altair_chart(c, use_container_width=True)

    st.caption("Note: Models and outputs here are illustrative. Replace synthetic data and toy DGP with your real pipeline.")
    cA,cB = st.columns(2)
    with cA:
        st.download_button("⬇️ Download baseline microdata (CSV)", data=df_base.to_csv(index=False).encode("utf-8"), file_name=f"{export_name}_baseline.csv", mime="text/csv", use_container_width=True)
    with cB:
        st.download_button("⬇️ Download post-intervention microdata (CSV)", data=df_post.to_csv(index=False).encode("utf-8"), file_name=f"{export_name}_post.csv", mime="text/csv", use_container_width=True)

# --------------------------- JSON View ---------------------------
def raw_json_view():
    st.subheader("🧾 Pipeline JSON (decisions so far)")
    st.json(build_pipeline_json(), expanded=False)

# --------------------------- Main ---------------------------
def main():
    st.title("📈 Economic Simulation — Prototype")
    st.caption("Select among pre-trained models, configure multiple interventions, run the simulation, and export the full pipeline JSON.")

    sidebar_controls()

    # Ensure data and pre-trained repo exist for selection
    if "training_data" not in st.session_state:
        st.session_state["training_data"] = generate_synthetic_population(st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42))
    seed_pretrained_models()

    # Ensure we always have a baseline config in session
    if "config" not in st.session_state:
        st.session_state["config"] = default_config()

    # Tabs (Model Estimation hidden unless flag is turned on)
    if SHOW_MODEL_ESTIMATION:
        tabs = st.tabs(["0) Model estimation", "1) Model selection", "2) Interventions", "3) Run & Results", "4) JSON"])
        with tabs[0]:
            # model_estimation_tab()  # (intentionally left commented; flip flag & add your function body to recover)
            st.info("Model Estimation page is disabled. Set SHOW_MODEL_ESTIMATION=True and restore the body to re-enable.")
        with tabs[1]:
            model_selection_tab()
        with tabs[2]:
            interventions_tab()
        with tabs[3]:
            run_and_visualize()
        with tabs[4]:
            raw_json_view()
    else:
        tabs = st.tabs(["1) Model selection", "2) Interventions", "3) Run & Results", "4) JSON"])
        with tabs[0]:
            model_selection_tab()
        with tabs[1]:
            interventions_tab()
        with tabs[2]:
            run_and_visualize()
        with tabs[3]:
            raw_json_view()

    st.divider()
    st.markdown("Built for demonstration • Replace synthetic data and toy risk with your pipeline.")

if __name__ == "__main__":
    main()
