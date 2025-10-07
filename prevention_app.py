# app.py
# -------------------------------------------------------------
# Economic Simulation Prototype (Streamlit)
#   • Model Estimation & Selection workflow
#   • Multi-intervention config
#   • Pipeline JSON export (decisions so far)
# -------------------------------------------------------------
from __future__ import annotations

import json
import math
import re
import time
from io import StringIO, BytesIO
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ML imports (XGBoost optional)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --------------------------- Page Setup ---------------------------
st.set_page_config(
    page_title="Simulation Engine for Preventative Health Interventions - Prototype",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Utilities ---------------------------
NUMERIC_LIKE = {"int64", "int32", "float64", "float32", "int16", "float16"}

def _safe_read_byteslike(file_obj: BytesIO | StringIO) -> str:
    try:
        if isinstance(file_obj, BytesIO):
            return file_obj.getvalue().decode("utf-8", errors="ignore")
        return file_obj.getvalue()
    except Exception:
        try:
            return file_obj.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def clean_loose_json(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r'\s*"\.{3}"\s*:\s*(".*?"|\{.*?\}|\[.*?\]|true|false|null|-?\d+\.?\d*)\s*,?', "", text, flags=re.DOTALL)
    text = re.sub(r'\s*"\.{3}"\s*,?', "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def load_config_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    cleaned = clean_loose_json(text)
    try:
        cfg = json.loads(cleaned)
        if not isinstance(cfg, dict):
            raise ValueError("Top-level JSON must be an object/dict.")
        return cfg, cleaned
    except Exception as e:
        raise ValueError(f"Could not parse JSON. Details: {e}")


def default_config() -> Dict[str, Any]:
    return {
        "meta": {"name": "Demo Scenario"},
        # unified place where we store decisions for export
        "pipeline": {
            "estimation": {},        # per-target configs as recorded by the user
            "trained": {},           # per-target metrics (no model objects)
            "selection": {},         # chosen model per target
            "interventions": [],     # list of interventions
        },
    }


def is_binary_series(s: pd.Series, threshold_unique: int = 2) -> bool:
    vals = s.dropna().unique()
    return len(vals) <= threshold_unique and set(vals).issubset({0, 1})


# --------------------------- Synthetic Data ---------------------------
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

    df = pd.DataFrame(
        {
            "age": ages.astype(int),
            "log_income": log_income.astype(float),
            "employment_status": employment_status.astype(str),
            "bmi": bmi.astype(float),
        }
    )
    df["prob_hattack_true"] = risk_heart_attack(df)
    df["hattack_ever_w10"] = (rng.random(size=n) < df["prob_hattack_true"]).astype(int)
    df["is_employed"] = (df["employment_status"] == "employed").astype(int)
    return df


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def risk_heart_attack(prob_inputs: pd.DataFrame) -> np.ndarray:
    bmi = prob_inputs["bmi"].to_numpy()
    age = prob_inputs["age"].to_numpy()
    a = -4.2
    b1 = 0.50
    b2 = 0.65
    z = a + b1 * ((bmi - 25.0) / 5.0) + b2 * ((age - 50.0) / 10.0)
    return logistic(z)


# --------------------------- Feature Engineering ---------------------------
def available_variables(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        if c in ("prob_hattack_true",):
            continue
        if is_binary_series(df[c]):
            out[c] = "binary"
        elif str(df[c].dtype) in NUMERIC_LIKE:
            out[c] = "numeric"
        else:
            out[c] = "categorical"
    return out


def compute_stats_table(df: pd.DataFrame, vtypes: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for c, t in vtypes.items():
        s = df[c]
        if t in ("numeric", "binary"):
            vals = s.dropna().to_numpy()
            if vals.size == 0:
                rows.append({"variable": c, "type": t, "min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None, "std": None})
            else:
                rows.append({
                    "variable": c,
                    "type": t,
                    "min": float(np.nanmin(vals)),
                    "p10": float(np.nanpercentile(vals, 10)),
                    "median": float(np.nanmedian(vals)),
                    "mean": float(np.nanmean(vals)),
                    "p90": float(np.nanpercentile(vals, 90)),
                    "max": float(np.nanmax(vals)),
                    "std": float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0),
                })
        else:
            rows.append({"variable": c, "type": t, "min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None, "std": None})
    return pd.DataFrame(rows)


def build_design_matrix(df: pd.DataFrame, target: str, spec: Dict[str, Any]) -> pd.DataFrame:
    base = [b for b in spec.get("base_features", []) if b != target and b in df.columns]
    df_work = pd.DataFrame(index=df.index)
    for b in base:
        df_work[b] = df[b]
    # transforms (numeric only)
    for b in spec.get("log", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            df_work[f"log_{b}"] = np.log1p(np.clip(df[b].to_numpy(), a_min=0, a_max=None))
    for b in spec.get("square", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            df_work[f"{b}_sq"] = np.square(df[b])
    # interactions (numeric)
    for (u, v) in spec.get("interactions", []):
        if u in df.columns and v in df.columns:
            if (str(df[u].dtype) in NUMERIC_LIKE) and (str(df[v].dtype) in NUMERIC_LIKE):
                df_work[f"{u}*{v}"] = df[u].to_numpy() * df[v].to_numpy()
    # one-hot encode categoricals
    X = pd.get_dummies(df_work, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# --------------------------- Interventions ---------------------------
def _range_to_tuple(val: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    try:
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return fallback


def apply_intervention(df: pd.DataFrame, inv: Dict[str, Any]) -> pd.DataFrame:
    """
    Generic numeric intervention:
      - inv["target"]: column name to change
      - inv["type"]: "percentage_decrease" or "absolute_change"
      - inv["amount"]: float
      - inv["filters"]: {"age":[min,max], "bmi":[min,max]}  (optional)
    """
    new_df = df.copy()
    target_var = inv.get("target")
    if (target_var is None) or (target_var not in new_df.columns):
        return new_df

    iv_type = inv.get("type", "percentage_decrease")
    amount = float(inv.get("amount", 0.0))
    filters = inv.get("filters", {}) or {}

    # build mask
    mask = pd.Series(True, index=new_df.index)
    if "age" in new_df.columns and "age" in filters:
        a_min, a_max = _range_to_tuple(filters.get("age"), (new_df["age"].min(), new_df["age"].max()))
        mask &= (new_df["age"] >= a_min) & (new_df["age"] <= a_max)
    if "bmi" in new_df.columns and "bmi" in filters:
        b_min, b_max = _range_to_tuple(filters.get("bmi"), (new_df["bmi"].min(), new_df["bmi"].max()))
        mask &= (new_df["bmi"] >= b_min) & (new_df["bmi"] <= b_max)

    if iv_type == "percentage_decrease":
        new_df.loc[mask, target_var] = new_df.loc[mask, target_var] * (1.0 - amount)
    elif iv_type == "absolute_change":
        new_df.loc[mask, target_var] = new_df.loc[mask, target_var] + amount
    return new_df


# --------------------------- Sidebar ---------------------------
def sidebar_controls(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("⚙️ Scenario & Data")
    st.sidebar.text_input("Scenario name", key="scenario_name", value=st.session_state.get("scenario_name", "Demo Scenario"))

    st.sidebar.caption("Upload a configuration JSON (optional). Pipeline decisions will load if present.")
    uploaded = st.sidebar.file_uploader("Choose a JSON file", type=["json"], key="cfg_uploader")
    if uploaded is not None:
        raw_text = _safe_read_byteslike(uploaded)
        try:
            parsed, cleaned = load_config_from_text(raw_text)
            st.session_state["raw_uploaded_text"] = raw_text
            st.session_state["cleaned_uploaded_text"] = cleaned

            # Merge into current config if keys present
            st.session_state.setdefault("config", cfg)
            pipe = parsed.get("pipeline") or {}
            if pipe:
                st.session_state["config"]["pipeline"] = pipe
                # Restore into session_state for live UI
                st.session_state["pending_estimations"] = list(pipe.get("estimation", {}).values())
                st.session_state["trained_models_meta"] = pipe.get("trained", {})
                st.session_state["chosen_models"] = pipe.get("selection", {})
                st.session_state["interventions"] = pipe.get("interventions", [])
            else:
                # Backward-compat for older structure
                if "intervention_config" in parsed:
                    # convert dict to list
                    inv_list = []
                    for k, inv in (parsed.get("intervention_config") or {}).items():
                        inv_list.append({
                            "target": k,
                            "type": inv.get("type","percentage_decrease"),
                            "amount": inv.get("amount",0.0),
                            "filters": inv.get("target_population",{})
                        })
                    st.session_state.setdefault("interventions", inv_list)

            st.sidebar.success("Configuration loaded.")
        except ValueError as e:
            st.sidebar.error(str(e))

    st.sidebar.divider()
    st.sidebar.caption("Global Simulation Settings")
    pop_n = st.sidebar.number_input("Population size", min_value=2_000, max_value=300_000, value=30_000, step=1_000, key="pop_n")
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1, key="seed")
    montecarlo_sims = st.sidebar.number_input("Monte Carlo simulations", min_value=100, max_value=10_000, value=1_000, step=100, key="mc_sims")

    st.sidebar.divider()
    st.sidebar.caption("Download pipeline configuration")
    st.sidebar.download_button(
        "💾 Download pipeline JSON",
        data=json.dumps(build_pipeline_json(), indent=2),
        file_name="pipeline_config.json",
        mime="application/json",
        use_container_width=True,
    )

    return st.session_state.get("config", cfg)


# --------------------------- Defaults Helper ---------------------------
def apply_defaults_to_estimation():
    # two defaults: bmi (continuous), hattack_ever_w10 (binary)
    defaults = []

    # BMI regression
    algos_reg = ["linear_regression"]
    if XGB_AVAILABLE:
        algos_reg.append("xgboost_regressor")
    defaults.append({
        "target": "bmi",
        "target_type": "continuous",
        "spec": {
            "base_features": ["age", "log_income", "is_employed"],
            "log": ["log_income"],
            "square": ["age"],
            "interactions": [("age", "log_income")],
        },
        "algorithms": algos_reg,
        "train_test_split": {"test_size": 0.2, "random_state": 42, "stratify": None},
    })

    # Heart attack classification
    algos_cls = ["logistic_regression"]
    if XGB_AVAILABLE:
        algos_cls.append("xgboost_classifier")
    defaults.append({
        "target": "hattack_ever_w10",
        "target_type": "binary",
        "spec": {
            "base_features": ["age", "bmi", "is_employed"],
            "log": [],
            "square": ["age"],
            "interactions": [("age", "bmi")],
        },
        "algorithms": algos_cls,
        "train_test_split": {"test_size": 0.2, "random_state": 42, "stratify": "hattack_ever_w10"},
    })

    st.session_state["pending_estimations"] = defaults
    # Also set the UI to the first default
    st.session_state["est_target"] = "bmi"
    st.session_state["est_base_features"] = defaults[0]["spec"]["base_features"]
    st.session_state["est_log_feats"] = defaults[0]["spec"]["log"]
    st.session_state["est_sq_feats"] = defaults[0]["spec"]["square"]
    st.session_state["est_interacts"] = [f"{u}|{v}" for (u,v) in defaults[0]["spec"]["interactions"]]
    st.session_state["est_use_lin"] = True
    st.session_state["est_use_xgb_reg"] = XGB_AVAILABLE
    st.session_state["est_use_logit"] = False
    st.session_state["est_use_xgb_cls"] = False
    st.session_state["est_test_size"] = 0.2
    st.session_state["est_random_state"] = 42
    st.session_state["est_stratify"] = None


def apply_defaults_to_selection():
    # If not trained, attempt to train current pending
    if not st.session_state.get("trained_models"):
        train_all_recorded_models(show_progress=False)
    # Choose default models: XGBoost where available else GLM
    trained = st.session_state.get("trained_models", {})
    chosen = {}
    for tgt, pack in trained.items():
        names = list(pack.get("models", {}).keys())
        pick = None
        for pref in ["XGBoost Regressor", "XGBoost Classifier", "Logistic Regression", "Linear Regression"]:
            if pref in names:
                pick = pref; break
        pick = pick or (names[0] if names else None)
        if pick:
            chosen[tgt] = pick
    st.session_state["chosen_models"] = chosen


def apply_defaults_to_interventions(df: pd.DataFrame):
    st.session_state["interventions"] = [
        {"target": "bmi", "type": "percentage_decrease", "amount": 0.2, "filters": {"age": [40, 60], "bmi": [30, 100]}},
        {"target": "bmi", "type": "percentage_decrease", "amount": 0.1, "filters": {"age": [60, 90], "bmi": [28, 100]}}
    ]


# --------------------------- Training Helpers ---------------------------
def train_all_recorded_models(show_progress: bool = True):
    df = st.session_state["training_data"]
    pend = st.session_state.get("pending_estimations", [])
    if not pend:
        st.warning("No recorded configurations to train.")
        return

    total_jobs = sum(len(p["algorithms"]) for p in pend)
    progress = st.progress(0) if show_progress else None
    done = 0

    st.session_state.setdefault("trained_models", {})
    st.session_state.setdefault("trained_models_meta", {})

    for p in pend:
        target = p["target"]
        ttype = p["target_type"]
        spec = p["spec"]
        algos = p["algorithms"]
        tts = p["train_test_split"]
        strat = tts.get("stratify") if ttype == "binary" else None

        y = df[target]
        X = build_design_matrix(df, target, spec)
        if ttype == "binary":
            y = y.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(tts.get("test_size", 0.2)),
            random_state=int(tts.get("random_state", 42)),
            stratify=y if (ttype=="binary" and strat is not None) else None
        )

        results: Dict[str, Any] = {
            "type": ttype,
            "feature_spec": spec,
            "train_test_split": {"test_size": float(tts.get("test_size", 0.2)), "random_state": int(tts.get("random_state", 42)), "stratify": (strat if ttype=="binary" else None)},
            "models": {},
            "columns": list(X.columns),
        }

        for algo in algos:
            if ttype == "continuous" and algo == "linear_regression":
                est = LinearRegression().fit(X_train, y_train)
                y_pred = est.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                results["models"]["Linear Regression"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": y_pred},
                }
                done += 1

            if ttype == "continuous" and algo == "xgboost_regressor" and XGB_AVAILABLE:
                est = XGBRegressor(
                    n_estimators=250, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=int(tts.get("random_state",42)),
                    reg_lambda=1.0, n_jobs=4
                ).fit(X_train, y_train)
                y_pred = est.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                results["models"]["XGBoost Regressor"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": y_pred},
                }
                done += 1

            if ttype == "binary" and algo == "logistic_regression":
                est = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                prob = est.predict_proba(X_test)[:,1]
                rmse = float(np.sqrt(mean_squared_error(y_test, prob)))
                mae = float(mean_absolute_error(y_test, prob))
                ll = float(log_loss(y_test, prob))
                prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=12, strategy="uniform")
                results["models"]["Logistic Regression"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "LogLoss": ll},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": prob},
                    "calibration": {"prob_true": prob_true, "prob_pred": prob_pred},
                }
                done += 1

            if ttype == "binary" and algo == "xgboost_classifier" and XGB_AVAILABLE:
                est = XGBClassifier(
                    n_estimators=350, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=int(tts.get("random_state",42)),
                    reg_lambda=1.0, n_jobs=4, eval_metric="logloss"
                ).fit(X_train, y_train)
                prob = est.predict_proba(X_test)[:,1]
                rmse = float(np.sqrt(mean_squared_error(y_test, prob)))
                mae = float(mean_absolute_error(y_test, prob))
                ll = float(log_loss(y_test, prob))
                prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=12, strategy="uniform")
                results["models"]["XGBoost Classifier"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "LogLoss": ll},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": prob},
                    "calibration": {"prob_true": prob_true, "prob_pred": prob_pred},
                }
                done += 1

            if show_progress:
                pct = int(done / max(total_jobs,1) * 100)
                progress.progress(min(pct, 100))
                time.sleep(0.08)  # small delay to make the bar visible

        # Store trained pack
        st.session_state["trained_models"][target] = results

        # Save a serializer-friendly meta summary for export
        meta = {
            "type": results["type"],
            "feature_spec": results["feature_spec"],
            "train_test_split": results["train_test_split"],
            "columns": results["columns"],
            "models": {name: {"metrics": info["metrics"]} for name, info in results["models"].items()},
        }
        st.session_state["trained_models_meta"][target] = meta


# --------------------------- Model Estimation Tab ---------------------------
def model_estimation_tab():
    top_cols = st.columns([1, 4, 1])
    with top_cols[1]:
        if st.button("Use default configuration", key="btn_default_estimation"):
            apply_defaults_to_estimation()
            st.success("Default estimation configurations added and UI pre-filled for BMI.")

    st.subheader("🧮 Model estimation")

    # Prepare training data
    if "training_data" not in st.session_state:
        st.info("Generating a synthetic training dataset based on your sidebar settings.")
        st.session_state["training_data"] = generate_synthetic_population(
            st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42)
        )
    df = st.session_state["training_data"]
    vtypes = available_variables(df)

    # Available variables with stats
    st.markdown("**Available variables in dataset**")
    stats_df = compute_stats_table(df, vtypes)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with st.expander("Preview first 20 rows", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    # Target & feature engineering UI with stable keys
    st.markdown("**Choose a target variable to estimate**")
    candidate_targets = [c for c,t in vtypes.items() if t in ("numeric", "binary")]
    default_target = "bmi" if "bmi" in candidate_targets else candidate_targets[0]
    target = st.selectbox("Target variable", options=candidate_targets, index=candidate_targets.index(default_target), key="est_target")
    target_type = "binary" if vtypes.get(target) == "binary" else "continuous"
    st.caption(f"Detected target type: **{target_type}**")

    st.markdown("**Feature engineering**")
    base_candidates = [c for c in vtypes.keys() if c != target]
    base_default = [c for c in ["age", "log_income", "is_employed", "bmi"] if c in base_candidates]
    base_features = st.multiselect("Base features", options=base_candidates, default=st.session_state.get("est_base_features", base_default), key="est_base_features")

    numeric_bases = [c for c in base_features if str(df[c].dtype) in NUMERIC_LIKE]
    col1, col2 = st.columns(2)
    with col1:
        log_feats = st.multiselect("Apply log(1+x) to", options=numeric_bases, default=st.session_state.get("est_log_feats", []), key="est_log_feats")
    with col2:
        sq_feats = st.multiselect("Apply square to", options=numeric_bases, default=st.session_state.get("est_sq_feats", []), key="est_sq_feats")

    st.markdown("**Interactions (pairwise among selected numeric features)**")
    seed_defaults = ["age", "bmi", "log_income"]
    default_candidates = [v for v in seed_defaults if v in numeric_bases]
    default_interacts = [f"{u}|{v}" for i,u in enumerate(default_candidates) for v in default_candidates[i+1:i+2]]
    # string encode pairs to keep a stable widget key
    all_pairs = [f"{u}|{v}" for i,u in enumerate(numeric_bases) for v in numeric_bases[i+1:]]
    selected_pairs = st.multiselect(
        "Select variables to fully interact (all pairwise products)",
        options=all_pairs,
        default=st.session_state.get("est_interacts", default_interacts),
        key="est_interacts",
    )
    interactions = []
    for pair in selected_pairs:
        if "|" in pair:
            u,v = pair.split("|",1)
            if u in numeric_bases and v in numeric_bases:
                interactions.append((u,v))

    st.markdown("**Choose algorithm(s)**")
    algos = []
    if target_type == "continuous":
        c1, c2 = st.columns(2)
        with c1:
            use_lin = st.checkbox("Linear Regression", value=st.session_state.get("est_use_lin", True), key="est_use_lin")
        with c2:
            if XGB_AVAILABLE:
                use_xgb = st.checkbox("XGBoost (Regressor)", value=st.session_state.get("est_use_xgb_reg", True), key="est_use_xgb_reg")
            else:
                st.checkbox("XGBoost (Regressor)", value=False, disabled=True)
                use_xgb = False
        if use_lin: algos.append("linear_regression")
        if use_xgb: algos.append("xgboost_regressor")
    else:
        c1, c2 = st.columns(2)
        with c1:
            use_logit = st.checkbox("Logistic Regression", value=st.session_state.get("est_use_logit", True), key="est_use_logit")
        with c2:
            if XGB_AVAILABLE:
                use_xgb = st.checkbox("XGBoost (Classifier)", value=st.session_state.get("est_use_xgb_cls", True), key="est_use_xgb_cls")
            else:
                st.checkbox("XGBoost (Classifier)", value=False, disabled=True)
                use_xgb = False
        if use_logit: algos.append("logistic_regression")
        if use_xgb: algos.append("xgboost_classifier")

    st.markdown("**Train / Test split**")
    colA, colB, colC = st.columns(3)
    with colA:
        test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=st.session_state.get("est_test_size", 0.2), step=0.05, key="est_test_size")
    with colB:
        random_state = st.number_input("Random state", value=st.session_state.get("est_random_state", 42), step=1, key="est_random_state")
    with colC:
        stratify_candidates = [None] + base_candidates
        stratify_opt = st.selectbox("Stratify (binary only)", options=stratify_candidates, index=0, key="est_stratify")
        if target_type != "binary":
            st.caption("Stratify is only used for classification.")

    spec = {"base_features": base_features, "log": log_feats, "square": sq_feats, "interactions": interactions}

    st.divider()
    # RECORD decision (do not train yet)
    colR1, colR2 = st.columns([1,2])
    with colR1:
        record = st.button("📌 Add/Update configuration for this target", type="primary", use_container_width=True, key="btn_record_config")
    with colR2:
        st.caption("Record all your choices below first; then train everything together.")

    if record:
        st.session_state.setdefault("pending_estimations", [])
        entry = {
            "target": target,
            "target_type": target_type,
            "spec": spec,
            "algorithms": algos,
            "train_test_split": {"test_size": float(test_size), "random_state": int(random_state), "stratify": stratify_opt if target_type=="binary" else None},
        }
        # Replace if target already exists
        pending = st.session_state["pending_estimations"]
        found = False
        for i, p in enumerate(pending):
            if p["target"] == target:
                pending[i] = entry; found = True; break
        if not found:
            pending.append(entry)
        st.success(f"Recorded configuration for `{target}`.")

    # Show all recorded decisions
    pend = st.session_state.get("pending_estimations", [])
    if pend:
        st.markdown("### Selected configurations")
        rows = []
        for p in pend:
            rows.append({
                "target": p["target"],
                "type": p["target_type"],
                "base_features": ", ".join(p["spec"]["base_features"]),
                "log": ", ".join(p["spec"]["log"]),
                "square": ", ".join(p["spec"]["square"]),
                "interactions": ", ".join([f"{u}*{v}" for (u,v) in p["spec"]["interactions"]]),
                "algorithms": ", ".join(p["algorithms"]),
                "test_size": p["train_test_split"]["test_size"],
                "random_state": p["train_test_split"]["random_state"],
                "stratify": p["train_test_split"]["stratify"],
            })
        rec_df = pd.DataFrame(rows)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        st.markdown("Train all selected configurations at once:")
        if st.button("🧪 Train selected models", type="primary", use_container_width=True, key="btn_train_all"):
            with st.spinner("Training models..."):
                train_all_recorded_models(show_progress=True)
            st.success("Training complete. Proceed to **Model selection** to choose which models to use.")

    # Update config export mirror
    pipe = st.session_state.get("config", default_config())["pipeline"]
    pipe["estimation"] = {p["target"]: p for p in st.session_state.get("pending_estimations", [])}
    pipe["trained"] = st.session_state.get("trained_models_meta", {})
    st.session_state["config"]["pipeline"] = pipe


# --------------------------- Model Selection Tab ---------------------------
def model_selection_tab():
    top_cols = st.columns([1, 4, 1])
    with top_cols[1]:
        if st.button("Use default configuration", key="btn_default_selection"):
            apply_defaults_to_selection()
            st.success("Default selections applied (prefers XGBoost where available).")

    st.subheader("🧩 Model selection")
    trained = st.session_state.get("trained_models", {})
    if not trained:
        st.info("No models trained yet. Use **Model estimation** to record and train first.")
        return

    variables = list(trained.keys())
    var = st.selectbox("Select a variable to inspect & choose model", options=variables, index=0, key="sel_target")
    pack = trained[var]
    mtype = pack["type"]
    models_dict = pack["models"]

    st.markdown("**Error metrics comparison**")
    rows = []
    for name, info in models_dict.items():
        met = info["metrics"]
        rows.append({"model": name, **met})
    met_df = pd.DataFrame(rows)

    if mtype == "continuous":
        col1, col2 = st.columns(2)
        with col1:
            rmse_chart = alt.Chart(met_df).mark_bar().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("RMSE:Q", title="RMSE"),
                tooltip=["model", alt.Tooltip("RMSE:Q", format=".4f")]
            ).properties(height=250)
            st.altair_chart(rmse_chart, use_container_width=True)
        with col2:
            mae_chart = alt.Chart(met_df).mark_bar().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("MAE:Q", title="MAE"),
                tooltip=["model", alt.Tooltip("MAE:Q", format=".4f")]
            ).properties(height=250)
            st.altair_chart(mae_chart, use_container_width=True)

        st.markdown("**Predicted vs Actual (test set)**")
        for name, info in models_dict.items():
            d = pd.DataFrame({"y_true": info["pred_true"]["y_true"], "y_pred": info["pred_true"]["y_pred"]})
            chart = alt.Chart(d).mark_circle(opacity=0.4).encode(
                x=alt.X("y_true:Q", title="Actual"),
                y=alt.Y("y_pred:Q", title="Predicted"),
                tooltip=[alt.Tooltip("y_true:Q", format=".2f"), alt.Tooltip("y_pred:Q", format=".2f")]
            ).properties(height=250, title=f"{name}")
            st.altair_chart(chart, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            rmse_chart = alt.Chart(met_df).mark_bar().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("RMSE:Q", title="RMSE (prob vs 0/1)"),
                tooltip=["model", alt.Tooltip("RMSE:Q", format=".4f")]
            ).properties(height=250)
            st.altair_chart(rmse_chart, use_container_width=True)
        with col2:
            mae_chart = alt.Chart(met_df).mark_bar().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("MAE:Q", title="MAE (prob vs 0/1)"),
                tooltip=["model", alt.Tooltip("MAE:Q", format=".4f")]
            ).properties(height=250)
            st.altair_chart(mae_chart, use_container_width=True)

        st.markdown("**Calibration curves**")
        layers = []
        for name, info in models_dict.items():
            cal = info.get("calibration")
            if cal is None:
                continue
            d = pd.DataFrame({"prob_pred": cal["prob_pred"], "prob_true": cal["prob_true"], "model": name})
            layers.append(d)
        if layers:
            dd = pd.concat(layers, ignore_index=True)
            diag = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line().encode(x="x:Q", y="y:Q")
            chart = diag + alt.Chart(dd).mark_line(point=True).encode(
                x=alt.X("prob_pred:Q", title="Predicted probability (binned)"),
                y=alt.Y("prob_true:Q", title="Empirical frequency"),
                color=alt.Color("model:N", title="Model")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

    # Selection UI + model detail table
    st.divider()
    st.markdown("**Select model for simulation**")
    model_names = list(models_dict.keys())
    chosen = st.radio("Choose one", options=model_names, index=0, key="sel_chosen_model")
    st.session_state.setdefault("chosen_models", {})
    st.session_state["chosen_models"][var] = chosen
    st.success(f"Selected: **{chosen}** for `{var}`")

    # Details: coefficients or feature importances
    st.markdown("**Model details**")
    info = models_dict[chosen]
    cols = pack["columns"]
    if chosen in ("Linear Regression", "Logistic Regression"):
        est = info["estimator"]
        coef = getattr(est, "coef_", None)
        intercept = getattr(est, "intercept_", None)
        if coef is not None:
            if coef.ndim > 1:
                coef = coef.ravel()
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
        # XGBoost importances
        est = info["estimator"]
        try:
            fi = est.feature_importances_
            df_imp = pd.DataFrame({"feature": cols, "importance": fi}).sort_values("importance", ascending=False)
            st.dataframe(df_imp, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Feature importances not available for this estimator.")

    # Decisions summary
    st.markdown("### Selections recorded")
    all_sel = st.session_state.get("chosen_models", {})
    if all_sel:
        rows = []
        for tgt, nm in all_sel.items():
            rows.append({"target": tgt, "chosen_model": nm})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Update export mirror
    pipe = st.session_state.get("config", default_config())["pipeline"]
    pipe["selection"] = st.session_state.get("chosen_models", {})
    st.session_state["config"]["pipeline"] = pipe


# --------------------------- Interventions Tab ---------------------------
def interventions_tab():
    top_cols = st.columns([1, 4, 1])
    with top_cols[1]:
        if st.button("Use default configuration", key="btn_default_interventions"):
            # build defaults using current data ranges
            df = st.session_state.get("training_data")
            if df is None:
                df = generate_synthetic_population(st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42))
                st.session_state["training_data"] = df
            apply_defaults_to_interventions(df)
            st.success("Added two default BMI interventions.")

    st.subheader("🧪 Interventions")
    st.caption("Add as many interventions as you like. For demo, filters support age and BMI.")
    st.session_state.setdefault("interventions", [])
    df = st.session_state.get("training_data")

    vtypes = available_variables(df) if df is not None else {}
    numeric_vars = [c for c,t in vtypes.items() if t in ("numeric","binary")]

    # List existing interventions
    to_delete = []
    for idx, inv in enumerate(st.session_state["interventions"]):
        with st.container(border=True):
            st.markdown(f"**Intervention {idx+1}**")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                inv["target"] = st.selectbox("Target variable", options=numeric_vars, index=numeric_vars.index(inv.get("target","bmi")) if inv.get("target","bmi") in numeric_vars else 0, key=f"inv_target_{idx}")
            with col2:
                inv["type"] = st.selectbox("Type", options=["percentage_decrease","absolute_change"], index=["percentage_decrease","absolute_change"].index(inv.get("type","percentage_decrease")), key=f"inv_type_{idx}")
            with col3:
                inv["amount"] = st.number_input("Amount", value=float(inv.get("amount", 0.1)), step=0.01, key=f"inv_amount_{idx}")
            st.markdown("**Filters**")
            f = inv.get("filters", {})
            a_min, a_max = _range_to_tuple(f.get("age"), (18, 90))
            b_min, b_max = _range_to_tuple(f.get("bmi"), (15, 60))
            age_min, age_max = st.slider("Age range", min_value=0, max_value=100, value=(int(a_min), int(a_max)), key=f"inv_age_{idx}")
            bmi_min, bmi_max = st.slider("BMI range", min_value=10, max_value=100, value=(int(b_min), int(b_max)), key=f"inv_bmi_{idx}")
            inv["filters"] = {"age": [age_min, age_max], "bmi": [bmi_min, bmi_max]}
            if st.button("Remove", key=f"inv_remove_{idx}"):
                to_delete.append(idx)
        st.write("")

    # delete requested
    for i in sorted(to_delete, reverse=True):
        del st.session_state["interventions"][i]

    # Add new
    if st.button("➕ Add intervention", key="btn_add_intervention"):
        st.session_state["interventions"].append({"target": "bmi", "type": "percentage_decrease", "amount": 0.1, "filters": {"age":[30,60], "bmi":[25,60]}})

    # Update export mirror
    pipe = st.session_state.get("config", default_config())["pipeline"]
    pipe["interventions"] = st.session_state.get("interventions", [])
    st.session_state["config"]["pipeline"] = pipe


# --------------------------- Run Simulation ---------------------------
def run_and_visualize(cfg: Dict[str, Any]) -> None:
    st.subheader("🚀 Run Simulation")

    pop_n = int(st.session_state.get("pop_n", 30_000))
    seed = int(st.session_state.get("seed", 42))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Generate a synthetic baseline population, apply your interventions, and use your selected models.")
        go = st.button("Run Simulation", type="primary", use_container_width=True)
    with c2:
        st.caption("Export results or share a quick snapshot.")
        export_name = st.text_input("Export name", value="demo_run")

    if not go:
        st.info("Train & select models, configure interventions, then click **Run Simulation**.")
        return

    # baseline data
    df_base = generate_synthetic_population(pop_n, seed=seed)

    # Predict BMI if a BMI model was selected
    chosen = st.session_state.get("chosen_models", {})
    trained = st.session_state.get("trained_models", {})

    if "bmi" in chosen and "bmi" in trained:
        pack = trained["bmi"]
        model_name = chosen["bmi"]
        est = pack["models"][model_name]["estimator"]
        Xb = build_design_matrix(df_base, "bmi", pack["feature_spec"])
        for col in pack["columns"]:
            if col not in Xb.columns: Xb[col] = 0.0
        Xb = Xb[pack["columns"]]
        df_base["bmi"] = est.predict(Xb)

    # Apply all interventions
    df_post = df_base.copy()
    for inv in st.session_state.get("interventions", []):
        df_post = apply_intervention(df_post, inv)

    # Predict heart-attack probabilities if selected, otherwise fallback
    if "hattack_ever_w10" in chosen and "hattack_ever_w10" in trained:
        pack_h = trained["hattack_ever_w10"]
        model_name_h = chosen["hattack_ever_w10"]
        est_h = pack_h["models"][model_name_h]["estimator"]
        # Baseline
        Xh0 = build_design_matrix(df_base, "hattack_ever_w10", pack_h["feature_spec"])
        for col in pack_h["columns"]:
            if col not in Xh0.columns: Xh0[col] = 0.0
        Xh0 = Xh0[pack_h["columns"]]
        if pack_h["type"] == "binary":
            if hasattr(est_h, "predict_proba"):
                df_base["prob_hattack"] = est_h.predict_proba(Xh0)[:,1]
            else:
                df_base["prob_hattack"] = est_h.predict(Xh0)
        else:
            df_base["prob_hattack"] = np.clip(est_h.predict(Xh0), 0, 1)

        # Post
        Xh1 = build_design_matrix(df_post, "hattack_ever_w10", pack_h["feature_spec"])
        for col in pack_h["columns"]:
            if col not in Xh1.columns: Xh1[col] = 0.0
        Xh1 = Xh1[pack_h["columns"]]
        if pack_h["type"] == "binary":
            if hasattr(est_h, "predict_proba"):
                df_post["prob_hattack"] = est_h.predict_proba(Xh1)[:,1]
            else:
                df_post["prob_hattack"] = est_h.predict(Xh1)
        else:
            df_post["prob_hattack"] = np.clip(est_h.predict(Xh1), 0, 1)
    else:
        df_base["prob_hattack"] = risk_heart_attack(df_base)
        df_post["prob_hattack"] = risk_heart_attack(df_post)

    rng = np.random.default_rng(seed + 123)
    df_base["hattack_event"] = (rng.random(size=len(df_base)) < df_base["prob_hattack"]).astype(int)
    df_post["hattack_event"] = (rng.random(size=len(df_post)) < df_post["prob_hattack"]).astype(int)

    # summaries
    s_base = {
        "n": len(df_base),
        "bmi_mean": float(df_base["bmi"].mean()),
    }
    s_post = {
        "n": len(df_post),
        "bmi_mean": float(df_post["bmi"].mean()),
    }
    prev_base = float(df_base["hattack_event"].mean())
    prev_post = float(df_post["hattack_event"].mean())
    delta_prev_abs = prev_post - prev_base
    delta_prev_rel = (prev_post / prev_base - 1.0) if prev_base > 0 else math.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Population", f"{len(df_base):,}")
    m2.metric("Mean BMI (baseline)", f"{s_base['bmi_mean']:.2f}", delta=f"{(s_post['bmi_mean']-s_base['bmi_mean']):+.2f}")
    m3.metric("Heart attack prevalence (baseline)", f"{prev_base*100:.2f}%",
              delta=f"{delta_prev_abs*100:+.2f}%")
    m4.metric("Relative change in prevalence", f"{delta_prev_rel*100:+.2f}%")

    st.divider()

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("**BMI distribution (baseline vs. post-intervention)**")
        b0 = df_base[["bmi"]].copy(); b0["state"] = "Baseline"
        b1 = df_post[["bmi"]].copy(); b1["state"] = "Post"
        tidy_bmi = pd.concat([b0, b1], ignore_index=True)
        bins = alt.Bin(maxbins=40)
        chart_bmi = alt.Chart(tidy_bmi).transform_bin(
            ["bmi_bin"], field="bmi", bin=bins
        ).mark_bar(opacity=0.6).encode(
            x=alt.X("bmi_bin:Q", title="BMI (binned)"),
            y=alt.Y("count()", stack=None, title="Count"),
            color=alt.Color("state:N", title="Scenario"),
            tooltip=["state", "count()"]
        ).properties(height=300)
        st.altair_chart(chart_bmi, use_container_width=True)

    with col_right:
        st.markdown("**Prevalence of heart attack (baseline vs. post)**")
        prev_df = pd.DataFrame({
            "scenario": ["Baseline", "Post"],
            "prevalence": [prev_base * 100.0, prev_post * 100.0],
        })
        chart_prev = alt.Chart(prev_df).mark_bar().encode(
            x=alt.X("scenario:N", title=None),
            y=alt.Y("prevalence:Q", title="Prevalence (%)"),
            tooltip=["scenario", alt.Tooltip("prevalence:Q", format=".2f")]
        ).properties(height=300)
        st.altair_chart(chart_prev, use_container_width=True)

    st.divider()

    st.markdown("**Breakdown by age group**")
    def _breakdown(df: pd.DataFrame) -> pd.DataFrame:
        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ["18–29", "30–39", "40–49", "50–59", "60–69", "70–89"]
        g = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
        out = (
            df.assign(age_group=g).groupby("age_group", as_index=False)
            .agg(n=("age", "size"), bmi_mean=("bmi", "mean"), prev=("hattack_event", "mean"))
        )
        out["prev"] = out["prev"] * 100.0
        return out

    br_base = _breakdown(df_base)
    br_post = _breakdown(df_post)
    merged = br_base.merge(br_post, on="age_group", suffixes=("_base", "_post"))

    c = alt.Chart(merged).transform_fold(
        ["prev_base", "prev_post"], as_=["scenario", "value"],
    ).mark_line(point=True).encode(
        x=alt.X("age_group:N", title="Age group"),
        y=alt.Y("value:Q", title="Prevalence (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("scenario:N", title=""),
        tooltip=["age_group:N", alt.Tooltip("value:Q", format=".2f"), "scenario:N"],
    ).properties(height=300)
    st.altair_chart(c, use_container_width=True)

    st.caption("Note: Models and outputs here are illustrative.")

    # Option to download results
    # col_a, col_b = st.columns(2)
    # with col_a:
    #     st.download_button(
    #         "⬇️ Download baseline microdata (CSV)",
    #         data=df_base.to_csv(index=False).encode("utf-8"),
    #         file_name=f"{export_name}_baseline.csv",
    #         mime="text/csv",
    #         use_container_width=True,
    #     )
    # with col_b:
    #     st.download_button(
    #         "⬇️ Download post-intervention microdata (CSV)",
    #         data=df_post.to_csv(index=False).encode("utf-8"),
    #         file_name=f"{export_name}_post.csv",
    #         mime="text/csv",
    #         use_container_width=True,
    #     )


# --------------------------- JSON View ---------------------------
def raw_json_view(cfg: Dict[str, Any]) -> None:
    st.subheader("🧾 JSON configuration")
    st.json(build_pipeline_json(), expanded=False)
    if "raw_uploaded_text" in st.session_state:
        with st.expander("Original uploaded text", expanded=False):
            st.code(st.session_state["raw_uploaded_text"], language="json")
    if "cleaned_uploaded_text" in st.session_state:
        with st.expander("Cleaned JSON used for parsing", expanded=False):
            st.code(st.session_state["cleaned_uploaded_text"], language="json")


def build_pipeline_json() -> Dict[str, Any]:
    scenario_name = st.session_state.get("scenario_name", "Demo Scenario")
    pipe = st.session_state.get("config", default_config()).get("pipeline", {})
    # Pull live mirrors from session
    estimation = {p["target"]: p for p in st.session_state.get("pending_estimations", [])}
    trained = st.session_state.get("trained_models_meta", {})
    selection = st.session_state.get("chosen_models", {})
    interventions = st.session_state.get("interventions", [])

    # Optionally include data summary for context (computed from synthetic training data)
    df = st.session_state.get("training_data")
    data_summary = None
    if df is not None:
        vtypes = available_variables(df)
        data_summary = compute_stats_table(df, vtypes).to_dict(orient="records")

    return {
        "meta": {"name": scenario_name},
        "pipeline": {
            "estimation": estimation,
            "trained": trained,
            "selection": selection,
            "interventions": interventions,
        },
        "data_summary": data_summary,
    }


# --------------------------- App Entry ---------------------------
def main():
    st.title("📈 Simulation Engine for Preventative Health Interventions - 🤖 Prototype")
    st.caption(
        "Estimate models from data, compare candidate models, select the ones to use, configure multiple interventions, "
        "simulate outcomes, analyse and export results."
    )

    cfg = sidebar_controls(default_config())

    # Ensure training data exists early for defaults
    if "training_data" not in st.session_state:
        st.session_state["training_data"] = generate_synthetic_population(
            st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42)
        )

    tabs = st.tabs(["0) Model estimation", "1) Model selection", "2) Interventions", "3) Run & Results", "4) JSON config"])
    with tabs[0]:
        model_estimation_tab()
    with tabs[1]:
        model_selection_tab()
    with tabs[2]:
        interventions_tab()
    with tabs[3]:
        run_and_visualize(cfg)
    with tabs[4]:
        raw_json_view(cfg)

    st.divider()
    st.markdown("Built for demonstration purposes only.")

if __name__ == "__main__":
    if "config" not in st.session_state:
        st.session_state["config"] = default_config()
    main()
