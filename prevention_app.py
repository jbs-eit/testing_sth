# app.py
# -------------------------------------------------------------
# Economic Simulation Prototype (Streamlit) — with Model Estimation & Selection
# -------------------------------------------------------------
# How to run:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py
# -------------------------------------------------------------
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from io import StringIO, BytesIO
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    brier_score_loss, roc_auc_score, log_loss, accuracy_score
)

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# --------------------------- Page Setup ---------------------------
st.set_page_config(
    page_title="Economic Simulation Prototype",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Utilities ---------------------------
def _safe_read_byteslike(file_obj: BytesIO | StringIO) -> str:
    """Return text from a Streamlit uploaded file (BytesIO or StringIO)."""
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
    """Make 'loose' JSON loadable: strip comments, ellipses, trailing commas."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)        # // comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)        # /* */ comments
    text = re.sub(r'\s*"\.{3}"\s*:\s*(".*?"|\{.*?\}|\[.*?\]|true|false|null|-?\d+\.?\d*)\s*,?', "", text, flags=re.DOTALL)
    text = re.sub(r'\s*"\.{3}"\s*,?', "", text)                   # "..." entries in arrays
    text = re.sub(r",\s*([}\]])", r"\1", text)                    # dangling commas
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
    """Minimal default config. The new workflow ignores specific structure here by design."""
    return {
        "data": "ukhls",
        "model_config": {},
        "intervention_config": {
            "bmi": {
                "type": "percentage_decrease",
                "amount": 0.2,
                "target_population": {"age": [40, 60], "bmi": [30, 100]},
            }
        },
    }


def _range_to_tuple(val: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    try:
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return fallback


# --------------------------- Synthetic Data + Model Stubs ---------------------------
def generate_synthetic_population(n: int, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic micro-population (illustrative only)."""
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 90, size=n)

    base_log_income = rng.normal(loc=10.5, scale=0.6, size=n)  # ~exp -> median ~$36k
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
    return df


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def risk_heart_attack(prob_inputs: pd.DataFrame) -> np.ndarray:
    """
    Toy probability of ever having a heart attack (illustrative, not clinical).
    logit(p) = a + b1*((bmi-25)/5) + b2*((age-50)/10)
    """
    bmi = prob_inputs["bmi"].to_numpy()
    age = prob_inputs["age"].to_numpy()
    a, b1, b2 = -4.2, 0.50, 0.65
    z = a + b1 * ((bmi - 25.0) / 5.0) + b2 * ((age - 50.0) / 10.0)
    return logistic(z)


# --------------------------- Feature engineering helpers ---------------------------
def get_variable_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return mapping of var -> {'binary','continuous','categorical'}"""
    out = {}
    for c in df.columns:
        ser = df[c]
        if pd.api.types.is_bool_dtype(ser) or (pd.api.types.is_integer_dtype(ser) and ser.dropna().isin([0,1]).all()):
            out[c] = "binary"
        elif pd.api.types.is_numeric_dtype(ser):
            out[c] = "continuous"
        else:
            out[c] = "categorical"
    return out


def safe_log(x: pd.Series) -> pd.Series:
    return np.log(np.clip(pd.to_numeric(x, errors="coerce"), 1e-12, None))


def build_feature_matrix(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    log_vars: List[str],
    square_vars: List[str],
    interactions: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Construct a design matrix with:
      - numeric features as-is
      - categorical features one-hot encoded (drop_first=True)
      - optional log/square expansions (numeric only)
      - optional pairwise interactions for numeric vars
    Returns X, y, and the final column order.
    """
    X_parts = []
    var_types = get_variable_types(df)

    # Base features
    for v in features:
        if v not in df.columns:
            continue
        if var_types.get(v) == "categorical":
            dummies = pd.get_dummies(df[v], prefix=v, drop_first=True)
            X_parts.append(dummies.astype(float))
        else:
            X_parts.append(pd.to_numeric(df[v], errors="coerce").to_frame(v))

        # log
        if v in log_vars and var_types.get(v) != "categorical":
            X_parts.append(safe_log(df[v]).rename(f"log_{v}").to_frame())

        # square
        if v in square_vars and var_types.get(v) != "categorical":
            X_parts.append((pd.to_numeric(df[v], errors="coerce")**2).rename(f"{v}_sq").to_frame())

    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=df.index)

    # Interactions (only numeric-numeric to keep this demo manageable)
    for a, b in interactions:
        if a in df.columns and b in df.columns:
            if pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
                X[f"{a}_x_{b}"] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    y = df[target] if target in df.columns else pd.Series(index=df.index, dtype=float)

    # Drop rows with any NA in X or y
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y, list(X.columns)


def ensure_same_columns(X_new: pd.DataFrame, cols_template: List[str]) -> pd.DataFrame:
    """Add any missing columns (0) and drop extras; then align order to template."""
    for c in cols_template:
        if c not in X_new.columns:
            X_new[c] = 0.0
    extra = [c for c in X_new.columns if c not in cols_template]
    if extra:
        X_new = X_new.drop(columns=extra)
    return X_new[cols_template]


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return a binned calibration table (mean predicted vs. observed fraction)."""
    df = pd.DataFrame({"y": y_true.astype(int), "p": y_prob})
    df = df.sort_values("p")
    bins = np.array_split(df, n_bins)
    rows = []
    for i, b in enumerate(bins, 1):
        if len(b) == 0:
            continue
        rows.append({
            "bin": i,
            "mean_pred": b["p"].mean(),
            "frac_pos": b["y"].mean(),
            "n": len(b),
        })
    return pd.DataFrame(rows)


# --------------------------- Session State ---------------------------
if "config" not in st.session_state:
    st.session_state["config"] = default_config()
if "trained_models" not in st.session_state:
    # structure: {target: {model_key: {...}}}
    st.session_state["trained_models"] = {}
if "selected_models" not in st.session_state:
    # structure: {target: "model_key"}
    st.session_state["selected_models"] = {}
if "estimation_specs" not in st.session_state:
    # structure: {target: {...spec...}}
    st.session_state["estimation_specs"] = {}
if "raw_uploaded_text" not in st.session_state:
    st.session_state["raw_uploaded_text"] = None
if "cleaned_uploaded_text" not in st.session_state:
    st.session_state["cleaned_uploaded_text"] = None


# --------------------------- Sidebar ---------------------------
def sidebar_config_controls(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("⚙️ Scenario & Configuration")

    st.sidebar.text_input("Scenario name", key="scenario_name", value=st.session_state.get("scenario_name", "Demo Scenario"))

    st.sidebar.caption("Upload a configuration JSON (optional)")
    uploaded = st.sidebar.file_uploader("Choose a JSON file", type=["json"], key="cfg_uploader")
    if uploaded is not None:
        raw_text = _safe_read_byteslike(uploaded)
        try:
            parsed, cleaned = load_config_from_text(raw_text)
            st.session_state["raw_uploaded_text"] = raw_text
            st.session_state["cleaned_uploaded_text"] = cleaned
            st.session_state["config"] = parsed
            st.sidebar.success("Configuration loaded.")
        except ValueError as e:
            st.sidebar.error(str(e))

    st.sidebar.download_button(
        label="💾 Download current config",
        data=json.dumps(st.session_state.get("config", cfg), indent=2),
        file_name="scenario_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.sidebar.divider()
    st.sidebar.caption("Simulation Settings")
    pop_n = st.sidebar.number_input("Population size (synthetic)", min_value=5_000, max_value=1_000_000, value=60_000, step=5_000)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

    st.session_state["pop_n"] = int(pop_n)
    st.session_state["seed"] = int(seed)

    return st.session_state.get("config", cfg)


# --------------------------- UI: Model Estimation ---------------------------
def model_estimation_tab():
    st.subheader("🧮 Model estimation")

    # Build / refresh an in-memory dataset for estimation
    pop_n = int(st.session_state.get("pop_n", 60_000))
    seed = int(st.session_state.get("seed", 42))
    df = generate_synthetic_population(pop_n, seed=seed)
    # Add a binary outcome for illustration
    df["prob_hattack"] = risk_heart_attack(df)
    rng = np.random.default_rng(seed + 10)
    df["hattack_event"] = (rng.random(size=len(df)) < df["prob_hattack"]).astype(int)

    var_types = get_variable_types(df)
    st.markdown("**Available variables in dataset**")
    with st.container(border=True):
        st.dataframe(pd.DataFrame({"variable": list(var_types.keys()), "type": list(var_types.values())}).sort_values("variable"), use_container_width=True, hide_index=True)

    st.markdown("**Choose a target variable to estimate**")
    # We allow continuous (regression) and binary (classification) targets
    target_candidates = [v for v, t in var_types.items() if t in ("continuous", "binary")]
    if not target_candidates:
        st.warning("No suitable target variables found.")
        return
    target = st.selectbox("Target variable", target_candidates, index=0)

    task = "classification" if var_types[target] == "binary" else "regression"
    st.caption(f"Detected task: **{task}**")

    # Independent variables
    candidate_X = [v for v in var_types.keys() if v != target and v not in ("prob_hattack",)]
    default_feats = ["age", "log_income"] if target != "bmi" else ["age", "log_income"]
    features = st.multiselect("Independent variables", candidate_X, default=[f for f in default_feats if f in candidate_X])

    # Transformations
    numeric_feats = [f for f in features if var_types[f] == "continuous"]
    with st.expander("Feature engineering", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            log_vars = st.multiselect("Apply log() to", numeric_feats, default=[v for v in numeric_feats if v.startswith("log_")])
        with col2:
            square_vars = st.multiselect("Add squared term", numeric_feats, default=[])
        with col3:
            st.markdown("**Interactions**")
            inter_a = st.selectbox("Var A", ["—"] + numeric_feats, index=0, key="inter_a")
            inter_b = st.selectbox("Var B", ["—"] + numeric_feats, index=0, key="inter_b")
            if "interactions" not in st.session_state:
                st.session_state["interactions"] = []
            if st.button("➕ Add interaction (A × B)") and inter_a != "—" and inter_b != "—" and inter_a != inter_b:
                st.session_state["interactions"].append((inter_a, inter_b))
            # Display current interactions for this target (session-wide)
            if st.session_state["interactions"]:
                st.write(pd.DataFrame(st.session_state["interactions"], columns=["A", "B"]))

    interactions = list(st.session_state.get("interactions", []))

    # Model choices
    st.markdown("**Choose models to train**")
    if task == "regression":
        available_models = ["linear_regression"] + (["xgboost_regressor"] if HAS_XGB else [])
    else:
        available_models = ["logistic_regression"] + (["xgboost_classifier"] if HAS_XGB else [])
    chosen_models = st.multiselect("Models", available_models, default=[available_models[0]])

    # Split
    st.markdown("**Train/test split**")
    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
    with c2:
        random_state = st.number_input("Random state", value=42, step=1)

    # Train button
    if st.button("🚀 Train selected models", type="primary", use_container_width=True, disabled=(len(features) == 0 or len(chosen_models) == 0)):
        # Persist specification
        st.session_state["estimation_specs"][target] = {
            "features": features,
            "log_vars": log_vars,
            "square_vars": square_vars,
            "interactions": interactions,
            "task": task,
            "test_size": float(test_size),
            "random_state": int(random_state),
        }

        # Build design matrix
        X, y, cols = build_feature_matrix(df, target, features, log_vars, square_vars, interactions)

        # Split (stratify if classification)
        strat = y if task == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=strat)

        trained_for_target: Dict[str, Any] = st.session_state["trained_models"].get(target, {})

        # Train each selected model
        for mkey in chosen_models:
            try:
                if task == "regression" and mkey == "linear_regression":
                    mdl = LinearRegression(n_jobs=None) if hasattr(LinearRegression, "__init__") else LinearRegression()
                    mdl.fit(X_tr, y_tr)
                    pred = mdl.predict(X_te)
                    metrics = {
                        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                        "MAE": float(mean_absolute_error(y_te, pred)),
                        "R2": float(r2_score(y_te, pred)),
                    }
                    cal = None  # calibration not applicable
                elif task == "regression" and mkey == "xgboost_regressor":
                    if not HAS_XGB:
                        st.warning("xgboost not available in this environment.")
                        continue
                    mdl = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        n_jobs=2,
                        random_state=int(random_state),
                    )
                    mdl.fit(X_tr, y_tr)
                    pred = mdl.predict(X_te)
                    metrics = {
                        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                        "MAE": float(mean_absolute_error(y_te, pred)),
                        "R2": float(r2_score(y_te, pred)),
                    }
                    cal = None
                elif task == "classification" and mkey == "logistic_regression":
                    mdl = LogisticRegression(max_iter=1000, solver="lbfgs")
                    mdl.fit(X_tr, y_tr.astype(int))
                    prob = mdl.predict_proba(X_te)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    metrics = {
                        "LogLoss": float(log_loss(y_te, prob, labels=[0,1])),
                        "ROC_AUC": float(roc_auc_score(y_te, prob)),
                        "Brier": float(brier_score_loss(y_te, prob)),
                        "Accuracy@0.5": float(accuracy_score(y_te, pred)),
                    }
                    cal = calibration_table(y_te.to_numpy(), prob)
                elif task == "classification" and mkey == "xgboost_classifier":
                    if not HAS_XGB:
                        st.warning("xgboost not available in this environment.")
                        continue
                    mdl = xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        n_jobs=2,
                        random_state=int(random_state),
                        eval_metric="logloss",
                        use_label_encoder=False,
                    )
                    mdl.fit(X_tr, y_tr.astype(int))
                    prob = mdl.predict_proba(X_te)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    metrics = {
                        "LogLoss": float(log_loss(y_te, prob, labels=[0,1])),
                        "ROC_AUC": float(roc_auc_score(y_te, prob)),
                        "Brier": float(brier_score_loss(y_te, prob)),
                        "Accuracy@0.5": float(accuracy_score(y_te, pred)),
                    }
                    cal = calibration_table(y_te.to_numpy(), prob)
                else:
                    st.warning(f"Unknown model key: {mkey}")
                    continue

                trained_for_target[mkey] = {
                    "task": task,
                    "model": mdl,
                    "metrics": metrics,
                    "calibration": cal,   # DataFrame or None
                    "feature_cols": cols, # training column order
                    "spec": {
                        "features": features,
                        "log_vars": log_vars,
                        "square_vars": square_vars,
                        "interactions": interactions,
                    },
                }
                st.success(f"Trained {mkey} for {target}.")
            except Exception as e:
                st.error(f"Training failed for {mkey}: {e}")

        st.session_state["trained_models"][target] = trained_for_target

    # Show a small preview
    with st.expander("Preview a sample of the dataset used for estimation", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)


# --------------------------- UI: Model Selection ---------------------------
def model_selection_tab():
    st.subheader("🧩 Model selection")

    trained = st.session_state.get("trained_models", {})
    if not trained:
        st.info("No models have been trained yet. Go to **Model estimation** first.")
        return

    target_list = sorted(list(trained.keys()))
    target = st.selectbox("Select a target", target_list, index=0)

    models_for_target = trained.get(target, {})
    if not models_for_target:
        st.warning("No models trained for this target yet.")
        return

    # Metrics comparison table
    st.markdown("**Error metrics (evaluated on holdout test set)**")
    rows = []
    for k, rec in models_for_target.items():
        row = {"model": k}
        for mk, mv in rec["metrics"].items():
            row[mk] = mv
        rows.append(row)
    metrics_df = pd.DataFrame(rows).set_index("model")
    st.dataframe(metrics_df, use_container_width=True)

    # Calibration curve (if classification)
    any_classif = any(rec["task"] == "classification" for rec in models_for_target.values())
    if any_classif:
        st.markdown("**Calibration curves**")
        for k, rec in models_for_target.items():
            if rec["task"] != "classification" or rec["calibration"] is None:
                continue
            caldf = rec["calibration"].copy()
            caldf["model"] = k
            line = alt.Chart(caldf).mark_line(point=True).encode(
                x=alt.X("mean_pred:Q", title="Mean predicted probability"),
                y=alt.Y("frac_pos:Q", title="Observed fraction positive"),
                color="model:N",
                tooltip=["model:N", alt.Tooltip("mean_pred:Q", format=".3f"), alt.Tooltip("frac_pos:Q", format=".3f"), "n:Q"],
            ).properties(height=300)
            # 45-degree reference
            ref = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_rule().encode(x="x", y="y")
            st.altair_chart(line + ref, use_container_width=True)

    # Choose the production model
    model_keys = list(models_for_target.keys())
    current_choice = st.session_state["selected_models"].get(target, model_keys[0])
    choice = st.radio("Select model for simulation", model_keys, index=model_keys.index(current_choice), horizontal=True)
    st.session_state["selected_models"][target] = choice
    st.success(f"Using **{choice}** for **{target}** in the simulation.")


# --------------------------- UI: Interventions ---------------------------
def intervention_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.subheader("🧪 Interventions")

    ic = cfg.get("intervention_config", {}) or {}

    existing_keys = list(ic.keys())
    if not existing_keys:
        st.info("No interventions configured yet. Add one below to explore the UI.")
        new_key = st.text_input("New intervention key (e.g., bmi)")
        if st.button("Add Intervention", use_container_width=True) and new_key:
            ic[new_key] = {
                "type": "percentage_decrease",
                "amount": 0.1,
                "target_population": {"age": [30, 60], "bmi": [25, 60]},
            }
            cfg["intervention_config"] = ic
    else:
        key = st.selectbox("Choose an intervention to edit", existing_keys, index=0)
        inv = ic.get(key, {})

        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                inv_type = st.selectbox("Intervention type", ["percentage_decrease", "absolute_change"], index=["percentage_decrease", "absolute_change"].index(inv.get("type", "percentage_decrease")))
                inv["type"] = inv_type

                if inv_type == "percentage_decrease":
                    amt = st.slider("Amount (percent decrease of target variable)", 0.0, 1.0, float(inv.get("amount", 0.2)), 0.01, help="0.2 means a 20% decrease in the target variable for the selected population.")
                else:
                    amt = st.number_input("Amount (absolute change)", value=float(inv.get("amount", 1.0)))
                inv["amount"] = float(amt)

            with c2:
                st.markdown("**Target population filters**")
                tgt = inv.get("target_population", {}) or {}
                a_min, a_max = _range_to_tuple(tgt.get("age"), (18, 90))
                b_min, b_max = _range_to_tuple(tgt.get("bmi"), (15, 60))
                a_min, a_max = st.slider("Age range", min_value=0, max_value=100, value=(int(a_min), int(a_max)))
                b_min, b_max = st.slider("BMI range", min_value=10, max_value=100, value=(int(b_min), int(b_max)))
                tgt["age"] = [int(a_min), int(a_max)]
                tgt["bmi"] = [int(b_min), int(b_max)]
                inv["target_population"] = tgt

        # Save back
        ic[key] = inv
        cfg["intervention_config"] = ic

    return cfg


# --------------------------- Simulation helpers ---------------------------
def apply_bmi_intervention(df: pd.DataFrame, intervention: Dict[str, Any]) -> pd.DataFrame:
    """Apply a 'percentage_decrease' intervention on BMI for a target population."""
    new_df = df.copy()
    if not intervention or intervention.get("type") not in {"percentage_decrease", "absolute_change"}:
        return new_df

    tgt = intervention.get("target_population", {}) or {}
    age_min, age_max = _range_to_tuple(tgt.get("age"), (18, 90))
    bmi_min, bmi_max = _range_to_tuple(tgt.get("bmi"), (15, 60))

    mask = (
        (new_df["age"] >= age_min)
        & (new_df["age"] <= age_max)
        & (new_df["bmi"] >= bmi_min)
        & (new_df["bmi"] <= bmi_max)
    )
    if intervention["type"] == "percentage_decrease":
        amount = float(intervention.get("amount", 0.0))
        new_df.loc[mask, "bmi"] = new_df.loc[mask, "bmi"] * (1.0 - amount)
    else:
        amount = float(intervention.get("amount", 0.0))
        new_df.loc[mask, "bmi"] = new_df.loc[mask, "bmi"] + amount

    return new_df


def predict_with_record(df: pd.DataFrame, target: str, record: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Use a trained model record to produce predictions (and probabilities if classifier)."""
    spec = record["spec"]
    X_new, _, _ = build_feature_matrix(
        df, target,
        spec["features"],
        spec["log_vars"],
        spec["square_vars"],
        spec["interactions"],
    )
    X_new = ensure_same_columns(X_new, record["feature_cols"])

    mdl = record["model"]
    if record["task"] == "regression":
        yhat = mdl.predict(X_new)
        return yhat, None
    else:
        # classifier
        try:
            proba = mdl.predict_proba(X_new)[:, 1]
        except Exception:
            # Some xgboost versions require predict with output margin etc.
            proba = mdl.predict(X_new)
            if proba.ndim == 2 and proba.shape[1] == 2:
                proba = proba[:, 1]
            elif proba.ndim != 1:
                proba = np.ravel(proba)
        return (proba >= 0.5).astype(int), proba


# --------------------------- UI: Run & Results ---------------------------
def run_and_visualize(cfg: Dict[str, Any]) -> None:
    st.subheader("🚀 Run Simulation")

    pop_n = int(st.session_state.get("pop_n", 60_000))
    seed = int(st.session_state.get("seed", 42))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Generate a synthetic baseline population, optionally recompute outcomes with your selected models, and apply interventions.")
        go = st.button("Run Simulation", type="primary", use_container_width=True)
    with c2:
        st.caption("Export results or share a quick snapshot.")
        export_name = st.text_input("Export name", value="demo_run")

    use_models = st.checkbox("Use selected models to compute baseline outcomes before interventions", value=True)

    if not go:
        st.info("Train and select models, configure interventions, then click **Run Simulation**.")
        return

    # Generate baseline
    df_base = generate_synthetic_population(pop_n, seed=seed)
    # Add default 'bmi' as measured; optionally recompute using selected model
    trained = st.session_state.get("trained_models", {})
    selected = st.session_state.get("selected_models", {})

    # If a model was selected for 'bmi' and user wants to apply models, recompute bmi
    if use_models and "bmi" in selected and "bmi" in trained and selected["bmi"] in trained["bmi"]:
        rec = trained["bmi"][selected["bmi"]]
        yhat, _ = predict_with_record(df_base, "bmi", rec)
        # Use predicted BMI as baseline BMI (demo choice)
        df_base["bmi"] = yhat

    # Baseline heart attack probability/model
    if use_models and "hattack_event" in selected and "hattack_event" in trained and selected["hattack_event"] in trained["hattack_event"]:
        rec_h = trained["hattack_event"][selected["hattack_event"]]
        # For classification, predict probabilities if available
        _, prob = predict_with_record(df_base, "hattack_event", rec_h)
        if prob is None:
            # Fall back to toy risk as probability
            df_base["prob_hattack"] = risk_heart_attack(df_base)
        else:
            df_base["prob_hattack"] = prob
    else:
        df_base["prob_hattack"] = risk_heart_attack(df_base)

    rng = np.random.default_rng(seed + 1)
    df_base["hattack_event"] = (rng.random(size=len(df_base)) < df_base["prob_hattack"]).astype(int)

    # Apply interventions (BMI only in this demo)
    df_post = df_base.copy()
    ic = cfg.get("intervention_config", {}) or {}
    for key, inv in ic.items():
        if key == "bmi":
            df_post = apply_bmi_intervention(df_post, inv)

    # Recompute outcome probabilities after intervention using selected model (or toy risk fallback)
    if use_models and "hattack_event" in selected and "hattack_event" in trained and selected["hattack_event"] in trained["hattack_event"]:
        rec_h = trained["hattack_event"][selected["hattack_event"]]
        _, prob_post = predict_with_record(df_post, "hattack_event", rec_h)
        if prob_post is None:
            df_post["prob_hattack"] = risk_heart_attack(df_post)
        else:
            df_post["prob_hattack"] = prob_post
    else:
        df_post["prob_hattack"] = risk_heart_attack(df_post)

    rng = np.random.default_rng(seed + 2)
    df_post["hattack_event"] = (rng.random(size=len(df_post)) < df_post["prob_hattack"]).astype(int)

    # Summaries
    prev_base = float(df_base["hattack_event"].mean())
    prev_post = float(df_post["hattack_event"].mean())

    delta_prev_abs = prev_post - prev_base
    delta_prev_rel = (prev_post / prev_base - 1.0) if prev_base > 0 else math.nan

    # ---- Top-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Population", f"{len(df_base):,}")
    m2.metric("Mean BMI (baseline)", f"{df_base['bmi'].mean():.2f}", delta=f"{(df_post['bmi'].mean()-df_base['bmi'].mean()):+.2f}")
    m3.metric("Heart attack prevalence (baseline)", f"{prev_base*100:.2f}%", delta=f"{delta_prev_abs*100:+.2f}%")
    m4.metric("Relative change in prevalence", f"{delta_prev_rel*100:+.2f}%")

    st.divider()

    # ---- Charts
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("**BMI distribution (baseline vs. post-intervention)**")
        b0 = df_base[["bmi"]].copy(); b0["state"] = "Baseline"
        b1 = df_post[["bmi"]].copy(); b1["state"] = "Post"
        tidy_bmi = pd.concat([b0, b1], ignore_index=True)

        bins = alt.Bin(maxbins=40)
        chart_bmi = alt.Chart(tidy_bmi).transform_bin(["bmi_bin"], field="bmi", bin=bins).mark_bar(opacity=0.6).encode(
            x=alt.X("bmi_bin:Q", title="BMI (binned)"),
            y=alt.Y("count()", stack=None, title="Count"),
            color=alt.Color("state:N", title="Scenario"),
            tooltip=["state", "count()"]
        ).properties(height=300)
        st.altair_chart(chart_bmi, use_container_width=True)

    with col_right:
        st.markdown("**Prevalence of heart attack (baseline vs. post)**")
        prev_df = pd.DataFrame({"scenario": ["Baseline", "Post"], "prevalence": [prev_base * 100.0, prev_post * 100.0]})
        chart_prev = alt.Chart(prev_df).mark_bar().encode(
            x=alt.X("scenario:N", title=None),
            y=alt.Y("prevalence:Q", title="Prevalence (%)"),
            tooltip=["scenario", alt.Tooltip("prevalence:Q", format=".2f")]
        ).properties(height=300)
        st.altair_chart(chart_prev, use_container_width=True)

    st.divider()

    # ---- By age group breakdown
    st.markdown("**Breakdown by age group**")
    def _breakdown(df: pd.DataFrame) -> pd.DataFrame:
        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ["18–29", "30–39", "40–49", "50–59", "60–69", "70–89"]
        g = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
        out = (
            df.assign(age_group=g)
            .groupby("age_group", as_index=False)
            .agg(n=("age", "size"), bmi_mean=("bmi", "mean"), prev=("hattack_event", "mean"))
        )
        out["prev"] = out["prev"] * 100.0
        return out

    br_base = _breakdown(df_base)
    br_post = _breakdown(df_post)
    merged = br_base.merge(br_post, on="age_group", suffixes=("_base", "_post"))

    c = alt.Chart(merged).transform_fold(["prev_base", "prev_post"], as_=["scenario", "value"]).mark_line(point=True).encode(
        x=alt.X("age_group:N", title="Age group"),
        y=alt.Y("value:Q", title="Prevalence (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("scenario:N", title=""),
        tooltip=["age_group:N", alt.Tooltip("value:Q", format=".2f"), "scenario:N"],
    ).properties(height=300)
    st.altair_chart(c, use_container_width=True)

    st.caption("Note: All models and outputs here are illustrative. Replace with your trained models and data pipeline.")

    # ---- Data export
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ Download baseline microdata (CSV)",
            data=df_base.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_name}_baseline.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "⬇️ Download post-intervention microdata (CSV)",
            data=df_post.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_name}_post.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------- UI: Raw JSON ---------------------------
def raw_json_view(cfg: Dict[str, Any]) -> None:
    st.subheader("🧾 Raw JSON (current in-session config)")
    st.json(cfg, expanded=False)
    if "raw_uploaded_text" in st.session_state and st.session_state["raw_uploaded_text"]:
        with st.expander("Original uploaded text", expanded=False):
            st.code(st.session_state["raw_uploaded_text"], language="json")
    if "cleaned_uploaded_text" in st.session_state and st.session_state["cleaned_uploaded_text"]:
        with st.expander("Cleaned JSON used for parsing", expanded=False):
            st.code(st.session_state["cleaned_uploaded_text"], language="json")


# --------------------------- App Entry ---------------------------
def main():
    st.title("📈 Economic Simulation — Prototype")
    st.caption(
        "New workflow: (0) Estimate models from your dataset, (1) compare and select models, "
        "(2) configure interventions, (3) run the simulation."
    )

    # Sidebar controls and optional upload
    cfg = sidebar_config_controls(default_config())

    # Tabs
    t0, t1, t2, t3, t4 = st.tabs([
        "0) Model estimation",
        "1) Model selection",
        "2) Interventions",
        "3) Run & Results",
        "4) JSON",
    ])
    with t0:
        model_estimation_tab()
    with t1:
        model_selection_tab()
    with t2:
        cfg = intervention_editor(cfg)
    with t3:
        run_and_visualize(cfg)
    with t4:
        raw_json_view(cfg)

    # Footer
    st.divider()
    st.markdown(
        "Built as a demonstration stub • Replace synthetic data and toy risk models with your own pipeline. "
        "This UI is purposely simple and extensible."
    )


if __name__ == "__main__":
    main()
