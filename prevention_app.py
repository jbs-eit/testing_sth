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

# ML imports (optional XGBoost)
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
    """
    Make a 'loose' JSON string loadable by Python's json by:
      - removing // line comments and /* ... */ block comments
      - removing key-value pairs where key is "..."
      - removing standalone "..." tokens in arrays
      - removing trailing commas before } or ]
    This improves UX when configs include ellipses for illustration.
    """
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
    """
    Minimal default config. You can upload any JSON; we only read 'intervention_config' here for demo.
    Model estimation/selection is session-driven, not JSON-driven.
    """
    return {
        "meta": {"name": "Demo Scenario"},
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
        if isinstance(val, list) and len(val) == 2:
            return float(val[0]), float(val[1])
        if isinstance(val, tuple) and len(val) == 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return fallback


def is_binary_series(s: pd.Series, threshold_unique: int = 2) -> bool:
    vals = s.dropna().unique()
    return len(vals) <= threshold_unique and set(vals).issubset({0, 1})


# --------------------------- Synthetic Data + DGP ---------------------------
def generate_synthetic_population(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic micro-population with a few relevant variables.
    This is *illustrative only* — not a real dataset.
    """
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
    # Derived binary target using a toy DGP
    df["prob_hattack_true"] = risk_heart_attack(df)
    df["hattack_ever_w10"] = (rng.random(size=n) < df["prob_hattack_true"]).astype(int)
    df["is_employed"] = (df["employment_status"] == "employed").astype(int)
    return df


def apply_bmi_intervention(df: pd.DataFrame, intervention: Dict[str, Any]) -> pd.DataFrame:
    new_df = df.copy()
    if not intervention or intervention.get("type") != "percentage_decrease":
        return new_df

    amount = float(intervention.get("amount", 0.0))
    tgt = intervention.get("target_population", {}) or {}

    age_min, age_max = _range_to_tuple(tgt.get("age"), (18, 90))
    bmi_min, bmi_max = _range_to_tuple(tgt.get("bmi"), (15, 60))

    mask = (
        (new_df["age"] >= age_min)
        & (new_df["age"] <= age_max)
        & (new_df["bmi"] >= bmi_min)
        & (new_df["bmi"] <= bmi_max)
    )
    new_df.loc[mask, "bmi"] = new_df.loc[mask, "bmi"] * (1.0 - amount)
    return new_df


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def risk_heart_attack(prob_inputs: pd.DataFrame) -> np.ndarray:
    """
    Illustrative probability of *ever* having a heart attack (not clinical).
    Uses BMI and Age with placeholder coefficients.
    """
    bmi = prob_inputs["bmi"].to_numpy()
    age = prob_inputs["age"].to_numpy()
    a = -4.2
    b1 = 0.50
    b2 = 0.65
    z = a + b1 * ((bmi - 25.0) / 5.0) + b2 * ((age - 50.0) / 10.0)
    return logistic(z)


def summarize_population(df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "n": len(df),
        "age_mean": float(df["age"].mean()),
        "age_p50": float(df["age"].median()),
        "bmi_mean": float(df["bmi"].mean()),
        "bmi_p50": float(df["bmi"].median()),
    }
    return out


# --------------------------- Feature Engineering ---------------------------
NUMERIC_LIKE = {"int64", "int32", "float64", "float32", "int16", "float16"}

def available_variables(df: pd.DataFrame) -> Dict[str, str]:
    """Return {var: type} for variables in df (numeric | categorical | binary)."""
    out = {}
    for c in df.columns:
        if c in ("prob_hattack_true",):  # exclude latent
            continue
        if is_binary_series(df[c]):
            out[c] = "binary"
        elif str(df[c].dtype) in NUMERIC_LIKE:
            out[c] = "numeric"
        else:
            out[c] = "categorical"
    return out


def build_design_matrix(df: pd.DataFrame, target: str, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    spec = {
      'base_features': [str, ...],
      'log': [str, ...],           # subset of base_features (numeric)
      'square': [str, ...],        # subset of base_features (numeric)
      'interactions': [(str,str), ...],  # pairs, numeric only
    }
    """
    base = list(spec.get("base_features", []))
    base = [b for b in base if b != target and b in df.columns]  # guard

    df_work = pd.DataFrame(index=df.index)
    # Add base features
    for b in base:
        df_work[b] = df[b]

    # Transforms (numeric only)
    for b in spec.get("log", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            # log1p to be safe
            df_work[f"log_{b}"] = np.log1p(np.clip(df[b].to_numpy(), a_min=0, a_max=None))

    for b in spec.get("square", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            df_work[f"{b}_sq"] = np.square(df[b])

    # Interactions (numeric only)
    for (u, v) in spec.get("interactions", []):
        if u in df.columns and v in df.columns:
            if (str(df[u].dtype) in NUMERIC_LIKE) and (str(df[v].dtype) in NUMERIC_LIKE):
                df_work[f"{u}*{v}"] = df[u].to_numpy() * df[v].to_numpy()

    # One-hot encode categoricals
    X = pd.get_dummies(df_work, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# --------------------------- Sidebar ---------------------------
def sidebar_controls(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("⚙️ Scenario & Data")
    st.sidebar.text_input("Scenario name", key="scenario_name", value=st.session_state.get("scenario_name", "Demo Scenario"))

    st.sidebar.caption("Upload a configuration JSON (optional): only interventions are read; models are session-driven in this demo.")
    uploaded = st.sidebar.file_uploader("Choose a JSON file", type=["json"], key="cfg_uploader")
    if uploaded is not None:
        raw_text = _safe_read_byteslike(uploaded)
        try:
            parsed, cleaned = load_config_from_text(raw_text)
            st.session_state["raw_uploaded_text"] = raw_text
            st.session_state["cleaned_uploaded_text"] = cleaned
            # Keep only intervention_config if present
            if "intervention_config" in parsed:
                st.session_state.setdefault("config", cfg)
                st.session_state["config"]["intervention_config"] = parsed["intervention_config"]
            st.sidebar.success("Configuration loaded (interventions).")
        except ValueError as e:
            st.sidebar.error(str(e))

    st.sidebar.divider()
    st.sidebar.caption("Synthetic Training/Simulation Settings")
    pop_n = st.sidebar.number_input("Population size", min_value=2_000, max_value=300_000, value=30_000, step=1_000)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    st.session_state["pop_n"] = int(pop_n)
    st.session_state["seed"] = int(seed)

    st.sidebar.divider()
    st.sidebar.caption("Downloads")
    st.sidebar.download_button(
        "💾 Download current interventions JSON",
        data=json.dumps(st.session_state.get("config", cfg), indent=2),
        file_name="intervention_config.json",
        mime="application/json",
        use_container_width=True,
    )

    return st.session_state.get("config", cfg)


# --------------------------- Model Estimation ---------------------------
def model_estimation_tab():
    st.subheader("🧮 Model estimation")

    # Ensure training data exists
    if "training_data" not in st.session_state:
        st.info("Generating a synthetic training dataset based on your sidebar settings.")
        st.session_state["training_data"] = generate_synthetic_population(
            st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42)
        )

    df = st.session_state["training_data"]
    vtypes = available_variables(df)

    with st.expander("Preview training data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown("**Choose a target variable to estimate**")
    # Potential targets: numeric or binary
    candidate_targets = [c for c,t in vtypes.items() if t in ("numeric", "binary")]
    target = st.selectbox("Target variable", options=candidate_targets, index=candidate_targets.index("bmi") if "bmi" in candidate_targets else 0)

    # Detect target type
    target_type = "binary" if vtypes.get(target) == "binary" else "continuous"
    st.caption(f"Detected target type: **{target_type}**")

    # Feature selection & transforms
    st.markdown("**Feature engineering**")
    base_candidates = [c for c in vtypes.keys() if c != target]
    base_features = st.multiselect("Base features", options=base_candidates, default=[c for c in ["age", "log_income", "is_employed", "bmi"] if c in base_candidates])

    # Transform toggles (numeric only)
    numeric_bases = [c for c in base_features if str(df[c].dtype) in NUMERIC_LIKE]
    col1, col2 = st.columns(2)
    with col1:
        log_feats = st.multiselect("Apply log(1+x) to", options=numeric_bases, default=[c for c in numeric_bases if c.startswith("income")])
    with col2:
        sq_feats = st.multiselect("Apply square to", options=numeric_bases, default=[])

    # Interactions
    st.markdown("**Interactions (pairwise among selected numeric features)**")
    interact_vars = st.multiselect("Select variables to fully interact (all pairwise products)", options=numeric_bases, default=["age", "bmi"] if "bmi" in numeric_bases else ["age"])
    interactions: List[Tuple[str,str]] = []
    if len(interact_vars) >= 2:
        for i in range(len(interact_vars)):
            for j in range(i+1, len(interact_vars)):
                interactions.append((interact_vars[i], interact_vars[j]))

    # Algorithm selection
    st.markdown("**Choose algorithm(s)**")
    algos = []
    if target_type == "continuous":
        c1, c2 = st.columns(2)
        with c1:
            use_lin = st.checkbox("Linear Regression", value=True)
        with c2:
            if XGB_AVAILABLE:
                use_xgb = st.checkbox("XGBoost (Regressor)", value=True)
            else:
                st.checkbox("XGBoost (Regressor)", value=False, disabled=True, help="Install xgboost to enable.")
                use_xgb = False
        if use_lin: algos.append("linear_regression")
        if use_xgb: algos.append("xgboost_regressor")
    else:
        c1, c2 = st.columns(2)
        with c1:
            use_logit = st.checkbox("Logistic Regression", value=True)
        with c2:
            if XGB_AVAILABLE:
                use_xgb = st.checkbox("XGBoost (Classifier)", value=True)
            else:
                st.checkbox("XGBoost (Classifier)", value=False, disabled=True, help="Install xgboost to enable.")
                use_xgb = False
        if use_logit: algos.append("logistic_regression")
        if use_xgb: algos.append("xgboost_classifier")

    # Train/test split
    st.markdown("**Train / Test split**")
    colA, colB, colC = st.columns(3)
    with colA:
        test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    with colB:
        random_state = st.number_input("Random state", value=42, step=1)
    with colC:
        stratify_opt = st.selectbox("Stratify (binary only)", options=[None] + base_candidates, index=0)
        if target_type != "binary":
            st.caption("Stratify is only used for classification.")

    spec = {
        "base_features": base_features,
        "log": log_feats,
        "square": sq_feats,
        "interactions": interactions,
    }

    st.divider()
    do_train = st.button("🧪 Train selected models", type="primary", use_container_width=True)
    if do_train and algos:
        # Prepare data
        y = df[target]
        X = build_design_matrix(df, target, spec)
        # Align target type
        if target_type == "binary":
            y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state),
            stratify=y if (target_type=="binary" and stratify_opt is not None) else None
        )

        results: Dict[str, Any] = {
            "type": target_type,
            "feature_spec": spec,
            "train_test_split": {"test_size": float(test_size), "random_state": int(random_state), "stratify": (stratify_opt if target_type=="binary" else None)},
            "models": {},
            "columns": list(X.columns),
        }

        # Train each selected algorithm
        for algo in algos:
            if target_type == "continuous" and algo == "linear_regression":
                est = LinearRegression()
                est.fit(X_train, y_train)
                y_pred = est.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                results["models"]["Linear Regression"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": y_pred},
                }

            if target_type == "continuous" and algo == "xgboost_regressor" and XGB_AVAILABLE:
                est = XGBRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=int(random_state),
                    reg_lambda=1.0, n_jobs=4
                )
                est.fit(X_train, y_train)
                y_pred = est.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                results["models"]["XGBoost Regressor"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": y_pred},
                }

            if target_type == "binary" and algo == "logistic_regression":
                est = LogisticRegression(max_iter=1000)
                est.fit(X_train, y_train)
                prob = est.predict_proba(X_test)[:,1]
                rmse = float(np.sqrt(mean_squared_error(y_test, prob)))  # RMSE on probabilities vs 0/1
                mae = float(mean_absolute_error(y_test, prob))           # MAE on probabilities vs 0/1
                ll = float(log_loss(y_test, prob))
                # Calibration
                prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=12, strategy="uniform")
                results["models"]["Logistic Regression"] = {
                    "estimator": est,
                    "metrics": {"RMSE": rmse, "MAE": mae, "LogLoss": ll},
                    "pred_true": {"y_true": y_test.to_numpy(), "y_pred": prob},
                    "calibration": {"prob_true": prob_true, "prob_pred": prob_pred},
                }

            if target_type == "binary" and algo == "xgboost_classifier" and XGB_AVAILABLE:
                est = XGBClassifier(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=int(random_state),
                    reg_lambda=1.0, n_jobs=4, eval_metric="logloss"
                )
                est.fit(X_train, y_train)
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

        # Save results to session
        st.session_state.setdefault("trained_models", {})
        st.session_state["trained_models"][target] = results

        st.success(f"Trained {len(results['models'])} model(s) for target '{target}'. Go to the **Model selection** tab to compare and choose one.")
        st.session_state["last_trained_target"] = target

    # Show a compact summary of what has been trained so far
    if "trained_models" in st.session_state and st.session_state["trained_models"]:
        st.divider()
        st.markdown("**Trained so far**")
        for tgt, pack in st.session_state["trained_models"].items():
            algo_names = list(pack.get("models", {}).keys())
            if not algo_names:
                continue
            st.write(f"• `{tgt}` — {pack['type']} — models: {', '.join(algo_names)}")


# --------------------------- Model Selection ---------------------------
def model_selection_tab():
    st.subheader("🧩 Model selection")
    trained = st.session_state.get("trained_models", {})
    if not trained:
        st.info("No models trained yet. Please go to **Model estimation** first.")
        return

    variables = list(trained.keys())
    var = st.selectbox("Select a variable to inspect & choose model", options=variables, index=variables.index(st.session_state.get("last_trained_target", variables[0])))
    pack = trained[var]
    mtype = pack["type"]
    models_dict = pack["models"]

    if not models_dict:
        st.warning("No models found for this variable. Train models in the previous tab.")
        return

    # Compare metrics
    st.markdown("**Error metrics comparison**")
    rows = []
    for name, info in models_dict.items():
        met = info["metrics"]
        rows.append({"model": name, **met})
    met_df = pd.DataFrame(rows)

    # Metrics chart(s)
    if mtype == "continuous":
        # RMSE & MAE bars
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

        # Predicted vs Actual scatter for each model
        st.markdown("**Predicted vs Actual (test set)**")
        for name, info in models_dict.items():
            d = pd.DataFrame({
                "y_true": info["pred_true"]["y_true"],
                "y_pred": info["pred_true"]["y_pred"]
            })
            chart = alt.Chart(d).mark_circle(opacity=0.4).encode(
                x=alt.X("y_true:Q", title="Actual"),
                y=alt.Y("y_pred:Q", title="Predicted"),
                tooltip=[alt.Tooltip("y_true:Q", format=".2f"), alt.Tooltip("y_pred:Q", format=".2f")]
            ).properties(height=250, title=f"{name}")
            st.altair_chart(chart, use_container_width=True)

    else:
        # Classification: RMSE & MAE on probabilities + calibration curves
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

        # Calibration curves overlay
        st.markdown("**Calibration curves**")
        layers = []
        for name, info in models_dict.items():
            cal = info.get("calibration", None)
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
        else:
            st.info("No calibration data found for these models.")

    # Selection UI
    st.divider()
    st.markdown("**Select model for simulation**")
    model_names = list(models_dict.keys())
    chosen = st.radio("Choose one", options=model_names, index=0)
    st.session_state.setdefault("chosen_models", {})
    st.session_state["chosen_models"][var] = chosen
    st.success(f"Selected: **{chosen}** for `{var}`")


# --------------------------- Interventions ---------------------------
def intervention_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.subheader("🧪 Interventions")

    ic = cfg.get("intervention_config", {}) or {}
    if not ic:
        st.info("No interventions configured yet. Add one below to explore the UI.")
        new_key = st.text_input("New intervention key (e.g., bmi)")
        if st.button("Add Intervention", use_container_width=True) and new_key:
            ic[new_key] = {"type":"percentage_decrease","amount":0.1,"target_population":{"age":[30,60],"bmi":[25,60]}}
            cfg["intervention_config"] = ic
    else:
        key = st.selectbox("Choose an intervention to edit", list(ic.keys()), index=0)
        inv = ic.get(key, {})

        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                inv_type = st.selectbox("Intervention type", ["percentage_decrease","absolute_change"], index=["percentage_decrease","absolute_change"].index(inv.get("type","percentage_decrease")))
                inv["type"] = inv_type

                if inv_type == "percentage_decrease":
                    amt = st.slider("Amount (percent decrease of target variable)", 0.0, 1.0, float(inv.get("amount", 0.2)), 0.01)
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

        ic[key] = inv
        cfg["intervention_config"] = ic

    return cfg


# --------------------------- Run Simulation ---------------------------
def run_and_visualize(cfg: Dict[str, Any]) -> None:
    st.subheader("🚀 Run Simulation")

    pop_n = int(st.session_state.get("pop_n", 30_000))
    seed = int(st.session_state.get("seed", 42))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Generate a synthetic baseline population, apply the configured interventions, and use your **selected** models to compute outcomes.")
        go = st.button("Run Simulation", type="primary", use_container_width=True)
    with c2:
        st.caption("Export results or share a quick snapshot.")
        export_name = st.text_input("Export name", value="demo_run")

    if not go:
        st.info("Pick models in **Model selection**, configure **Interventions**, then click **Run Simulation**.")
        return

    # Generate baseline data
    df_base = generate_synthetic_population(pop_n, seed=seed)

    # If a BMI model has been selected, use it to predict BMI (baseline), then let interventions modify it
    chosen = st.session_state.get("chosen_models", {})
    trained = st.session_state.get("trained_models", {})

    if "bmi" in chosen and "bmi" in trained:
        pack = trained["bmi"]
        model_name = chosen["bmi"]
        est = pack["models"][model_name]["estimator"]
        # Build features for bmi prediction
        Xb = build_design_matrix(df_base, "bmi", pack["feature_spec"])
        # Align columns (training columns)
        for col in pack["columns"]:
            if col not in Xb.columns:
                Xb[col] = 0.0
        Xb = Xb[pack["columns"]]
        df_base["bmi"] = est.predict(Xb)

    # Apply interventions that might touch BMI
    ic = cfg.get("intervention_config", {}) or {}
    df_post = df_base.copy()
    for key, inv in ic.items():
        if key == "bmi":
            df_post = apply_bmi_intervention(df_post, inv)

    # If a heart-attack model is selected, use it to compute probabilities.
    # Otherwise, fall back to the toy risk function.
    if "hattack_ever_w10" in chosen and "hattack_ever_w10" in trained:
        pack_h = trained["hattack_ever_w10"]
        model_name_h = chosen["hattack_ever_w10"]
        est_h = pack_h["models"][model_name_h]["estimator"]
        # Baseline
        Xh0 = build_design_matrix(df_base, "hattack_ever_w10", pack_h["feature_spec"])
        for col in pack_h["columns"]:
            if col not in Xh0.columns:
                Xh0[col] = 0.0
        Xh0 = Xh0[pack_h["columns"]]
        if pack_h["type"] == "binary":
            if hasattr(est_h, "predict_proba"):
                df_base["prob_hattack"] = est_h.predict_proba(Xh0)[:,1]
            else:
                df_base["prob_hattack"] = est_h.predict(Xh0)  # fallback for calibrated models
        else:
            df_base["prob_hattack"] = np.clip(est_h.predict(Xh0), 0, 1)

        # Post
        Xh1 = build_design_matrix(df_post, "hattack_ever_w10", pack_h["feature_spec"])
        for col in pack_h["columns"]:
            if col not in Xh1.columns:
                Xh1[col] = 0.0
        Xh1 = Xh1[pack_h["columns"]]
        if pack_h["type"] == "binary":
            if hasattr(est_h, "predict_proba"):
                df_post["prob_hattack"] = est_h.predict_proba(Xh1)[:,1]
            else:
                df_post["prob_hattack"] = est_h.predict(Xh1)
        else:
            df_post["prob_hattack"] = np.clip(est_h.predict(Xh1), 0, 1)
    else:
        # Fallback: toy function
        df_base["prob_hattack"] = risk_heart_attack(df_base)
        df_post["prob_hattack"] = risk_heart_attack(df_post)

    rng = np.random.default_rng(seed + 123)
    df_base["hattack_event"] = (rng.random(size=len(df_base)) < df_base["prob_hattack"]).astype(int)
    df_post["hattack_event"] = (rng.random(size=len(df_post)) < df_post["prob_hattack"]).astype(int)

    # Summaries
    s_base = summarize_population(df_base)
    s_post = summarize_population(df_post)
    prev_base = float(df_base["hattack_event"].mean())
    prev_post = float(df_post["hattack_event"].mean())

    delta_prev_abs = prev_post - prev_base
    delta_prev_rel = (prev_post / prev_base - 1.0) if prev_base > 0 else math.nan

    # ---- Top-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Population", f"{len(df_base):,}")
    m2.metric("Mean BMI (baseline)", f"{s_base['bmi_mean']:.2f}", delta=f"{(s_post['bmi_mean']-s_base['bmi_mean']):+.2f}")
    m3.metric("Heart attack prevalence (baseline)", f"{prev_base*100:.2f}%",
              delta=f"{delta_prev_abs*100:+.2f}%")
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

    # ---- By age group breakdown
    st.markdown("**Breakdown by age group**")
    def _breakdown(df: pd.DataFrame) -> pd.DataFrame:
        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ["18–29", "30–39", "40–49", "50–59", "60–69", "70–89"]
        g = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
        out = (
            df.assign(age_group=g)
            .groupby("age_group", as_index=False)
            .agg(
                n=("age", "size"),
                bmi_mean=("bmi", "mean"),
                prev=("hattack_event", "mean"),
            )
        )
        out["prev"] = out["prev"] * 100.0
        return out

    br_base = _breakdown(df_base)
    br_post = _breakdown(df_post)
    merged = br_base.merge(br_post, on="age_group", suffixes=("_base", "_post"))

    c = alt.Chart(merged).transform_fold(
        ["prev_base", "prev_post"],
        as_=["scenario", "value"],
    ).mark_line(point=True).encode(
        x=alt.X("age_group:N", title="Age group"),
        y=alt.Y("value:Q", title="Prevalence (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("scenario:N", title=""),
        tooltip=["age_group:N", alt.Tooltip("value:Q", format=".2f"), "scenario:N"],
    ).properties(height=300)
    st.altair_chart(c, use_container_width=True)

    st.caption("Note: Models and outputs here are illustrative. Replace synthetic data and toy DGP with your real pipeline.")

    # ---- Export
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


# --------------------------- Raw JSON View ---------------------------
def raw_json_view(cfg: Dict[str, Any]) -> None:
    st.subheader("🧾 Raw JSON (interventions only in this demo)")
    st.json(cfg, expanded=False)
    if "raw_uploaded_text" in st.session_state:
        with st.expander("Original uploaded text", expanded=False):
            st.code(st.session_state["raw_uploaded_text"], language="json")
    if "cleaned_uploaded_text" in st.session_state:
        with st.expander("Cleaned JSON used for parsing", expanded=False):
            st.code(st.session_state["cleaned_uploaded_text"], language="json")


# --------------------------- App Entry ---------------------------
def main():
    st.title("📈 Economic Simulation — Prototype")
    st.caption(
        "Upload interventions JSON (optional), estimate models from data, compare candidates, "
        "select the ones to use, configure interventions, and simulate outcomes."
    )

    cfg = sidebar_controls(default_config())

    # Tabs in the requested order:
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

    st.divider()
    st.markdown(
        "Built for demonstration • Replace synthetic data and toy risk with your pipeline. "
        "Model artifacts are kept in-session for exploration."
    )


if __name__ == "__main__":
    if "config" not in st.session_state:
        st.session_state["config"] = default_config()
    main()
