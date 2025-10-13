# app.py
# -------------------------------------------------------------
# Economic Simulation Prototype (Streamlit)
#   • Model Selection from a Pre-trained Catalog (no estimation UI)
#   • Multi-intervention configuration
#   • Pipeline JSON export (decisions + catalog metadata)
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
    page_title="Economic Simulation — Model Selection Demo",
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
        "pipeline": {
            # A catalog of pre-trained models is created at runtime; we only export metadata here.
            "catalog": {},            # per-target: list of models + metrics (no estimator objects)
            "selection": {},          # chosen model per target
            "interventions": [],      # list of interventions
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
def build_design_matrix(df: pd.DataFrame, target: str, spec: Dict[str, Any]) -> pd.DataFrame:
    base = [b for b in spec.get("base_features", []) if b != target and b in df.columns]
    df_work = pd.DataFrame(index=df.index)
    for b in base:
        df_work[b] = df[b]
    for b in spec.get("log", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            df_work[f"log_{b}"] = np.log1p(np.clip(df[b].to_numpy(), a_min=0, a_max=None))
    for b in spec.get("square", []):
        if b in df.columns and str(df[b].dtype) in NUMERIC_LIKE:
            df_work[f"{b}_sq"] = np.square(df[b])
    for (u, v) in spec.get("interactions", []):
        if u in df.columns and v in df.columns:
            if (str(df[u].dtype) in NUMERIC_LIKE) and (str(df[v].dtype) in NUMERIC_LIKE):
                df_work[f"{u}*{v}"] = df[u].to_numpy() * df[v].to_numpy()
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
    new_df = df.copy()
    target_var = inv.get("target")
    if (target_var is None) or (target_var not in new_df.columns):
        return new_df

    iv_type = inv.get("type", "percentage_decrease")
    amount = float(inv.get("amount", 0.0))
    filters = inv.get("filters", {}) or {}

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

    st.sidebar.caption("Upload a pipeline JSON (optional). Selections & interventions will load if present.")
    uploaded = st.sidebar.file_uploader("Choose a JSON file", type=["json"], key="cfg_uploader")
    if uploaded is not None:
        raw_text = _safe_read_byteslike(uploaded)
        try:
            parsed, cleaned = load_config_from_text(raw_text)
            st.session_state["raw_uploaded_text"] = raw_text
            st.session_state["cleaned_uploaded_text"] = cleaned

            st.session_state.setdefault("config", cfg)
            pipe = parsed.get("pipeline") or {}
            if pipe:
                # Restore into session_state for live UI
                st.session_state["catalog_meta"] = pipe.get("catalog", {})
                st.session_state["chosen_models"] = pipe.get("selection", {})
                st.session_state["interventions"] = pipe.get("interventions", [])
            st.sidebar.success("Configuration loaded.")
        except ValueError as e:
            st.sidebar.error(str(e))

    st.sidebar.divider()
    st.sidebar.caption("Synthetic Training/Simulation Settings")
    pop_n = st.sidebar.number_input("Population size", min_value=2_000, max_value=300_000, value=30_000, step=1_000, key="pop_n")
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1, key="seed")

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


# --------------------------- Catalog (Pre-trained models) ---------------------------
def _catalog_spec() -> Dict[str, Any]:
    """Return a fixed set of targets, specs, and algorithms to pre-train for the demo."""
    algos_reg = ["linear_regression"] + (["xgboost_regressor"] if XGB_AVAILABLE else [])
    algos_cls = ["logistic_regression"] + (["xgboost_classifier"] if XGB_AVAILABLE else [])
    return {
        # Continuous
        "bmi": {
            "target_type": "continuous",
            "spec": {
                "base_features": ["age", "log_income", "is_employed"],
                "log": [],
                "square": ["age"],
                "interactions": [("age", "log_income")],
            },
            "algorithms": algos_reg,
        },
        "log_income": {
            "target_type": "continuous",
            "spec": {
                "base_features": ["age", "is_employed", "bmi"],
                "log": [],
                "square": ["age"],
                "interactions": [("age", "bmi")],
            },
            "algorithms": algos_reg,
        },
        # Binary
        "hattack_ever_w10": {
            "target_type": "binary",
            "spec": {
                "base_features": ["age", "bmi", "is_employed"],
                "log": [],
                "square": ["age"],
                "interactions": [("age", "bmi")],
            },
            "algorithms": algos_cls,
        },
        "is_employed": {
            "target_type": "binary",
            "spec": {
                "base_features": ["age", "log_income", "bmi"],
                "log": [],
                "square": ["age"],
                "interactions": [("age", "log_income")],
            },
            "algorithms": algos_cls,
        },
    }


def _train_one(df: pd.DataFrame, target: str, ttype: str, spec: Dict[str, Any], algos: List[str], random_state: int = 42):
    """Train a small set of algorithms per target on synthetic data."""
    y = df[target]
    X = build_design_matrix(df, target, spec)
    if ttype == "binary":
        y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if ttype=="binary" else None
    )

    results: Dict[str, Any] = {
        "type": ttype,
        "feature_spec": spec,
        "train_test_split": {"test_size": 0.2, "random_state": random_state, "stratify": (target if ttype=="binary" else None)},
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

        if ttype == "continuous" and algo == "xgboost_regressor" and XGB_AVAILABLE:
            est = XGBRegressor(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=random_state,
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

        if ttype == "binary" and algo == "xgboost_classifier" and XGB_AVAILABLE:
            est = XGBClassifier(
                n_estimators=350, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=random_state,
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

    return results


def build_catalog_models(df: pd.DataFrame, force: bool = False):
    """Build a pre-trained catalog on synthetic data (once per session)."""
    if ("catalog_models" in st.session_state) and (not force):
        return

    st.session_state["catalog_models"] = {}
    st.session_state["catalog_meta"] = {}
    spec = _catalog_spec()

    with st.spinner("Preparing catalog of pre-trained models..."):
        for tgt, cfg in spec.items():
            res = _train_one(
                df=df,
                target=tgt,
                ttype=cfg["target_type"],
                spec=cfg["spec"],
                algos=cfg["algorithms"],
                random_state=42,
            )
            st.session_state["catalog_models"][tgt] = res
            # Store exporter-friendly metadata
            meta = {
                "type": res["type"],
                "feature_spec": res["feature_spec"],
                "train_test_split": res["train_test_split"],
                "columns": res["columns"],
                "models": {name: {"metrics": info["metrics"]} for name, info in res["models"].items()},
            }
            st.session_state["catalog_meta"][tgt] = meta
        time.sleep(0.2)


# --------------------------- Defaults Helper ---------------------------
def apply_defaults_to_selection():
    # Choose sensible defaults: prefer XGBoost where available; else GLM.
    catalog = st.session_state.get("catalog_models", {})
    chosen = {}
    for tgt, pack in catalog.items():
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


# --------------------------- Model Selection Tab ---------------------------
def model_selection_tab():
    top_cols = st.columns([1, 4, 1])
    with top_cols[1]:
        if st.button("Use default configuration", key="btn_default_selection"):
            apply_defaults_to_selection()
            st.success("Default selections applied (prefers XGBoost where available).")

    st.subheader("🧩 Model selection (catalog)")
    catalog = st.session_state.get("catalog_models", {})
    if not catalog:
        st.info("Catalog is empty. It will be generated automatically at startup.")
        return

    variables = list(catalog.keys())
    var = st.selectbox("Select a variable to inspect & choose model", options=variables, index=0, key="sel_target")
    pack = catalog[var]
    mtype = pack["type"]
    models_dict = pack["models"]

    # Compare metrics
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
    default_idx = 0
    if var in st.session_state.get("chosen_models", {}):
        try:
            default_idx = model_names.index(st.session_state["chosen_models"][var])
        except ValueError:
            default_idx = 0
    chosen = st.radio("Choose one", options=model_names, index=default_idx, key=f"sel_chosen_model_{var}")
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
            if getattr(coef, "ndim", 1) > 1:
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
        rows = [{"target": t, "chosen_model": m} for t, m in all_sel.items()]
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
            df = st.session_state.get("training_data")
            if df is None:
                df = generate_synthetic_population(st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42))
                st.session_state["training_data"] = df
            apply_defaults_to_interventions(df)
            st.success("Added two default BMI interventions.")

    st.subheader("🧪 Interventions")
    st.caption("Add multiple interventions. Filters support age and BMI for the demo.")
    st.session_state.setdefault("interventions", [])
    df = st.session_state.get("training_data")

    # List existing interventions
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

    for i in sorted(to_delete, reverse=True):
        del st.session_state["interventions"][i]

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
        st.info("Choose models in **Model selection**, configure **Interventions**, then click **Run Simulation**.")
        return

    # baseline data
    df_base = generate_synthetic_population(pop_n, seed=seed)

    # Predict BMI if a BMI model was selected
    chosen = st.session_state.get("chosen_models", {})
    catalog = st.session_state.get("catalog_models", {})

    if "bmi" in chosen and "bmi" in catalog:
        pack = catalog["bmi"]
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
    if "hattack_ever_w10" in chosen and "hattack_ever_w10" in catalog:
        pack_h = catalog["hattack_ever_w10"]
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

    st.caption("Note: Models and outputs here are illustrative. Replace synthetic data and toy DGP with your real pipeline.")

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


# --------------------------- JSON View ---------------------------
def raw_json_view(cfg: Dict[str, Any]) -> None:
    st.subheader("🧾 Pipeline JSON (decisions so far)")
    st.json(build_pipeline_json(), expanded=False)
    if "raw_uploaded_text" in st.session_state:
        with st.expander("Original uploaded text", expanded=False):
            st.code(st.session_state["raw_uploaded_text"], language="json")
    if "cleaned_uploaded_text" in st.session_state:
        with st.expander("Cleaned JSON used for parsing", expanded=False):
            st.code(st.session_state["cleaned_uploaded_text"], language="json")


def build_pipeline_json() -> Dict[str, Any]:
    scenario_name = st.session_state.get("scenario_name", "Demo Scenario")
    selection = st.session_state.get("chosen_models", {})
    interventions = st.session_state.get("interventions", [])
    catalog_meta = st.session_state.get("catalog_meta", {})

    return {
        "meta": {"name": scenario_name},
        "pipeline": {
            "catalog": catalog_meta,     # metrics & specs for pre-trained models
            "selection": selection,      # chosen models per target
            "interventions": interventions,
        },
    }


# --------------------------- (Commented Out) Model Estimation Tab ---------------------------
# NOTE: Per request, the Model Estimation page has been removed from the UI.
# The original estimation workflow is preserved below as commented code,
# so you can restore it later if needed. To re-enable:
#   1) Uncomment the function body.
#   2) Add the tab back in `main()`.
#
# def model_estimation_tab():
#     \"\"\"
#     Original model estimation UI (target, feature engineering, algorithms,
#     record configurations, and train-all). It depended on user-driven training.
#     We now rely on a pre-trained catalog instead.
#     \"\"\"
#     pass


# --------------------------- App Entry ---------------------------
def main():
    st.title("📈 Economic Simulation — Prototype (Model Selection Demo)")
    st.caption(
        "Choose among pre-trained models per variable, configure multiple interventions, "
        "simulate outcomes, and export the complete pipeline JSON (selections + interventions + catalog metadata)."
    )

    cfg = sidebar_controls(default_config())

    # Ensure training data exists and build the catalog once
    if "training_data" not in st.session_state:
        st.session_state["training_data"] = generate_synthetic_population(
            st.session_state.get("pop_n", 30_000), st.session_state.get("seed", 42)
        )
    build_catalog_models(st.session_state["training_data"], force=False)

    # Tabs (Model Estimation removed; code commented above for recovery)
    tabs = st.tabs([
        # "1) Model selection", "2) Interventions", "3) Run & Results", "4) JSON"
        "1) Model selection", "2) Interventions", "3) Run & Results"
    ])
    with tabs[0]:
        model_selection_tab()
    with tabs[1]:
        interventions_tab()
    with tabs[2]:
        run_and_visualize(cfg)
    # with tabs[3]:
    #     raw_json_view(cfg)

    st.divider()
    st.markdown("Built for demonstration • Replace synthetic data and toy DGP with your pipeline.")

if __name__ == "__main__":
    if "config" not in st.session_state:
        st.session_state["config"] = default_config()
    main()
