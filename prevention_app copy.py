# app.py
# -------------------------------------------------------------
# Economic Simulation Prototype (Streamlit)
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
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

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
        # Streamlit uploader sometimes returns a BytesIO-like object with .read()
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
    This is only to improve UX when users provide illustrative JSON with ellipses.
    """
    if not isinstance(text, str):
        return text

    # Remove // comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    # Remove /* block comments */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Remove key: value where key is "..."
    text = re.sub(r'\s*"\\.{3}"\s*:\s*(".*?"|\{.*?\}|\[.*?\]|true|false|null|-?\d+\.?\d*)\s*,?', "", text, flags=re.DOTALL)

    # Remove standalone "..." entries in arrays (with optional trailing comma)
    text = re.sub(r'\s*"\\.{3}"\s*,?', "", text)

    # Remove dangling commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()


def load_config_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Attempt to parse JSON, with a tolerance for illustrative 'loose' JSON.
    Returns (config_dict, cleaned_text). Raises ValueError on failure.
    """
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
    A tiny default config aligned with the user's structure.
    Replace or extend as needed. This is used if no file is uploaded.
    """
    return {
        "data": "ukhls",
        "model_config": {
            "bmi": {
                "model_type": "linear_regression",
                "dataset": "innovation_panel",
                "independent_vars": ["log_income", "age"],
                "train_test_split": {
                    "test_size": 0.2,
                    "stratify": "employment_status",
                    "random_state": 35,
                },
            },
            "hattack_ever_w10": {
                "model_type": "logistic_regression",
                "dataset": "ukhls",
                "independent_vars": ["bmi", "age"],
                "train_test_split": {
                    "test_size": 0.2,
                    "stratify": "h_hattack_hcondever_w10",
                    "random_state": 35,
                },
            },
        },
        "intervention_config": {
            "bmi": {
                "type": "percentage_decrease",
                "amount": 0.2,
                "target_population": {
                    "age": [40, 60],
                    "bmi": [30, 100],
                },
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


# --------------------------- Synthetic Data + Model Stubs ---------------------------
def generate_synthetic_population(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic micro-population with a few relevant variables.
    This is *illustrative only* — not a real dataset.
    """
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 90, size=n)

    # Log-income (approx log-normal) varies with age a bit
    # We'll create a base log income, then add small age trend
    base_log_income = rng.normal(loc=10.5, scale=0.6, size=n)  # ~ exp -> median ~$36k
    age_effect = (ages - 45) / 45 * 0.15
    log_income = base_log_income + age_effect

    # Employment status categorical 'employed'/'other'
    employed = rng.random(size=n) < (0.7 - 0.002 * np.clip(ages - 40, 0, None))
    employment_status = np.where(employed, "employed", "other")

    # BMI: higher at older ages on average, with noise
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


def apply_bmi_intervention(df: pd.DataFrame, intervention: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a 'percentage_decrease' intervention on BMI for a target population.
    This modifies BMI only for rows matching the target filters.
    """
    new_df = df.copy()
    if not intervention or intervention.get("type") != "percentage_decrease":
        return new_df

    amount = float(intervention.get("amount", 0.0))  # e.g., 0.2 for 20%
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


def risk_heart_attack(prob_inputs: pd.DataFrame) -> pd.Series:
    """
    Illustrative probability of *ever* having a heart attack (not clinical).
    Uses BMI and Age with placeholder coefficients to create a plausible baseline.
    logit(p) = a + b1*((bmi-25)/5) + b2*((age-50)/10)
    Tuned to give ~10-15% overall prevalence in the synthetic data.
    """
    bmi = prob_inputs["bmi"].to_numpy()
    age = prob_inputs["age"].to_numpy()

    a = -4.2  # intercept
    b1 = 0.50  # bmi effect
    b2 = 0.65  # age effect

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


# --------------------------- Streamlit UI Builders ---------------------------
def sidebar_config_controls(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Sidebar for scenario-level controls and upload/download."""
    st.sidebar.header("⚙️ Scenario & Configuration")

    st.sidebar.text_input("Scenario name", key="scenario_name", value=st.session_state.get("scenario_name", "Demo Scenario"))

    st.sidebar.caption("Upload a configuration JSON")
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

    # Download current config
    dl = st.sidebar.download_button(
        label="💾 Download current config",
        data=json.dumps(st.session_state.get("config", cfg), indent=2),
        file_name="scenario_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.sidebar.divider()
    st.sidebar.caption("Simulation Settings")
    pop_n = st.sidebar.number_input("Population size (synthetic)", min_value=1000, max_value=1_000_000, value=50_000, step=1000)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

    st.session_state["pop_n"] = int(pop_n)
    st.session_state["seed"] = int(seed)

    return st.session_state.get("config", cfg)


def model_overview(cfg: Dict[str, Any]) -> None:
    st.subheader("📚 Model Configuration")

    mc = cfg.get("model_config", {}) or {}
    if not mc:
        st.warning("No `model_config` found in your JSON.")
        return

    model_names = sorted(mc.keys())
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = model_names[0]

    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Models**")
            choice = st.radio(
                "Select a model",
                model_names,
                index=model_names.index(st.session_state["selected_model"]),
                label_visibility="collapsed",
            )
            st.session_state["selected_model"] = choice

        with col2:
            m = mc.get(st.session_state["selected_model"], {})
            st.markdown(f"**Selected Model:** `{st.session_state['selected_model']}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Type", m.get("model_type", "—"))
            c2.metric("Dataset", m.get("dataset", "—"))
            c3.metric("Features", len(m.get("independent_vars", [])))

            with st.expander("View / Edit training split", expanded=False):
                tts = (m.get("train_test_split", {}) or {})
                new_test_size = st.slider("Test size", 0.05, 0.95, float(tts.get("test_size", 0.2)), 0.05)
                new_random_state = st.number_input("Random state", value=int(tts.get("random_state", 42)), step=1)
                new_stratify = st.text_input("Stratify column", value=str(tts.get("stratify", "")))
                # Persist changes in session only (for demo)
                m.setdefault("train_test_split", {})
                m["train_test_split"]["test_size"] = float(new_test_size)
                m["train_test_split"]["random_state"] = int(new_random_state)
                m["train_test_split"]["stratify"] = new_stratify

            with st.expander("View / Edit feature list", expanded=False):
                feats = m.get("independent_vars", [])
                feats_txt = st.text_area("Independent variables (comma-separated)", value=", ".join(map(str, feats)))
                feats_new = [f.strip() for f in feats_txt.split(",") if f.strip()]
                m["independent_vars"] = feats_new

            # Save back
            cfg["model_config"][st.session_state["selected_model"]] = m


def intervention_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.subheader("🧪 Intervention Configuration")

    ic = cfg.get("intervention_config", {}) or {}

    # Only one example intervention (bmi) is provided in the illustrative JSON,
    # but we render this generically and allow editing.
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


def run_and_visualize(cfg: Dict[str, Any]) -> None:
    st.subheader("🚀 Run Simulation")

    pop_n = int(st.session_state.get("pop_n", 50_000))
    seed = int(st.session_state.get("seed", 42))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Click to generate a synthetic baseline population and apply the configured interventions.")
        go = st.button("Run Simulation", type="primary", use_container_width=True)
    with c2:
        st.caption("Export results or share a quick snapshot.")
        export_name = st.text_input("Export name", value="demo_run")

    if not go:
        st.info("Configure your models and interventions, then click **Run Simulation**.")
        return

    # Generate baseline data
    df_base = generate_synthetic_population(pop_n, seed=seed)
    df_base["prob_hattack"] = risk_heart_attack(df_base)
    df_base["hattack_event"] = (np.random.default_rng(seed + 1).random(size=len(df_base)) < df_base["prob_hattack"]).astype(int)

    # Apply interventions
    df_post = df_base.copy()
    ic = cfg.get("intervention_config", {}) or {}
    for key, inv in ic.items():
        if key == "bmi":
            df_post = apply_bmi_intervention(df_post, inv)

    # Recompute probabilities after interventions (only BMI affects our toy risk here)
    df_post["prob_hattack"] = risk_heart_attack(df_post)
    df_post["hattack_event"] = (np.random.default_rng(seed + 2).random(size=len(df_post)) < df_post["prob_hattack"]).astype(int)

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
        # Create tidy data for Altair
        b0 = df_base[["bmi"]].copy()
        b0["state"] = "Baseline"
        b1 = df_post[["bmi"]].copy()
        b1["state"] = "Post"
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

    st.caption("Note: All models, coefficients, and outputs here are illustrative. Replace with your trained models and data pipeline.")

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


def raw_json_view(cfg: Dict[str, Any]) -> None:
    st.subheader("🧾 Raw JSON (current in-session config)")
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
        "Interactive mock-up that reads your configuration JSON, lets you edit interventions, "
        "and runs an illustrative synthetic simulation to showcase the proposed web app."
    )

    # Sidebar controls and (optional) file upload
    cfg = sidebar_config_controls(default_config())

    # Tabs for flow
    t1, t2, t3, t4 = st.tabs(["1) Model setup", "2) Interventions", "3) Run & Results", "4) JSON"])
    with t1:
        model_overview(cfg)
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
    if "config" not in st.session_state:
        st.session_state["config"] = default_config()
    main()
