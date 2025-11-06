
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px

# Reproducibility (will be set once, perhaps in session_state or app initialization)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

@st.cache_data
def generate_llm_data(num_samples):
    """Generates a synthetic dataset simulating LLM interactions."""
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    
    columns = [
        'timestamp', 'prompt', 'llm_output', 'true_label', 
        'model_confidence', 'model_accuracy', 
        'explanation_quality_score', 'faithfulness_metric', 'xai_technique'
    ]
    
    data = {
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(num_samples)],
        'prompt': [f"Prompt {i+1} about a safety concern." for i in range(num_samples)],
        'llm_output': [f"LLM output for safety concern {i+1}." for i in range(num_samples)],
        'true_label': [random.choice(['Safe', 'Unsafe', 'Ambiguous']) for _ in range(num_samples)],
        'model_confidence': [random.uniform(0.5, 1.0) for _ in range(num_samples)],
        'model_accuracy': [random.choice([0, 1]) for _ in range(num_samples)],
        'explanation_quality_score': [random.uniform(0.0, 1.0) for _ in range(num_samples)],
        'faithfulness_metric': [random.uniform(0.0, 1.0) for _ in range(num_samples)],
        'xai_technique': [random.choice(['LIME', 'SHAP', 'GradCAM']) for _ in range(num_samples)],
    }
    return pd.DataFrame(data, columns=columns)

def validate_and_summarize_data(dataframe: pd.DataFrame):
    """
    Performs basic validation and summarizes the dataset.
    - Checks presence of expected columns
    - Validates data types for critical numeric fields
    - Checks missing values in critical fields
    - Prints descriptive statistics and categorical distributions (adapted for Streamlit)
    """
    st.subheader("Data Validation & Summary Statistics")
    expected_columns = {
        'timestamp', 'prompt', 'llm_output', 'true_label',
        'model_confidence', 'model_accuracy', 'explanation_quality_score',
        'faithfulness_metric', 'xai_technique'
    }

    missing = expected_columns - set(dataframe.columns)
    if missing:
        st.warning(f"**[Warning] Missing expected columns: {missing}**")
    else:
        st.success("All expected columns are present.")

    # Validate dtypes for critical numeric fields
    critical_numeric = ['model_confidence', 'explanation_quality_score', 'faithfulness_metric', 'model_accuracy']
    for col in critical_numeric:
        if col in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                st.warning(f"**[Warning] Column '{col}' is not numeric.**")
        else:
            st.warning(f"**[Warning] Column '{col}' not found for numeric validation.**")

    # Missing value checks
    na_counts_critical = dataframe[critical_numeric].isna().sum()
    if na_counts_critical.sum() == 0:
        st.info("No missing values found in critical numeric fields.")
    else:
        st.warning(f"**[Warning] Missing values in critical numeric fields:\n{na_counts_critical.to_string()}**")

    # Descriptive statistics
    st.markdown("### Descriptive statistics for numeric columns:")
    st.dataframe(dataframe.select_dtypes(include='number').describe().T)

    # Value counts for key categoricals
    for col in ['true_label', 'xai_technique']:
        if col in dataframe.columns:
            st.markdown(f"### Value counts for `{col}`:")
            st.dataframe(dataframe[col].value_counts())

def run_page1(filtered_df):
    st.header("Page 1: Introduction & Data Setup")

    st.markdown(r"""
    ## Executive Summary
    This application provides a compact, end-to-end, business-oriented introduction to Explainable AI (XAI) for Large Language Models (LLMs) using synthetic data and lightweight visualizations. It focuses on the context of **Agentic AI for Safety Monitoring**, where understanding model behavior is crucial.

    What you will learn and do:
    - Understand core XAI concepts for LLMs: interpretability vs transparency and why it matters for risk, trust, and governance in safety-critical applications.
    - Generate a realistic synthetic dataset of LLM interactions to explore XAI signals at scale.
    - Simulate and visualize two cornerstone XAI techniques: saliency maps and counterfactuals (in subsequent pages).
    - Analyze faithfulness over time and the trade-off between explanation quality and model accuracy (in subsequent pages).
    - Apply practical filtering by explanation verbosity and model confidence to focus reviews on high-value cases.

    Constraints honored:
    - Runs on a mid-spec laptop (<5 minutes) with open-source libraries only.
    - All plots use color-blind-friendly palettes with legible fonts and PNG fallbacks.
    - Each step includes a short narrative and code comments explaining what and why.
    """)

    st.markdown(r"""
    ## Data and Inputs Overview

    -   **Inputs**: This application uses fully synthetic data by default to simulate LLM prompts, outputs, and XAI-related metrics (confidence, accuracy, explanation quality, faithfulness, technique label). An option to upload custom data will also be provided.
    -   **Why synthetic**: It allows us to isolate explainability concepts while ensuring fast, reproducible execution on a laptop. We avoid the cost and variability of running real LLMs.
    -   **Business value**: Teams can prototype review workflows (e.g., trust & safety, model governance) without waiting on infrastructure; results translate directly to production telemetry schemas.
    -   **Assumptions**: Scores are in $[0, 1]$ and reflect conceptual tendencies (not model internals). Timestamps are uniformly spread to illustrate trend analyses. Technique labels are categorical tags used for stratified monitoring.

    All code executes locally and uses only open-source packages.
    """)

    st.markdown(r"""
    ## Methodology Overview

    We simulate a lightweight analytics pipeline for explainable LLM behavior using synthetic data. The pipeline mirrors a typical model governance workflow relevant for safety monitoring:

    1)  Generate interaction telemetry: prompts, outputs, confidence, accuracy, explanation quality, faithfulness, technique tags.
    2)  Attach token-level saliency signals (synthetic) and visualize local explanations.
    3)  Create counterfactuals to reason about minimal input changes and outcome shifts.
    4)  Monitor trends (faithfulness over time) and trade-offs (explanation quality vs. accuracy).
    5)  Filter by verbosity and confidence to focus human review on the most impactful cases.

    Key formulae and rationale:
    -   Saliency importance (conceptual): $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$ captures how sensitive the model output $Y$ is to small changes in token $x_i$. High $|\partial Y/\partial x_i|$ implies stronger influence, guiding auditors to critical tokens.
    -   Counterfactual minimal change: Find $X' = X + \Delta X$ such that $\text{Model}(X') = Y' \ne Y$ and $||\Delta X||$ is minimal. This supports “what-if” analysis for decision recourse and fairness reviews.
    -   Accuracy (simulated binary): $$ A = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i] $$ Even when simplified, plotting $A$ against explanation quality $Q_{exp}$ surfaces potential trade-offs.

    Business linkage:
    -   Explainability reduces operational risk (model audits), accelerates root-cause analysis (incident response), and improves user trust (comms & compliance). The visuals we produce are the backbone of explainability reporting in production dashboards.
    """)

    st.markdown(r"""
    ## 3.1 Introduction to Explainable AI (XAI)

    Explainable AI (XAI) aims to make model behavior understandable to humans. For Large Language Models (LLMs), explanations help stakeholders assess risks, justify decisions, and comply with governance, especially in safety-critical domains.

    -   **Interpretability**: understanding why a specific decision was made. Focuses on the relationship between inputs and outputs.
    -   **Transparency**: understanding how the model works internally (architecture, parameters, training). For LLMs, full transparency is often infeasible due to scale.

    In practice, we treat LLMs as black boxes and use interpretability techniques (e.g., saliency, counterfactuals) to reason about outputs and calibrate trust. This is particularly important for safety monitoring, where unexpected or harmful behaviors must be quickly identified and understood.
    """)

    st.markdown(r"""
    ## 3.3 Overview of Synthetic Data Generation

    We will generate synthetic LLM interaction data to safely illustrate XAI concepts without requiring a live model. This provides a controlled, fast-to-run sandbox that mirrors telemetry you might collect in production (prompts, outputs, confidence, accuracy, explanation quality, faithfulness, technique labels).

    Business relevance for safety monitoring:
    -   Enables rapid prototyping of monitoring and governance reports specifically for safety-related incidents.
    -   Reduces cost and risk while designing explainability workflows for safety interventions.
    -   Produces reproducible examples to train reviewers and align stakeholders on interpreting safety-critical explanations.
    """)
    
    st.markdown(r"""
    ## 3.4 Generating Core Synthetic LLM Interaction Data — Context & Business Value
    We need a realistic yet lightweight dataset that mirrors what product analytics and governance teams monitor: prompts, outputs, confidence, accuracy, explanation quality, faithfulness, and technique labels. This enables:
    -   Rapid stress-testing of explainability reports before production data pipelines exist.
    -   Training analysts on how to interpret explainability signals.
    -   Evaluating trade-offs between accuracy and explanation quality.

    Formulae (for reference):
    -   Simulated accuracy per row: $A_i \sim \text{Bernoulli}(p = c_i)$, where $c_i$ is model confidence for record i. Aggregate accuracy: $A_{model} = \frac{1}{N}\sum_i A_i$.
    -   Scores range in $[0,1]$ and are constructed to be plausible, not derived from any real model internals.
    """)

    st.subheader("Synthetic Data Generation Controls")
    data_source = st.radio(
        "Data Source Selection",
        ("Generate Synthetic Data", "Upload Custom Data (CSV)"),
        help="Choose whether to generate synthetic LLM interaction data or upload your own CSV file."
    )

    if data_source == "Generate Synthetic Data":
        num_samples = st.slider(
            "Number of Synthetic LLM Interactions",
            100, 5000, 500,
            help="Adjust the number of simulated LLM interactions for the dataset."
        )
        if st.button("Generate Data", help="Click to generate the synthetic LLM interaction dataset."):
            with st.spinner("Generating synthetic data..."):
                st.session_state['df_llm_data'] = generate_llm_data(num_samples)
                st.success(f"Generated {num_samples} synthetic LLM interactions.")

    elif data_source == "Upload Custom Data (CSV)":
        uploaded_file = st.file_uploader(
            "Upload your LLM interaction data (CSV)",
            type=["csv"],
            help="Upload a CSV file containing LLM interaction data. The file should have columns like 'timestamp', 'prompt', 'llm_output', 'model_confidence', etc.",
            accept_multiple_files=False
        )
        if uploaded_file is not None:
            with st.spinner("Uploading and processing data..."):
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    st.session_state['df_llm_data'] = df_uploaded
                    st.success("Custom data uploaded successfully!")
                except Exception as e:
                    st.error(f"Error uploading file: {e}")

    if not st.session_state['df_llm_data'].empty:
        st.markdown("### Initial Inspection of Generated/Uploaded Data")
        st.write(f"Shape of the dataset: {st.session_state['df_llm_data'].shape}")
        st.dataframe(st.session_state['df_llm_data'].head())

        st.markdown(r"""
        ## 3.6 Explanation of the Synthetic Dataset
        The generated dataset (`df_llm_data`) mirrors common telemetry fields used in model monitoring:
        -   `timestamp`: when the interaction occurred; supports trend analysis and seasonality checks.
        -   `prompt`: input to the LLM; useful for content stratification and policy audits.
        -   `llm_output`: model response; used for local explanations (e.g., saliency at the token level).
        -   `true_label`: a synthetic categorical label to simulate downstream evaluation.
        -   `model_confidence` in $[0,1]$: proxy for model’s certainty; often used to route human review.
        -   `model_accuracy` $\in \{0,1\}$: simulated correctness flag; enables outcome-level reporting and trade-off studies.
        -   `explanation_quality_score` in $[0,1]$: proxy for how coherent/useful an explanation is.
        -   `faithfulness_metric` in $[0,1]$: conceptual alignment between explanations and model behavior.
        -   `xai_technique`: categorical tag for explanations (e.g., LIME/SHAP/GradCAM in this synthetic example).

        Business takeaway: These fields are enough to build explainability dashboards, route triage by confidence, and demonstrate governance controls without access to proprietary data or live models.
        """)

        st.markdown(r"""
        ## Optional: Quick Sanity Check Metric (Accuracy)
        As a simple governance check, we can compare a heuristic “predicted correctness” (e.g., confidence $\ge 0.75$) to the simulated binary accuracy flag via accuracy score. This is not a model evaluation, just a diagnostic to illustrate how monitoring hooks into basic metrics.

        Formula:
        -   Heuristic correctness: $ \hat{c}_i = \mathbb{1}[\text{confidence}_i \ge 0.75] $
        -   Accuracy vs. label: $ A = \frac{1}{N} \sum_i \mathbb{1}[\hat{c}_i = \text{model\_accuracy}_i] $
        """)

        heuristic_threshold = st.slider(
            "Heuristic Confidence Threshold for Accuracy",
            0.0, 1.0, 0.75, 0.05,
            help="Define the confidence threshold to predict correctness for the sanity check."
        )

        heuristic_correct = (st.session_state['df_llm_data']['model_confidence'] >= heuristic_threshold).astype(int)
        match_rate = (heuristic_correct == st.session_state['df_llm_data']['model_accuracy']).mean()
        st.info(f"Heuristic (confidence $\ge$ {heuristic_threshold}) vs. simulated accuracy agreement: **{match_rate:.3f}**")

        st.markdown(r"""
        ## Optional Result Interpretation: Heuristic Agreement
        The printed agreement rate shows how often a simple confidence-based heuristic aligns with the simulated accuracy flag. A higher value suggests confidence is a reasonable proxy for correctness; a lower value warns that confidence alone may be misleading. In governance, such diagnostics inform threshold setting for human-in-the-loop review.
        """)

        st.markdown("---")
        st.markdown(r"""
        ## 3.9 Data Validation and Summary Statistics — Executable Code
        After data generation or upload, it's crucial to validate the data structure and summarize its key characteristics. This step ensures data quality and provides a quick overview of the dataset's contents before proceeding with XAI analysis.
        """)
        validate_and_summarize_data(st.session_state['df_llm_data'])
    else:
        st.info("Please generate or upload data to proceed.")

