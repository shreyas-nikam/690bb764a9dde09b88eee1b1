
# Streamlit Application Requirements Specification: Explainable AI for LLMs

## 1. Application Overview

This Streamlit application will provide an interactive, end-to-end, business-oriented introduction to Explainable AI (XAI) for Large Language Models (LLMs) within the context of **Agentic AI for Safety Monitoring**. It leverages synthetic data and lightweight visualizations to demonstrate core XAI concepts and practical application.

### Learning Goals

The application is designed to help users:
- Understand key XAI concepts for LLMs, including interpretability vs. transparency, and their relevance for risk, trust, and governance.
- Generate realistic synthetic datasets of LLM interactions to explore XAI signals at scale.
- Simulate and visualize cornerstone XAI techniques: saliency maps and counterfactuals.
- Analyze faithfulness over time and understand the trade-off between explanation quality and model accuracy.
- Apply practical filtering mechanisms (e.g., by explanation verbosity and model confidence) to focus reviews on high-value cases.
- Design testing and validation for adaptive systems and apply explainability frameworks to LLMs.
- Understand the importance of XAI for AI-security threats and implementing defenses in agentic systems.

## 2. User Interface Requirements

The application will feature a clear, intuitive layout with a sidebar for global controls and a main content area for narrative, data display, and visualizations.

### Layout and Navigation Structure

-   **Sidebar (`st.sidebar`):** Will host global controls such as data generation parameters, data upload, and filtering options.
-   **Main Content Area:** Structured sequentially to guide the user through the XAI concepts, data generation, validation, technique application, and visualization.
    -   **Introduction:** Overview of XAI for LLMs.
    -   **Data Setup:** Synthetic data generation or upload.
    -   **Data Validation & Overview:** Display of generated/uploaded data, validation, and summary statistics.
    -   **XAI Concepts:** Explanation of interpretability vs. transparency, saliency maps, and counterfactuals.
    -   **Saliency Map Simulation:** Interactive visualization of token saliency.
    -   **Counterfactual Explanation:** Interactive demonstration of counterfactual generation.
    -   **Trend & Trade-off Analysis:** Visualizations of faithfulness over time and explanation quality vs. model accuracy.
    -   **Filtering & Review Focus:** Controls for filtering data based on XAI metrics.

### Input Widgets and Controls

1.  **Data Source Selection (`st.radio` or `st.selectbox`):**
    -   Option 1: Generate Synthetic Data.
    -   Option 2: Upload Custom Data (CSV).
    -   *Help Text:* "Choose whether to generate synthetic LLM interaction data or upload your own CSV file."

2.  **Synthetic Data Generation Parameters:**
    -   **Number of Samples (`st.slider`):**
        -   Label: "Number of Synthetic LLM Interactions"
        -   Range: 100 to 5000 (default 500)
        -   *Help Text:* "Adjust the number of simulated LLM interactions for the dataset."
    -   **Generate Data Button (`st.button`):**
        -   Label: "Generate Data"
        -   *Help Text:* "Click to generate the synthetic LLM interaction dataset."

3.  **Data Upload (`st.file_uploader`):**
    -   Label: "Upload your LLM interaction data (CSV)"
    -   Accepted types: `['csv']`
    -   Max size: 5 MB
    -   *Help Text:* "Upload a CSV file containing LLM interaction data. The file should have columns like 'timestamp', 'prompt', 'llm_output', 'model_confidence', etc."

4.  **Saliency Map Controls:**
    -   **Sample Output Index (`st.number_input` or `st.selectbox`):**
        -   Label: "Select Output for Saliency Map"
        -   Range: 0 to `len(df_llm_data)-1`
        -   Default: 0
        -   *Help Text:* "Choose an index from the dataset to visualize its saliency map."
    -   **Saliency Threshold (`st.slider`):**
        -   Label: "Saliency Highlight Threshold $(\\tau)$"
        -   Range: 0.0 to 1.0 (step 0.05, default 0.7)
        -   *Help Text:* "Tokens with saliency scores at or above this threshold will be highlighted."

5.  **Counterfactual Explanation Controls:**
    -   **Sample Record Index (`st.number_input` or `st.selectbox`):**
        -   Label: "Select Record for Counterfactuals"
        -   Range: 0 to `len(df_llm_data)-1`
        -   Default: 0
        -   *Help Text:* "Choose an index to generate a counterfactual explanation for its prompt and output."

6.  **Filtering Controls (Sidebar):**
    -   **Minimum Explanation Quality (`st.slider`):**
        -   Label: "Min Explanation Quality Score"
        -   Range: 0.0 to 1.0 (step 0.01, default 0.0)
        -   *Help Text:* "Filter to show only records with explanation quality scores above this value."
    -   **Minimum Model Confidence (`st.slider`):**
        -   Label: "Min Model Confidence"
        -   Range: 0.0 to 1.0 (step 0.01, default 0.0)
        -   *Help Text:* "Filter to show only records with model confidence scores above this value."
    -   **XAI Technique Filter (`st.multiselect`):**
        -   Label: "Filter by XAI Technique"
        -   Options: Unique values from `df_llm_data['xai_technique']`
        -   Default: All selected
        -   *Help Text:* "Select which XAI techniques to include in the analysis and visualizations."

7.  **Heuristic Sanity Check Threshold (`st.slider`):**
    -   Label: "Heuristic Confidence Threshold for Accuracy"
    -   Range: 0.0 to 1.0 (step 0.05, default 0.75)
    -   *Help Text:* "Define the confidence threshold to predict correctness for the sanity check."

### Visualization Components

1.  **Data Tables (`st.dataframe`):**
    -   Display head of the `df_llm_data` dataframe.
    -   Display head of the `df_saliency` dataframe.
    -   Display descriptive statistics and value counts from `validate_and_summarize_data`.
2.  **Saliency Map Output (`st.markdown(unsafe_allow_html=True)`):**
    -   Render the HTML output from `visualize_saliency_map` to show highlighted tokens.
3.  **Counterfactual Explanation Output (`st.write` or `st.markdown`):**
    -   Display the original prompt, original output, counterfactual prompt, and counterfactual output in a clear format.
4.  **Faithfulness Metric Trend Plot (`st.pyplot`):**
    -   **Type:** Line plot (`sns.lineplot`).
    -   **Data:** `df_llm_data` (filtered).
    -   **X-axis:** `timestamp`.
    -   **Y-axis:** `faithfulness_metric`.
    -   **Hue:** `xai_technique`.
    -   **Title:** "Faithfulness Metric over Time by XAI Technique".
    -   **Style:** Color-blind-friendly palette, font size $\ge 12$ pt, clear labels and legend.
5.  **Relationship Plot (`st.pyplot`):**
    -   **Type:** Scatter plot (`sns.scatterplot`).
    -   **Data:** `df_llm_data` (filtered).
    -   **X-axis:** `model_accuracy` (or average accuracy per time bucket for trend).
    -   **Y-axis:** `explanation_quality_score`.
    -   **Title:** "Explanation Quality vs. Model Accuracy".
    -   **Style:** Color-blind-friendly palette, font size $\ge 12$ pt, clear labels.
6.  **Aggregated Comparison Plot (`st.pyplot`):**
    -   **Type:** Bar chart (`sns.barplot`).
    -   **Data:** `df_llm_data` (grouped by `xai_technique`, showing average `faithfulness_metric`).
    -   **X-axis:** `xai_technique`.
    -   **Y-axis:** Average `faithfulness_metric`.
    -   **Title:** "Average Faithfulness by XAI Technique".
    -   **Style:** Color-blind-friendly palette, font size $\ge 12$ pt, clear labels.

### Interactive Elements and Feedback Mechanisms

-   All plots will be interactive where Streamlit allows (e.g., via `altair` or by using `plotly.express` for more native interactivity if `st.pyplot` is insufficient). For Matplotlib/Seaborn, a static image will be displayed with `st.pyplot`.
-   Validation messages, descriptive statistics, and heuristic agreement rates will be displayed dynamically using `st.write` or `st.info`.
-   Loading spinners (`st.spinner`) will be used during long computations like data generation.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

-   Every input widget will have a descriptive `help` parameter provided, explaining its purpose and potential impact on the analysis.
-   Key concepts (e.g., Interpretability, Transparency, Saliency, Counterfactuals, Faithfulness) will be introduced with clear `st.markdown` sections providing context and definitions, often including mathematical formulae.
-   Visualizations will have clear titles, labeled axes, and legends as specified in the `plt.rcParams` setup in the notebook content.

### Save the States of the Fields Properly

-   `st.session_state` will be extensively used to preserve the state of all user inputs (e.g., `num_samples`, `saliency_threshold`, selected indices, filters) across reruns, ensuring changes are not lost when the script re-executes.
-   Generated dataframes (`df_llm_data`, `df_saliency`) will also be stored in `st.session_state` to avoid regenerating them unnecessarily.

## 4. Notebook Content and Code Requirements

This section outlines the direct integration of content and code from the Jupyter notebook into the Streamlit application.

### Environment Setup & Reproducibility

```python
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker # Not directly used in Streamlit app for synthetic data due to simpler generation
from datetime import datetime, timedelta
# IPython.display is not directly used in Streamlit, will be replaced by st.markdown(unsafe_allow_html=True)
from sklearn.metrics import accuracy_score
import streamlit as st # Streamlit import

# Reproducibility (will be set once, perhaps in session_state or app initialization)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Plotting style: color-blind-friendly and legible (applied globally)
sns.set_theme(style='whitegrid', palette='cividis', context='talk')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# Pandas display options for compact readability (less critical for Streamlit display, but good practice)
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 120)
```

### Application Title and Executive Summary

```python
st.title("Explainable AI (XAI) for LLMs: A Practical, Executable Walkthrough")

st.markdown("""
## Executive Summary
This application provides a compact, end-to-end, business-oriented introduction to Explainable AI (XAI) for Large Language Models (LLMs) using synthetic data and lightweight visualizations.

What you will learn and do:
- Understand core XAI concepts for LLMs: interpretability vs transparency and why it matters for risk, trust, and governance.
- Generate a realistic synthetic dataset of LLM interactions to explore XAI signals at scale.
- Simulate and visualize two cornerstone XAI techniques: saliency maps and counterfactuals.
- Analyze faithfulness over time and the trade-off between explanation quality and model accuracy.
- Apply practical filtering by explanation verbosity and model confidence to focus reviews on high-value cases.

Constraints honored:
- Runs on a mid-spec laptop (<5 minutes) with open-source libraries only.
- All plots use color-blind-friendly palettes with legible fonts and PNG fallbacks.
- Each step includes a short narrative and code comments explaining what and why.
""")
```

### Data and Inputs Overview

```python
st.markdown("""
## Data and Inputs Overview

-   **Inputs**: This application uses fully synthetic data by default to simulate LLM prompts, outputs, and XAI-related metrics (confidence, accuracy, explanation quality, faithfulness, technique label). An option to upload custom data will also be provided. No external data or API keys are required.
-   **Why synthetic**: It allows us to isolate explainability concepts while ensuring fast, reproducible execution on a laptop. We avoid the cost and variability of running real LLMs.
-   **Business value**: Teams can prototype review workflows (e.g., trust & safety, model governance) without waiting on infrastructure; results translate directly to production telemetry schemas.
-   **Assumptions**:
    -   Scores are in $[0, 1]$ and reflect conceptual tendencies (not model internals).
    -   Timestamps are uniformly spread to illustrate trend analyses.
    -   Technique labels are categorical tags used for stratified monitoring.

All code executes locally in under five minutes and uses only open-source packages.
""")
```

### Methodology Overview

```python
st.markdown("""
## Methodology Overview

We simulate a lightweight analytics pipeline for explainable LLM behavior using synthetic data. The pipeline mirrors a typical model governance workflow:

1)  Generate interaction telemetry: prompts, outputs, confidence, accuracy, explanation quality, faithfulness, technique tags.
2)  Attach token-level saliency signals (synthetic) and visualize local explanations.
3)  Create counterfactuals to reason about minimal input changes and outcome shifts.
4)  Monitor trends (faithfulness over time) and trade-offs (explanation quality vs. accuracy).
5)  Filter by verbosity and confidence to focus human review on the most impactful cases.

Key formulae and rationale:
-   Saliency importance (conceptual): $$ S(x_i) = \\left| \\frac{\\partial Y}{\\partial x_i} \\right| $$ captures how sensitive the model output $Y$ is to small changes in token $x_i$. High $|\\partial Y/\\partial x_i|$ implies stronger influence, guiding auditors to critical tokens.
-   Counterfactual minimal change: Find $X' = X + \\Delta X$ such that $\\text{Model}(X') = Y' \\ne Y$ and $||\\Delta X||$ is minimal. This supports “what-if” analysis for decision recourse and fairness reviews.
-   Accuracy (simulated binary): $$ A = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{1}[\\hat{y}_i = y_i] $$ Even when simplified, plotting $A$ against explanation quality $Q_{exp}$ surfaces potential trade-offs.

Business linkage:
-   Explainability reduces operational risk (model audits), accelerates root-cause analysis (incident response), and improves user trust (comms & compliance). The visuals we produce are the backbone of explainability reporting in production dashboards.
""")
```

### 3.1 Introduction to Explainable AI (XAI)

```python
st.markdown("""
## 3.1 Introduction to Explainable AI (XAI)

Explainable AI (XAI) aims to make model behavior understandable to humans. For Large Language Models (LLMs), explanations help stakeholders assess risks, justify decisions, and comply with governance.

-   **Interpretability**: understanding why a specific decision was made. Focuses on the relationship between inputs and outputs.
-   **Transparency**: understanding how the model works internally (architecture, parameters, training). For LLMs, full transparency is often infeasible due to scale.

In practice, we treat LLMs as black boxes and use interpretability techniques (e.g., saliency, counterfactuals) to reason about outputs and calibrate trust.
""")
```

### 3.3 Overview of Synthetic Data Generation

```python
st.markdown("""
## 3.3 Overview of Synthetic Data Generation

We will generate synthetic LLM interaction data to safely illustrate XAI concepts without requiring a live model. This provides a controlled, fast-to-run sandbox that mirrors telemetry you might collect in production (prompts, outputs, confidence, accuracy, explanation quality, faithfulness, technique labels).

Business relevance:
-   Enables rapid prototyping of monitoring and governance reports.
-   Reduces cost and risk while designing explainability workflows.
-   Produces reproducible examples to train reviewers and align stakeholders.
""")
```

### 3.4 Generating Core Synthetic LLM Interaction Data — Context & Business Value

```python
st.markdown("""
## 3.4 Generating Core Synthetic LLM Interaction Data — Context & Business Value
We need a realistic yet lightweight dataset that mirrors what product analytics and governance teams monitor: prompts, outputs, confidence, accuracy, explanation quality, faithfulness, and technique labels. This enables:
-   Rapid stress-testing of explainability reports before production data pipelines exist.
-   Training analysts on how to interpret explainability signals.
-   Evaluating trade-offs between accuracy and explanation quality.

Formulae (for reference):
-   Simulated accuracy per row: $A_i \\sim \\text{Bernoulli}(p = c_i)$, where $c_i$ is model confidence for record i. Aggregate accuracy: $A_{model} = \\frac{1}{N}\\sum_i A_i$.
-   Scores range in $[0,1]$ and are constructed to be plausible, not derived from any real model internals.
""")

# Code for generate_llm_data
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
        'prompt': [f"Prompt {i+1}" for i in range(num_samples)],
        'llm_output': [f"Output {i+1}" for i in range(num_samples)],
        'true_label': [random.choice(['A', 'B', 'C']) for _ in range(num_samples)],
        'model_confidence': [random.uniform(0.5, 1.0) for _ in range(num_samples)],
        'model_accuracy': [random.choice([0, 1]) for _ in range(num_samples)],
        'explanation_quality_score': [random.uniform(0, 1) for _ in range(num_samples)],
        'faithfulness_metric': [random.uniform(0, 1) for _ in range(num_samples)],
        'xai_technique': [random.choice(['LIME', 'SHAP', 'GradCAM']) for _ in range(num_samples)],
    }
    return pd.DataFrame(data, columns=columns)

# 3.5 Executing Synthetic Data Generation and Initial Inspection
# In Streamlit, this will be driven by user input widgets
# Example usage (within the app's main flow):
# num_samples = st.sidebar.slider("Number of Synthetic LLM Interactions", 100, 5000, 500)
# if st.sidebar.button("Generate Data"):
#     with st.spinner("Generating synthetic data..."):
#         df_llm_data = generate_llm_data(num_samples)
#         st.session_state['df_llm_data'] = df_llm_data
# st.write("Shape:", st.session_state['df_llm_data'].shape)
# st.dataframe(st.session_state['df_llm_data'].head())
# st.write("DataFrame Info:")
# st.text(st.session_state['df_llm_data'].info()) # info prints to stdout, use text for Streamlit

```

### 3.6 Explanation of the Synthetic Dataset

```python
st.markdown("""
## 3.6 Explanation of the Synthetic Dataset
The generated dataset (df_llm_data) mirrors common telemetry fields used in model monitoring:
-   `timestamp`: when the interaction occurred; supports trend analysis and seasonality checks.
-   `prompt`: input to the LLM; useful for content stratification and policy audits.
-   `llm_output`: model response; used for local explanations (e.g., saliency at the token level).
-   `true_label`: a synthetic categorical label to simulate downstream evaluation.
-   `model_confidence` in $[0,1]$: proxy for model’s certainty; often used to route human review.
-   `model_accuracy` $\\in \\{0,1\\}$: simulated correctness flag; enables outcome-level reporting and trade-off studies.
-   `explanation_quality_score` in $[0,1]$: proxy for how coherent/useful an explanation is.
-   `faithfulness_metric` in $[0,1]$: conceptual alignment between explanations and model behavior.
-   `xai_technique`: categorical tag for explanations (e.g., LIME/SHAP/GradCAM in this synthetic example).

Business takeaway: These fields are enough to build explainability dashboards, route triage by confidence, and demonstrate governance controls without access to proprietary data or live models.
""")
```

### Optional: Quick Sanity Check Metric (Accuracy)

```python
st.markdown("""
## Optional: Quick Sanity Check Metric (Accuracy)
As a simple governance check, we can compare a heuristic “predicted correctness” (e.g., confidence $\\ge 0.75$) to the simulated binary accuracy flag via accuracy score. This is not a model evaluation, just a diagnostic to illustrate how monitoring hooks into basic metrics.

Formula:
-   Heuristic correctness: $ \\hat{c}_i = \\mathbb{1}[\\text{confidence}_i \\ge 0.75] $
-   Accuracy vs. label: $ A = \\frac{1}{N} \\sum_i \\mathbb{1}[\\hat{c}_i = \\text{model\\_accuracy}_i] $
""")

# Code for sanity check
# Example usage (within the app's main flow):
# threshold = st.slider("Heuristic Confidence Threshold for Accuracy", 0.0, 1.0, 0.75, 0.05)
# if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
#     heuristic_correct = (st.session_state['df_llm_data']['model_confidence'] >= threshold).astype(int)
#     match_rate = (heuristic_correct == st.session_state['df_llm_data']['model_accuracy']).mean()
#     st.write(f"Heuristic vs. simulated accuracy agreement at threshold {threshold}: {match_rate:.3f}")

```

### 3.7 Generating Synthetic Saliency Data — Context & Business Value

```python
st.markdown("""
## 3.7 Generating Synthetic Saliency Data — Context & Business Value
We simulate token-level saliency scores to conceptually indicate which words in an LLM output were most influential. While these scores are synthetic (random), they let us:
-   Demonstrate how token attribution can be visualized and audited.
-   Prototype UI/UX for explanation overlays without accessing model internals.
-   Train reviewers on how to interpret local explanations.

Approach: Split each output on whitespace to form tokens and assign each token a random score in $[0,1]$. Higher scores indicate higher conceptual importance.
""")

# Code for generate_saliency_data
@st.cache_data
def generate_saliency_data(llm_outputs):
    """Generates synthetic token-level saliency scores for LLM output strings."""
    data = []
    for idx, output in enumerate(llm_outputs):
        tokens = output.split()
        for token in tokens:
            saliency_score = np.random.rand()
            data.append((idx, token, saliency_score))
    return pd.DataFrame(data, columns=['output_index', 'token', 'saliency_score'])

# 3.8 Executing Synthetic Saliency Data Generation and Initial Inspection
# Example usage (within the app's main flow):
# if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
#     sample_outputs = st.session_state['df_llm_data']['llm_output'].head(10)
#     df_saliency = generate_saliency_data(sample_outputs)
#     st.session_state['df_saliency'] = df_saliency
#     st.write("Saliency shape:", st.session_state['df_saliency'].shape)
#     st.dataframe(st.session_state['df_saliency'].head(10))
```

### 3.8 Interpretation: What the Saliency Table Shows

```python
st.markdown("""
## 3.8 Interpretation: What the Saliency Table Shows
-   Each row corresponds to a token from one of the first 10 outputs and its synthetic saliency score in $[0,1]$.
-   Higher scores suggest a stronger conceptual contribution of that token to the output.
-   We will use these scores to visually highlight important tokens in a sample output, mimicking a saliency map overlay often used in review tools.
""")
```

### 3.9 Data Validation and Summary Statistics — Executable Code

```python
# Code for validate_and_summarize_data
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
        st.warning(f"[Warning] Missing expected columns: {missing}")
    else:
        st.success("All expected columns are present.")

    # Validate dtypes for critical numeric fields
    critical_numeric = ['model_confidence', 'explanation_quality_score', 'faithfulness_metric']
    for col in critical_numeric:
        if col in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                st.warning(f"[Warning] Column '{col}' is not numeric.")
        else:
            st.warning(f"[Warning] Column '{col}' not found for numeric validation.")

    # Missing value checks
    if all(c in dataframe.columns for c in critical_numeric):
        na_counts = dataframe[critical_numeric].isna().sum()
        if na_counts.sum() == 0:
            st.info("No missing values found in critical numeric fields.")
        else:
            st.warning("[Warning] Missing values in critical numeric fields:\n", na_counts)

    # Descriptive statistics
    st.markdown("\n**Descriptive statistics for numeric columns:**")
    st.dataframe(dataframe.select_dtypes(include='number').describe().T)

    # Value counts for key categoricals
    for col in ['true_label', 'xai_technique']:
        if col in dataframe.columns:
            st.markdown(f"\n**Value counts for `{col}`:**")
            st.dataframe(dataframe[col].value_counts())

# 3.10 Executing Data Validation and Summary Statistics
# Example usage (within the app's main flow):
# if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
#     validate_and_summarize_data(st.session_state['df_llm_data'])
```

### 3.11 Interpretability vs. Transparency

```python
st.markdown("""
## 3.11 Interpretability vs. Transparency
-   **Interpretability**: understanding why the model produced a specific output for a given input; focuses on input–output relationships.
-   **Transparency**: understanding how the model works internally (architecture, training data, parameters). For large LLMs, this is often infeasible in practice.

Business impact: When full transparency is not possible, strong interpretability (clear rationales, salient inputs, counterfactuals) supports audits, incident response, and policy compliance without exposing proprietary internals.
""")
```

### 3.12 Introduction to XAI Techniques (Saliency and Counterfactuals)

```python
st.markdown("""
## 3.12 Introduction to XAI Techniques (Saliency and Counterfactuals)

-   **Saliency Maps**: Highlight which input tokens most influenced an output. Conceptually, token importance can be related to local sensitivity of the output with respect to an input token:
    $$ S(x_i) = \\left| \\frac{\\partial Y}{\\partial x_i} \\right| $$
    Higher values indicate stronger influence on the model’s decision, helping reviewers quickly locate decisive words.

-   **Counterfactual Explanations**: Show how minimal changes to an input could change the model’s output. Formally, we seek a small perturbation $\\Delta X$ such that:
    $$ \\text{Model}(X + \\Delta X) = Y' \\ne Y, \\quad \\text{with minimal } ||\\Delta X|| $$
    This supports “what-if” analysis, recourse options, and fairness assessments.

Business value: Saliency sharpens local interpretability for specific outputs; counterfactuals reveal levers that alter outcomes—both are essential for audits, compliance, and user trust.
""")
```

### 3.13 Applying XAI Technique: Saliency Map Simulation — Context & Business Value

```python
st.markdown("""
## 3.13 Applying XAI Technique: Saliency Map Simulation — Context & Business Value
Saliency maps help reviewers quickly identify which words in an output likely drove the model’s behavior. This accelerates root-cause analysis in incident reviews and supports explainability audits without revealing proprietary internals.

Formula (conceptual sensitivity):
-   Token importance: $$ S(x_i) = \\left| \\frac{\\partial Y}{\\partial x_i} \\right| $$
-   In this notebook we simulate $S(x_i)$ with random values in $[0,1]$ and highlight tokens with scores above a threshold $\\tau$.
""")

# Code for visualize_saliency_map
def visualize_saliency_map(llm_output, token_scores, threshold=0.5):
    """
    Highlights tokens in an LLM output string based on saliency scores using HTML.
    Tokens with scores at or above the specified threshold are highlighted.
    """
    tokens = llm_output.split()
    highlighted_tokens = [
        f"<span style=\"background-color: yellow;\">{token}</span>" if score >= threshold else token
        for token, score in zip(tokens, token_scores)
    ]
    highlighted_text = " ".join(highlighted_tokens)
    return highlighted_text # Return raw HTML string for st.markdown

# 3.14 Executing Saliency Map Simulation
# Example usage (within the app's main flow):
# if 'df_llm_data' in st.session_state and 'df_saliency' in st.session_state and not st.session_state['df_llm_data'].empty:
#     st.subheader("Saliency Map Visualization")
#     sample_index = st.number_input("Select Output Index for Saliency Map", 0, len(st.session_state['df_llm_data'])-1, 0)
#     saliency_threshold = st.slider("Saliency Highlight Threshold (τ)", 0.0, 1.0, 0.7, 0.05)

#     sample_output = st.session_state['df_llm_data'].loc[sample_index, 'llm_output']
#     sample_scores_df = st.session_state['df_saliency'][st.session_state['df_saliency']['output_index'] == sample_index]

#     if not sample_scores_df.empty:
#         sample_scores = sample_scores_df['saliency_score'].tolist()
#         tokens = sample_output.split()
#         if len(sample_scores) < len(tokens):
#             sample_scores = sample_scores + [0.0] * (len(tokens) - len(sample_scores))
#         elif len(sample_scores) > len(tokens):
#             sample_scores = sample_scores[:len(tokens)]

#         html_vis = visualize_saliency_map(sample_output, sample_scores, threshold=saliency_threshold)
#         st.markdown(html_vis, unsafe_allow_html=True)
#     else:
#         st.info("No saliency scores found for the selected output index.")

```

### 3.15 Interpreting the Saliency Map

```python
st.markdown("""
## 3.15 Interpreting the Saliency Map
The highlighted words represent tokens with saliency scores above the chosen threshold ($\\tau = 0.7$). In a real system, these would indicate which parts of the text most influenced the model’s output. This helps reviewers:
-   Quickly spot decisive phrases.
-   Compare highlighted rationale with policy or business rules.
-   Identify unexpected drivers of model behavior that may need mitigation or re-training.
""")
```

### Optional Result Interpretation: Heuristic Agreement

```python
st.markdown("""
## Optional Result Interpretation: Heuristic Agreement
The printed agreement rate shows how often a simple confidence-based heuristic aligns with the simulated accuracy flag. A higher value suggests confidence is a reasonable proxy for correctness; a lower value warns that confidence alone may be misleading. In governance, such diagnostics inform threshold setting for human-in-the-loop review.
""")
```

### 3.16 Applying XAI Technique: Counterfactual Explanation — Context & Business Value

```python
st.markdown("""
## 3.16 Applying XAI Technique: Counterfactual Explanation — Context & Business Value
Counterfactuals answer the question “What minimal change to the input would have changed the output?” They are critical for recourse (what a user could do differently), fairness (sensitivity to protected attributes), and debugging (what levers flip the decision).

Formula (goal):
-   Find a perturbation $\\Delta X$ such that the model output flips: $$ \\text{Model}(X + \\Delta X) = Y' \\ne Y $$
-   With minimal change: minimize $||\\Delta X||$ subject to the flip. In this notebook, we simulate a simple, illustrative counterfactual.
""")

# Code for generate_counterfactual_explanation
def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """
    Generates a counterfactual explanation by proposing a modified input prompt
    and its resulting output.
    """
    # Simulate a minimal modification to the original prompt
    counterfactual_prompt = original_prompt + " (What if incorrect?)"

    # Simulate generating a counterfactual output
    if current_model_accuracy > 0.5:
        counterfactual_output = "A different plausible explanation."
    else:
        counterfactual_output = "An incorrect explanation."

    return {
        "original_prompt": original_prompt,
        "original_output": original_output,
        "counterfactual_prompt": counterfactual_prompt,
        "counterfactual_output": counterfactual_output
    }

# 3.17 Executing Counterfactual Explanation Simulation
# Example usage (within the app's main flow):
# if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
#     st.subheader("Counterfactual Explanation")
#     cf_idx = st.number_input("Select Record Index for Counterfactual", 0, len(st.session_state['df_llm_data'])-1, 0)
#     original_prompt = st.session_state['df_llm_data'].loc[cf_idx, 'prompt']
#     original_output = st.session_state['df_llm_data'].loc[cf_idx, 'llm_output']
#     current_model_accuracy = float(st.session_state['df_llm_data'].loc[cf_idx, 'model_accuracy'])

#     cf_result = generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy)

#     st.write('**Original Prompt:**', cf_result['original_prompt'])
#     st.write('**Original Output:**', cf_result['original_output'])
#     st.write('\n**Counterfactual Prompt:**', cf_result['counterfactual_prompt'])
#     st.write('**Counterfactual Output:**', cf_result['counterfactual_output'])
```

### 3.18 Interpreting the Counterfactual Explanation

```python
st.markdown("""
## 3.18 Interpreting the Counterfactual Explanation
The counterfactual demonstrates how a small prompt change could plausibly lead to a different outcome. In practice, this:
-   Helps determine levers for decision recourse (what a user could change).
-   Reveals sensitivity to specific terms or conditions for fairness auditing.
-   Supports root-cause analysis by contrasting original vs. perturbed inputs and outcomes.
""")
```

### 3.19 Core Visual: Faithfulness Metric Over Time — Context & Business Value

```python
st.markdown("""
## 3.19 Core Visual: Faithfulness Metric Over Time — Context & Business Value
Monitoring explanation faithfulness over time helps detect drift in how well explanations align with modeled behavior. This is essential for governance dashboards and incident review.

Formula and setup:
-   Faithfulness (simulated): $F_t \\in [0,1]$ for interaction at time t. Higher is better.
-   We stratify by XAI technique (e.g., LIME, SHAP, GradCAM) to spot consistency gaps across methods.

We will produce a line chart with a color-blind-friendly palette and save a PNG fallback for reporting workflows.
""")

# Code for plot_faithfulness_trend
def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    Generates a line plot showing trends of a specified metric over time for different categories.
    Displays the plot and saves it as a PNG file.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot faithfulness trend.")
        return

    for column in [x_axis, y_axis, hue_column]:
        if column not in dataframe.columns:
            st.error(f"Column '{column}' is missing from DataFrame, cannot plot faithfulness trend.")
            return

    if not pd.api.types.is_numeric_dtype(dataframe[y_axis]):
        st.error(f"Column '{y_axis}' must have numeric data type, cannot plot faithfulness trend.")
        return

    if not pd.api.types.is_datetime64_any_dtype(dataframe[x_axis]) and not pd.api.types.is_numeric_dtype(dataframe[x_axis]):
        st.error(f"Column '{x_axis}' must have datetime or numeric data type, cannot plot faithfulness trend.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue_column, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend(title=hue_column)
    ax.grid(visible=True)
    
    # Save as PNG fallback
    plt.savefig('plot_faithfulness_trend.png', bbox_inches='tight', dpi=150)
    st.pyplot(fig) # Display in Streamlit
    plt.close(fig) # Close figure to free memory

# 3.20 Executing Faithfulness Trend Plot
# Example usage (within the app's main flow, after filtering):
# if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
#     filtered_df = st.session_state['df_llm_data'].copy()
#     # Apply filters based on sidebar widgets (min_quality, min_confidence, xai_technique_filter)
#     min_quality = st.sidebar.slider("Min Explanation Quality Score", 0.0, 1.0, 0.0)
#     min_confidence = st.sidebar.slider("Min Model Confidence", 0.0, 1.0, 0.0)
#     xai_techniques = filtered_df['xai_technique'].unique().tolist()
#     selected_techniques = st.sidebar.multiselect("Filter by XAI Technique", xai_techniques, default=xai_techniques)

#     filtered_df = filtered_df[
#         (filtered_df['explanation_quality_score'] >= min_quality) &
#         (filtered_df['model_confidence'] >= min_confidence) &
#         (filtered_df['xai_technique'].isin(selected_techniques))
#     ]

#     st.subheader("Core Visualizations")
#     plot_faithfulness_trend(
#         dataframe=filtered_df,
#         x_axis='timestamp',
#         y_axis='faithfulness_metric',
#         hue_column='xai_technique',
#         title='Faithfulness Metric over Time by XAI Technique'
#     )
```

### Additional Visualizations (Relationship Plot, Aggregated Comparison)

```python
# Placeholder for Relationship Plot (Explanation Quality vs. Model Accuracy)
def plot_explanation_accuracy_relationship(dataframe):
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot explanation quality vs. accuracy.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dataframe, x='model_accuracy', y='explanation_quality_score', hue='xai_technique', ax=ax, alpha=0.6)
    ax.set_title("Explanation Quality vs. Model Accuracy")
    ax.set_xlabel("Simulated Model Accuracy (0=Incorrect, 1=Correct)")
    ax.set_ylabel("Explanation Quality Score")
    ax.legend(title="XAI Technique")
    ax.grid(visible=True)
    st.pyplot(fig)
    plt.close(fig)

# Placeholder for Aggregated Comparison Plot (Average Faithfulness by XAI Technique)
def plot_average_faithfulness_by_technique(dataframe):
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot average faithfulness by technique.")
        return
    
    avg_faithfulness = dataframe.groupby('xai_technique')['faithfulness_metric'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=avg_faithfulness, x='xai_technique', y='faithfulness_metric', ax=ax)
    ax.set_title("Average Faithfulness by XAI Technique")
    ax.set_xlabel("XAI Technique")
    ax.set_ylabel("Average Faithfulness Metric")
    ax.grid(visible=True)
    st.pyplot(fig)
    plt.close(fig)

# Example usage (within the app's main flow after filtering):
# plot_explanation_accuracy_relationship(filtered_df)
# plot_average_faithfulness_by_technique(filtered_df)
```
