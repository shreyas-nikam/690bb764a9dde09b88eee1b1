id: 690bb764a9dde09b88eee1b1_documentation
summary: Agentic AI for Safety Monitoring Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Explaining Agentic AI for Safety Monitoring

## 1. Introduction to QuLab and Agentic AI Safety Monitoring
Duration: 0:08:00

Welcome to the **QuLab: Agentic AI for Safety Monitoring: An Explainable AI Lab** codelab! In this lab, we will embark on a comprehensive journey to understand and interact with a Streamlit application designed to explore the critical role of Explainable AI (XAI) in monitoring Large Language Models (LLMs) within agentic systems, with a particular focus on safety.

As AI agents gain more autonomy and become integrated into critical systems, ensuring their decisions are transparent, justifiable, and inherently safe becomes paramount. This application provides a hands-on, end-to-end walkthrough using synthetic data to demonstrate how core XAI concepts and techniques can be applied in a practical setting.

<aside class="positive">
<b>Why is this important?</b> Agentic AI systems can make complex decisions with limited human oversight. XAI provides the essential tools to "look inside" these black-box LLMs, offering insights into their reasoning and potential vulnerabilities. This is crucial for:
<ul>
    <li><b>Enhanced Trust:</b> Building confidence in AI systems among stakeholders.</li>
    <li><b>Improved Governance:</b> Meeting regulatory and ethical requirements for AI explainability.</li>
    <li><b>Faster Incident Response:</b> Rapidly identifying and mitigating unsafe or undesirable behaviors.</li>
    <li><b>Bias Detection & Mitigation:</b> Proactively addressing potential biases in AI decision-making.</li>
    <li><b>Robustness against Attacks:</b> Understanding how models might be vulnerable to adversarial inputs.</li>
</ul>
</aside>

**What you will learn and do in this codelab:**

*   **Core XAI Concepts**: Distinguish between interpretability and transparency, and understand their importance for risk management, trust, and AI governance.
*   **Synthetic Data Generation**: Learn how to create realistic datasets of LLM interactions and associated XAI signals at scale to simulate real-world scenarios.
*   **XAI Technique Simulation**: Interact with demonstrations of cornerstone techniques like saliency maps (to understand token importance) and counterfactual explanations (to explore "what-if" scenarios).
*   **Trend & Trade-off Analysis**: Visualize how explanation quality and faithfulness evolve over time and their relationship with model accuracy.
*   **Practical Filtering**: Apply filters based on XAI metrics and model confidence to focus human review efforts on high-value or high-risk cases.
*   **AI Security & Defenses**: Understand how XAI supports identifying and mitigating AI-security threats in agentic systems by providing insights into model behavior.

This lab aims to equip you with a foundational understanding and practical examples to integrate XAI into your AI safety monitoring strategies. All code executes locally and uses only open-source packages, making it a highly accessible and reproducible learning experience.

## 2. Setting Up the Environment and Understanding the Application Structure
Duration: 0:05:00

Before diving into the application's functionalities, let's set up your environment and understand the overall structure of the Streamlit application.

### Prerequisites

Ensure you have Python installed (Python 3.8+ recommended).

### 2.1 Clone the Repository (Conceptual)

Assuming you have access to the code, you would typically clone the repository. For this codelab, we are provided with the `app.py` and `application_pages/` directory.

### 2.2 Install Dependencies

The application relies on several Python libraries. You can install them using pip:

```bash
pip install streamlit pandas numpy plotly
```

### 2.3 Running the Streamlit Application

Navigate to the directory containing `app.py` and run the application using the Streamlit command:

```bash
streamlit run app.py
```

This command will open the application in your default web browser.

### 2.4 Application Structure Overview

The application is structured into a main `app.py` file and a `application_pages` directory containing individual page logic.

*   **`app.py`**:
    *   Initializes the Streamlit page configuration (`st.set_page_config`).
    *   Sets up the main title and introductory markdown.
    *   Initializes `st.session_state` variables (`df_llm_data`, `df_saliency`) to persist data across user interactions and page navigations.
    *   Implements **Global Filtering Controls** in the sidebar, which apply to data displayed across all pages.
    *   Manages **Navigation** between different pages using a `st.sidebar.selectbox`.
    *   Imports and runs the respective page functions based on user selection.

    ```python
    # app.py (excerpt)
    import streamlit as st
    import pandas as pd
    import numpy as np
    import random
    import warnings
    from datetime import datetime, timedelta

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Reproducibility (will be set once, perhaps in session_state or app initialization)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    st.set_page_config(page_title="QuLab", layout="wide")
    st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
    st.sidebar.divider()
    st.title("QuLab")
    st.divider()
    # ... introductory markdown ...

    # Initialize session state for data if not already present
    if 'df_llm_data' not in st.session_state:
        st.session_state['df_llm_data'] = pd.DataFrame()
    if 'df_saliency' not in st.session_state:
        st.session_state['df_saliency'] = pd.DataFrame()

    # Global Filtering Controls in Sidebar
    st.sidebar.markdown("## Global Filters")
    if not st.session_state['df_llm_data'].empty:
        min_quality = st.sidebar.slider("Min Explanation Quality Score", 0.0, 1.0, 0.0, 0.01, help="Filter to show only records with explanation quality scores above this value.")
        min_confidence = st.sidebar.slider("Min Model Confidence", 0.0, 1.0, 0.0, 0.01, help="Filter to show only records with model confidence scores above this value.")

        xai_techniques = st.session_state['df_llm_data']['xai_technique'].unique().tolist()
        selected_techniques = st.sidebar.multiselect("Filter by XAI Technique", xai_techniques, default=xai_techniques, help="Select which XAI techniques to include in the analysis and visualizations.")

        # Apply global filters
        filtered_df = st.session_state['df_llm_data'][
            (st.session_state['df_llm_data']['explanation_quality_score'] >= min_quality) &
            (st.session_state['df_llm_data']['model_confidence'] >= min_confidence) &
            (st.session_state['df_llm_data']['xai_technique'].isin(selected_techniques))
        ].copy()
    else:
        filtered_df = pd.DataFrame() # Empty if no data yet

    # Navigation
    page = st.sidebar.selectbox(label="Navigation", options=["Page 1: Introduction & Data Setup", "Page 2: XAI Concepts & Saliency Map", "Page 3: Counterfactuals & Trend Analysis"])

    if page == "Page 1: Introduction & Data Setup":
        from application_pages.page1 import run_page1
        run_page1(filtered_df)
    elif page == "Page 2: XAI Concepts & Saliency Map":
        from application_pages.page2 import run_page2
        run_page2(filtered_df)
    elif page == "Page 3: Counterfactuals & Trend Analysis":
        from application_pages.page3 import run_page3
        run_page3(filtered_df)
    ```

*   **`application_pages/page1.py`**: Handles data generation, upload, initial inspection, and validation.
*   **`application_pages/page2.py`**: Focuses on XAI concepts, specifically saliency maps.
*   **`application_pages/page3.py`**: Explores counterfactual explanations and various trend analyses.

<aside class="positive">
<b>Tip: Session State and Global Filters</b>
The use of `st.session_state` is crucial for maintaining data (`df_llm_data`, `df_saliency`) across page navigations in Streamlit. The `filtered_df` is derived from `st.session_state['df_llm_data']` based on the global filters in the sidebar. This filtered data is then passed to the respective page functions, ensuring all visualizations and analyses operate on the user-selected subset of data.
</aside>

## 3. Page 1: Introduction & Data Setup
Duration: 0:15:00

The first page of the application, "Introduction & Data Setup," is where you begin your journey by generating or uploading the core LLM interaction data. This data forms the foundation for all subsequent XAI analyses.

### 3.1 Executive Summary and Data Overview

Upon navigating to "Page 1: Introduction & Data Setup", you'll see an executive summary outlining the purpose of the application and the business value of XAI for agentic AI safety monitoring. It emphasizes the use of synthetic data for rapid prototyping and explains the constraints and assumptions.

The "Data and Inputs Overview" section clarifies why synthetic data is used (to simulate realistic scenarios without the cost and variability of real LLMs) and highlights its business value for prototyping review workflows.

### 3.2 Generating Core Synthetic LLM Interaction Data

This section allows you to generate a synthetic dataset or upload your own. The synthetic data mirrors common telemetry fields critical for model monitoring and governance teams.

#### The `generate_llm_data` Function

The `generate_llm_data` function in `application_pages/page1.py` is responsible for creating this synthetic dataset. It populates columns like `timestamp`, `prompt`, `llm_output`, `model_confidence`, `model_accuracy`, `explanation_quality_score`, `faithfulness_metric`, and `xai_technique` with plausible random values.

```python
# application_pages/page1.py (excerpt)
@st.cache_data
def generate_llm_data(num_samples):
    """Generates a synthetic dataset simulating LLM interactions."""
    # ... input validation ...
    
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
```

The app provides a slider to control the `num_samples` (number of LLM interactions) from 100 to 5000. Click the **"Generate Data"** button to populate `st.session_state['df_llm_data']` with this synthetic information.

<aside class="negative">
<b>Warning: Data Generation Required</b>
You must generate or upload data in Page 1 before proceeding to other pages, as they rely on `st.session_state['df_llm_data']` to be populated.
</aside>

### 3.3 Uploading Custom Data

Alternatively, you can select "Upload Custom Data (CSV)" and provide your own CSV file. This flexibility allows you to apply the same XAI analysis framework to your specific datasets, provided they have similar column structures.

```python
# application_pages/page1.py (excerpt)
    data_source = st.radio(
        "Data Source Selection",
        ("Generate Synthetic Data", "Upload Custom Data (CSV)"),
        help="Choose whether to generate synthetic LLM interaction data or upload your own CSV file."
    )

    if data_source == "Generate Synthetic Data":
        # ... synthetic data generation controls ...
    elif data_source == "Upload Custom Data (CSV)":
        uploaded_file = st.file_uploader(
            "Upload your LLM interaction data (CSV)",
            type=["csv"],
            help="Upload a CSV file containing LLM interaction data. The file should have columns like 'timestamp', 'prompt', 'llm_output', 'model_confidence', etc.",
            accept_multiple_files=False
        )
        if uploaded_file is not None:
            # ... file processing ...
```

### 3.4 Initial Inspection and Explanation of the Dataset

Once data is generated or uploaded, the app displays the first few rows and provides an explanation of each column:

*   `timestamp`: When the interaction occurred; supports trend analysis.
*   `prompt`: Input to the LLM; useful for content stratification and policy audits.
*   `llm_output`: Model response; used for local explanations (e.g., saliency at the token level).
*   `true_label`: A synthetic categorical label to simulate downstream evaluation.
*   `model_confidence` in $[0,1]$: Proxy for model’s certainty; often used to route human review.
*   `model_accuracy` $\in \{0,1\}$: Simulated correctness flag; enables outcome-level reporting and trade-off studies.
*   `explanation_quality_score` in $[0,1]$: Proxy for how coherent/useful an explanation is.
*   `faithfulness_metric` in $[0,1]$: Conceptual alignment between explanations and model behavior.
*   `xai_technique`: Categorical tag for explanations (e.g., LIME, SHAP, GradCAM in this synthetic example).

<aside class="positive">
<b>Business Takeaway:</b> These fields are comprehensive enough to build explainability dashboards, route triage by confidence, and demonstrate governance controls without needing access to proprietary data or live models.
</aside>

### 3.5 Optional: Quick Sanity Check Metric (Accuracy)

The application includes a simple diagnostic check: comparing a heuristic "predicted correctness" (e.g., confidence $\ge 0.75$) to the simulated binary `model_accuracy` flag.

The heuristic correctness is defined as:
$$ \hat{c}_i = \mathbb{1}[\text{confidence}_i \ge \text{threshold}] $$
The agreement rate with the true label is:
$$ A = \frac{1}{N} \sum_i \mathbb{1}[\hat{c}_i = \text{model\_accuracy}_i] $$

You can adjust the `Heuristic Confidence Threshold for Accuracy` using a slider to observe how this agreement rate changes. This diagnostic helps inform threshold settings for human-in-the-loop review in real-world governance scenarios.

### 3.6 Data Validation and Summary Statistics

Finally, the `validate_and_summarize_data` function performs basic data validation and provides summary statistics for the generated/uploaded dataset. This ensures data quality and gives a quick overview of its characteristics.

```python
# application_pages/page1.py (excerpt)
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
```

This summary includes:
*   Presence of expected columns.
*   Validation of data types for critical numeric fields.
*   Checks for missing values.
*   Descriptive statistics for numeric columns.
*   Value counts for categorical columns like `true_label` and `xai_technique`.

## 4. Page 2: XAI Concepts & Saliency Map
Duration: 0:10:00

On "Page 2: XAI Concepts & Saliency Map", we delve into two fundamental XAI concepts: interpretability versus transparency, and then focus on saliency maps as a practical technique.

### 4.1 Interpretability vs. Transparency

The page starts by distinguishing between:

*   **Interpretability**: Understanding *why* a model made a specific decision for a given input; focuses on input–output relationships.
*   **Transparency**: Understanding *how* the model works internally (its architecture, training data, parameters). For large LLMs, full transparency is often not feasible.

<aside class="positive">
<b>Business Impact:</b> When full transparency is not possible, strong interpretability through techniques like saliency maps and counterfactuals is vital. It supports audits, incident response, and policy compliance without exposing proprietary model internals, especially critical for safety monitoring of agentic systems.
</aside>

### 4.2 Introduction to XAI Techniques: Saliency Maps

**Saliency Maps** are introduced as a technique that highlights which input tokens most influenced an LLM's output. Conceptually, token importance can be related to the local sensitivity of the output with respect to an input token:

$$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$

Here, $S(x_i)$ represents the saliency score for token $x_i$, and it captures how sensitive the model output $Y$ is to small changes in $x_i$. Higher values indicate stronger influence, helping reviewers quickly locate decisive words, which is crucial for identifying critical parts of an LLM's response that indicate safety risks.

### 4.3 Generating Synthetic Saliency Data

Similar to the main LLM interaction data, saliency data is also synthetically generated. The `generate_saliency_data` function in `application_pages/page2.py` takes the LLM outputs and assigns a random saliency score (between 0 and 1) to each token within those outputs.

```python
# application_pages/page2.py (excerpt)
@st.cache_data
def generate_saliency_data(llm_outputs):
    """Generates synthetic token-level saliency scores for LLM output strings."""
    data = []
    for idx, output in enumerate(llm_outputs):
        tokens = str(output).split() # Ensure output is string
        if not tokens:
            # Handle empty strings to avoid errors later
            data.append((idx, "", 0.0))
            continue
        for token in tokens:
            saliency_score = np.random.rand()
            data.append((idx, token, saliency_score))
    return pd.DataFrame(data, columns=['output_index', 'token', 'saliency_score'])
```

<aside class="positive">
<b>Note:</b> For consistent indexing, the `generate_saliency_data` function is called using the full `st.session_state['df_llm_data']` from `app.py`, even if `filtered_df` is being passed to `run_page2`. This ensures that the `output_index` in the saliency data consistently maps to the original dataframe's index.
</aside>

The app displays the head of the generated `df_saliency` dataframe, showing `output_index`, `token`, and `saliency_score`. Higher scores conceptually indicate a stronger contribution of that token to the LLM's output.

### 4.4 Saliency Map Visualization

The core of this page is the interactive saliency map visualization. The `visualize_saliency_map` function takes an LLM output, the generated saliency scores, the output's index, and a threshold. It then highlights tokens whose saliency scores meet or exceed this threshold.

```python
# application_pages/page2.py (excerpt)
def visualize_saliency_map(llm_output, token_scores_df, output_index, threshold=0.5):
    """
    Highlights tokens in an LLM output string based on saliency scores using HTML.
    Tokens with scores at or above the specified threshold are highlighted.
    """
    if not isinstance(llm_output, str):
        return "Invalid LLM output provided."
    
    tokens = llm_output.split()
    
    # Filter saliency scores for the specific output_index
    scores_for_output = token_scores_df[token_scores_df['output_index'] == output_index]
    
    # Create a dictionary for quick lookup of scores
    token_score_map = {row['token']: row['saliency_score'] for _, row in scores_for_output.iterrows()}

    highlighted_tokens = []
    for token in tokens:
        # Get the score for the current token, default to 0 if not found
        score = token_score_map.get(token, 0.0)
        if score >= threshold:
            highlighted_tokens.append(f"<span style=\"background-color: yellow;\">{token}</span>")
        else:
            highlighted_tokens.append(token)
            
    highlighted_text = " ".join(highlighted_tokens)
    return highlighted_text # Return raw HTML string for st.markdown
```

You can select an `Output Index` from the dataset using a number input and adjust the `Saliency Highlight Threshold ($\tau$)` with a slider. Tokens with scores at or above the chosen threshold will be highlighted in yellow, visually representing their conceptual importance.

### 4.5 Interpreting the Saliency Map

The highlighted words help reviewers, particularly in safety monitoring, to:
*   Quickly spot decisive phrases related to safety risks.
*   Compare the highlighted rationale with policy or business rules for compliance.
*   Identify unexpected drivers of model behavior that may indicate vulnerabilities or unsafe tendencies, requiring mitigation or re-training.

## 5. Page 3: Counterfactuals & Trend Analysis
Duration: 0:12:00

"Page 3: Counterfactuals & Trend Analysis" introduces another powerful XAI technique—counterfactual explanations—and then moves into trend analysis visualizations that are crucial for ongoing AI safety monitoring.

### 5.1 Applying XAI Technique: Counterfactual Explanation

**Counterfactuals** answer the question: "What minimal change to the input would have changed the output?" This technique is vital for understanding recourse, fairness, and debugging model behavior. In agentic AI safety, counterfactuals can illuminate how minor changes in instructions or environmental context might lead an agent to an unsafe outcome.

Formally, the goal is to find a perturbation $\Delta X$ such that the model output flips:

$$ \text{Model}(X + \Delta X) = Y' \ne Y $$

While minimizing the change: minimize $||\Delta X||$ subject to the output flip.

#### The `generate_counterfactual_explanation` Function

The application simulates a simple counterfactual using the `generate_counterfactual_explanation` function in `application_pages/page3.py`. It proposes a subtly modified input prompt and a corresponding altered output, demonstrating how small changes could conceptually lead to different outcomes.

```python
# application_pages/page3.py (excerpt)
def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """
    Generates a counterfactual explanation by proposing a modified input prompt
    and its resulting output.
    """
    # Simulate a minimal modification to the original prompt
    counterfactual_prompt = original_prompt + " (What if we rephrased the safety concern?)"

    # Simulate generating a counterfactual output
    if current_model_accuracy > 0.5:
        counterfactual_output = "A subtly different, but still safe, explanation."
    else:
        counterfactual_output = "An unsafe or undesirable response due to the rephrasing."

    return {
        "original_prompt": original_prompt,
        "original_output": original_output,
        "counterfactual_prompt": counterfactual_prompt,
        "counterfactual_output": counterfactual_output
    }
```

You can select a `Record Index for Counterfactuals` from the available data. The app will then display the original prompt and output, followed by the simulated counterfactual prompt and output.

### 5.2 Interpreting the Counterfactual Explanation

The counterfactual demonstration reveals how a small prompt change could plausibly lead to a different outcome. This is invaluable for safety monitoring as it:
*   Helps determine levers for decision recourse (what a user could change to get a safe outcome).
*   Reveals sensitivity to specific terms or conditions, which is crucial for fairness auditing and identifying adversarial attacks.
*   Supports root-cause analysis by contrasting original vs. perturbed inputs and outcomes, helping to understand why an agent might deviate from safe behavior.

### 5.3 Core Visual: Faithfulness Metric Over Time

This section provides key visualizations for monitoring XAI metrics. The first is a line plot showing the `faithfulness_metric` over time, stratified by `xai_technique`.

#### The `plot_faithfulness_trend` Function

```python
# application_pages/page3.py (excerpt)
def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    Generates a line plot showing trends of a specified metric over time for different categories using Plotly.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot faithfulness trend.")
        return

    # ... input validation ...

    fig = px.line(dataframe, x=x_axis, y=y_axis, color=hue_column, title=title)
    fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    st.plotly_chart(fig, use_container_width=True)
```

Monitoring explanation faithfulness over time helps detect drift in how well explanations align with modeled behavior. This is essential for governance dashboards and incident review, especially for understanding the long-term reliability of safety explanations in agentic systems. A decreasing trend in faithfulness might indicate that the model's explanations are becoming less reliable or that the model's behavior has subtly shifted.

### 5.4 Explanation Quality vs. Model Accuracy

This scatter plot illustrates the potential trade-off between the quality of an explanation and the model's accuracy.

#### The `plot_explanation_accuracy_relationship` Function

```python
# application_pages/page3.py (excerpt)
def plot_explanation_accuracy_relationship(dataframe):
    """
    Generates a scatter plot of explanation quality vs. model accuracy using Plotly.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot explanation quality vs. accuracy.")
        return
    
    # ... column validation ...

    fig = px.scatter(dataframe, x='model_accuracy', y='explanation_quality_score', color='xai_technique',
                     title="Explanation Quality vs. Model Accuracy",
                     labels={'model_accuracy': "Simulated Model Accuracy (0=Incorrect, 1=Correct)", 'explanation_quality_score': "Explanation Quality Score"},
                     opacity=0.6)
    fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    st.plotly_chart(fig, use_container_width=True)
```

Understanding this relationship is vital for balancing interpretability requirements with model performance goals in safety-critical applications. For instance, a highly accurate model with poor explanations might be harder to debug if it makes a safety error, suggesting a need to invest more in explanation quality or choose different XAI techniques.

### 5.5 Average Faithfulness by XAI Technique

This bar chart provides an aggregated comparison of the average faithfulness metric across different XAI techniques.

#### The `plot_average_faithfulness_by_technique` Function

```python
# application_pages/page3.py (excerpt)
def plot_average_faithfulness_by_technique(dataframe):
    """
    Generates a bar chart of average faithfulness by XAI technique using Plotly.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot average faithfulness by technique.")
        return
    
    # ... column validation ...
    
    avg_faithfulness = dataframe.groupby('xai_technique')['faithfulness_metric'].mean().reset_index()
    fig = px.bar(avg_faithfulness, x='xai_technique', y='faithfulness_metric',
                 title="Average Faithfulness by XAI Technique",
                 labels={'xai_technique': "XAI Technique", 'faithfulness_metric': "Average Faithfulness Metric"})
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    st.plotly_chart(fig, use_container_width=True)
```

This chart helps in evaluating which explanation techniques consistently provide better alignment with the model's behavior, guiding the selection of techniques for robust safety monitoring and ensuring that the chosen XAI methods are indeed reliable for understanding agentic AI decisions.

## 6. Global Filtering and Conclusion
Duration: 0:05:00

The Streamlit application includes powerful **Global Filtering Controls** in the sidebar, which significantly enhance its utility for monitoring and analysis.

### 6.1 Global Filtering Controls

In `app.py`, these filters are applied to the `df_llm_data` before it is passed as `filtered_df` to any of the application pages.

```python
# app.py (excerpt)
    st.sidebar.markdown("## Global Filters")
    if not st.session_state['df_llm_data'].empty:
        min_quality = st.sidebar.slider("Min Explanation Quality Score", 0.0, 1.0, 0.0, 0.01, help="Filter to show only records with explanation quality scores above this value.")
        min_confidence = st.sidebar.slider("Min Model Confidence", 0.0, 1.0, 0.0, 0.01, help="Filter to show only records with model confidence scores above this value.")

        xai_techniques = st.session_state['df_llm_data']['xai_technique'].unique().tolist()
        selected_techniques = st.sidebar.multiselect("Filter by XAI Technique", xai_techniques, default=xai_techniques, help="Select which XAI techniques to include in the analysis and visualizations.")

        # Apply global filters
        filtered_df = st.session_state['df_llm_data'][
            (st.session_state['df_llm_data']['explanation_quality_score'] >= min_quality) &
            (st.session_state['df_llm_data']['model_confidence'] >= min_confidence) &
            (st.session_state['df_llm_data']['xai_technique'].isin(selected_techniques))
        ].copy()
    else:
        filtered_df = pd.DataFrame() # Empty if no data yet
