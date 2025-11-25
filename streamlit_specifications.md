
# Streamlit Application Specification: Explainable AI (XAI) for LLMs

## 1. Application Overview

This Streamlit application will provide an interactive exploration of Explainable AI (XAI) concepts and techniques applied to Large Language Models (LLMs), leveraging synthetic data to simulate real-world scenarios.

### Learning Goals

Upon completion of interacting with this application, users will be able to:
- Understand the core insights presented regarding XAI and LLMs.
- Distinguish clearly between interpretability and transparency in the context of AI models.
- Review and conceptually apply different XAI techniques, specifically saliency maps and counterfactual explanations.
- Analyze the inherent trade-offs between model performance (e.g., accuracy) and the explainability of its decisions.

## 2. User Interface Requirements

### Layout and Navigation Structure

The application will be structured primarily in a single main page, with logical sections organized using Streamlit's `st.header`, `st.subheader`, and `st.expander` components for a clean and digestible user experience. A sidebar will be used for global data generation parameters and general application settings.

*   **Sidebar:**
    *   Application Title: "XAI for LLMs"
    *   Data Generation Parameters:
        *   Number of synthetic LLM interaction samples (`num_samples`).
    *   Data Overview & Validation Trigger button.
    *   Global Filters (e.g., for plots).
*   **Main Content Area:**
    *   **Introduction:** Overview, Learning Goals.
    *   **Data Generation & Inspection:** Display of synthetic data, summary statistics.
    *   **Core XAI Concepts:** Interpretability vs. Transparency, Introduction to XAI Techniques (Saliency Maps, Counterfactuals).
    *   **XAI Technique Simulations:**
        *   Saliency Map visualization with interactive threshold.
        *   Counterfactual Explanation simulation.
    *   **Core Visualizations:**
        *   Faithfulness Metric over Time (Trend Plot) with interactive filters.
        *   Explanation Quality vs. Model Accuracy (Relationship Plot).
        *   Aggregated Saliency Comparison (Heatmap) with interactive `top_n_tokens`.
    *   **Interactive Parameters:** Explanation Verbosity filtering, Model Confidence filtering.
    *   **Conclusion & References.**

### Input Widgets and Controls

The application will feature several interactive controls to allow users to modify data generation and analysis parameters dynamically.

*   **`st.slider` for numerical ranges:**
    *   `num_samples`: For generating the core LLM interaction data (e.g., range 100-2000, default 500).
    *   `saliency_threshold`: For visualizing saliency maps (e.g., range 0.0-1.0, default 0.7).
    *   `top_n_tokens`: For the aggregated saliency heatmap (e.g., range 5-20, default 10).
    *   `verbosity_threshold`: For filtering explanations by quality score (e.g., range 0.0-1.0, default 0.9).
    *   `confidence_threshold`: For filtering data by model confidence (e.g., range 0.0-1.0, default 0.95).
*   **`st.selectbox` or `st.number_input` for selecting samples:**
    *   `sample_index`: To select a specific LLM interaction for detailed Saliency Map or Counterfactual Explanation (e.g., based on available indices in `df_llm_data`).
*   **`st.button` for triggering actions:**
    *   "Generate/Regenerate Data": To re-run `generate_llm_data` with new `num_samples`.
    *   "Run Data Validation": To execute `validate_and_summarize_data`.
*   **`st.expander` for collapsible sections:**
    *   To keep the main content area clean, detailed explanations or output tables will be placed inside expanders.

### Visualization Components

The application will utilize `matplotlib` and `seaborn` for generating plots, displayed using `st.pyplot`. Dataframes will be displayed using `st.dataframe` or `st.table`. Rich text with HTML highlighting will be displayed using `st.markdown(..., unsafe_allow_html=True)`.

*   **Tables:**
    *   `df_llm_data.head()`: Displayed using `st.dataframe`.
    *   `df_llm_data.info()` summary: Displayed as text using `st.write`.
    *   `df_saliency.head()`: Displayed using `st.dataframe`.
    *   `dataframe.describe()` (numerical summary): Displayed using `st.dataframe`.
    *   `dataframe[col].value_counts()` (categorical summary): Displayed using `st.dataframe`.
    *   Filtered dataframes (`filtered_by_verbosity_df`, `filtered_by_confidence_df`): Displayed using `st.dataframe`.
*   **Plots:**
    *   **Trend Plot:** "Faithfulness Metric over Time" (line plot) using `st.pyplot`.
    *   **Relationship Plot:** "Explanation Quality Score vs. Model Accuracy" (scatter plot) using `st.pyplot`.
    *   **Aggregated Comparison:** "Aggregated Influence of Top N Tokens" (heatmap) using `st.pyplot`.
*   **Text/HTML Output:**
    *   Saliency Map visualization: Highlighted text output using `st.markdown(html_content, unsafe_allow_html=True)`.
    *   Counterfactual Explanation: Original and counterfactual prompts/outputs displayed as `st.write` text.

### Interactive Elements and Feedback Mechanisms

*   **Dynamic Updates:** Plots and tables will re-render automatically when associated input parameters (sliders, selectboxes) are changed.
*   **Loading Indicators:** `st.spinner` or `st.status` will be used for long-running operations (e.g., data generation) to provide user feedback.
*   **Error Handling:** Implement `try-except` blocks around critical operations and display error messages using `st.error`. Warnings from data validation will be displayed using `st.warning`.
*   **Informative Text:** `st.info` will be used to provide inline help or context for sections.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

*   **Widget Help Text:** All interactive input widgets (`st.slider`, `st.selectbox`, `st.button`) will include `help` parameters to provide brief descriptions of their function, consistent with the Jupyter Notebook's "inline help text or tooltips" requirement.
*   **Plot Annotations:** Plots generated using `matplotlib`/`seaborn` will include clear titles, labeled axes, and legends as already implemented in the provided plotting functions.
*   **Explanatory Markdown:** Extensive use of `st.markdown` will provide narrative context, interpretations of visualizations, and explanations of XAI concepts, mimicking the detailed markdown cells of the notebook.

### Save the States of the Fields Properly so that Changes are Not Lost

*   The application will leverage Streamlit's `st.session_state` to persist the values of all input widgets (sliders, selectboxes) across reruns, ensuring that user selections are maintained.
*   Generated dataframes (`df_llm_data`, `df_saliency`) will also be stored in `st.session_state` after their initial creation or regeneration to avoid unnecessary recomputation.

## 4. Notebook Content and Code Requirements

This section outlines how the content from the Jupyter Notebook will be integrated into the Streamlit application, including extracted code stubs and markdown.

### Application Initialization

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from IPython.display import HTML # Used for saliency, will be adapted to st.markdown
from datetime import datetime, timedelta

# --- Configuration for Streamlit App ---
st.set_page_config(layout="wide", page_title="XAI for LLMs")
sns.set_theme(style="whitegrid", palette='viridis') # Apply color-blind friendly palette
plt.rcParams.update({'font.size': 12}) # Set font size >= 12 pt
```

### Markdown Content Integration

All Markdown cells from the Jupyter Notebook will be directly translated into `st.markdown` calls within the Streamlit application, preserving their headings and content.

#### Main Title and Overview
```python
st.title("Explainable AI (XAI) for LLMs")

st.markdown("## 1. Notebook Overview")
st.markdown("### Learning Goals")
st.markdown("""
This application aims to provide a practical understanding of Explainable AI (XAI) concepts and techniques as applied to Large Language Models (LLMs). Upon completion, users will be able to:
- Understand the core insights presented regarding XAI and LLMs.
- Distinguish clearly between interpretability and transparency in the context of AI models.
- Review and conceptually apply different XAI techniques, specifically saliency maps and counterfactual explanations.
- Analyze the inherent trade-offs between model performance (e.g., accuracy) and the explainability of its decisions.
""")

st.markdown("## 2. Setup and Library Imports")
st.markdown("This section describes the necessary Python libraries for data generation, manipulation, analysis, and visualization. The environment is pre-configured for executing the steps.")
```

#### Synthetic Data Generation Overview
```python
st.markdown("## 3. Overview of Synthetic Data Generation")
st.markdown("This section explains the rationale behind using synthetic data. Given the complexity and computational cost of working with real LLMs for illustrative purposes, synthetic data will be generated to simulate LLM inputs, outputs, and various XAI-related metrics. This approach allows for a controlled environment to explore XAI concepts.")
```

#### Explanation of Synthetic Data Structure
```python
st.markdown("## 6. Explanation of Synthetic Data Generation")
st.markdown("""
This section explains the structure of the generated synthetic dataset, detailing each column and its purpose in simulating LLM interactions and XAI metrics. It highlights how these synthetic values will enable the exploration of XAI concepts.

- **timestamp**: The date and time of the interaction.
- **prompt**: The input text given to the LLM.
- **llm_output**: The text generated by the LLM.
- **true_label**: A ground-truth label for classification tasks.
- **model_confidence**: The model's confidence in its prediction (a score between 0 and 1).
- **model_accuracy**: Whether the model's prediction was correct (1) or not (0).
- **explanation_quality_score**: A score representing the quality of the explanation ($Q_{exp}$).
- **faithfulness_metric**: A score representing how consistent the explanation is with the model's behavior.
- **xai_technique**: The XAI technique used for the explanation.
""")
```

#### XAI Concepts: Interpretability vs. Transparency
```python
st.markdown("## 11. Interpretability vs. Transparency")
st.markdown("""
This section delves into the fundamental distinction between interpretability and transparency in AI.
- **Interpretability** refers to understanding *why* a model made a specific decision. It's often about understanding the relationship between inputs and outputs.
- **Transparency** refers to comprehending the *inner workings* of a model, including its architecture, algorithms, and parameters.

For LLMs, true transparency is often infeasible due to their scale and complexity. While transparency is about "seeing inside the black box," interpretability is about "making sense of the black box's actions."
""")
```

#### Introduction to XAI Techniques
```python
st.markdown("## 12. Introduction to XAI Techniques")
st.markdown("""
This section introduces two key XAI techniques: Saliency Maps and Counterfactual Explanations.
- **Saliency Maps:** These techniques highlight input features (e.g., words, pixels) that are most influential in determining a model's output. Conceptually, for an input $X = (x_1, x_2, \dots, x_n)$ and model output $Y$, the saliency for token $x_i$ can be represented as:
  $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
  where $S(x_i)$ indicates the importance of token $x_i$ to the output $Y$.

- **Counterfactual Explanations:** These provide explanations by showing what minimal change to the input would have resulted in a different, desired output. This answers the question "What if...?" For an input $X$ leading to an output $Y$, a counterfactual explanation suggests a perturbed input $X'$ such that the model's output changes to a desired $Y' \ne Y$, i.e., $ \text{Model}(X + \Delta X) = Y' $ with minimal $ \Delta X $.
""")
```

#### Explanation of Saliency Map Visualization
```python
st.markdown("## 15. Explanation of Saliency Map Visualization")
st.markdown("This section interprets the visualized saliency map. It explains that the highlighted words are conceptually the most influential tokens in the LLM's decision for that particular output, based on our synthetic scoring.")
```

#### Explanation of Counterfactual Explanation
```python
st.markdown("## 18. Explanation of Counterfactual Explanation")
st.markdown("This section interprets the simulated counterfactual explanation. It explains that by making a small, defined change to the input prompt, a different output is achieved, conceptually showing what input conditions would lead to an alternative model decision.")
```

#### Analysis of Faithfulness Trend Plot
```python
st.markdown("## 21. Analysis of Faithfulness Trend Plot")
st.markdown("This section provides an analysis of the generated trend plot. It discusses how the faithfulness metric fluctuates over time and compares the consistency of different conceptual XAI techniques with the underlying (synthetic) model behavior. In this synthetic data, the trends are random, but in a real-world scenario, we would look for techniques that consistently maintain high faithfulness.")
```

#### Analysis of Explanation Quality vs. Model Accuracy Plot
```python
st.markdown("## 24. Analysis of Explanation Quality vs. Model Accuracy Plot")
st.markdown("This section analyzes the scatter plot, discussing any observed correlations or trade-offs between model accuracy and the perceived quality of explanations. It highlights how achieving high performance might sometimes come at the cost of explainability, and vice versa. In this synthetic dataset, the relationship is random, but in real applications, we might see a negative correlation, indicating a trade-off.")
```

#### Analysis of Aggregated Saliency Heatmap
```python
st.markdown("## 27. Analysis of Aggregated Saliency Heatmap")
st.markdown("This section interprets the heatmap, identifying which synthetic tokens consistently show high saliency scores. This provides insights into which input features are generally considered more influential by the (simulated) LLM. The heatmap shows the tokens with the highest average saliency scores across all the outputs we analyzed.")
```

#### Interactive Parameter Simulation: Explanation Verbosity
```python
st.markdown("## 28. Interactive Parameter Simulation: Explanation Verbosity")
st.markdown("""
This section introduces the concept of 'Explanation Verbosity' ($V_{exp} \in [0, 1]$), which can be used to filter or adjust the level of detail in explanations. A higher $V_{exp}$ would mean more detailed explanations. This section demonstrates how a user parameter could filter explanations based on their quality score, acting as a proxy for verbosity.
""")
```

#### Conclusion
```python
st.markdown("## 32. Conclusion")
st.markdown("""
This concluding section summarizes the key learnings from the application, reinforcing the distinction between interpretability and transparency, the application of XAI techniques, and the understanding of trade-offs between model performance and explainability for LLMs. Through synthetic data, we have simulated and visualized key XAI concepts, providing a foundation for applying these ideas to real-world models.
""")
```

#### References
```python
st.markdown("## 33. References")
st.markdown("""
- [1] Unit 5: Explainable and Trustworthy AI, Provided Resource Document. This unit discusses interpretability vs transparency, XAI techniques (saliency maps, counterfactual explanations, faithfulness metrics), and the trade-offs between explainability and model performance, noting the absence of explainability in generative AI.
""")
```

### Extracted Code Stubs and Streamlit Usage

#### Data Generation Functions
These functions will be defined in the Streamlit script and called based on user interaction or initial load.

```python
@st.cache_data # Cache data generation to improve performance
def generate_llm_data(num_samples):
    """Creates a synthetic pandas DataFrame simulating LLM interactions and associated XAI metrics."""
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples cannot be negative.")

    fake = Faker()
    
    if num_samples == 0:
        return pd.DataFrame(columns=[
        'timestamp', 'prompt', 'llm_output', 'true_label', 'model_confidence',
        'model_accuracy', 'explanation_quality_score', 'faithfulness_metric', 'xai_technique'
    ])

    start_date = datetime.now()
    timestamps = [fake.date_time_this_year() for _ in range(num_samples)]

    data = {
        'timestamp': sorted(timestamps),
        'prompt': [fake.sentence() for _ in range(num_samples)],
        'llm_output': [fake.text(max_nb_chars=100) for _ in range(num_samples)],
        'true_label': np.random.choice(['Positive', 'Negative', 'Neutral'], size=num_samples),
        'model_confidence': np.random.uniform(0.5, 1.0, size=num_samples),
        'model_accuracy': np.random.randint(0, 2, size=num_samples),
        'explanation_quality_score': np.random.uniform(0.5, 1.0, size=num_samples),
        'faithfulness_metric': np.random.uniform(0.6, 1.0, size=num_samples),
        'xai_technique': np.random.choice(['Saliency Map', 'Counterfactual', 'LIME'], size=num_samples)
    }

    return pd.DataFrame(data)

@st.cache_data # Cache saliency data generation
def generate_saliency_data(llm_outputs):
    """Generates synthetic token-level saliency scores for a given series of LLM text outputs."""
    saliency_records = [
        [index, token, np.random.rand()]
        for index, text in llm_outputs.items()
        for token in text.split()
    ]
    return pd.DataFrame(
        saliency_records,
        columns=['output_index', 'token', 'saliency_score']
    )
```

#### Data Validation and Summary Statistics
```python
def validate_and_summarize_data(dataframe):
    """Performs data integrity checks and prints summary statistics."""
    st.markdown("### 9. Data Validation and Summary Statistics")
    st.markdown("This section ensures the generated data is well-formed. It confirms expected column names and data types, checks for missing values in critical fields, and provides summary statistics for numeric and categorical columns. This step is crucial for data integrity before further analysis.")

    expected_columns = {
        'model_confidence': 'float64',
        'explanation_quality_score': 'float64',
        'faithfulness_metric': 'float64',
        'true_label': 'object',
        'xai_technique': 'object'
    }

    for col, dtype in expected_columns.items():
        if col not in dataframe.columns:
            st.warning(f"Warning: Missing expected column: {col}")
        else:
            if dataframe[col].dtype != dtype:
                st.warning(f"Warning: Column '{col}' has incorrect dtype: {dataframe[col].dtype}, expected {dtype}")

    if dataframe[['model_confidence', 'explanation_quality_score', 'faithfulness_metric']].isnull().any().any():
        st.error("Missing values found in critical fields")
    else:
        st.success("No missing values found in critical fields.")

    if dataframe.empty:
        st.warning("DataFrame is empty")
        return

    st.markdown("#### Numerical Summary:")
    st.dataframe(dataframe.describe())

    st.markdown("#### Categorical Summary:")
    for col in dataframe.select_dtypes(include=['object']).columns:
        if col in ['prompt', 'llm_output']: continue
        st.write(f"**{col} value counts:**")
        st.dataframe(dataframe[col].value_counts())
```

#### XAI Technique Simulation Functions

**Saliency Map Visualization:**
```python
def visualize_saliency_map(llm_output, token_scores, threshold=0.5):
    """Renders an LLM output string as HTML, visually highlighting tokens whose saliency scores exceed a specified threshold."""
    highlighted_parts = []
    for item in token_scores:
        if len(item) == 2:
            token, score = item
            if score >= threshold:
                highlighted_parts.append(f'<span style="background-color: yellow;">{token}</span>')
            else:
                highlighted_parts.append(token)
    
    html_content = " ".join(highlighted_parts)
    st.markdown(html_content, unsafe_allow_html=True)
```

**Counterfactual Explanation Simulation:**
```python
def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """Simulates a counterfactual explanation by creating a slightly modified version of an original prompt and a corresponding altered output."""
    if not isinstance(original_prompt, str):
        raise TypeError("original_prompt must be a string.")
    if not isinstance(original_output, str):
        raise TypeError("original_output must be a string.")
    if not isinstance(current_model_accuracy, (int, float)):
        raise TypeError("current_model_accuracy must be a float or an integer.")

    counterfactual_prompt = f"What if the question was: {original_prompt.replace('a', 'another')}"
    counterfactual_output = f"An alternative answer might be: {original_output[::-1]}"

    return {
        'original_prompt': original_prompt,
        'original_output': original_output,
        'counterfactual_prompt': counterfactual_prompt,
        'counterfactual_output': counterfactual_output
    }
```

#### Plotting Functions

```python
def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """Generates and displays a line plot to visualize the trend of a faithfulness metric over time."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=dataframe,
        x=x_axis,
        y=y_axis,
        hue=hue_column,
        marker='o',
        palette='viridis',
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=hue_column.replace('_', ' ').title(), fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

def plot_quality_vs_accuracy(dataframe, x_axis, y_axis, title):
    """Creates and displays a scatter plot to examine the relationship and potential trade-offs between model accuracy and explanation quality score."""
    if not isinstance(dataframe, pd.DataFrame):
        raise AttributeError("The 'dataframe' argument must be a pandas DataFrame.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dataframe, x=x_axis, y=y_axis, palette='viridis', ax=ax)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

def plot_aggregated_saliency_heatmap(saliency_dataframe, top_n_tokens, title):
    """Generates and displays a heatmap visualizing the aggregated influence of the most important input tokens across multiple LLM outputs."""
    if saliency_dataframe.empty:
        raise ValueError("dataframe cannot be empty")
    
    if not isinstance(top_n_tokens, int) or top_n_tokens <= 0:
        raise ValueError("top_n_tokens must be a positive integer")

    aggregated_saliency = saliency_dataframe.groupby('token')['saliency_score'].mean()
    top_tokens = aggregated_saliency.sort_values(ascending=False).head(top_n_tokens)

    heatmap_data = pd.DataFrame(top_tokens).rename(columns={'saliency_score': 'Aggregated Saliency'})

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        cbar_kws={'label': 'Aggregated Saliency Score'},
        ax=ax
    )

    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Token', fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
```

#### Interactive Filtering Functions

```python
def filter_by_verbosity(dataframe, verbosity_threshold):
    """Filters a DataFrame based on a proxy for explanation verbosity."""
    return dataframe[dataframe['explanation_quality_score'] >= verbosity_threshold]

def filter_by_confidence(dataframe, confidence_threshold):
    """Filters a DataFrame to include only rows where 'model_confidence' meets a specified minimum threshold."""
    return dataframe[dataframe['model_confidence'] >= confidence_threshold]
```

### Streamlit Application Flow (Conceptual)

1.  **Sidebar for Global Controls:**
    *   `num_samples_input = st.sidebar.slider("Number of synthetic LLM samples:", 100, 2000, 500, help="Adjust the number of LLM interaction records to generate.")`
    *   `if st.sidebar.button("Generate/Update Data", help="Click to generate new synthetic data."):`
        *   `with st.spinner("Generating LLM data..."):`
            *   `st.session_state['df_llm_data'] = generate_llm_data(num_samples_input)`
            *   `st.session_state['df_saliency'] = generate_saliency_data(st.session_state['df_llm_data']['llm_output'].head(max(10, num_samples_input // 50))) # Generate saliency for a subset`
    *   (Initial check for `st.session_state` to generate data on first run if not present).

2.  **Display Introduction & Learning Goals.**

3.  **Data Inspection Section:**
    *   `st.subheader("Data Overview")`
    *   `st.dataframe(st.session_state['df_llm_data'].head())`
    *   `if st.button("Run Data Validation", help="Perform checks on generated data."):`
        *   `validate_and_summarize_data(st.session_state['df_llm_data'])`

4.  **XAI Concepts and Simulations:**
    *   **Interpretability vs. Transparency** markdown.
    *   **Introduction to XAI Techniques** markdown.
    *   **Saliency Map Simulation:**
        *   `sample_index = st.number_input("Select sample index for Saliency Map:", 0, len(st.session_state['df_llm_data'])-1, 0)`
        *   `llm_output_sample = st.session_state['df_llm_data'].loc[sample_index, 'llm_output']`
        *   `saliency_scores_sample = st.session_state['df_saliency'][st.session_state['df_saliency']['output_index'] == sample_index][['token', 'saliency_score']].values.tolist()`
        *   `saliency_threshold = st.slider("Saliency Highlight Threshold:", 0.0, 1.0, 0.7, help="Tokens with saliency scores above this threshold will be highlighted.")`
        *   `st.write("Original Text:")`
        *   `st.write(llm_output_sample)`
        *   `st.write(f"\nSaliency Map (threshold = {saliency_threshold}):")`
        *   `visualize_saliency_map(llm_output_sample, saliency_scores_sample, saliency_threshold)`
        *   **Explanation of Saliency Map Visualization** markdown.
    *   **Counterfactual Explanation Simulation:**
        *   `original_prompt = st.session_state['df_llm_data'].loc[sample_index, 'prompt']`
        *   `original_output = st.session_state['df_llm_data'].loc[sample_index, 'llm_output']`
        *   `model_accuracy = st.session_state['df_llm_data'].loc[sample_index, 'model_accuracy']`
        *   `counterfactual = generate_counterfactual_explanation(original_prompt, original_output, model_accuracy)`
        *   Display `counterfactual` dictionary content.
        *   **Explanation of Counterfactual Explanation** markdown.

5.  **Core Visualizations Section:**
    *   `plot_faithfulness_trend(st.session_state['df_llm_data'], 'timestamp', 'faithfulness_metric', 'xai_technique', 'Faithfulness Metric over Time')`
    *   **Analysis of Faithfulness Trend Plot** markdown.
    *   `plot_quality_vs_accuracy(st.session_state['df_llm_data'], 'model_accuracy', 'explanation_quality_score', 'Explanation Quality Score vs. Model Accuracy')`
    *   **Analysis of Explanation Quality vs. Model Accuracy Plot** markdown.
    *   `top_n_tokens_heatmap = st.slider("Number of Top Tokens for Heatmap:", 5, 20, 10, help="Select how many top tokens to display in the aggregated saliency heatmap.")`
    *   `plot_aggregated_saliency_heatmap(st.session_state['df_saliency'], top_n_tokens_heatmap, f'Aggregated Influence of Top {top_n_tokens_heatmap} Tokens')`
    *   **Analysis of Aggregated Saliency Heatmap** markdown.

6.  **Interactive Parameter Simulation Section:**
    *   **Explanation Verbosity** markdown.
    *   `verbosity_threshold = st.slider("Explanation Verbosity Threshold ($V_{exp}$):", 0.0, 1.0, 0.9, help="Filter explanations by quality score. Higher threshold means more detailed/higher quality explanations.")`
    *   `filtered_by_verbosity_df = filter_by_verbosity(st.session_state['df_llm_data'], verbosity_threshold)`
    *   `st.write(f"Number of records with verbosity >= {verbosity_threshold}: {len(filtered_by_verbosity_df)}")`
    *   `st.dataframe(filtered_by_verbosity_df.head())`
    *   **Model Confidence Filtering** (similar structure as verbosity filter).

7.  **Conclusion & References.**
