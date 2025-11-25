import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from datetime import datetime, timedelta

# --- Configuration for Streamlit App ---
st.set_page_config(layout="wide", page_title="XAI for LLMs")
sns.set_theme(style="whitegrid", palette='viridis') # Apply color-blind friendly palette
plt.rcParams.update({'font.size': 12}) # Set font size >= 12 pt

# --- Session State Initialization ---
if 'df_llm_data' not in st.session_state:
    st.session_state['df_llm_data'] = pd.DataFrame()
if 'df_saliency' not in st.session_state:
    st.session_state['df_saliency'] = pd.DataFrame()

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
    saliency_records = []
    for index, text in llm_outputs.items():
        if isinstance(text, str): # Ensure text is a string
            for token in text.split():
                saliency_records.append([index, token, np.random.rand()])
    
    if not saliency_records:
        return pd.DataFrame(columns=['output_index', 'token', 'saliency_score'])

    return pd.DataFrame(
        saliency_records,
        columns=['output_index', 'token', 'saliency_score']
    )

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
        st.warning("Saliency data is empty, cannot generate heatmap.")
        return
    
    if not isinstance(top_n_tokens, int) or top_n_tokens <= 0:
        raise ValueError("top_n_tokens must be a positive integer")

    aggregated_saliency = saliency_dataframe.groupby('token')['saliency_score'].mean()
    if aggregated_saliency.empty:
        st.warning("No tokens found in saliency data for heatmap.")
        return

    top_tokens = aggregated_saliency.sort_values(ascending=False).head(top_n_tokens)

    heatmap_data = pd.DataFrame(top_tokens).rename(columns={'saliency_score': 'Aggregated Saliency'})

    if heatmap_data.empty:
        st.warning("No top tokens to display in heatmap.")
        return

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

def filter_by_verbosity(dataframe, verbosity_threshold):
    """Filters a DataFrame based on a proxy for explanation verbosity."""
    if dataframe.empty:
        return pd.DataFrame()
    return dataframe[dataframe['explanation_quality_score'] >= verbosity_threshold]

def filter_by_confidence(dataframe, confidence_threshold):
    """Filters a DataFrame to include only rows where 'model_confidence' meets a specified minimum threshold."""
    if dataframe.empty:
        return pd.DataFrame()
    return dataframe[dataframe['model_confidence'] >= confidence_threshold]

def main():
    st.title("Explainable AI (XAI) for LLMs")

    # --- Sidebar for Global Controls ---
    st.sidebar.title("XAI for LLMs")
    num_samples_input = st.sidebar.slider(
        "Number of synthetic LLM samples:", 
        100, 
        2000, 
        500, 
        help="Adjust the number of LLM interaction records to generate."
    )

    if st.sidebar.button("Generate/Update Data", help="Click to generate new synthetic data."):
        with st.spinner("Generating LLM data..."):
            st.session_state['df_llm_data'] = generate_llm_data(num_samples_input)
            # Generate saliency for a subset to avoid excessive computation
            # Using .head() is fine for demonstration; in real app, might sample or generate on demand
            if not st.session_state['df_llm_data'].empty:
                st.session_state['df_saliency'] = generate_saliency_data(
                    st.session_state['df_llm_data']['llm_output'].head(max(10, num_samples_input // 50))
                )
            else:
                st.session_state['df_saliency'] = pd.DataFrame()
        st.sidebar.success("Data generated successfully!")

    # Initial data generation on first run if not present
    if st.session_state['df_llm_data'].empty and num_samples_input > 0:
        with st.spinner("Initializing LLM data..."):
            st.session_state['df_llm_data'] = generate_llm_data(num_samples_input)
            if not st.session_state['df_llm_data'].empty:
                st.session_state['df_saliency'] = generate_saliency_data(
                    st.session_state['df_llm_data']['llm_output'].head(max(10, num_samples_input // 50))
                )
            else:
                st.session_state['df_saliency'] = pd.DataFrame()
        st.sidebar.success("Initial data generated successfully!")

    st.sidebar.divider()

    # --- Main Content Area ---
    st.markdown("## 1. Notebook Overview")
    st.markdown("### Learning Goals")
    st.markdown('''
This application aims to provide a practical understanding of Explainable AI (XAI) concepts and techniques as applied to Large Language Models (LLMs). Upon completion, users will be able to:
- Understand the core insights presented regarding XAI and LLMs.
- Distinguish clearly between interpretability and transparency in the context of AI models.
- Review and conceptually apply different XAI techniques, specifically saliency maps and counterfactual explanations.
- Analyze the inherent trade-offs between model performance (e.g., accuracy) and the explainability of its decisions.
''')

    st.markdown("## 2. Setup and Library Imports")
    st.markdown("This section describes the necessary Python libraries for data generation, manipulation, analysis, and visualization. The environment is pre-configured for executing the steps.")

    st.markdown("## 3. Overview of Synthetic Data Generation")
    st.markdown("This section explains the rationale behind using synthetic data. Given the complexity and computational cost of working with real LLMs for illustrative purposes, synthetic data will be generated to simulate LLM inputs, outputs, and various XAI-related metrics. This approach allows for a controlled environment to explore XAI concepts.")

    # --- Data Generation & Inspection ---
    st.header("4. Data Generation & Inspection")
    if not st.session_state['df_llm_data'].empty:
        st.markdown("### 5. Display of Synthetic Data")
        st.markdown("A glimpse of the generated synthetic dataset:")
        st.dataframe(st.session_state['df_llm_data'].head())
    else:
        st.info("Please generate data using the sidebar control.")

    st.markdown("## 6. Explanation of Synthetic Data Generation")
    st.markdown(r'''
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
''')

    if st.button("Run Data Validation", help="Perform checks on generated data."):
        if not st.session_state['df_llm_data'].empty:
            validate_and_summarize_data(st.session_state['df_llm_data'])
        else:
            st.warning("Please generate data first to run validation.")

    # --- Core XAI Concepts ---
    st.header("10. Core XAI Concepts")
    st.markdown("## 11. Interpretability vs. Transparency")
    st.markdown('''
This section delves into the fundamental distinction between interpretability and transparency in AI.
- **Interpretability** refers to understanding *why* a model made a specific decision. It's often about understanding the relationship between inputs and outputs.
- **Transparency** refers to comprehending the *inner workings* of a model, including its architecture, algorithms, and parameters.

For LLMs, true transparency is often infeasible due to their scale and complexity. While transparency is about "seeing inside the black box," interpretability is about "making sense of the black box's actions."
''')

    st.markdown("## 12. Introduction to XAI Techniques")
    st.markdown(r'''
This section introduces two key XAI techniques: Saliency Maps and Counterfactual Explanations.
- **Saliency Maps:** These techniques highlight input features (e.g., words, pixels) that are most influential in determining a model's output. Conceptually, for an input $X = (x_1, x_2, \dots, x_n)$ and model output $Y$, the saliency for token $x_i$ can be represented as:
  $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
  where $S(x_i)$ indicates the importance of token $x_i$ to the output $Y$.

- **Counterfactual Explanations:** These provide explanations by showing what minimal change to the input would have resulted in a different, desired output. This answers the question "What if...?" For an input $X$ leading to an output $Y$, a counterfactual explanation suggests a perturbed input $X'$ such that the model's output changes to a desired $Y' \ne Y$, i.e., $ \text{Model}(X + \Delta X) = Y' $ with minimal $ \Delta X $.
''')

    # --- XAI Technique Simulations ---
    st.header("13. XAI Technique Simulations")
    if not st.session_state['df_llm_data'].empty and not st.session_state['df_saliency'].empty:
        st.markdown("### 14. Saliency Map Visualization")
        sample_index_options = list(st.session_state['df_llm_data'].index)
        selected_sample_index = st.number_input(
            "Select sample index for Saliency Map:", 
            min_value=min(sample_index_options) if sample_index_options else 0,
            max_value=max(sample_index_options) if sample_index_options else 0,
            value=sample_index_options[0] if sample_index_options else 0,
            help="Select a specific LLM interaction sample to visualize its saliency map."
        )

        if not st.session_state['df_llm_data'].empty and selected_sample_index in st.session_state['df_llm_data'].index:
            llm_output_sample = st.session_state['df_llm_data'].loc[selected_sample_index, 'llm_output']
            
            saliency_scores_for_sample_df = st.session_state['df_saliency'][
                st.session_state['df_saliency']['output_index'] == selected_sample_index
            ]
            
            saliency_scores_sample = saliency_scores_for_sample_df[['token', 'saliency_score']].values.tolist()
            
            saliency_threshold = st.slider(
                "Saliency Highlight Threshold:", 
                0.0, 
                1.0, 
                0.7, 
                help="Tokens with saliency scores above this threshold will be highlighted in yellow."
            )

            st.write("Original Text:")
            st.write(llm_output_sample)
            st.write(f"\nSaliency Map (threshold = {saliency_threshold}):")
            visualize_saliency_map(llm_output_sample, saliency_scores_sample, saliency_threshold)

            st.markdown("## 15. Explanation of Saliency Map Visualization")
            st.markdown("This section interprets the visualized saliency map. It explains that the highlighted words are conceptually the most influential tokens in the LLM's decision for that particular output, based on our synthetic scoring.")
        else:
            st.info("Please ensure data is generated and a valid sample index is selected for Saliency Map visualization.")

        st.markdown("### 16. Counterfactual Explanation Simulation")
        if not st.session_state['df_llm_data'].empty and selected_sample_index in st.session_state['df_llm_data'].index:
            original_prompt = st.session_state['df_llm_data'].loc[selected_sample_index, 'prompt']
            original_output = st.session_state['df_llm_data'].loc[selected_sample_index, 'llm_output']
            model_accuracy = st.session_state['df_llm_data'].loc[selected_sample_index, 'model_accuracy']
            
            counterfactual = generate_counterfactual_explanation(original_prompt, original_output, model_accuracy)
            
            st.write("#### Original Interaction:")
            st.write(f"**Prompt:** {counterfactual['original_prompt']}")
            st.write(f"**Output:** {counterfactual['original_output']}")
            
            st.write("#### Counterfactual Interaction:")
            st.write(f"**Counterfactual Prompt:** {counterfactual['counterfactual_prompt']}")
            st.write(f"**Counterfactual Output:** {counterfactual['counterfactual_output']}")

            st.markdown("## 18. Explanation of Counterfactual Explanation")
            st.markdown("This section interprets the simulated counterfactual explanation. It explains that by making a small, defined change to the input prompt, a different output is achieved, conceptually showing what input conditions would lead to an alternative model decision.")
        else:
            st.info("Please ensure data is generated and a valid sample index is selected for Counterfactual Explanation simulation.")
    else:
        st.info("Please generate data in the sidebar to simulate XAI techniques.")

    # --- Core Visualizations ---
    st.header("19. Core Visualizations")
    if not st.session_state['df_llm_data'].empty:
        st.markdown("### 20. Faithfulness Metric over Time (Trend Plot)")
        plot_faithfulness_trend(
            st.session_state['df_llm_data'], 
            'timestamp', 
            'faithfulness_metric', 
            'xai_technique', 
            'Faithfulness Metric over Time'
        )
        st.markdown("## 21. Analysis of Faithfulness Trend Plot")
        st.markdown("This section provides an analysis of the generated trend plot. It discusses how the faithfulness metric fluctuates over time and compares the consistency of different conceptual XAI techniques with the underlying (synthetic) model behavior. In this synthetic data, the trends are random, but in a real-world scenario, we would look for techniques that consistently maintain high faithfulness.")

        st.markdown("### 22. Explanation Quality Score vs. Model Accuracy (Relationship Plot)")
        plot_quality_vs_accuracy(
            st.session_state['df_llm_data'], 
            'model_accuracy', 
            'explanation_quality_score', 
            'Explanation Quality Score vs. Model Accuracy'
        )
        st.markdown("## 24. Analysis of Explanation Quality vs. Model Accuracy Plot")
        st.markdown("This section analyzes the scatter plot, discussing any observed correlations or trade-offs between model accuracy and the perceived quality of explanations. It highlights how achieving high performance might sometimes come at the cost of explainability, and vice versa. In this synthetic dataset, the relationship is random, but in real applications, we might see a negative correlation, indicating a trade-off.")

        st.markdown("### 25. Aggregated Influence of Top N Tokens (Heatmap)")
        top_n_tokens_heatmap = st.slider(
            "Number of Top Tokens for Heatmap:", 
            5, 
            20, 
            10, 
            help="Select how many top tokens to display in the aggregated saliency heatmap."
        )
        plot_aggregated_saliency_heatmap(
            st.session_state['df_saliency'], 
            top_n_tokens_heatmap, 
            f'Aggregated Influence of Top {top_n_tokens_heatmap} Tokens'
        )
        st.markdown("## 27. Analysis of Aggregated Saliency Heatmap")
        st.markdown("This section interprets the heatmap, identifying which synthetic tokens consistently show high saliency scores. This provides insights into which input features are generally considered more influential by the (simulated) LLM. The heatmap shows the tokens with the highest average saliency scores across all the outputs we analyzed.")
    else:
        st.info("Please generate data in the sidebar to view core visualizations.")


    # --- Interactive Parameter Simulation ---
    st.header("28. Interactive Parameter Simulation")
    if not st.session_state['df_llm_data'].empty:
        st.markdown("## 28. Interactive Parameter Simulation: Explanation Verbosity")
        st.markdown(r'''
This section introduces the concept of 'Explanation Verbosity' ($V_{exp} \in [0, 1]$), which can be used to filter or adjust the level of detail in explanations. A higher $V_{exp}$ would mean more detailed explanations. This section demonstrates how a user parameter could filter explanations based on their quality score, acting as a proxy for verbosity.
''')
        verbosity_threshold = st.slider(
            "Explanation Verbosity Threshold ($V_{exp}$):", 
            0.0, 
            1.0, 
            0.9, 
            help="Filter explanations by quality score. Higher threshold means more detailed/higher quality explanations."
        )
        filtered_by_verbosity_df = filter_by_verbosity(st.session_state['df_llm_data'], verbosity_threshold)
        st.write(f"Number of records with verbosity >= {verbosity_threshold}: {len(filtered_by_verbosity_df)}")
        if not filtered_by_verbosity_df.empty:
            st.dataframe(filtered_by_verbosity_df.head())
        else:
            st.info("No records match the selected verbosity threshold.")

        st.markdown("### 29. Interactive Parameter Simulation: Model Confidence Filtering")
        st.markdown("This section demonstrates how to filter LLM interactions based on the model's confidence in its predictions. Users can adjust a threshold to only view explanations for predictions where the model had a certain level of confidence, which is crucial for safety monitoring in critical applications.")
        confidence_threshold = st.slider(
            "Model Confidence Threshold:", 
            0.0, 
            1.0, 
            0.95, 
            help="Filter data by model confidence. Only records with confidence above this threshold will be shown."
        )
        filtered_by_confidence_df = filter_by_confidence(st.session_state['df_llm_data'], confidence_threshold)
        st.write(f"Number of records with confidence >= {confidence_threshold}: {len(filtered_by_confidence_df)}")
        if not filtered_by_confidence_df.empty:
            st.dataframe(filtered_by_confidence_df.head())
        else:
            st.info("No records match the selected confidence threshold.")
    else:
        st.info("Please generate data in the sidebar to use interactive parameter simulations.")

    # --- Conclusion & References ---
    st.header("32. Conclusion & References")
    st.markdown("## 32. Conclusion")
    st.markdown('''
This concluding section summarizes the key learnings from the application, reinforcing the distinction between interpretability and transparency, the application of XAI techniques, and the understanding of trade-offs between model performance and explainability for LLMs. Through synthetic data, we have simulated and visualized key XAI concepts, providing a foundation for applying these ideas to real-world models.
''')

    st.markdown("## 33. References")
    st.markdown('''
- [1] Unit 5: Explainable and Trustworthy AI, Provided Resource Document. This unit discusses interpretability vs transparency, XAI techniques (saliency maps, counterfactual explanations, faithfulness metrics), and the trade-offs between explainability and model performance, noting the absence of explainability in generative AI.
''')

