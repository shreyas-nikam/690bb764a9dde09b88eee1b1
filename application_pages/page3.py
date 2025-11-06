
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

# Reproducibility (will be set once, perhaps in session_state or app initialization)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    Generates a line plot showing trends of a specified metric over time for different categories using Plotly.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot faithfulness trend.")
        return

    for column in [x_axis, y_axis, hue_column]:
        if column not in dataframe.columns:
            st.error(f"Column \'{column}\' is missing from DataFrame, cannot plot faithfulness trend.")
            return

    if not pd.api.types.is_numeric_dtype(dataframe[y_axis]):
        st.error(f"Column \'{y_axis}\' must have numeric data type, cannot plot faithfulness trend.")
        return

    fig = px.line(dataframe, x=x_axis, y=y_axis, color=hue_column, title=title)
    fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_explanation_accuracy_relationship(dataframe):
    """
    Generates a scatter plot of explanation quality vs. model accuracy using Plotly.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot explanation quality vs. accuracy.")
        return
    
    if 'model_accuracy' not in dataframe.columns or 'explanation_quality_score' not in dataframe.columns or 'xai_technique' not in dataframe.columns:
        st.error("Missing required columns for explanation quality vs. accuracy plot.")
        return

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

def plot_average_faithfulness_by_technique(dataframe):
    """
    Generates a bar chart of average faithfulness by XAI technique using Plotly.
    """
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot average faithfulness by technique.")
        return
    
    if 'xai_technique' not in dataframe.columns or 'faithfulness_metric' not in dataframe.columns:
        st.error("Missing required columns for average faithfulness by technique plot.")
        return
    
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

def run_page3(filtered_df):
    st.header("Page 3: Counterfactuals & Trend Analysis")

    st.markdown(r"""
    ## 3.16 Applying XAI Technique: Counterfactual Explanation — Context & Business Value
    Counterfactuals answer the question “What minimal change to the input would have changed the output?” They are critical for recourse (what a user could do differently), fairness (sensitivity to protected attributes), and debugging (what levers flip the decision). In the context of safety monitoring for agentic AI, counterfactuals can help us understand what minor changes in an instruction or environment might lead to an unsafe outcome.

    Formula (goal):
    -   Find a perturbation $\Delta X$ such that the model output flips: $$ \text{Model}(X + \Delta X) = Y' \ne Y $$
    -   With minimal change: minimize $||\Delta X||$ subject to the flip. In this application, we simulate a simple, illustrative counterfactual.
    """)

    if not filtered_df.empty and 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
        st.subheader("Counterfactual Explanation Simulation")
        
        # Use the original (unfiltered) dataframe for selection to ensure all indices are available
        # This ensures consistency in index selection across pages and with actual data generation
        df_for_selection = st.session_state['df_llm_data']
        sample_indices_available = df_for_selection.index.tolist()

        if not sample_indices_available:
            st.warning("No LLM interaction data available for counterfactual generation. Please generate data in Page 1.")
            return
            
        selected_record_display_index = st.number_input(
            "Select Record Index for Counterfactuals", 
            min_value=0, 
            max_value=len(sample_indices_available)-1, 
            value=0,
            help="Choose an index from the dataset to generate a counterfactual explanation for its prompt and output."
        )
        
        # Map display index to actual dataframe index
        cf_idx = sample_indices_available[selected_record_display_index]

        original_prompt = df_for_selection.loc[cf_idx, 'prompt']
        original_output = df_for_selection.loc[cf_idx, 'llm_output']
        current_model_accuracy = float(df_for_selection.loc[cf_idx, 'model_accuracy'])

        cf_result = generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy)

        st.write('**Original Prompt:**', cf_result['original_prompt'])
        st.write('**Original Output:**', cf_result['original_output'])
        st.write('---')
        st.write('**Counterfactual Prompt:**', cf_result['counterfactual_prompt'])
        st.write('**Counterfactual Output:**', cf_result['counterfactual_output'])

        st.markdown(r"""
        ## 3.18 Interpreting the Counterfactual Explanation
        The counterfactual demonstrates how a small prompt change could plausibly lead to a different outcome. In practice, this is invaluable for safety monitoring as it:
        -   Helps determine levers for decision recourse (what a user could change to get a safe outcome).
        -   Reveals sensitivity to specific terms or conditions, which is crucial for fairness auditing and identifying adversarial attacks in agentic systems.
        -   Supports root-cause analysis by contrasting original vs. perturbed inputs and outcomes, helping to understand why an agent might deviate from safe behavior.
        """)
    else:
        st.info("Please ensure data is generated/uploaded in 'Page 1: Introduction & Data Setup' to simulate counterfactuals.")
        return # Exit if no data

    st.markdown("---")

    st.markdown(r"""
    ## 3.19 Core Visual: Faithfulness Metric Over Time — Context & Business Value
    Monitoring explanation faithfulness over time helps detect drift in how well explanations align with modeled behavior. This is essential for governance dashboards and incident review, especially for understanding the long-term reliability of safety explanations in agentic systems.

    Formula and setup:
    -   Faithfulness (simulated): $F_t \in [0,1]$ for interaction at time t. Higher is better.
    -   We stratify by XAI technique (e.g., LIME, SHAP, GradCAM) to spot consistency gaps across methods.

    We will produce a line chart with a color-blind-friendly palette.
    """)

    st.subheader("Core Visualizations (Filtered Data)")
    if not filtered_df.empty:
        plot_faithfulness_trend(
            dataframe=filtered_df,
            x_axis='timestamp',
            y_axis='faithfulness_metric',
            hue_column='xai_technique',
            title='Faithfulness Metric over Time by XAI Technique'
        )
    else:
        st.info("No data available for faithfulness trend plot after applying filters. Adjust filters or generate data.")

    st.markdown("---")
    st.markdown(r"""
    ### Explanation Quality vs. Model Accuracy
    This scatter plot illustrates the potential trade-off between the quality of an explanation and the model's accuracy. Understanding this relationship is vital for balancing interpretability requirements with model performance goals in safety-critical applications. For instance, a highly accurate model with poor explanations might be harder to debug if it makes a safety error.
    """)
    if not filtered_df.empty:
        plot_explanation_accuracy_relationship(filtered_df)
    else:
        st.info("No data available for explanation quality vs. model accuracy plot after applying filters.")

    st.markdown("---")
    st.markdown(r"""
    ### Average Faithfulness by XAI Technique
    This bar chart provides an aggregated comparison of the average faithfulness metric across different XAI techniques. It helps in evaluating which explanation techniques consistently provide better alignment with the model's behavior, guiding the selection of techniques for robust safety monitoring.
    """)
    if not filtered_df.empty:
        plot_average_faithfulness_by_technique(filtered_df)
    else:
        st.info("No data available for average faithfulness by technique plot after applying filters.")

