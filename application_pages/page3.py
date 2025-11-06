import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

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

def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    Generates a line plot showing trends of a specified metric over time for different categories.
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

    # Convert timestamp to datetime if not already
    if x_axis in dataframe.columns and not pd.api.types.is_datetime64_any_dtype(dataframe[x_axis]):
        try:
            dataframe[x_axis] = pd.to_datetime(dataframe[x_axis])
        except Exception as e:
            st.error(f"Could not convert \'{x_axis}\' to datetime: {e}")
            return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue_column, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(x_axis.replace("_", " ").title())
    ax.set_ylabel(y_axis.replace("_", " ").title())
    ax.legend(title=hue_column.replace("_", " ").title())
    ax.grid(visible=True)
    
    st.pyplot(fig) # Display in Streamlit
    plt.close(fig) # Close figure to free memory

def plot_explanation_accuracy_relationship(dataframe):
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot explanation quality vs. accuracy.")
        return
    
    for column in ['model_accuracy', 'explanation_quality_score', 'xai_technique']:
        if column not in dataframe.columns:
            st.error(f"Column \'{column}\' is missing from DataFrame, cannot plot explanation quality vs. accuracy.")
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

def plot_average_faithfulness_by_technique(dataframe):
    if dataframe.empty:
        st.warning("DataFrame is empty, cannot plot average faithfulness by technique.")
        return
    
    for column in ['xai_technique', 'faithfulness_metric']:
        if column not in dataframe.columns:
            st.error(f"Column \'{column}\' is missing from DataFrame, cannot plot average faithfulness by technique.")
            return

    avg_faithfulness = dataframe.groupby('xai_technique', observed=False)['faithfulness_metric'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=avg_faithfulness, x='xai_technique', y='faithfulness_metric', ax=ax)
    ax.set_title("Average Faithfulness by XAI Technique")
    ax.set_xlabel("XAI Technique")
    ax.set_ylabel("Average Faithfulness Metric")
    ax.grid(visible=True)
    st.pyplot(fig)
    plt.close(fig)

def run_page3():
    st.title("Trend & Trade-off Analysis")

    st.markdown("""
    ## 3.19 Core Visual: Faithfulness Metric Over Time — Context & Business Value
    Monitoring explanation faithfulness over time helps detect drift in how well explanations align with modeled behavior. This is essential for governance dashboards and incident review.

    Formula and setup:
    -   Faithfulness (simulated): $F_t \in [0,1]$ for interaction at time t. Higher is better.
    -   We stratify by XAI technique (e.g., LIME, SHAP, GradCAM) to spot consistency gaps across methods.

    We will produce a line chart with a color-blind-friendly palette and save a PNG fallback for reporting workflows.
    """)

    if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
        st.subheader("Filtering Controls")

        df_llm_data = st.session_state['df_llm_data'].copy()
        
        min_quality = st.slider(
            "Minimum Explanation Quality Score", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.01,
            help="Filter to show only records with explanation quality scores above this value."
        )
        min_confidence = st.slider(
            "Minimum Model Confidence", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.01,
            help="Filter to show only records with model confidence scores above this value."
        )

        # Ensure xai_technique column exists before getting unique values
        xai_techniques = []
        if 'xai_technique' in df_llm_data.columns:
            xai_techniques = df_llm_data['xai_technique'].unique().tolist()
        
        selected_techniques = st.multiselect(
            "Filter by XAI Technique", 
            options=xai_techniques, 
            default=xai_techniques,
            help="Select which XAI techniques to include in the analysis and visualizations."
        )

        filtered_df = df_llm_data[
            (df_llm_data['explanation_quality_score'] >= min_quality) &
            (df_llm_data['model_confidence'] >= min_confidence)
        ]

        if 'xai_technique' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['xai_technique'].isin(selected_techniques))
            ]
        elif selected_techniques: # If selected_techniques is not empty but column is missing
            st.warning("Cannot filter by XAI Technique: 'xai_technique' column not found in data.")

        st.subheader("Core Visualizations")
        plot_faithfulness_trend(
            dataframe=filtered_df,
            x_axis='timestamp',
            y_axis='faithfulness_metric',
            hue_column='xai_technique',
            title='Faithfulness Metric over Time by XAI Technique'
        )

        plot_explanation_accuracy_relationship(filtered_df)
        plot_average_faithfulness_by_technique(filtered_df)

    else:
        st.info("Please generate or upload data on the 'Data Generation & Validation' page to proceed with trend and trade-off analysis.")
