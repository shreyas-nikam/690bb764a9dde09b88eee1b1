import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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

import pandas as pd
import numpy as np

def generate_saliency_data(llm_outputs):
    """Generates synthetic token-level saliency scores for LLM output strings."""
    
    data = []
    
    for idx, output in enumerate(llm_outputs):
        tokens = output.split()
        for token in tokens:
            saliency_score = np.random.rand()
            data.append((idx, token, saliency_score))
    
    return pd.DataFrame(data, columns=['output_index', 'token', 'saliency_score'])

import pandas as pd

def validate_and_summarize_data(dataframe):
    """
    Validates and summarizes input DataFrame.
    Arguments:
        dataframe: The pandas DataFrame to validate and summarize.
    Output:
        None. Prints validation messages and summary statistics to the console.
    """
    # Check if the DataFrame is empty
    if dataframe.empty:
        raise AssertionError("DataFrame is empty")
    
    # Check for expected columns
    expected_columns = {'numeric_column', 'categorical_column', 'critical_numeric'}
    missing_columns = expected_columns - set(dataframe.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns: {missing_columns}")

    # Check data types
    if not pd.api.types.is_numeric_dtype(dataframe['numeric_column']):
        raise AssertionError("'numeric_column' should be of numeric type")
    if not pd.api.types.is_numeric_dtype(dataframe['critical_numeric']):
        raise AssertionError("'critical_numeric' should be of numeric type")
    
    # Check for missing critical values
    if dataframe['critical_numeric'].isnull().any():
        raise AssertionError("Missing values in 'critical_numeric' column")
    
    # Print summary statistics
    print("Summary Statistics:")
    print(dataframe.describe())
    
    # Print value counts for categorical columns
    for column in dataframe.select_dtypes(include=['object']).columns:
        print(f"Value counts for {column}:")
        print(dataframe[column].value_counts())

from IPython.display import HTML

def visualize_saliency_map(llm_output, token_scores, threshold):
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
    return HTML(highlighted_text)

def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """
    Generates a counterfactual explanation by proposing a modified input prompt
    and its resulting output.

    Arguments:
    original_prompt: The initial input prompt to the LLM.
    original_output: The LLM's response to the original prompt.
    current_model_accuracy: The simulated accuracy of the model for the original output.

    Output:
    A dictionary containing 'original_prompt', 'original_output', 'counterfactual_prompt', and 'counterfactual_output'.
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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    Generates a line plot showing trends of a specified metric over time for different categories.
    Displays the plot and saves it as a PNG file.
    """
    # Validate dataframe
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")

    # Check necessary columns
    for column in [x_axis, y_axis, hue_column]:
        if column not in dataframe.columns:
            raise KeyError(f"Column '{column}' is missing from DataFrame.")

    # Validate data types
    if not pd.api.types.is_numeric_dtype(dataframe[y_axis]):
        raise TypeError(f"Column '{y_axis}' must have numeric data type.")

    if not pd.api.types.is_datetime64_any_dtype(dataframe[x_axis]) and not pd.api.types.is_numeric_dtype(dataframe[x_axis]):
        raise ValueError(f"Column '{x_axis}' must have datetime or numeric data type.")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue_column)
    
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend(title=hue_column)
    plt.grid(visible=True)
    
    plt.savefig('plot_faithfulness_trend.png')
    plt.show()

import matplotlib.pyplot as plt

def plot_quality_vs_accuracy(dataframe, x_axis, y_axis, title):
    """
    Generates a scatter plot to examine the relationship between two metrics.

    Arguments:
        dataframe: The pandas DataFrame containing the data.
        x_axis: The column name for the x-axis.
        y_axis: The column name for the y-axis.
        title: The title of the plot.

    Output:
        None. Displays the plot and saves it as a PNG file.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(dataframe[x_axis], dataframe[y_axis], alpha=0.7)
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.grid(True)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.show()
    except KeyError as e:
        raise KeyError(f"Missing column in DataFrame: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_aggregated_saliency_heatmap(saliency_dataframe, top_n_tokens, title):
    """
    Creates a heatmap visualizing the aggregated influence of top input tokens.
    
    Arguments:
    saliency_dataframe: A pandas DataFrame containing token-level saliency scores ('token', 'saliency_score').
    top_n_tokens: The number of top tokens to include in the heatmap.
    title: The title of the plot.
    
    Output:
    None. Displays the plot and saves it as a PNG file.
    """
    if not isinstance(top_n_tokens, int):
        raise TypeError("top_n_tokens must be an integer")
    if top_n_tokens < 0:
        raise ValueError("top_n_tokens must be non-negative")
    if top_n_tokens > len(saliency_dataframe):
        raise IndexError("top_n_tokens exceeds the number of available tokens")

    # Select top N tokens based on saliency_score
    top_tokens_df = saliency_dataframe.nlargest(top_n_tokens, 'saliency_score')

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(data=top_tokens_df[['saliency_score']].T, annot=True, fmt=".2f", 
                xticklabels=top_tokens_df['token'], cmap='YlGnBu')
    plt.title(title)
    plt.yticks([], [])
    plt.xlabel('Tokens')
    plt.ylabel('Saliency Score')

    # Save as PNG
    plt.savefig('saliency_heatmap.png')
    plt.show()

import pandas as pd

def filter_by_verbosity(dataframe, verbosity_threshold):
    """
    Filters the input DataFrame based on the 'explanation_quality_score' column.
    Only records where the score meets or exceeds the threshold are returned.
    """
    if 'explanation_quality_score' not in dataframe.columns:
        raise KeyError("The required column 'explanation_quality_score' is missing from the DataFrame.")
    
    try:
        return dataframe[dataframe['explanation_quality_score'] >= verbosity_threshold]
    except TypeError:
        raise ValueError("The 'explanation_quality_score' column must contain numeric values.")

import pandas as pd

def filter_by_confidence(dataframe, confidence_threshold):
    """
    Filters the input DataFrame based on a simulated model confidence score.
    
    Arguments:
        dataframe: The pandas DataFrame to filter.
        confidence_threshold: A float representing the minimum model confidence score required for a record to be included.
    
    Output:
        A filtered pandas DataFrame.
    """
    if not isinstance(confidence_threshold, (int, float)):
        raise TypeError("Confidence threshold must be a number.")
        
    filtered_df = dataframe[dataframe['model_confidence'] >= confidence_threshold]
    return filtered_df