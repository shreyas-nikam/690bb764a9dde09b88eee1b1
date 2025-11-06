import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_llm_data(num_samples):
    """    Creates a synthetic pandas DataFrame simulating LLM interactions and associated XAI metrics such as model confidence, explanation quality, and faithfulness.
Arguments:
num_samples (int): The number of synthetic data records to generate.
Output:
pandas.DataFrame: A DataFrame containing columns for timestamp, prompt, llm_output, model_confidence, model_accuracy, and various XAI metrics.
    """
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples cannot be negative.")

    columns = [
        'timestamp', 'prompt', 'llm_output', 'model_confidence',
        'model_accuracy', 'explanation_quality', 'faithfulness'
    ]
    
    if num_samples == 0:
        return pd.DataFrame(columns=columns)

    start_date = datetime.now()
    timestamps = [start_date - timedelta(days=np.random.randint(0, 365), seconds=np.random.randint(0, 86400)) for _ in range(num_samples)]

    sample_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "Write a short story about a robot.",
        "Summarize the plot of Hamlet.",
        "Translate 'hello world' into Spanish."
    ]
    
    sample_outputs = [
        "The capital of France is Paris.",
        "It's a theory by Einstein about space and time.",
        "Unit 734 felt a strange new emotion: joy.",
        "A prince seeks revenge for his father's murder.",
        "'Hola, mundo.'"
    ]

    data = {
        'timestamp': sorted(timestamps),
        'prompt': np.random.choice(sample_prompts, size=num_samples),
        'llm_output': np.random.choice(sample_outputs, size=num_samples),
        'model_confidence': np.random.uniform(0.75, 1.0, size=num_samples),
        'model_accuracy': np.random.uniform(0.8, 1.0, size=num_samples),
        'explanation_quality': np.random.uniform(0.5, 1.0, size=num_samples),
        'faithfulness': np.random.uniform(0.6, 1.0, size=num_samples)
    }

    return pd.DataFrame(data)

import pandas as pd
import numpy as np


def generate_saliency_data(llm_outputs):
    """
    Generates synthetic token-level saliency scores for a given series of LLM text outputs.

    Arguments:
        llm_outputs (pandas.Series): A series of strings, where each string is a simulated LLM output.
    Output:
        pandas.DataFrame: A DataFrame with columns 'output_index', 'token', and 'saliency_score'.
    """

    # Use a list comprehension to build a list of records for the DataFrame.
    # This iterates through each text output, splits it into tokens,
    # and creates a record for each token with its original index and a random score.
    saliency_records = [
        [index, token, np.random.rand()]
        for index, text in llm_outputs.items()
        for token in text.split()  # split() handles whitespace and empty strings
    ]

    # Create the DataFrame from the list of records.
    # This approach correctly handles an empty input series, resulting in an
    # empty DataFrame with the specified columns.
    return pd.DataFrame(
        saliency_records,
        columns=['output_index', 'token', 'saliency_score']
    )

import pandas as pd

def validate_and_summarize_data(dataframe):
    """
    Performs data integrity checks on a given DataFrame by verifying expected column names
    and data types, checking for missing values in critical fields, and printing summary
    statistics for both numerical and categorical columns.
    """

    # Expected columns and their data types
    expected_columns = {
        'model_confidence': 'float64',
        'explanation_quality_score': 'float64',
        'faithfulness_metric': 'float64',
        'true_label': 'object',
        'xai_technique': 'object'
    }

    # Check for missing columns
    for col, dtype in expected_columns.items():
        if col not in dataframe.columns:
            print(f"Warning: Missing expected column: {col}")
        else:
            # Check for incorrect data types
            if dataframe[col].dtype != dtype:
                print(f"Warning: Column '{col}' has incorrect dtype")

    # Check for missing values
    if dataframe.isnull().values.any():
        missing_in_critical = dataframe[['model_confidence', 'explanation_quality_score', 'faithfulness_metric']].isnull().any().any()
        if missing_in_critical:
            print("Missing values found in critical fields")
    else:
        print("No missing values found")

    if dataframe.empty:
        print("DataFrame is empty")
        return

    # Summarize numerical data
    print("Numerical summary:")
    print(dataframe.describe())

    # Summarize categorical data
    print("Categorical summary:")
    for col in dataframe.select_dtypes(include=['object']).columns:
        print(f"\n{col} value counts:")
        print(dataframe[col].value_counts())

from IPython.display import HTML

def visualize_saliency_map(llm_output, token_scores, threshold):
    """    Renders an LLM output string as HTML, visually highlighting tokens whose saliency scores exceed a specified threshold. This provides a visual representation of a saliency map.
Arguments:
llm_output (str): The text output from the LLM.
token_scores (list): A list of tuples, where each tuple contains a token (str) and its corresponding saliency score (float).
threshold (float): The score above which a token will be highlighted.
Output:
IPython.display.HTML: A displayable object that renders the highlighted text within a Jupyter environment.
    """
    highlighted_parts = []
    for token, score in token_scores:
        if score >= threshold:
            # Wrap the token in a span with a yellow background if its score is above or equal to the threshold
            highlighted_parts.append(f'<span style="background-color: yellow;">{token}</span>')
        else:
            # Otherwise, just add the token as is
            highlighted_parts.append(token)
    
    # Join the parts with spaces to form the final HTML string
    html_content = " ".join(highlighted_parts)
    
    # Return the content as a displayable HTML object
    return HTML(html_content)

def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """    Simulates a counterfactual explanation by creating a slightly modified version of an original prompt and a corresponding altered output, demonstrating how a minimal input change could lead to a different outcome.
Arguments:
original_prompt (str): The initial input text.
original_output (str): The initial LLM output text.
current_model_accuracy (float): The accuracy associated with the original output.
Output:
dict: A dictionary containing the 'original_prompt', 'original_output', 'counterfactual_prompt', and 'counterfactual_output'.
    """

    # Validate input types to handle invalid data as per test cases.
    if not isinstance(original_prompt, str):
        raise TypeError("original_prompt must be a string.")
    if not isinstance(original_output, str):
        raise TypeError("original_output must be a string.")
    if not isinstance(current_model_accuracy, (int, float)):
        raise TypeError("current_model_accuracy must be a float or an integer.")

    # Create a simple, deterministic modification for the prompt to ensure it differs.
    counterfactual_prompt = f"What if the question was: {original_prompt}"

    # Create a simple, deterministic modification for the output.
    counterfactual_output = f"An alternative answer might be: {original_output}"

    # Return the dictionary with original and counterfactual data.
    return {
        'original_prompt': original_prompt,
        'original_output': original_output,
        'counterfactual_prompt': counterfactual_prompt,
        'counterfactual_output': counterfactual_output
    }

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """    Generates and displays a line plot to visualize the trend of a faithfulness metric over time, with separate lines for different XAI techniques. This helps in analyzing the consistency of explanations.
Arguments:
dataframe (pandas.DataFrame): The source data for plotting.
x_axis (str): The name of the column to use for the x-axis (e.g., 'timestamp').
y_axis (str): The name of the column to use for the y-axis (e.g., 'faithfulness_metric').
hue_column (str): The name of the categorical column to differentiate the lines (e.g., 'xai_technique').
title (str): The title of the plot.
Output:
None: This function displays the plot and saves it to a file.
    """

    # A try...finally block ensures that plt.close() is called to free resources,
    # regardless of whether plotting succeeds or fails.
    try:
        # Set the visual style and figure size for the plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Create the line plot using seaborn.
        # This call handles the core plotting logic and will naturally raise
        # the exceptions expected by the test cases for invalid inputs:
        # - AttributeError for non-DataFrame input.
        # - KeyError for missing columns.
        # - TypeError for non-numeric y-axis.
        ax = sns.lineplot(
            data=dataframe,
            x=x_axis,
            y=y_axis,
            hue=hue_column,
            marker='o' # Add markers for better data point visibility
        )

        # Set plot title and labels for clarity
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel(y_axis.replace('_', ' ').title())

        # Improve layout and readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot to a file. A sanitized version of the title is used as the filename.
        safe_filename = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        filename = f"{safe_filename.replace(' ', '_').lower()}.png"
        plt.savefig(filename)

        # Display the plot
        plt.show()

    finally:
        # Close the current figure to free up memory
        plt.close()

import pandas as pd
import seaborn
import matplotlib.pyplot

def plot_quality_vs_accuracy(dataframe, x_axis, y_axis, title):
    """    Creates and displays a scatter plot to examine the relationship and potential trade-offs between model accuracy and explanation quality score.
Arguments:
dataframe (pandas.DataFrame): The source data containing the metrics to plot.
x_axis (str): The name of the column for the x-axis (e.g., 'model_accuracy').
y_axis (str): The name of the column for the y-axis (e.g., 'explanation_quality_score').
title (str): The title for the plot.
Output:
None: This function displays the plot and saves it to a file.
    """
    if not isinstance(dataframe, pd.DataFrame):
        # This explicit check helps in raising an AttributeError for non-DataFrame inputs,
        # which aligns with test_plot_quality_vs_accuracy_invalid_dataframe_type.
        # Otherwise, a different error might be raised by seaborn.
        raise AttributeError("The 'dataframe' argument must be a pandas DataFrame.")

    try:
        # Create the scatter plot using seaborn
        seaborn.scatterplot(data=dataframe, x=x_axis, y=y_axis)
        
        # Set the title of the plot
        matplotlib.pyplot.title(title)
        
        # Save the plot to a file. A default filename is used as none is specified.
        matplotlib.pyplot.savefig('quality_vs_accuracy.png')
        
        # Display the plot
        matplotlib.pyplot.show()

    except KeyError as e:
        # Re-raise KeyError if a specified column is not in the DataFrame
        raise KeyError(f"Column not found in DataFrame: {e}")
    except TypeError as e:
        # Re-raise TypeError if data is not numeric, which is raised by plotting library
        raise TypeError(f"Non-numeric data found in plot columns: {e}")
    except Exception as e:
        # Catch any other unexpected errors during plotting
        # This helps in debugging but in this context, main errors are covered above.
        # For the test cases, this generic catch is not strictly necessary but is good practice.
        print(f"An unexpected error occurred: {e}")
        raise

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_aggregated_saliency_heatmap(saliency_dataframe, top_n_tokens, title):
    """    Generates and displays a heatmap visualizing the aggregated influence of the most important input tokens across multiple LLM outputs. This helps identify tokens that are consistently influential.
Arguments:
saliency_dataframe (pandas.DataFrame): A DataFrame containing token-level saliency scores.
top_n_tokens (int): The number of most salient tokens to include in the heatmap.
title (str): The title for the plot.
Output:
None: This function displays the plot and saves it to a file.
    """
    # Validate inputs
    if saliency_dataframe.empty:
        raise ValueError("dataframe cannot be empty")
    
    if not isinstance(top_n_tokens, int) or top_n_tokens <= 0:
        raise ValueError("top_n_tokens must be a positive integer")

    try:
        # Aggregate saliency scores by token (calculating the mean)
        aggregated_saliency = saliency_dataframe.groupby('token')['saliency_score'].mean()
    except KeyError:
        # This will be raised if 'token' or 'saliency_score' columns are missing
        raise KeyError("Input DataFrame must contain 'token' and 'saliency_score' columns.")

    # Sort by saliency score to find the most influential tokens
    sorted_saliency = aggregated_saliency.sort_values(ascending=False)

    # Select the top N tokens. .head() gracefully handles cases where
    # top_n_tokens is larger than the number of unique tokens.
    top_tokens = sorted_saliency.head(top_n_tokens)

    # Prepare data for plotting (seaborn heatmap requires a 2D array-like structure)
    heatmap_data = pd.DataFrame(top_tokens)
    heatmap_data.columns = ['Aggregated Saliency']

    # Generate the heatmap
    plt.figure(figsize=(8, max(4, len(top_tokens) * 0.5)))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        cbar_kws={'label': 'Aggregated Saliency Score'}
    )

    plt.title(title, fontsize=14)
    plt.ylabel('Token')
    plt.xlabel('Influence')
    plt.yticks(rotation=0) # Ensure token names are readable
    plt.tight_layout()

    # Save the figure and display the plot
    filename = f"{title.replace(' ', '_').lower()}_heatmap.png"
    plt.savefig(filename)
    plt.show()

def filter_by_verbosity(dataframe, verbosity_threshold):
                """    Filters a DataFrame based on a proxy for explanation verbosity. It returns rows where the 'explanation_quality_score' is greater than or equal to a given threshold.
Arguments:
dataframe (pandas.DataFrame): The input DataFrame to filter.
verbosity_threshold (float): The minimum explanation quality score required to be included in the output.
Output:
pandas.DataFrame: A new DataFrame containing only the rows that meet the verbosity threshold.
                """
                # Create a boolean mask to filter rows based on the threshold.
                mask = dataframe['explanation_quality_score'] >= verbosity_threshold
                
                # Apply the mask to return the filtered DataFrame.
                return dataframe[mask]

import pandas as pd

def filter_by_confidence(dataframe, confidence_threshold):
    """
    Filters a DataFrame to include only rows where 'model_confidence' meets a specified minimum threshold.

    Arguments:
        dataframe (pandas.DataFrame): The input DataFrame to filter. It must contain a 'model_confidence' column.
        confidence_threshold (float): The minimum model confidence score required for a row to be included.

    Output:
        pandas.DataFrame: A new DataFrame containing only the rows that meet the confidence threshold.
    """
    # This will naturally raise a KeyError if the 'model_confidence' column is missing,
    # satisfying the test case for that scenario.
    return dataframe[dataframe['model_confidence'] >= confidence_threshold]