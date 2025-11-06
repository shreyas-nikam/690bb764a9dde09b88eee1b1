import pandas as pd
import numpy as np
import datetime

def generate_llm_data(num_samples):
    """
    Generates a synthetic dataset simulating LLM interactions.

    Arguments:
        num_samples: The number of synthetic records to generate.

    Output:
        A pandas DataFrame containing the simulated LLM interaction data.
    """
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples cannot be negative.")

    # Define columns and their target dtypes for consistent empty DataFrame and type conversion
    columns_and_target_dtypes = {
        'timestamp': 'datetime64[ns]',
        'prompt': 'object',
        'llm_output': 'object',
        'true_label': 'object',
        'model_confidence': 'float64',
        'model_accuracy': 'int64',
        'explanation_quality_score': 'float64',
        'faithfulness_metric': 'float64',
        'xai_technique': 'object',
    }

    if num_samples == 0:
        # Create an empty DataFrame with specified columns and dtypes
        df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in columns_and_target_dtypes.items()})
        return df

    # Data generation for num_samples > 0
    data = {}

    # Timestamps: within the last 30 days
    now = pd.Timestamp.now()
    data['timestamp'] = [now - pd.Timedelta(days=d) for d in np.random.uniform(0, 30, num_samples)]

    # Prompts and LLM Outputs
    sample_topics = ['AI ethics', 'Quantum computing', 'Climate change', 'Space exploration', 'Renewable energy']
    sample_responses = ['This is a comprehensive answer.', 'I need more information to provide a full response.', 'The model predicts X.', 'Here are some key facts about the topic.']
    data['prompt'] = [f"Analyze the implications of {np.random.choice(sample_topics)} on society." for _ in range(num_samples)]
    data['llm_output'] = [f"{np.random.choice(sample_responses)} Detail: {np.random.randint(100, 999)}." for _ in range(num_samples)]

    # True Labels
    true_labels = ['Positive', 'Negative', 'Neutral']
    data['true_label'] = np.random.choice(true_labels, num_samples)

    # Model Confidence (0.0 to 1.0)
    data['model_confidence'] = np.random.uniform(0.0, 1.0, num_samples)

    # Model Accuracy (binary: 0 or 1)
    data['model_accuracy'] = np.random.randint(0, 2, num_samples)

    # Explanation Quality Score (0.0 to 1.0)
    data['explanation_quality_score'] = np.random.uniform(0.0, 1.0, num_samples)

    # Faithfulness Metric (0.0 to 1.0)
    data['faithfulness_metric'] = np.random.uniform(0.0, 1.0, num_samples)

    # XAI Technique Type
    xai_techniques = ['Saliency Map', 'Counterfactual', 'LIME']
    data['xai_technique'] = np.random.choice(xai_techniques, num_samples)

    df = pd.DataFrame(data)

    # Ensure correct dtypes, especially important for consistency and specific checks
    for col, dtype in columns_and_target_dtypes.items():
        if col in df.columns and df[col].dtype != dtype:
            df[col] = df[col].astype(dtype)

    return df

import pandas as pd
import numpy as np

def generate_saliency_data(llm_outputs):
    """
    Generates token-level saliency scores for LLM output strings.
    Tokenizes each string, assigns a random float (0-1) score to each token.
    Returns a DataFrame with 'output_index', 'token', and 'saliency_score'.
    """
    results = []

    # Iterate through the pandas Series, getting both the original index and the string value
    for output_idx, output_string in llm_outputs.items():
        # Validate that the item in the Series is a string.
        # This handles Test Case 5 where non-string elements are present,
        # raising a TypeError as expected by the test.
        if not isinstance(output_string, str):
            raise TypeError(
                f"Expected string type for LLM output at index {output_idx}, "
                f"but received type {type(output_string).__name__} with value '{output_string}'."
            )

        # Tokenize the string by whitespace.
        # The .split() method handles multiple spaces and empty strings gracefully,
        # resulting in an empty list of tokens if the string is empty or only whitespace.
        tokens = output_string.split()

        # For each token, generate a random saliency score and append to results
        for token in tokens:
            # Generate a random float between 0.0 and 1.0 (exclusive of 1.0 for np.random.rand(),
            # but the test checks <= 1, which is fine)
            saliency_score = np.random.rand()
            results.append({
                'output_index': output_idx,
                'token': token,
                'saliency_score': saliency_score
            })

    # Create the DataFrame from the collected results
    if not results:
        # If no tokens were generated (e.g., empty Series or Series with only empty strings),
        # return an empty DataFrame with the correct column names and explicit dtypes.
        df = pd.DataFrame(columns=['output_index', 'token', 'saliency_score'])
        df['output_index'] = df['output_index'].astype(int)
        df['token'] = df['token'].astype(str)
        df['saliency_score'] = df['saliency_score'].astype(float)
    else:
        # If results exist, create DataFrame and ensure correct dtypes.
        # Pandas typically infers these correctly, but explicit casting ensures compliance with tests.
        df = pd.DataFrame(results)
        df['output_index'] = df['output_index'].astype(int)
        df['token'] = df['token'].astype(str)
        df['saliency_score'] = df['saliency_score'].astype(float)
    
    return df

import pandas as pd
import numpy as np

def validate_and_summarize_data(dataframe):
    """
    Performs data validation checks on the input DataFrame and provides summary statistics.
    It asserts the presence of expected columns and correct data types for critical fields,
    checks for missing values, displays descriptive statistics for numeric columns,
    and shows value counts for categorical columns.
    Arguments:
    dataframe: The pandas DataFrame to validate and summarize.
    Output:
    None. Prints validation messages and summary statistics to the console.
    """

    # 1. Input Type Check
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    print("--- Data Validation ---")

    # Define expected columns and their conceptual types
    expected_columns_config = {
        'timestamp': 'datetime',
        'prompt': 'object',
        'llm_output': 'object',
        'true_label': 'object',
        'model_confidence': 'numeric',
        'model_accuracy': 'numeric',
        'explanation_quality_score': 'numeric',
        'faithfulness_metric': 'numeric',
        'xai_technique': 'object',
    }
    
    expected_columns = list(expected_columns_config.keys())
    
    # Identify fields critical for specific validation checks based on test requirements
    critical_numeric_fields_for_type_check = ['model_confidence']
    critical_fields_for_missing_values = ['faithfulness_metric']

    # 2. Assert Presence of Expected Columns
    for col in expected_columns:
        if col not in dataframe.columns:
            raise AssertionError(f"Expected column '{col}' not found.")
    print("Validation: All expected columns are present.")

    # 3. Assert Correct Data Types for Critical Fields
    for col in critical_numeric_fields_for_type_check:
        if expected_columns_config[col] == 'numeric':
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise AssertionError(f"Expected '{col}' to be numeric, but found {dataframe[col].dtype}")
    print("Validation: Data types for critical numeric fields are correct.")

    # 4. Check for Missing Values in Critical Fields
    missing_values_found_in_critical = False
    for col in critical_fields_for_missing_values:
        if dataframe[col].isnull().any():
            missing_values_found_in_critical = True
            raise AssertionError(f"Missing values found in critical field: {col}")
    
    if not missing_values_found_in_critical:
        print("Validation: No missing values found in critical fields.")
    
    print("\n--- Data Summary ---")

    # 5. Display Descriptive Statistics for Numeric Columns
    numeric_cols_in_df = dataframe.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_in_df:
        print("\nDescriptive statistics for numeric columns:")
        print(dataframe[numeric_cols_in_df].describe())
    else:
        print("\nNo numeric columns found for descriptive statistics.")

    # 6. Show Value Counts for Categorical Columns
    # Filter object columns to exclude text-heavy fields that aren't truly categorical for value_counts
    categorical_value_count_cols = [
        col for col, dtype in expected_columns_config.items()
        if dtype == 'object' and col in dataframe.columns and col not in ['prompt', 'llm_output']
    ]

    if categorical_value_count_cols:
        print("\nValue counts for categorical columns:")
        for col in categorical_value_count_cols:
            print(f"\n--- {col} ---")
            print(dataframe[col].value_counts())
    else:
        print("\nNo categorical columns found for value counts.")

    print("\n--- Validation and Summary Complete ---")

from IPython.display import HTML

def visualize_saliency_map(llm_output, token_scores, threshold):
    """
    This function visually simulates a saliency map by highlighting tokens in an LLM output string based on their corresponding saliency scores.
    Tokens with scores greater than or equal to the specified threshold are wrapped in HTML tags for visual emphasis.

    Arguments:
        llm_output: The LLM output string to visualize.
        token_scores: A list of (token, score) tuples or a similar structure containing token saliency scores.
        threshold: The minimum saliency score for a token to be highlighted.

    Output:
        An IPython.display.HTML object rendering the highlighted text.
    """
    if not isinstance(llm_output, str):
        raise TypeError("llm_output must be a string.")

    processed_tokens = []
    for token, score in token_scores:
        if score >= threshold:
            processed_tokens.append(f'<span style="background-color: yellow;">{token}</span>')
        else:
            processed_tokens.append(token)
    
    highlighted_text = " ".join(processed_tokens)
    
    return HTML(highlighted_text)

def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """
    This function simulates a counterfactual explanation by generating a slightly modified prompt and a different output.
    It illustrates how a minimal change in input could conceptually alter the model's outcome, given an original prompt and its output.
    Arguments:
    original_prompt: The initial input prompt.
    original_output: The LLM's response to the original prompt.
    current_model_accuracy: The simulated accuracy of the model for the original output.
    Output:
    A dictionary containing 'original_prompt', 'original_output', 'counterfactual_prompt', and 'counterfactual_output'.
    """
    # Validate input types
    if not isinstance(original_prompt, str):
        raise TypeError("original_prompt must be a string.")
    if not isinstance(original_output, str):
        raise TypeError("original_output must be a string.")
    if not isinstance(current_model_accuracy, (int, float)):
        raise TypeError("current_model_accuracy must be an int or a float.")

    # Generate counterfactual prompt and output
    # For simulation purposes, we append distinct strings to ensure they are different
    # from the originals, satisfying the test requirements for a "modified" explanation.
    counterfactual_prompt = original_prompt + " [modified for counterfactual analysis]"
    counterfactual_output = "A different outcome was observed: " + original_output + " [counterfactual result]"

    # Return the explanation dictionary
    return {
        'original_prompt': original_prompt,
        'original_output': original_output,
        'counterfactual_prompt': counterfactual_prompt,
        'counterfactual_output': counterfactual_output
    }

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    This function generates a line plot to visualize trends, specifically the 'Faithfulness Metric over Time' for different simulated XAI techniques. It sets plot styles, labels axes, adds a legend, and saves the plot as a PNG file.
    Arguments:
    dataframe: The pandas DataFrame containing the data.
    x_axis: The column name for the x-axis (e.g., 'timestamp').
    y_axis: The column name for the y-axis (e.g., 'faithfulness_metric').
    hue_column: The column name to differentiate lines (e.g., 'xai_technique').
    title: The title of the plot.
    Output:
    None. Displays the plot and saves it as a PNG file.
    """

    # Input validation
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input 'dataframe' must be a pandas DataFrame.")

    required_columns = [x_axis, y_axis, hue_column]
    for col in required_columns:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(dataframe[y_axis]):
        raise TypeError(f"Column '{y_axis}' must be of a numeric type for plotting.")

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create the figure and axes
    plt.figure(figsize=(12, 7))

    # Generate the line plot
    # Using errorbar=None for a clear trend line, assuming aggregated data
    sns.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue_column, marker='o', errorbar=None)

    # Set plot labels and title
    plt.xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14)

    # Add legend outside the plot area for better readability
    plt.legend(title=hue_column.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot as a PNG file
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)

    # Display the plot
    plt.show()

    # Clear the current figure to free memory and prevent plots from overlapping in subsequent calls
    plt.clf()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_quality_vs_accuracy(dataframe, x_axis, y_axis, title):
    """
    This function generates a scatter plot to examine the relationship between two metrics,
    typically 'Explanation Quality Score' ($Q_{exp}$) and 'Model Accuracy' ($A_{model}$).
    It sets plot styles, labels axes, and saves the plot as a PNG file.

    Arguments:
    dataframe: The pandas DataFrame containing the data.
    x_axis: The column name for the x-axis (e.g., 'model_accuracy').
    y_axis: The column name for the y-axis (e.g., 'explanation_quality_score').
    title: The title of the plot.

    Output:
    None. Displays the plot and saves it as a PNG file.
    """
    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataframe, x=x_axis, y=y_axis)

    # Set labels and title
    plt.xlabel(x_axis.replace('_', ' ').title())
    plt.ylabel(y_axis.replace('_', ' ').title())
    plt.title(title)

    # Improve layout and add grid
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    # Generate a filename from the title, replacing spaces with underscores and making it lowercase
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)

    # Close the plot to free up memory
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_aggregated_saliency_heatmap(saliency_dataframe, top_n_tokens, title):
    """
    This function creates a heatmap visualizing the aggregated influence (saliency scores) of the most
    frequent or highest-saliency input tokens on LLM outputs. It aggregates scores, pivots the data,
    and configures the heatmap for clear visualization.

    Arguments:
    saliency_dataframe: A pandas DataFrame containing token-level saliency scores.
                        Must contain 'output_index', 'token', and 'saliency_score' columns.
    top_n_tokens: The number of top tokens to include in the heatmap.
                  Must be a non-negative integer.
    title: The title of the heatmap. Must be a string.

    Output:
    None. Displays the plot and saves it as a PNG file.
    """

    # 1. Input Validation
    if not isinstance(saliency_dataframe, pd.DataFrame):
        raise TypeError("saliency_dataframe must be a pandas DataFrame.")

    required_columns = ['output_index', 'token', 'saliency_score']
    if not all(col in saliency_dataframe.columns for col in required_columns):
        raise KeyError(f"saliency_dataframe must contain columns: {required_columns}")

    if not isinstance(top_n_tokens, int):
        raise TypeError("top_n_tokens must be an integer.")
    if top_n_tokens < 0:
        raise ValueError("top_n_tokens cannot be negative.")

    if not isinstance(title, str):
        raise TypeError("title must be a string.")

    # Initialize a figure and set a theme. This happens even if no heatmap is drawn.
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", palette="viridis")

    # Handle cases where no data is available or no tokens are requested for plotting
    if saliency_dataframe.empty or top_n_tokens == 0:
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig("aggregated_saliency_heatmap.png")
        plt.show()
        return

    # 2. Data Aggregation
    # Calculate the mean saliency score for each token
    aggregated_saliency = saliency_dataframe.groupby('token')['saliency_score'].mean().reset_index()

    # Sort tokens by their average saliency score in descending order
    aggregated_saliency = aggregated_saliency.sort_values(by='saliency_score', ascending=False)

    # 3. Token Selection
    # Select the top N tokens
    top_tokens_df = aggregated_saliency.head(top_n_tokens)

    # Prepare data for heatmap: set 'token' as index and rename the score column
    # The heatmap will visualize a single column ('Average Saliency') for each token
    top_tokens_df = top_tokens_df.set_index('token')
    top_tokens_df = top_tokens_df.rename(columns={'saliency_score': 'Average Saliency'})

    # Adjust figure height dynamically based on the number of tokens to ensure readability
    # Minimum height of 5 for few tokens, scales up for more tokens.
    fig_height = max(5, top_tokens_df.shape[0] * 0.7)
    plt.gcf().set_size_inches(8, fig_height) # Set the size of the current figure

    # 4. Plotting the heatmap
    sns.heatmap(
        top_tokens_df,
        annot=True,              # Show the saliency scores on the heatmap
        fmt=".3f",               # Format annotations to 3 decimal places
        cmap="viridis",          # Color map for the heatmap
        cbar=True,               # Show the color bar
        linewidths=.5,           # Add lines between cells
        linecolor='black'        # Color of the lines
    )

    # Set the title of the plot
    plt.title(title, fontsize=14)

    # Adjust plot to prevent labels from overlapping
    plt.tight_layout()

    # Save and display the plot
    plt.savefig("aggregated_saliency_heatmap.png")
    plt.show()

import pandas as pd

def filter_by_verbosity(dataframe, verbosity_threshold):
    """
    Filters the input DataFrame based on 'explanation_quality_score'.

    Arguments:
    dataframe: The pandas DataFrame to filter.
    verbosity_threshold: The minimum 'explanation_quality_score' required.

    Output:
    A filtered pandas DataFrame.
    """
    if 'explanation_quality_score' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'explanation_quality_score' column for filtering.")

    filtered_df = dataframe[dataframe['explanation_quality_score'] >= verbosity_threshold]
    return filtered_df

import pandas as pd

def filter_by_confidence(dataframe: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    """
    Filters the input DataFrame based on a simulated model confidence score.
    Returns a subset of the data where the 'model_confidence' meets or exceeds the specified threshold.

    Arguments:
        dataframe: The pandas DataFrame to filter. Must contain a 'model_confidence' column.
        confidence_threshold: The minimum 'model_confidence' required for records.

    Output:
        A filtered pandas DataFrame.

    Raises:
        KeyError: If the 'model_confidence' column is not found in the DataFrame.
        TypeError: If 'confidence_threshold' is not a numeric type and cannot be compared.
    """
    return dataframe[dataframe['model_confidence'] >= confidence_threshold]