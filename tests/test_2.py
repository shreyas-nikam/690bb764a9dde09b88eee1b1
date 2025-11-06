import pytest
import pandas as pd
import numpy as np
from definition_71d7c31a6f9f4aba934b3e4ec445f271 import validate_and_summarize_data

@pytest.fixture
def sample_dataframe():
    """Provides a valid sample DataFrame for testing."""
    data = {
        'model_confidence': [0.9, 0.8, 0.95, 0.7],
        'explanation_quality_score': [0.85, 0.92, 0.78, 0.88],
        'faithfulness_metric': [0.99, 0.95, 0.98, 0.91],
        'true_label': ['Positive', 'Negative', 'Positive', 'Neutral'],
        'xai_technique': ['Saliency Map', 'LIME', 'Saliency Map', 'Counterfactual']
    }
    return pd.DataFrame(data)

def test_validate_and_summarize_valid_data(capsys, sample_dataframe):
    """
    Test with a valid DataFrame containing all expected columns, correct dtypes,
    and no missing values.
    """
    validate_and_summarize_data(sample_dataframe)
    captured = capsys.readouterr()
    
    # Check for success messages and summaries
    assert "Validation checks passed" in captured.out or "No missing values found" in captured.out
    # Check for numerical summary from .describe()
    assert "model_confidence" in captured.out
    assert "mean" in captured.out
    assert "std" in captured.out
    # Check for categorical summary from .value_counts()
    assert "xai_technique" in captured.out
    assert "Saliency Map" in captured.out
    assert "LIME" in captured.out

def test_with_missing_critical_column(capsys, sample_dataframe):
    """
    Test with a DataFrame that is missing a critical column ('model_confidence').
    The function should log a warning but not crash.
    """
    df_missing_col = sample_dataframe.drop(columns=['model_confidence'])
    validate_and_summarize_data(df_missing_col)
    captured = capsys.readouterr()
    
    assert "Warning: Missing expected column: model_confidence" in captured.out
    # Should still process the remaining columns
    assert "faithfulness_metric" in captured.out
    assert "xai_technique" in captured.out

def test_with_incorrect_dtype(capsys, sample_dataframe):
    """
    Test with a DataFrame where a numerical column has an incorrect object/string dtype.
    The function should log a warning.
    """
    df_wrong_dtype = sample_dataframe.copy()
    df_wrong_dtype['model_confidence'] = df_wrong_dtype['model_confidence'].astype(str)
    validate_and_summarize_data(df_wrong_dtype)
    captured = capsys.readouterr()
    
    assert "Warning: Column 'model_confidence' has incorrect dtype" in captured.out
    # The describe output will look different for an object column, but should still run
    assert "top" in captured.out  # .describe() on object columns shows 'top', 'freq'
    assert "unique" in captured.out

def test_with_missing_values(capsys, sample_dataframe):
    """
    Test with a DataFrame containing NaN values in a critical column.
    The function should detect and report the missing values.
    """
    df_with_nan = sample_dataframe.copy()
    df_with_nan.loc[1, 'model_confidence'] = np.nan
    validate_and_summarize_data(df_with_nan)
    captured = capsys.readouterr()
    
    assert "Missing values found in critical fields" in captured.out
    # Check if the describe output for the column with NaN reflects the correct count
    # Expecting output that shows the count for 'model_confidence' is 3, not 4.
    assert "model_confidence      3.0" in captured.out.replace(" ", "")

def test_with_empty_dataframe(capsys):
    """
    Test with an empty DataFrame that has the expected column structure.
    The function should handle this case gracefully without errors.
    """
    empty_df = pd.DataFrame(columns=[
        'model_confidence', 'explanation_quality_score', 'faithfulness_metric',
        'true_label', 'xai_technique'
    ])
    validate_and_summarize_data(empty_df)
    captured = capsys.readouterr()
    
    # Should not crash and should indicate no data or counts of 0
    assert "DataFrame is empty" in captured.out or "count           0.0" in captured.out
    # It should correctly report no missing values
    assert "No missing values found" in captured.out