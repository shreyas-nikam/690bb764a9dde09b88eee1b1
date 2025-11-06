import pytest
import pandas as pd
import numpy as np
from definition_657d0c1e4b6f42bd955b057b6e43cdde import generate_llm_data

# Define expected column names for easy reuse and to ensure order is not strictly enforced
expected_columns = {
    'timestamp', 'prompt', 'llm_output', 'true_label', 
    'model_confidence', 'model_accuracy', 'explanation_quality_score', 
    'faithfulness_metric', 'xai_technique'
}

@pytest.mark.parametrize("num_samples, expected_type, expected_rows, expected_exception", [
    (5, pd.DataFrame, 5, None),      # Valid input: standard number of samples
    (0, pd.DataFrame, 0, None),      # Edge case: zero samples
    (1, pd.DataFrame, 1, None),      # Edge case: single sample
    (-1, None, None, ValueError),    # Invalid input: negative samples
    ("invalid", None, None, TypeError) # Invalid input: non-integer type for num_samples
])
def test_generate_llm_data(num_samples, expected_type, expected_rows, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception) as excinfo:
            generate_llm_data(num_samples)
        assert isinstance(excinfo.type, expected_exception)
        return

    df = generate_llm_data(num_samples)

    # Test 1: Type of return value
    assert isinstance(df, expected_type), f"Expected return type {expected_type}, but got {type(df)}"

    # Test 2: Number of rows
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, but got {len(df)}"

    # Test 3: Column names verification
    actual_columns_set = set(df.columns)
    assert actual_columns_set == expected_columns, \
        f"Expected columns {expected_columns}, but got {actual_columns_set}"

    if expected_rows > 0:
        # Test 4: Data types for key columns
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Column 'timestamp' has incorrect dtype"
        assert pd.api.types.is_string_dtype(df['prompt']), "Column 'prompt' has incorrect dtype"
        assert pd.api.types.is_string_dtype(df['llm_output']), "Column 'llm_output' has incorrect dtype"
        assert pd.api.types.is_float_dtype(df['model_confidence']), "Column 'model_confidence' has incorrect dtype"
        assert pd.api.types.is_integer_dtype(df['model_accuracy']), "Column 'model_accuracy' has incorrect dtype"
        assert pd.api.types.is_float_dtype(df['explanation_quality_score']), "Column 'explanation_quality_score' has incorrect dtype"
        assert pd.api.types.is_float_dtype(df['faithfulness_metric']), "Column 'faithfulness_metric' has incorrect dtype"
        assert pd.api.types.is_string_dtype(df['xai_technique']), "Column 'xai_technique' has incorrect dtype"
        assert pd.api.types.is_string_dtype(df['true_label']), "Column 'true_label' has incorrect dtype"

        # Test 5: Value ranges and categorical options for relevant columns
        assert df['model_confidence'].between(0.0, 1.0).all(), "Values in 'model_confidence' are not between 0.0 and 1.0"
        assert df['model_accuracy'].isin([0, 1]).all(), "Values in 'model_accuracy' are not binary (0 or 1)"
        assert df['explanation_quality_score'].between(0.0, 1.0).all(), "Values in 'explanation_quality_score' are not between 0.0 and 1.0"
        assert df['faithfulness_metric'].between(0.0, 1.0).all(), "Values in 'faithfulness_metric' are not between 0.0 and 1.0"
        
        expected_xai_techniques = ['Saliency Map', 'Counterfactual', 'LIME']
        assert df['xai_technique'].isin(expected_xai_techniques).all(), \
            f"Values in 'xai_technique' contain unexpected categories. Expected: {expected_xai_techniques}"
        
        expected_true_labels = ['Positive', 'Negative', 'Neutral'] # Based on notebook specification example
        assert df['true_label'].isin(expected_true_labels).all(), \
            f"Values in 'true_label' contain unexpected categories. Expected: {expected_true_labels}"
