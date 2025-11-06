import pytest
import pandas as pd
from definition_38b7384a639b43e9a757aab672145511 import generate_llm_data

EXPECTED_COLUMNS = [
    'timestamp', 'prompt', 'llm_output', 'true_label', 'model_confidence',
    'model_accuracy', 'explanation_quality_score', 'faithfulness_metric',
    'xai_technique'
]

def test_generate_llm_data_successful_generation():
    """
    Tests if the function correctly generates a DataFrame with the specified number of samples and columns.
    """
    num_samples = 100
    df = generate_llm_data(num_samples)
    
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert len(df) == num_samples, f"DataFrame should have {num_samples} rows"
    assert list(df.columns) == EXPECTED_COLUMNS, "DataFrame columns are incorrect"
    assert not df.isnull().values.any(), "Generated data should not contain null values"

def test_generate_llm_data_zero_samples():
    """
    Tests the edge case where zero samples are requested, expecting an empty DataFrame.
    """
    num_samples = 0
    df = generate_llm_data(num_samples)
    
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert len(df) == num_samples, "DataFrame should be empty for 0 samples"
    assert list(df.columns) == EXPECTED_COLUMNS, "Empty DataFrame should still have the correct columns"

def test_generate_llm_data_data_types_and_ranges():
    """
    Tests if the generated data in columns conforms to expected types and value ranges.
    """
    df = generate_llm_data(10)

    assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "'timestamp' column should have datetime type"
    assert df['model_confidence'].between(0.5, 1.0).all(), "'model_confidence' should be between 0.5 and 1.0"
    assert df['model_accuracy'].isin([0, 1]).all(), "'model_accuracy' should be either 0 or 1"
    assert df['explanation_quality_score'].between(0, 1).all(), "'explanation_quality_score' should be between 0 and 1"
    assert df['faithfulness_metric'].between(0, 1).all(), "'faithfulness_metric' should be between 0 and 1"
    assert df['xai_technique'].isin(['Saliency Map', 'Counterfactual', 'LIME']).all(), "Invalid 'xai_technique' values found"

@pytest.mark.parametrize("invalid_input, expected_exception", [
    (-5, ValueError),
    (10.5, TypeError),
    ("ten", TypeError),
    ([10], TypeError)
])
def test_generate_llm_data_invalid_input(invalid_input, expected_exception):
    """
    Tests that the function raises appropriate errors for invalid 'num_samples' inputs.
    """
    with pytest.raises(expected_exception):
        generate_llm_data(invalid_input)

def test_generate_llm_data_uniqueness():
    """
    Tests if subsequent calls generate different, non-identical data, indicating randomness.
    """
    df1 = generate_llm_data(10)
    df2 = generate_llm_data(10)
    
    assert not df1.equals(df2), "Two separate calls should not produce identical DataFrames"