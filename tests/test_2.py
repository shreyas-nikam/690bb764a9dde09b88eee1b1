import pytest
import pandas as pd
import numpy as np
import io
import sys

# definition_6e3906dbcafb4e19be40f305d4f3f933 block
from definition_6e3906dbcafb4e19be40f305d4f3f933 import validate_and_summarize_data
# End of definition_6e3906dbcafb4e19be40f305d4f3f933 block

# Helper function to create a test DataFrame based on scenario
def create_test_dataframe(num_samples=10, scenario='valid'):
    # Base data for a valid DataFrame with all expected columns and appropriate dtypes
    base_data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_samples, freq='D')),
        'prompt': [f"Prompt {i}" for i in range(num_samples)],
        'llm_output': [f"Output {i}" for i in range(num_samples)],
        'true_label': np.random.choice(['Positive', 'Negative', 'Neutral'], num_samples),
        'model_confidence': np.random.uniform(0.5, 1.0, num_samples),
        'model_accuracy': np.random.randint(0, 2, num_samples),
        'explanation_quality_score': np.random.uniform(0.0, 1.0, num_samples),
        'faithfulness_metric': np.random.uniform(0.0, 1.0, num_samples),
        'xai_technique': np.random.choice(['Saliency Map', 'Counterfactual', 'LIME'], num_samples),
    }

    df = pd.DataFrame(base_data)

    if scenario == 'missing_col':
        # Drop a critical column, e.g., 'model_confidence'
        df = df.drop(columns=['model_confidence'])
    elif scenario == 'bad_type':
        # Change 'model_confidence' to an incorrect data type (e.g., string instead of numeric)
        df['model_confidence'] = df['model_confidence'].astype(str)
    elif scenario == 'missing_values':
        # Introduce NaN in a critical field, e.g., 'faithfulness_metric'
        df.loc[0, 'faithfulness_metric'] = np.nan
    
    return df

# Parametrize test cases for different scenarios
@pytest.mark.parametrize("dataframe_input, expected_exception, expected_message_part", [
    # Test Case 1: Valid DataFrame - Expected functionality
    # Should print validation messages and summary statistics without errors.
    (create_test_dataframe(scenario='valid'), None, "No missing values found in critical fields."),
    
    # Test Case 2: DataFrame with Missing Critical Column
    # Should raise an AssertionError if a required column is missing.
    (create_test_dataframe(scenario='missing_col'), AssertionError, "Expected column 'model_confidence' not found."),
    
    # Test Case 3: DataFrame with Incorrect Data Type for a critical field
    # Should raise an AssertionError if a critical column has the wrong data type.
    (create_test_dataframe(scenario='bad_type'), AssertionError, "Expected 'model_confidence' to be numeric, but found object"),
    
    # Test Case 4: DataFrame with Missing Values in Critical Fields
    # Should raise an AssertionError if missing values are found in critical fields.
    (create_test_dataframe(scenario='missing_values'), AssertionError, "Missing values found in critical field: faithfulness_metric"),
    
    # Test Case 5: Non-DataFrame input
    # Should raise a TypeError if the input is not a pandas DataFrame.
    (None, TypeError, "dataframe must be a pandas DataFrame"),
])
def test_validate_and_summarize_data(dataframe_input, expected_exception, expected_message_part, capsys):
    if expected_exception:
        # Expect an exception to be raised
        with pytest.raises(expected_exception) as excinfo:
            validate_and_summarize_data(dataframe_input)
        assert expected_message_part in str(excinfo.value)
    else:
        # For valid cases, the function should print outputs without raising exceptions
        validate_and_summarize_data(dataframe_input)
        # Capture stdout to check for expected printed messages
        captured = capsys.readouterr()
        assert expected_message_part in captured.out
        assert "Descriptive statistics for numeric columns:" in captured.out
        assert "Value counts for categorical columns:" in captured.out
