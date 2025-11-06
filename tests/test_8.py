import pytest
import pandas as pd
from definition_c8dac6edfa444f75842a8263849137cd import filter_by_verbosity

# Helper to get an empty DataFrame with specific columns and dtypes
def get_empty_df_with_types(columns_and_types):
    """
    Creates an empty pandas DataFrame with specified columns and their dtypes.
    columns_and_types example: {'id': 'int64', 'explanation_quality_score': 'float64'}
    """
    df = pd.DataFrame()
    for col, dtype in columns_and_types.items():
        df[col] = pd.Series(dtype=dtype)
    return df

@pytest.mark.parametrize("dataframe_input, threshold_input, expected", [
    # Test Case 1: Standard filtering - some records pass, some don't
    # Input: DataFrame with mixed quality scores. Threshold 0.6.
    # Expected: DataFrame with scores >= 0.6, original indices preserved.
    (pd.DataFrame({'id': [1, 2, 3, 4], 'explanation_quality_score': [0.5, 0.7, 0.3, 0.9]}),
     0.6,
     pd.DataFrame({'id': [2, 4], 'explanation_quality_score': [0.7, 0.9]}, index=[1, 3])),

    # Test Case 2: All records returned - threshold is very low (0.0)
    # Input: DataFrame with various scores. Threshold 0.0 (inclusive).
    # Expected: Original DataFrame, as all scores are >= 0.0.
    (pd.DataFrame({'id': [1, 2, 3], 'explanation_quality_score': [0.5, 0.7, 0.3]}),
     0.0,
     pd.DataFrame({'id': [1, 2, 3], 'explanation_quality_score': [0.5, 0.7, 0.3]})),

    # Test Case 3: No records returned - threshold is very high (1.0)
    # Input: DataFrame with scores <= 0.9. Threshold 1.0.
    # Expected: An empty DataFrame with the same columns and dtypes as the input.
    (pd.DataFrame({'id': [1, 2, 3], 'explanation_quality_score': [0.5, 0.7, 0.3]}),
     1.0,
     get_empty_df_with_types({'id': 'int64', 'explanation_quality_score': 'float64'})),

    # Test Case 4: Empty DataFrame input
    # Input: An empty DataFrame with specific columns and dtypes. Threshold 0.5.
    # Expected: An empty DataFrame with the same columns and dtypes as the input.
    (get_empty_df_with_types({'id': 'int64', 'explanation_quality_score': 'float64'}),
     0.5,
     get_empty_df_with_types({'id': 'int64', 'explanation_quality_score': 'float64'})),

    # Test Case 5: DataFrame without the required 'explanation_quality_score' column
    # Input: DataFrame missing the target column. Threshold 0.5.
    # Expected: ValueError, as the function cannot filter on a non-existent column.
    (pd.DataFrame({'id': [1, 2], 'other_score': [0.5, 0.7]}),
     0.5,
     ValueError),
])
def test_filter_by_verbosity(dataframe_input, threshold_input, expected):
    try:
        result_df = filter_by_verbosity(dataframe_input, threshold_input)
        # If execution reaches here, no exception was expected, so assert DataFrame equality.
        pd.testing.assert_frame_equal(result_df, expected)
    except Exception as e:
        # An exception was caught, check if its type matches the expected exception.
        assert isinstance(e, expected), f"Expected {expected.__name__}, but got {type(e).__name__}: {e}"
