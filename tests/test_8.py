import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from definition_68dd4f6a129c4b608524956f934bbade import filter_by_verbosity

# Base data for creating the test DataFrame for most cases
base_data = {
    'prompt': ['p1', 'p2', 'p3', 'p4'],
    'explanation_quality_score': [0.2, 0.5, 0.8, 0.9],
    'other_col': [1, 2, 3, 4]
}

@pytest.mark.parametrize("input_data, verbosity_threshold, expected_data", [
    # Test case 1: Basic filtering where some rows are dropped, ensuring other columns are preserved.
    (
        base_data,
        0.7,
        {'prompt': ['p3', 'p4'], 'explanation_quality_score': [0.8, 0.9], 'other_col': [3, 4]}
    ),
    # Test case 2: Edge case - threshold is lower than all scores, returns the original DataFrame.
    (
        base_data,
        0.0,
        base_data
    ),
    # Test case 3: Edge case - threshold is higher than all scores, returns an empty DataFrame.
    (
        base_data,
        1.0,
        {'prompt': [], 'explanation_quality_score': [], 'other_col': []}
    ),
    # Test case 4: Edge case - threshold exactly matches a score, testing for inclusive filtering (>=).
    (
        base_data,
        0.8,
        {'prompt': ['p3', 'p4'], 'explanation_quality_score': [0.8, 0.9], 'other_col': [3, 4]}
    ),
    # Test case 5: Edge case - input DataFrame is empty.
    (
        {'prompt': [], 'explanation_quality_score': [], 'other_col': []},
        0.5,
        {'prompt': [], 'explanation_quality_score': [], 'other_col': []}
    ),
])
def test_filter_by_verbosity(input_data, verbosity_threshold, expected_data):
    """
    Tests filter_by_verbosity with various inputs including normal cases,
    edge cases for the threshold, and an empty DataFrame.
    """
    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(expected_data)

    result_df = filter_by_verbosity(input_df, verbosity_threshold)

    # Use pandas testing utility to compare DataFrames.
    # We reset index to ensure alignment after filtering and set check_dtype=False
    # because constructing empty dataframes from dicts can lead to object dtypes.
    assert_frame_equal(
        result_df.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_dtype=False
    )
