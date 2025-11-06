import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from definition_48dbf0161c8d4297b2b45a48735d772d import filter_by_confidence

# Test data setup
DF_BASE = pd.DataFrame({
    "model_confidence": [0.5, 0.75, 0.9, 1.0],
    "data": ["low", "medium", "high", "perfect"]
})
DF_EMPTY = pd.DataFrame({"model_confidence": [], "data": []}).astype(DF_BASE.dtypes)

@pytest.mark.parametrize("dataframe, confidence_threshold, expected", [
    # Test case 1: Standard functionality - filters rows below the threshold
    (DF_BASE, 0.8, pd.DataFrame({"model_confidence": [0.9, 1.0], "data": ["high", "perfect"]}, index=[2, 3])),
    
    # Test case 2: Edge case - threshold is 0.0, should return the original DataFrame
    (DF_BASE, 0.0, DF_BASE),
    
    # Test case 3: Edge case - threshold is higher than any value, should return an empty DataFrame
    (DF_BASE, 1.1, DF_EMPTY),

    # Test case 4: Edge case - input DataFrame is empty, should return an empty DataFrame
    (DF_EMPTY, 0.5, DF_EMPTY),
    
    # Test case 5: Error case - input DataFrame is missing the 'model_confidence' column
    (pd.DataFrame({"other_col": [1, 2]}), 0.5, KeyError),
])
def test_filter_by_confidence(dataframe, confidence_threshold, expected):
    """
    Tests the filter_by_confidence function with various inputs, including edge cases and expected errors.
    """
    try:
        result_df = filter_by_confidence(dataframe, confidence_threshold)
        assert_frame_equal(result_df, expected)
    except Exception as e:
        assert isinstance(e, expected)