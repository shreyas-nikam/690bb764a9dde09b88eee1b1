import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from definition_270f429e2dd449b0ac1c649a1c6a0492 import generate_saliency_data

def test_generate_saliency_data_happy_path():
    """
    Tests the basic functionality with a standard pandas Series input.
    Verifies the output DataFrame's structure, content, and data types.
    """
    llm_outputs = pd.Series(["hello world", "test sentence"])
    result_df = generate_saliency_data(llm_outputs)

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ['output_index', 'token', 'saliency_score']
    assert len(result_df) == 4  # 2 tokens from first string, 2 from second
    assert result_df['output_index'].to_list() == [0, 0, 1, 1]
    assert result_df['token'].to_list() == ['hello', 'world', 'test', 'sentence']
    assert pd.api.types.is_float_dtype(result_df['saliency_score'])
    assert all(0.0 <= score <= 1.0 for score in result_df['saliency_score'])

def test_generate_saliency_data_empty_series():
    """
    Tests the edge case where the input is an empty pandas Series.
    The function should return an empty DataFrame with the correct columns and dtypes.
    """
    llm_outputs = pd.Series([], dtype=str)
    result_df = generate_saliency_data(llm_outputs)

    expected_df = pd.DataFrame({
        'output_index': pd.Series([], dtype='int64'),
        'token': pd.Series([], dtype='object'),
        'saliency_score': pd.Series([], dtype='float64')
    })
    
    assert_frame_equal(result_df, expected_df, check_index_type=False)

def test_generate_saliency_data_with_empty_and_whitespace_strings():
    """
    Tests behavior with strings that result in no tokens (empty or whitespace-only).
    Ensures that these strings do not produce any rows in the output DataFrame.
    """
    llm_outputs = pd.Series(["first sentence", "", "  ", "last one"], index=[10, 20, 30, 40])
    result_df = generate_saliency_data(llm_outputs)

    assert len(result_df) == 4  # "first", "sentence", "last", "one"
    assert result_df['output_index'].to_list() == [10, 10, 40, 40]
    assert result_df['token'].to_list() == ['first', 'sentence', 'last', 'one']

def test_generate_saliency_data_preserves_custom_index():
    """
    Tests that the function correctly uses a custom non-sequential Series index
    for the 'output_index' column in the resulting DataFrame.
    """
    llm_outputs = pd.Series(["custom index test"], index=[99])
    result_df = generate_saliency_data(llm_outputs)

    assert len(result_df) == 3
    assert all(idx == 99 for idx in result_df['output_index'])
    assert result_df['token'].to_list() == ['custom', 'index', 'test']

def test_generate_saliency_data_invalid_input_type():
    """
    Tests that the function raises an AttributeError when the input is not a
    pandas Series, as it relies on Series-specific methods.
    """
    with pytest.raises(AttributeError):
        # A list does not have the .items() method required by the function.
        generate_saliency_data(["this is", "not a pandas series"])

