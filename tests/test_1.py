import pytest
import pandas as pd
import numpy as np

# Keep a placeholder definition_7ab21017b8bb4e22ad517a8b557e68fa for the import of the module. Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_7ab21017b8bb4e22ad517a8b557e68fa import generate_saliency_data
# End of definition_7ab21017b8bb4e22ad517a8b557e68fa block

@pytest.mark.parametrize(
    "llm_outputs_input, expected_type, expected_exception, expected_row_count_if_df",
    [
        # Test Case 1: Standard valid input with multiple outputs and tokens
        # Expects a DataFrame with 8 rows (5 tokens from first string, 3 from second).
        (
            pd.Series(["This is a test sentence.", "Another example output."]),
            pd.DataFrame,
            None,
            8
        ),
        # Test Case 2: Empty pandas Series
        # Expects an empty DataFrame.
        (
            pd.Series([]),
            pd.DataFrame,
            None,
            0
        ),
        # Test Case 3: Pandas Series with empty strings, strings with only spaces, and a valid string
        # Expects a DataFrame with 2 rows (only from "Hello world").
        (
            pd.Series(["", "   ", "Hello world"]),
            pd.DataFrame,
            None,
            2
        ),
        # Test Case 4: Non-pandas Series input (e.g., a list of strings)
        # Expects an AttributeError because the function likely assumes `llm_outputs` is a Series
        # and attempts to use Series-specific methods.
        (
            ["Just a list item"],
            None,
            AttributeError,
            None
        ),
        # Test Case 5: Pandas Series containing non-string elements (e.g., int, None)
        # Expects a TypeError when the internal tokenization (e.g., `.split()`) is applied to non-string types.
        (
            pd.Series(["valid string", 123, None, "another one"]),
            None,
            TypeError,
            None
        ),
    ]
)
def test_generate_saliency_data(llm_outputs_input, expected_type, expected_exception, expected_row_count_if_df):
    if expected_exception:
        # If an exception is expected, assert that the function call raises it
        with pytest.raises(expected_exception):
            generate_saliency_data(llm_outputs_input)
    else:
        # Assuming the function is correctly implemented according to its specification
        result_df = generate_saliency_data(llm_outputs_input)

        # Assert the overall type of the returned value
        assert isinstance(result_df, expected_type)

        # Define the expected column names for the output DataFrame
        expected_cols = ['output_index', 'token', 'saliency_score']

        # Check if the DataFrame contains all expected columns and no extra ones
        assert all(col in result_df.columns for col in expected_cols)
        assert len(result_df.columns) == len(expected_cols)

        if expected_row_count_if_df == 0:
            # If an empty DataFrame is expected, verify it is indeed empty
            assert result_df.empty
        else:
            # For non-empty expected output, verify it's not empty and has the correct number of rows
            assert not result_df.empty
            assert len(result_df) == expected_row_count_if_df

            # Check data types of the columns
            assert pd.api.types.is_integer_dtype(result_df['output_index'])
            assert pd.api.types.is_string_dtype(result_df['token'])
            assert pd.api.types.is_float_dtype(result_df['saliency_score'])

            # Check that saliency scores are within the expected range [0, 1]
            assert (result_df['saliency_score'] >= 0).all()
            assert (result_df['saliency_score'] <= 1).all()

            # Check that token strings are not empty after tokenization
            assert (result_df['token'].astype(str).str.len() > 0).all()