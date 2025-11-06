import pytest
import pandas as pd
from definition_a8b504b7c76c41ce924bcccab32778b1 import generate_saliency_data

@pytest.mark.parametrize("llm_outputs, expected", [
    # Test case 1: Standard functionality with a typical pandas Series.
    (
        pd.Series(["Hello world", "This is a test"], index=[10, 20]),
        # Expected: A validation function for the resulting DataFrame.
        lambda df: (
            isinstance(df, pd.DataFrame) and
            len(df) == 6 and
            list(df.columns) == ['output_index', 'token', 'saliency_score'] and
            df['output_index'].tolist() == [10, 10, 20, 20, 20, 20] and
            df['token'].tolist() == ["Hello", "world", "This", "is", "a", "test"] and
            (df['saliency_score'] >= 0).all() and (df['saliency_score'] <= 1).all()
        )
    ),
    # Test case 2: Edge case with an empty pandas Series.
    (
        pd.Series([], dtype=str),
        # Expected: A validation function for an empty DataFrame with correct columns.
        lambda df: (
            isinstance(df, pd.DataFrame) and
            df.empty and
            list(df.columns) == ['output_index', 'token', 'saliency_score']
        )
    ),
    # Test case 3: Edge case with a Series containing an empty string.
    (
        pd.Series(["First sentence", "", "last one"]),
        # Expected: The output should ignore the empty string and only process valid ones.
        lambda df: (
            len(df) == 4 and
            df['token'].tolist() == ["First", "sentence", "last", "one"] and
            df['output_index'].tolist() == [0, 0, 2, 2]
        )
    ),
    # Test case 4: Edge case with a Series containing only whitespace.
    (
        pd.Series(["  "]),
        # Expected: Whitespace-only strings should result in no tokens.
        lambda df: df.empty
    ),
    # Test case 5: Invalid input type (list instead of pandas Series).
    (
        ["this is", "not a series"],
        # Expected: An AttributeError because a list lacks Series methods.
        AttributeError
    )
])
def test_generate_saliency_data(llm_outputs, expected):
    """
    Tests the generate_saliency_data function with various inputs.
    """
    if isinstance(expected, type) and issubclass(expected, BaseException):
        with pytest.raises(expected):
            generate_saliency_data(llm_outputs)
    else:
        result_df = generate_saliency_data(llm_outputs)
        # 'expected' is a validation function that returns True on success
        assert expected(result_df)