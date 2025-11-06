import pytest
import pandas as pd
from unittest.mock import MagicMock

# Import the function to be tested
from definition_4b79718df0d04a49a86a2c323c926629 import plot_aggregated_saliency_heatmap

# Mocking plotting libraries to test function calls without generating images
@pytest.fixture(autouse=True)
def mock_plotting(mocker):
    mocker.patch(\"seaborn.heatmap\")
    mocker.patch(\"matplotlib.pyplot.show\")
    mocker.patch(\"matplotlib.pyplot.savefig\")
    mocker.patch(\"matplotlib.pyplot.figure\")
    mocker.patch(\"matplotlib.pyplot.title\")
    mocker.patch(\"matplotlib.pyplot.xlabel\")
    mocker.patch(\"matplotlib.pyplot.ylabel\")

def test_happy_path_with_valid_inputs(mock_plotting):
    \"\"\"
    Tests the function with a standard, valid DataFrame and top_n value.
    It should call the plotting functions without raising an error.
    \"\"\"
    data = {
        'token': ['apple', 'banana', 'cherry', 'apple', 'banana', 'date'],
        'saliency_score': [0.9, 0.8, 0.2, 0.85, 0.82, 0.5]
    }
    saliency_df = pd.DataFrame(data)
    
    try:
        plot_aggregated_saliency_heatmap(saliency_df, top_n_tokens=3, title=\"Valid Heatmap\")
    except Exception as e:
        pytest.fail(f\"Function raised an unexpected exception: {e}\")

def test_top_n_tokens_larger_than_unique_tokens(mock_plotting):
    \"\"\"
    Tests the edge case where top_n_tokens is greater than the number of unique tokens.
    The function should handle this gracefully and plot all available unique tokens.
    \"\"\"
    data = {
        'token': ['token_a', 'token_b', 'token_a'],
        'saliency_score': [0.9, 0.8, 0.7]
    }
    saliency_df = pd.DataFrame(data)
    
    try:
        plot_aggregated_saliency_heatmap(saliency_df, top_n_tokens=5, title=\"Top N Too Large\")
    except Exception as e:
        pytest.fail(f\"Function failed with more requested tokens than available: {e}\")

def test_empty_dataframe_input():
    \"\"\"
    Tests that the function raises a ValueError when provided with an empty DataFrame,
    as a heatmap cannot be generated from no data.
    \"\"\"
    empty_df = pd.DataFrame({'token': [], 'saliency_score': []})
    with pytest.raises(ValueError, match=r\"(?i)dataframe cannot be empty\"):
        plot_aggregated_saliency_heatmap(empty_df, top_n_tokens=5, title=\"Empty Data Test\")

def test_dataframe_missing_required_columns():
    \"\"\"
    Tests that the function raises a KeyError if the input DataFrame is missing
    the required 'token' or 'saliency_score' columns.
    \"\"\"
    invalid_df = pd.DataFrame({'word': ['a', 'b'], 'score': [0.1, 0.2]})
    with pytest.raises(KeyError):
        plot_aggregated_saliency_heatmap(invalid_df, top_n_tokens=2, title=\"Missing Columns Test\")

@pytest.mark.parametrize(\"invalid_top_n\", [0, -1, -10])
def test_invalid_top_n_tokens_value(invalid_top_n):
    \"\"\"
    Tests that the function raises a ValueError if top_n_tokens is zero or negative,
    as the number of tokens to plot must be a positive integer.
    \"\"\"
    saliency_df = pd.DataFrame({
        'token': ['apple', 'banana'],
        'saliency_score': [0.9, 0.8]
    })
    with pytest.raises(ValueError, match=r\"(?i)top_n_tokens must be a positive integer\"):
        plot_aggregated_saliency_heatmap(saliency_df, invalid_top_n, \"Invalid N Test\")

\"\"\"