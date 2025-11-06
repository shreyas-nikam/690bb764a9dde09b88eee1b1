import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Keep this block as it is.
# definition_f1b021ca601845a7aa7a2e7a991a0c4b
from definition_f1b021ca601845a7aa7a2e7a991a0c4b import plot_aggregated_saliency_heatmap
# </your_module>


@pytest.fixture
def sample_saliency_df():
    """Provides a sample saliency dataframe for testing."""
    data = {
        'token': ['hello', 'world', 'this', 'is', 'a', 'test', 'hello', 'world', 'is', 'a'],
        'saliency_score': [0.9, 0.8, 0.3, 0.7, 0.6, 0.2, 0.95, 0.85, 0.75, 0.65]
    }
    return pd.DataFrame(data)


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('seaborn.heatmap')
def test_plot_aggregated_saliency_heatmap_happy_path(mock_heatmap, mock_savefig, mock_show, sample_saliency_df):
    """
    Tests the function with valid inputs to ensure plotting functions are called correctly.
    """
    top_n = 3
    title = "Test Heatmap"
    plot_aggregated_saliency_heatmap(sample_saliency_df, top_n_tokens=top_n, title=title)

    mock_heatmap.assert_called_once()
    mock_savefig.assert_called_once_with('aggregated_saliency_heatmap.png')
    mock_show.assert_called_once()
    
    # Check that the data passed to heatmap has the correct shape (top_n tokens)
    call_args, _ = mock_heatmap.call_args
    heatmap_data = call_args[0]
    assert heatmap_data.shape[0] == top_n


def test_plot_aggregated_saliency_heatmap_empty_dataframe():
    """
    Tests the function with an empty DataFrame, expecting a ValueError.
    """
    empty_df = pd.DataFrame({'token': [], 'saliency_score': []})
    with pytest.raises(ValueError, match="saliency_dataframe cannot be empty"):
        plot_aggregated_saliency_heatmap(empty_df, top_n_tokens=5, title="Empty")


def test_plot_aggregated_saliency_heatmap_missing_columns():
    """
    Tests the function with a DataFrame missing required columns, expecting a KeyError.
    """
    invalid_df = pd.DataFrame({'word': ['a', 'b'], 'score': [0.1, 0.2]})
    with pytest.raises(KeyError, match="'token'"):
        plot_aggregated_saliency_heatmap(invalid_df, top_n_tokens=1, title="Invalid DF")


@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('seaborn.heatmap')
def test_plot_aggregated_saliency_heatmap_top_n_exceeds_unique(mock_heatmap, mock_savefig, mock_show, sample_saliency_df):
    """
    Tests that the function handles cases where top_n_tokens is larger than the number of unique tokens.
    It should plot all unique tokens available.
    """
    unique_tokens_count = sample_saliency_df['token'].nunique()
    plot_aggregated_saliency_heatmap(sample_saliency_df, top_n_tokens=unique_tokens_count + 5, title="Top N Exceeds")
    
    mock_heatmap.assert_called_once()
    call_args, _ = mock_heatmap.call_args
    heatmap_data = call_args[0]
    assert heatmap_data.shape[0] == unique_tokens_count


def test_plot_aggregated_saliency_heatmap_invalid_input_types(sample_saliency_df):
    """
    Tests the function with invalid data types for its arguments, expecting TypeErrors.
    """
    with pytest.raises(TypeError, match="saliency_dataframe must be a pandas DataFrame"):
        plot_aggregated_saliency_heatmap([1, 2, 3], top_n_tokens=5, title="Title")

    with pytest.raises(TypeError, match="top_n_tokens must be an integer"):
        plot_aggregated_saliency_heatmap(sample_saliency_df, top_n_tokens="five", title="Title")

    with pytest.raises(ValueError, match="top_n_tokens must be positive"):
        plot_aggregated_saliency_heatmap(sample_saliency_df, top_n_tokens=0, title="Title")