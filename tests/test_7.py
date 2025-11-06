import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Block for definition_fd1186e78b8644f08efcc72dd6a24b47 - DO NOT REPLACE or REMOVE
from definition_fd1186e78b8644f08efcc72dd6a24b47 import plot_aggregated_saliency_heatmap
# END Block for definition_fd1186e78b8644f08efcc72dd6a24b47

@pytest.fixture
def mock_matplotlib_seaborn(mocker):
    """Fixture to mock matplotlib and seaborn plotting functions."""
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.savefig')
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.xlabel')
    mocker.patch('matplotlib.pyplot.ylabel')
    mocker.patch('matplotlib.pyplot.xticks')
    mocker.patch('matplotlib.pyplot.tick_params')
    mocker.patch('matplotlib.pyplot.tight_layout')
    mocker.patch('seaborn.set_theme')
    mocker.patch('seaborn.heatmap')

def create_saliency_df(num_rows=10, num_outputs=3):
    """Helper to create a sample saliency DataFrame."""
    data = {
        'output_index': np.random.randint(0, num_outputs, num_rows),
        'token': [f'token_{i % 5}' for i in range(num_rows)], # 5 unique tokens
        'saliency_score': np.random.rand(num_rows)
    }
    return pd.DataFrame(data)

# Test Cases (at most 5)

def test_plot_aggregated_saliency_heatmap_happy_path(mock_matplotlib_seaborn):
    """
    Test case 1: Basic functionality with valid data and positive top_n_tokens.
    Verifies that plotting functions are called as expected and heatmap receives valid data.
    """
    saliency_df = create_saliency_df(num_rows=20, num_outputs=5)
    top_n = 3
    title = "Sample Saliency Heatmap"

    plot_aggregated_saliency_heatmap(saliency_df, top_n, title)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure.assert_called_once()
    sns.set_theme.assert_called_once_with(style="whitegrid", palette="viridis")
    sns.heatmap.assert_called_once()
    
    actual_data_passed = sns.heatmap.call_args[0][0]
    assert isinstance(actual_data_passed, pd.DataFrame)
    assert 'Average Saliency' in actual_data_passed.columns
    assert len(actual_data_passed) <= top_n # Number of tokens plotted should be <= top_n
    
    plt.title.assert_called_once_with(title, fontsize=14)
    plt.savefig.assert_called_once_with("aggregated_saliency_heatmap.png")
    plt.show.assert_called_once()


def test_plot_aggregated_saliency_heatmap_empty_dataframe(mock_matplotlib_seaborn):
    """
    Test case 2: saliency_dataframe is empty.
    Verifies that it handles empty data gracefully without errors, and no heatmap is drawn.
    """
    saliency_df = pd.DataFrame(columns=['output_index', 'token', 'saliency_score'])
    top_n = 5
    title = "Empty Saliency Heatmap"

    plot_aggregated_saliency_heatmap(saliency_df, top_n, title)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure.assert_called_once()
    sns.set_theme.assert_called_once()
    sns.heatmap.assert_not_called() 

    plt.title.assert_called_once_with(title, fontsize=14)
    plt.savefig.assert_called_once_with("aggregated_saliency_heatmap.png")
    plt.show.assert_called_once()


def test_plot_aggregated_saliency_heatmap_top_n_tokens_zero(mock_matplotlib_seaborn):
    """
    Test case 3: top_n_tokens is 0.
    Verifies that it runs without error and no heatmap is drawn.
    """
    saliency_df = create_saliency_df(num_rows=20, num_outputs=5)
    top_n = 0
    title = "Zero Top Tokens Heatmap"

    plot_aggregated_saliency_heatmap(saliency_df, top_n, title)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure.assert_called_once()
    sns.set_theme.assert_called_once()
    sns.heatmap.assert_not_called() 

    plt.title.assert_called_once_with(title, fontsize=14)
    plt.savefig.assert_called_once_with("aggregated_saliency_heatmap.png")
    plt.show.assert_called_once()


@pytest.mark.parametrize("missing_column", [
    'token',
    'saliency_score',
    'output_index'
])
def test_plot_aggregated_saliency_heatmap_missing_required_column(mock_matplotlib_seaborn, missing_column):
    """
    Test case 4: saliency_dataframe is missing a required column.
    Expects a KeyError.
    """
    saliency_df = create_saliency_df(num_rows=10)
    saliency_df = saliency_df.drop(columns=[missing_column])
    top_n = 3
    title = "Missing Column Heatmap"

    with pytest.raises(KeyError) as excinfo:
        plot_aggregated_saliency_heatmap(saliency_df, top_n, title)
    
    assert f"saliency_dataframe must contain columns: ['token', 'saliency_score', 'output_index']" in str(excinfo.value)


@pytest.mark.parametrize("saliency_dataframe_arg, top_n_tokens_arg, title_arg, expected_exception, error_message_part", [
    (123, 5, "Title", TypeError, "saliency_dataframe must be a pandas DataFrame."), 
    (create_saliency_df(), "not_an_int", "Title", TypeError, "top_n_tokens must be an integer."), 
    (create_saliency_df(), 3.5, "Title", TypeError, "top_n_tokens must be an integer."), 
    (create_saliency_df(), -1, "Title", ValueError, "top_n_tokens cannot be negative."), 
    (create_saliency_df(), 5, 123, TypeError, "title must be a string.") 
])
def test_plot_aggregated_saliency_heatmap_invalid_argument_types_or_values(
    mock_matplotlib_seaborn, saliency_dataframe_arg, top_n_tokens_arg, title_arg, expected_exception, error_message_part):
    """
    Test case 5: Invalid types or values for saliency_dataframe, top_n_tokens, or title.
    Expects TypeError or ValueError.
    """
    # Use the pre-created DataFrame or the invalid argument directly
    if not isinstance(saliency_dataframe_arg, pd.DataFrame): 
        saliency_df = saliency_dataframe_arg
    else:
        saliency_df = saliency_dataframe_arg

    with pytest.raises(expected_exception) as excinfo:
        plot_aggregated_saliency_heatmap(saliency_df, top_n_tokens_arg, title_arg)
    
    assert error_message_part in str(excinfo.value)