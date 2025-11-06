import pytest
import pandas as pd
from unittest.mock import patch

# Keep the following block as is:
# ---------------------------------
from definition_82e0fb0b2d0d4100b36865ac7ed40623 import plot_faithfulness_trend
# ---------------------------------

@pytest.fixture
def sample_dataframe():
    """Provides a sample valid DataFrame for testing."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'faithfulness_metric': [0.8, 0.9, 0.85],
        'xai_technique': ['LIME', 'SHAP', 'LIME']
    }
    return pd.DataFrame(data)

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('seaborn.lineplot')
def test_plot_faithfulness_trend_happy_path(mock_lineplot, mock_savefig, mock_show, sample_dataframe):
    """
    Tests the successful execution of the plot function with valid inputs,
    ensuring that the underlying plotting libraries are called correctly.
    """
    plot_faithfulness_trend(
        sample_dataframe, 
        'timestamp', 
        'faithfulness_metric', 
        'xai_technique', 
        'Test Title'
    )
    mock_lineplot.assert_called_once()
    # Check that seaborn's lineplot was called with the correct data and axes
    call_args, call_kwargs = mock_lineplot.call_args
    assert call_kwargs['x'] == 'timestamp'
    assert call_kwargs['y'] == 'faithfulness_metric'
    assert call_kwargs['hue'] == 'xai_technique'
    pd.testing.assert_frame_equal(call_kwargs['data'], sample_dataframe)
    mock_savefig.assert_called_once()
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
@patch('seaborn.lineplot')
def test_plot_faithfulness_trend_empty_dataframe(mock_lineplot, mock_show):
    """
    Tests that the function handles an empty DataFrame gracefully without errors.
    """
    empty_df = pd.DataFrame({
        'timestamp': pd.Series(dtype='datetime64[ns]'),
        'faithfulness_metric': pd.Series(dtype='float'),
        'xai_technique': pd.Series(dtype='object')
    })
    try:
        plot_faithfulness_trend(empty_df, 'timestamp', 'faithfulness_metric', 'xai_technique', 'Empty')
        mock_lineplot.assert_called_once()
        mock_show.assert_called_once()
    except Exception as e:
        pytest.fail(f"Function raised an unexpected exception with empty DataFrame: {e}")

def test_plot_faithfulness_trend_missing_column(sample_dataframe):
    """
    Tests that the function raises a ValueError or KeyError when a specified column is missing.
    """
    with pytest.raises((ValueError, KeyError)):
        plot_faithfulness_trend(
            sample_dataframe, 
            'timestamp', 
            'faithfulness_metric', 
            'non_existent_column',  # Missing hue column
            'Missing Column Plot'
        )

def test_plot_faithfulness_trend_invalid_dataframe_type():
    """
    Tests that the function raises an AttributeError when a non-DataFrame is passed.
    """
    not_a_dataframe = [("a", 1), ("b", 2)]
    with pytest.raises(AttributeError):
        plot_faithfulness_trend(not_a_dataframe, 'x', 'y', 'hue', 'Invalid Type')

def test_plot_faithfulness_trend_non_numeric_y_axis(sample_dataframe):
    """
    Tests that a TypeError is raised when the y-axis column contains non-numeric data,
    as plotting functions cannot process it.
    """
    sample_dataframe['faithfulness_metric'] = ['low', 'medium', 'high']
    with pytest.raises(TypeError):
        plot_faithfulness_trend(
            sample_dataframe, 
            'timestamp', 
            'faithfulness_metric', 
            'xai_technique', 
            'Non-numeric Data'
        )

