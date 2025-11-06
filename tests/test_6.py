import pytest
import pandas as pd
from unittest.mock import MagicMock

from definition_d0c9997da7754c9b9882dd20aa1ea5f1 import plot_quality_vs_accuracy

@pytest.fixture
def sample_dataframe():
    """Provides a sample pandas DataFrame for testing."""
    data = {
        'model_accuracy': [0.85, 0.90, 0.78, 0.95, 0.88],
        'explanation_quality_score': [0.7, 0.8, 0.6, 0.9, 0.75],
    }
    return pd.DataFrame(data)

def test_plot_quality_vs_accuracy_success(sample_dataframe, monkeypatch):
    """
    Test case for successful execution with valid inputs.
    Verifies that underlying plotting libraries are called with correct arguments.
    """
    mock_scatterplot = MagicMock()
    mock_title = MagicMock()
    mock_savefig = MagicMock()
    mock_show = MagicMock()
    
    # Patch the plotting functions to prevent GUI pop-ups and file I/O during tests
    monkeypatch.setattr("seaborn.scatterplot", mock_scatterplot)
    monkeypatch.setattr("matplotlib.pyplot.title", mock_title)
    monkeypatch.setattr("matplotlib.pyplot.savefig", mock_savefig)
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

    title_text = "Quality vs. Accuracy"
    plot_quality_vs_accuracy(
        dataframe=sample_dataframe,
        x_axis='model_accuracy',
        y_axis='explanation_quality_score',
        title=title_text
    )

    mock_scatterplot.assert_called_once()
    call_kwargs = mock_scatterplot.call_args.kwargs
    assert call_kwargs['x'] == 'model_accuracy'
    assert call_kwargs['y'] == 'explanation_quality_score'
    assert call_kwargs['data'].equals(sample_dataframe)
    
    mock_title.assert_called_once_with(title_text)
    mock_savefig.assert_called_once()
    mock_show.assert_called_once()

def test_plot_quality_vs_accuracy_missing_column(sample_dataframe):
    """
    Test case for a missing column in the DataFrame.
    Expects a KeyError when a specified axis column does not exist.
    """
    with pytest.raises(KeyError):
        plot_quality_vs_accuracy(
            dataframe=sample_dataframe,
            x_axis='model_accuracy',
            y_axis='non_existent_column',
            title='Missing Column Test'
        )

def test_plot_quality_vs_accuracy_invalid_dataframe_type():
    """
    Test case for providing an invalid type for the dataframe argument.
    Expects an AttributeError when a list is passed instead of a DataFrame.
    """
    with pytest.raises(AttributeError):
        plot_quality_vs_accuracy(
            dataframe=['not', 'a', 'dataframe'],
            x_axis='x',
            y_axis='y',
            title='Invalid Type Test'
        )

def test_plot_quality_vs_accuracy_empty_dataframe(monkeypatch):
    """
    Edge case test to ensure the function handles an empty DataFrame gracefully.
    It should execute without raising an error.
    """
    mock_scatterplot = MagicMock()
    mock_show = MagicMock()
    monkeypatch.setattr("seaborn.scatterplot", mock_scatterplot)
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)
    monkeypatch.setattr("matplotlib.pyplot.savefig", MagicMock())
    
    empty_df = pd.DataFrame({'model_accuracy': [], 'explanation_quality_score': []})
    
    try:
        plot_quality_vs_accuracy(
            dataframe=empty_df,
            x_axis='model_accuracy',
            y_axis='explanation_quality_score',
            title="Empty Data Plot"
        )
    except Exception as e:
        pytest.fail(f"Function raised an unexpected exception with empty DataFrame: {e}")

    mock_scatterplot.assert_called_once()
    mock_show.assert_called_once()

def test_plot_quality_vs_accuracy_non_numeric_data(monkeypatch):
    """
    Edge case test with non-numeric data in specified columns.
    Expects a TypeError to be raised by the underlying plotting library.
    """
    df = pd.DataFrame({
        'model_accuracy': [0.8, 0.9],
        'explanation_quality_score': ['low', 'high']  # String data
    })
    
    # Mock show to prevent plot window
    monkeypatch.setattr("matplotlib.pyplot.show", MagicMock())

    with pytest.raises(TypeError):
        plot_quality_vs_accuracy(
            dataframe=df,
            x_axis='model_accuracy',
            y_axis='explanation_quality_score',
            title='Non-Numeric Data Test'
        )

