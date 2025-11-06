import pytest
import pandas as pd

from definition_2d59174ad8b54815a79ec171ede71f9d import plot_quality_vs_accuracy

@pytest.mark.parametrize(
    "dataframe, x_axis, y_axis, title, expected_outcome",
    [
        # Test Case 1: Happy path with valid DataFrame and columns.
        # Expected: Function executes without error, implying successful plotting.
        (pd.DataFrame({
            'model_accuracy': [0.7, 0.8, 0.9, 0.6, 0.75],
            'explanation_quality_score': [0.5, 0.6, 0.8, 0.4, 0.7]
        }), 'model_accuracy', 'explanation_quality_score', 'Quality vs Accuracy Plot', None),

        # Test Case 2: Empty DataFrame but with the expected columns.
        # Expected: Should not raise an error, seaborn handles plotting an empty dataset gracefully.
        (pd.DataFrame({
            'model_accuracy': [],
            'explanation_quality_score': []
        }), 'model_accuracy', 'explanation_quality_score', 'Empty Data Plot', None),

        # Test Case 3: DataFrame missing the specified x_axis column.
        # Expected: A KeyError will be raised when pandas/seaborn try to access the non-existent column.
        (pd.DataFrame({
            'other_metric': [1, 2, 3],
            'explanation_quality_score': [0.5, 0.6, 0.8]
        }), 'model_accuracy', 'explanation_quality_score', 'Missing X Column', KeyError),

        # Test Case 4: y_axis column contains non-numeric data (e.g., strings).
        # Expected: TypeError or ValueError from seaborn/matplotlib when attempting to plot non-numeric data.
        (pd.DataFrame({
            'model_accuracy': [0.7, 0.8, 0.9],
            'explanation_quality_score': ['low', 'medium', 'high']
        }), 'model_accuracy', 'explanation_quality_score', 'Non-Numeric Y-Axis', (TypeError, ValueError)),

        # Test Case 5: Invalid type for the 'dataframe' argument (e.g., None instead of a DataFrame).
        # Expected: AttributeError or TypeError when trying to access DataFrame methods/attributes.
        (None, 'model_accuracy', 'explanation_quality_score', 'Invalid DataFrame Input', (AttributeError, TypeError)),
    ]
)
def test_plot_quality_vs_accuracy(dataframe, x_axis, y_axis, title, expected_outcome):
    try:
        # Call the function under test
        plot_quality_vs_accuracy(dataframe, x_axis, y_axis, title)
        # If no exception occurred, assert that the expected outcome was None (i.e., success)
        assert expected_outcome is None
    except Exception as e:
        # If an exception occurred, assert that its type matches the expected exception(s)
        assert isinstance(e, expected_outcome)

