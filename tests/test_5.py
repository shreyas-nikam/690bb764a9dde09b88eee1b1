import pytest
import pandas as pd
from unittest.mock import patch, ANY

from definition_a0fce57d167a40519aff768bd37237de import plot_faithfulness_trend

# --- Test Data Setup ---
valid_df = pd.DataFrame({
    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
    'faithfulness_metric': [0.8, 0.85, 0.9],
    'xai_technique': ['LIME', 'SHAP', 'LIME']
})

empty_df = pd.DataFrame({
    'timestamp': pd.Series(dtype='datetime64[ns]'),
    'faithfulness_metric': pd.Series(dtype='float64'),
    'xai_technique': pd.Series(dtype='object')
})

# --- Test Cases ---
@pytest.mark.parametrize("dataframe, x_axis, y_axis, hue_column, title, expected_exception", [
    # 1. Happy path: Valid inputs should execute without error.
    (valid_df, 'timestamp', 'faithfulness_metric', 'xai_technique', 'Faithfulness Trend', None),

    # 2. Edge case: An empty DataFrame should be handled gracefully without errors.
    (empty_df, 'timestamp', 'faithfulness_metric', 'xai_technique', 'Empty Plot', None),

    # 3. Error case: DataFrame missing the specified y-axis column.
    (valid_df.drop(columns=['faithfulness_metric']), 'timestamp', 'faithfulness_metric', 'xai_technique', 'Error Plot', (KeyError, ValueError)),

    # 4. Error case: DataFrame missing the specified hue column.
    (valid_df, 'timestamp', 'faithfulness_metric', 'non_existent_hue', 'Error Plot', (KeyError, ValueError)),

    # 5. Error case: The 'dataframe' argument is not a pandas DataFrame.
    ([1, 2, 3], 'x', 'y', 'hue', 'Error Plot', AttributeError),
])
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
@patch('seaborn.lineplot')
def test_plot_faithfulness_trend(mock_lineplot, mock_savefig, mock_show, dataframe, x_axis, y_axis, hue_column, title, expected_exception):
    """
    Tests the plot_faithfulness_trend function across various scenarios including
    happy path, edge cases, and error conditions.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title)
    else:
        try:
            plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title)
            # Assert that core plotting functions were called for successful cases
            mock_lineplot.assert_called()
            mock_savefig.assert_called_with(ANY)
            mock_show.assert_called_once()
        except Exception as e:
            pytest.fail(f"Function raised an unexpected exception: {e}")

