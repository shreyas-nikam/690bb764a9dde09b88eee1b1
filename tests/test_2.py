import pytest
from definition_bdc3ada2e9204186979a391fb92f60a2 import validate_and_summarize_data
import pandas as pd

@pytest.fixture
def example_dataframe():
    return pd.DataFrame({
        'numeric_column': [1, 2, 3, None],
        'categorical_column': ['A', 'B', 'A', 'B'],
        'critical_numeric': [1.0, None, 3.5, 0.0]
    })

def test_validate_and_summarize_data_success(example_dataframe):
    dataframe = example_dataframe
    # Simulate a successful validation
    dataframe['critical_numeric'].fillna(0, inplace=True)
    validate_and_summarize_data(dataframe)
    # Assume that the function prints information, so there's no return to check

def test_missing_critical_numeric(example_dataframe):
    dataframe = example_dataframe
    with pytest.raises(AssertionError):
        validate_and_summarize_data(dataframe)

def test_extra_column(example_dataframe):
    dataframe = example_dataframe
    dataframe['extra_column'] = [1, 2, 3, 4]
    # Should pass as extra columns don't affect the validation of expected columns
    validate_and_summarize_data(dataframe)

def test_unexpected_datatype(example_dataframe):
    dataframe = example_dataframe
    dataframe['numeric_column'] = ['one', 'two', 'three', 'four']
    with pytest.raises(AssertionError):
        validate_and_summarize_data(dataframe)

def test_empty_dataframe():
    dataframe = pd.DataFrame()
    with pytest.raises(AssertionError):
        validate_and_summarize_data(dataframe)