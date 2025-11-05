import pytest
from definition_d57cd71afdd945ca9131aaf3d98f2e89 import filter_by_confidence
import pandas as pd

@pytest.fixture
def create_test_dataframe():
    return pd.DataFrame({
        'model_confidence': [0.9, 0.7, 0.5, 0.95, 0.2, 0.85],
        'data': ['a', 'b', 'c', 'd', 'e', 'f']
    })

@pytest.mark.parametrize("threshold, expected_count", [
    (0.8, 3), 
    (0.5, 5),
    (1.0, 0), 
    (0.0, 6), 
    (0.95, 1)
])
def test_filter_by_confidence(create_test_dataframe, threshold, expected_count):
    result_df = filter_by_confidence(create_test_dataframe, threshold)
    assert len(result_df) == expected_count

@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9])
def test_filter_by_confidence_values(create_test_dataframe, threshold):
    result_df = filter_by_confidence(create_test_dataframe, threshold)
    assert all(result_df['model_confidence'] >= threshold)

def test_filter_by_confidence_no_data(create_test_dataframe):
    empty_df = pd.DataFrame(columns=['model_confidence', 'data'])
    result_df = filter_by_confidence(empty_df, 0.7)
    assert result_df.empty

def test_filter_by_confidence_invalid_threshold(create_test_dataframe):
    with pytest.raises(TypeError):
        filter_by_confidence(create_test_dataframe, 'invalid')