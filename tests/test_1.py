import pytest
import pandas as pd
from definition_abd059c6a80e48739e6971eb14ea4a64 import generate_saliency_data

@pytest.fixture
def sample_llm_outputs():
    return pd.Series([
        "Hello world",
        "This is a test",
        ""
    ])

@pytest.mark.parametrize("llm_outputs, expected_columns, expected_length", [
    (pd.Series(["Hello world"]), ['output_index', 'token', 'saliency_score'], 2),
    (pd.Series(["Hello world", "This is a test"]), ['output_index', 'token', 'saliency_score'], 5),
    (pd.Series([]), ['output_index', 'token', 'saliency_score'], 0),
    (pd.Series([""]), ['output_index', 'token', 'saliency_score'], 0),
    (pd.Series(["a" * 1000]), ['output_index', 'token', 'saliency_score'], 1)
])
def test_generate_saliency_data(llm_outputs, expected_columns, expected_length):
    df = generate_saliency_data(llm_outputs)
    assert list(df.columns) == expected_columns
    assert len(df) == expected_length

def test_generate_saliency_data_random_scores(sample_llm_outputs):
    df = generate_saliency_data(sample_llm_outputs)
    assert df['saliency_score'].min() >= 0.0
    assert df['saliency_score'].max() <= 1.0

def test_generate_saliency_data_output_index(sample_llm_outputs):
    df = generate_saliency_data(sample_llm_outputs)
    assert sorted(df['output_index'].unique()) == list(range(len(sample_llm_outputs)))