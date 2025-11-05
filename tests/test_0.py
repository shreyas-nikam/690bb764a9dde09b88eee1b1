import pytest
import pandas as pd
from definition_bd864f0e80de4866bb87d7b4f97f1e7c import generate_llm_data

@pytest.mark.parametrize("num_samples, expected", [
    (0, pd.DataFrame),
    (10, pd.DataFrame),
    (-5, ValueError),
    ("100", TypeError),
    (None, TypeError),
])

def test_generate_llm_data(num_samples, expected):
    try:
        result = generate_llm_data(num_samples)
        assert isinstance(result, expected)
        if isinstance(result, pd.DataFrame):
            # Verify columns are as expected
            expected_columns = [
                'timestamp', 'prompt', 'llm_output', 'true_label', 
                'model_confidence', 'model_accuracy', 
                'explanation_quality_score', 'faithfulness_metric', 'xai_technique'
            ]
            assert list(result.columns) == expected_columns
            # Check if DataFrame is the right size
            assert len(result) == num_samples
    except Exception as e:
        assert isinstance(e, expected)