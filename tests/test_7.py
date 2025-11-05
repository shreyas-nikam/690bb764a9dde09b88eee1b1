import pytest
import pandas as pd
from definition_92d7d67c61b74402907c8f943e2f451e import plot_aggregated_saliency_heatmap

@pytest.fixture
def saliency_dataframe():
    return pd.DataFrame({
        'token': ['token1', 'token2', 'token3', 'token4'],
        'saliency_score': [0.9, 0.8, 0.2, 0.1]
    })

@pytest.mark.parametrize("top_n_tokens,expected_exception", [
    (2, None),
    (10, IndexError),  
    (-1, ValueError),  
    ('3', TypeError),  
    (None, TypeError)
])
def test_plot_aggregated_saliency_heatmap_exceptions(saliency_dataframe, top_n_tokens, expected_exception):
    try:
        plot_aggregated_saliency_heatmap(saliency_dataframe, top_n_tokens, "Test Title")
    except Exception as exc:
        assert isinstance(exc, expected_exception)

def test_plot_aggregated_saliency_heatmap_no_exception(saliency_dataframe):
    try:
        plot_aggregated_saliency_heatmap(saliency_dataframe, 3, "Test Title")
    except Exception as exc:
        assert False, f"Unexpected exception raised: {exc}"