import pytest
from definition_45faa801e2ce489f888670f0ec9ec901 import visualize_saliency_map
from IPython.display import HTML

@pytest.mark.parametrize("llm_output, token_scores, threshold, expected_output", [
    # Test normal case with threshold highlighting
    ("This is a test", [0.2, 0.5, 0.8, 0.9], 0.7, HTML("This is <span style=\"background-color: yellow;\">a</span> <span style=\"background-color: yellow;\">test</span>")),
    
    # Test case where no token meets the threshold
    ("Completely safe text", [0.1, 0.2, 0.3], 0.5, HTML("Completely safe text")),

    # Edge case with empty string and no tokens
    ("", [], 0.5, HTML("")),

    # Edge case with threshold of 0 and all tokens highlighted
    ("Highlight everything", [0.3, 0.4, 0.5], 0.0, HTML("<span style=\"background-color: yellow;\">Highlight</span> <span style=\"background-color: yellow;\">everything</span>")),

    # Edge case with threshold of 1 where no token would be highlighted
    ("No highlights", [0.3, 0.9, 0.7], 1.0, HTML("No highlights")),
])

def test_visualize_saliency_map(llm_output, token_scores, threshold, expected_output):
    result = visualize_saliency_map(llm_output, token_scores, threshold)
    assert str(result) == str(expected_output)