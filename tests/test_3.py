import pytest
from definition_ee8d487e7bfa4ef1bc582b693323af11 import visualize_saliency_map

# Assuming the function is refactored to return the HTML string for testability,
# rather than directly calling IPython.display. This is a standard practice
# to separate logic from presentation.

@pytest.mark.parametrize("llm_output, token_scores, threshold, expected", [
    # Test case 1: Basic functionality with some tokens highlighted
    ("This is a test sentence", [0.9, 0.2, 0.8, 0.1, 0.95], 0.7, 
     '<span style="background-color: yellow;">This</span> is <span style="background-color: yellow;">a</span> test <span style="background-color: yellow;">sentence</span>'),

    # Test case 2: Edge case where no tokens are above the threshold
    ("All scores are low", [0.1, 0.2, 0.3, 0.4], 0.5, 
     "All scores are low"),

    # Test case 3: Edge case where all tokens are above the threshold
    ("Highlight every single word", [0.9, 0.8, 0.95, 0.85], 0.7, 
     '<span style="background-color: yellow;">Highlight</span> <span style="background-color: yellow;">every</span> <span style="background-color: yellow;">single</span> <span style="background-color: yellow;">word</span>'),

    # Test case 4: Edge case with empty input string and scores
    ("", [], 0.5, 
     ""),

    # Test case 5: Error case with mismatched number of tokens and scores
    ("Mismatched length", [0.9], 0.5, 
     ValueError)
])
def test_visualize_saliency_map(llm_output, token_scores, threshold, expected):
    """
    Tests the visualize_saliency_map function for various scenarios including
    correct HTML generation, edge cases, and error handling.
    """
    try:
        result = visualize_saliency_map(llm_output, token_scores, threshold)
        assert result == expected
    except Exception as e:
        assert isinstance(e, expected)

