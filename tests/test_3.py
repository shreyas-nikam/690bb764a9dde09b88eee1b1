import pytest
from IPython.display import HTML
from definition_742010575a164ec48a024f10d0fa99af import visualize_saliency_map

# Test cases for visualize_saliency_map function
@pytest.mark.parametrize("llm_output, token_scores, threshold, expected_html", [
    # Test case 1: Standard functionality with a mix of scores above and below the threshold.
    (
        "The quick brown fox",
        [('The', 0.9), ('quick', 0.2), ('brown', 0.8), ('fox', 0.4)],
        0.5,
        '<span style="background-color: yellow;">The</span> quick <span style="background-color: yellow;">brown</span> fox'
    ),
    # Test case 2: Edge case where no token scores meet the threshold, expecting no highlighting.
    (
        "All scores low",
        [('All', 0.1), ('scores', 0.2), ('low', 0.3)],
        0.4,
        'All scores low'
    ),
    # Test case 3: Edge case where all token scores are above the threshold, expecting all tokens to be highlighted.
    (
        "All scores high",
        [('All', 0.9), ('scores', 0.8), ('high', 0.7)],
        0.6,
        '<span style="background-color: yellow;">All</span> <span style="background-color: yellow;">scores</span> <span style="background-color: yellow;">high</span>'
    ),
    # Test case 4: Edge case with empty inputs, expecting an empty HTML output.
    (
        "",
        [],
        0.5,
        ''
    ),
    # Test case 5: Boundary condition where some scores are exactly equal to the threshold.
    (
        "A score on the edge",
        [('A', 0.9), ('score', 0.5), ('on', 0.4), ('the', 0.5), ('edge', 0.6)],
        0.5,
        '<span style="background-color: yellow;">A</span> <span style="background-color: yellow;">score</span> on <span style="background-color: yellow;">the</span> <span style="background-color: yellow;">edge</span>'
    )
])
def test_visualize_saliency_map(llm_output, token_scores, threshold, expected_html):
    """
    Tests the visualize_saliency_map function for various scenarios including
    standard operation, edge cases, and boundary conditions.
    """
    # The function is expected to build the HTML string from the token_scores list.
    # The llm_output parameter is used contextually but the primary source for the
    # rendered text is the list of tokens.
    result = visualize_saliency_map(llm_output, token_scores, threshold)
    
    # Verify that the output is an IPython.display.HTML object
    assert isinstance(result, HTML)
    
    # Verify that the generated HTML content is correct
    assert result.data == expected_html

