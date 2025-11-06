import pytest
from IPython.display import HTML
from definition_aa2e9ae0c188400384d58b63a9243b3e import visualize_saliency_map

@pytest.mark.parametrize(
    "llm_output, token_scores, threshold, expected_result",
    [
        # Test Case 1: Basic functionality - some tokens highlighted
        # Expected: "This", "a", "sentence." are highlighted.
        (
            "This is a test sentence.",
            [("This", 0.8), ("is", 0.3), ("a", 0.9), ("test", 0.4), ("sentence.", 0.7)],
            0.6,
            HTML('<span style="background-color: yellow;">This</span> is <span style="background-color: yellow;">a</span> test <span style="background-color: yellow;">sentence.</span>')
        ),
        # Test Case 2: Edge case - no tokens highlighted (all scores below threshold)
        # Expected: No highlighting, plain text.
        (
            "Hello world.",
            [("Hello", 0.1), ("world.", 0.2)],
            0.5,
            HTML("Hello world.")
        ),
        # Test Case 3: Edge case - all tokens highlighted (all scores above threshold)
        # Expected: All tokens highlighted.
        (
            "Important message.",
            [("Important", 0.9), ("message.", 0.8)],
            0.5,
            HTML('<span style="background-color: yellow;">Important</span> <span style="background-color: yellow;">message.</span>')
        ),
        # Test Case 4: Edge case - empty LLM output and token scores
        # Expected: An empty HTML object.
        (
            "",
            [],
            0.5,
            HTML("")
        ),
        # Test Case 5: Error handling - invalid llm_output type
        # Expected: TypeError, as string methods (.split()) would be called on an int.
        (
            123,  # Invalid type: integer instead of string
            [("token", 0.6)],
            0.5,
            TypeError
        ),
    ]
)
def test_visualize_saliency_map(llm_output, token_scores, threshold, expected_result):
    # If the expected_result is an Exception type, we expect the function to raise it.
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            visualize_saliency_map(llm_output, token_scores, threshold)
    else:
        # Otherwise, we expect a successful return value (HTML object)
        result = visualize_saliency_map(llm_output, token_scores, threshold)
        
        # Assert that the result is an IPython.display.HTML object
        assert isinstance(result, HTML)
        
        # Assert that the data attribute of the HTML object matches the expected HTML string
        assert result.data == expected_result.data