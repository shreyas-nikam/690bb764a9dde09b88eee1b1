import pytest
from definition_fd489ee044214b56a1f1a8977df21fdc import generate_counterfactual_explanation

@pytest.mark.parametrize(
    "original_prompt, original_output, current_model_accuracy, expected_result",
    [
        # Test Case 1: Standard inputs - expected functionality
        ("What is the capital of France?", "Paris is the capital city.", 0.95, True),
        # Test Case 2: Edge case - Empty strings for prompt and output
        ("", "", 0.5, True),
        # Test Case 3: Invalid type for original_prompt
        (123, "This is an output.", 0.8, TypeError),
        # Test Case 4: Invalid type for original_output
        ("This is a prompt.", ["list", "of", "words"], 0.75, TypeError),
        # Test Case 5: Invalid type for current_model_accuracy
        ("This is a prompt.", "This is an output.", "high", TypeError),
    ]
)
def test_generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy, expected_result):
    if expected_result is True:
        # Expected a successful execution and a specific dictionary structure
        result = generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy)

        # Assert result is a dictionary
        assert isinstance(result, dict)

        # Assert required keys are present
        expected_keys = {'original_prompt', 'original_output', 'counterfactual_prompt', 'counterfactual_output'}
        assert set(result.keys()) == expected_keys

        # Assert original inputs are preserved in the dictionary
        assert result['original_prompt'] == original_prompt
        assert result['original_output'] == original_output

        # Assert counterfactuals are strings and are different from originals
        # This tests the core functionality of generating *modified* explanations
        assert isinstance(result['counterfactual_prompt'], str)
        assert result['counterfactual_prompt'] != original_prompt

        assert isinstance(result['counterfactual_output'], str)
        assert result['counterfactual_output'] != original_output
    else:
        # Expected an exception
        with pytest.raises(expected_result):
            generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy)