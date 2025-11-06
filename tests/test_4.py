import pytest
from definition_491c241a92804ee0b535f82403ce3dc8 import generate_counterfactual_explanation

def test_generate_counterfactual_explanation_happy_path():
    """
    Tests the function with typical, valid inputs to ensure it returns the correct structure
    and that the counterfactuals are different from the originals.
    """
    prompt = "What is the weather like in London?"
    output = "In London, the weather is currently cloudy with a chance of rain."
    accuracy = 0.85
    
    result = generate_counterfactual_explanation(prompt, output, accuracy)
    
    assert isinstance(result, dict)
    assert 'original_prompt' in result
    assert 'original_output' in result
    assert 'counterfactual_prompt' in result
    assert 'counterfactual_output' in result
    
    assert result['original_prompt'] == prompt
    assert result['original_output'] == output
    assert result['counterfactual_prompt'] != prompt
    assert result['counterfactual_output'] != output

def test_generate_counterfactual_explanation_empty_inputs():
    """
    Tests the edge case where input strings are empty. The function should still
    generate non-empty counterfactuals.
    """
    prompt = ""
    output = ""
    accuracy = 0.5
    
    result = generate_counterfactual_explanation(prompt, output, accuracy)
    
    assert isinstance(result, dict)
    assert result['original_prompt'] == ""
    assert result['original_output'] == ""
    assert result['counterfactual_prompt'] != ""
    assert result['counterfactual_output'] != ""
    assert len(result['counterfactual_prompt']) > 0
    assert len(result['counterfactual_output']) > 0

def test_generate_counterfactual_explanation_numeric_string_inputs():
    """
    Tests the edge case with inputs that are strings of numbers. This ensures
    the modification logic handles non-alphabetic characters correctly.
    """
    prompt = "12345"
    output = "54321"
    accuracy = 0.99
    
    result = generate_counterfactual_explanation(prompt, output, accuracy)
    
    assert isinstance(result, dict)
    assert result['original_prompt'] == prompt
    assert result['original_output'] == output
    assert result['counterfactual_prompt'] != prompt
    assert result['counterfactual_output'] != output

@pytest.mark.parametrize("accuracy", [0.0, 1.0, 0.45])
def test_generate_counterfactual_explanation_various_accuracies(accuracy):
    """
    Tests that the function works correctly across a range of model accuracy values,
    including the boundary values 0.0 and 1.0.
    """
    prompt = "A sample prompt."
    output = "A sample output."
    
    result = generate_counterfactual_explanation(prompt, output, accuracy)
    
    assert isinstance(result, dict)
    assert result['counterfactual_prompt'] != prompt
    assert result['counterfactual_output'] != output

@pytest.mark.parametrize("prompt, output, accuracy", [
    (None, "output", 0.5),
    ("prompt", 123, 0.5),
    ("prompt", "output", "high"),
    (["prompt"], "output", 0.5),
])
def test_generate_counterfactual_explanation_invalid_types(prompt, output, accuracy):
    """
    Tests that the function raises a TypeError for inputs of incorrect types.
    """
    with pytest.raises(TypeError):
        generate_counterfactual_explanation(prompt, output, accuracy)