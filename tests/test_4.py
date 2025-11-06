import pytest
from definition_dc198cc7f55443faa44ad1a2e284a325 import generate_counterfactual_explanation

@pytest.mark.parametrize("original_prompt, original_output, current_model_accuracy, expected_keys", [
    ("Why is the sky blue?", "The sky appears blue to human eyes during the day.", 0.95, 
     {'original_prompt', 'original_output', 'counterfactual_prompt', 'counterfactual_output'}),
    ("How does AI work?", "AI uses machine learning to simulate human intelligence.", 0.89, 
     {'original_prompt', 'original_output', 'counterfactual_prompt', 'counterfactual_output'}),
])

def test_generate_counterfactual_explanation_structure(original_prompt, original_output, current_model_accuracy, expected_keys):
    result = generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy)
    assert isinstance(result, dict)
    assert set(result.keys()) == expected_keys

def test_generate_counterfactual_explanation_prompt_change():
    result = generate_counterfactual_explanation("Explain gravity", "Gravity is a force.", 0.75)
    assert result['original_prompt'] != result['counterfactual_prompt']

def test_generate_counterfactual_explanation_output_change():
    result = generate_counterfactual_explanation("Define photosynthesis", "Photosynthesis is a process.", 0.85)
    assert result['original_output'] != result['counterfactual_output']

@pytest.mark.parametrize("invalid_input", [
    (123, "Output", 0.9),  # Non-string original_prompt
    ("Prompt", 456, 0.9),  # Non-string original_output
    ("Prompt", "Output", "high"),  # Non-float current_model_accuracy
])
def test_generate_counterfactual_explanation_invalid_inputs(invalid_input):
    with pytest.raises(Exception):
        generate_counterfactual_explanation(*invalid_input)