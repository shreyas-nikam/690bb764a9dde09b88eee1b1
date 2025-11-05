import pytest
from definition_6600b9f2cb744868b52c7d5b396c9517 import generate_counterfactual_explanation

@pytest.mark.parametrize("original_prompt, original_output, model_accuracy, expected_keys", [
    ("What is the capital of France?", "Paris", 0.9, 
     {"original_prompt", "original_output", "counterfactual_prompt", "counterfactual_output"}),
    ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side.", 0.8, 
     {"original_prompt", "original_output", "counterfactual_prompt", "counterfactual_output"}),
    ("Summarize the article.", "The article discusses the impact of AI on society.", 0.95, 
     {"original_prompt", "original_output", "counterfactual_prompt", "counterfactual_output"}),
    ("Explain gravity.", "Gravity is a force that attracts objects towards each other.", 1.0, 
     {"original_prompt", "original_output", "counterfactual_prompt", "counterfactual_output"}),
    ("Translate 'Hello' to French.", "Bonjour", 0.85, 
     {"original_prompt", "original_output", "counterfactual_prompt", "counterfactual_output"})
])

def test_generate_counterfactual_explanation(original_prompt, original_output, model_accuracy, expected_keys):
    result = generate_counterfactual_explanation(original_prompt, original_output, model_accuracy)
    assert isinstance(result, dict)
    assert set(result.keys()) == expected_keys
    assert result['original_prompt'] == original_prompt
    assert result['original_output'] == original_output
    assert isinstance(result['counterfactual_prompt'], str)
    assert isinstance(result['counterfactual_output'], str)