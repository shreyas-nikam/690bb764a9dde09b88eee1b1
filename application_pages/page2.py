import streamlit as st
import pandas as pd
import numpy as np
import random

# Reproducibility (will be set once, perhaps in session_state or app initialization)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def visualize_saliency_map(llm_output, token_scores, threshold=0.5):
    """
    Highlights tokens in an LLM output string based on saliency scores using HTML.
    Tokens with scores at or above the specified threshold are highlighted.
    """
    tokens = llm_output.split()
    highlighted_tokens = [
        f"<span style=\"background-color: yellow;\">{token}</span>" if score >= threshold else token
        for token, score in zip(tokens, token_scores)
    ]
    highlighted_text = " ".join(highlighted_tokens)
    return highlighted_text # Return raw HTML string for st.markdown

def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """
    Generates a counterfactual explanation by proposing a modified input prompt
    and its resulting output.
    """
    # Simulate a minimal modification to the original prompt
    counterfactual_prompt = original_prompt + " (What if incorrect?)"

    # Simulate generating a counterfactual output
    if current_model_accuracy > 0.5:
        counterfactual_output = "A different plausible explanation."
    else:
        counterfactual_output = "An incorrect explanation."

    return {
        "original_prompt": original_prompt,
        "original_output": original_output,
        "counterfactual_prompt": counterfactual_prompt,
        "counterfactual_output": counterfactual_output
    }

def run_page2():
    st.title("XAI Techniques: Saliency Maps & Counterfactuals")

    st.markdown("""
    ## 3.11 Interpretability vs. Transparency
    -   **Interpretability**: understanding why the model produced a specific output for a given input; focuses on input–output relationships.
    -   **Transparency**: understanding how the model works internally (architecture, training data, parameters). For large LLMs, this is often infeasible in practice.

    Business impact: When full transparency is not possible, strong interpretability (clear rationales, salient inputs, counterfactuals) supports audits, incident response, and policy compliance without exposing proprietary internals.
    """)

    st.markdown("""
    ## 3.12 Introduction to XAI Techniques (Saliency and Counterfactuals)

    -   **Saliency Maps**: Highlight which input tokens most influenced an output. Conceptually, token importance can be related to local sensitivity of the output with respect to an input token:
        $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
        Higher values indicate stronger influence on the model’s decision, helping reviewers quickly locate decisive words.

    -   **Counterfactual Explanations**: Show how minimal changes to an input could change the model’s output. Formally, we seek a small perturbation $\Delta X$ such that:
        $$ \text{Model}(X + \Delta X) = Y' \ne Y, \quad \text{with minimal } ||\Delta X|| $$
        This supports “what-if” analysis, recourse options, and fairness assessments.

    Business value: Saliency sharpens local interpretability for specific outputs; counterfactuals reveal levers that alter outcomes—both are essential for audits, compliance, and user trust.
    """)

    if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty and \
       'df_saliency' in st.session_state and not st.session_state['df_saliency'].empty:
        st.markdown("""
        ## 3.13 Applying XAI Technique: Saliency Map Simulation — Context & Business Value
        Saliency maps help reviewers quickly identify which words in an output likely drove the model’s behavior. This accelerates root-cause analysis in incident reviews and supports explainability audits without revealing proprietary internals.

        Formula (conceptual sensitivity):
        -   Token importance: $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
        -   In this notebook we simulate $S(x_i)$ with random values in $[0,1]$ and highlight tokens with scores above a threshold $\tau$.
        """)

        st.subheader("Saliency Map Visualization")
        
        # Ensure index is within bounds
        max_index_saliency = len(st.session_state['df_llm_data']) - 1
        if max_index_saliency < 0:
            st.info("No data available for saliency map visualization.")
            return

        sample_index = st.number_input(
            "Select Output for Saliency Map", 
            min_value=0, 
            max_value=max_index_saliency, 
            value=0,
            help="Choose an index from the dataset to visualize its saliency map."
        )

        saliency_threshold = st.slider(
            r"Saliency Highlight Threshold $(\tau)$", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Tokens with saliency scores at or above this threshold will be highlighted."
        )

        sample_output = st.session_state['df_llm_data'].loc[sample_index, 'llm_output']
        sample_scores_df = st.session_state['df_saliency'][st.session_state['df_saliency']['output_index'] == sample_index]

        if not sample_scores_df.empty:
            sample_scores = sample_scores_df['saliency_score'].tolist()
            tokens = sample_output.split()
            if len(sample_scores) < len(tokens):
                sample_scores = sample_scores + [0.0] * (len(tokens) - len(sample_scores))
            elif len(sample_scores) > len(tokens):
                sample_scores = sample_scores[:len(tokens)]

            html_vis = visualize_saliency_map(sample_output, sample_scores, threshold=saliency_threshold)
            st.markdown(html_vis, unsafe_allow_html=True)
        else:
            st.info("No saliency scores found for the selected output index. Saliency data is generated for the first 10 outputs. Please select an index between 0 and 9.")

        st.markdown("""
        ## 3.15 Interpreting the Saliency Map
        The highlighted words represent tokens with saliency scores above the chosen threshold ($\tau = 0.7$). In a real system, these would indicate which parts of the text most influenced the model’s output. This helps reviewers:
        -   Quickly spot decisive phrases.
        -   Compare highlighted rationale with policy or business rules.
        -   Identify unexpected drivers of model behavior that may need mitigation or re-training.
        """)

        st.markdown("""
        ## Optional Result Interpretation: Heuristic Agreement
        The printed agreement rate shows how often a simple confidence-based heuristic aligns with the simulated accuracy flag. A higher value suggests confidence is a reasonable proxy for correctness; a lower value warns that confidence alone may be misleading. In governance, such diagnostics inform threshold setting for human-in-the-loop review.
        """)

        st.markdown("""
        ## 3.16 Applying XAI Technique: Counterfactual Explanation — Context & Business Value
        Counterfactuals answer the question “What minimal change to the input would have changed the output?” They are critical for recourse (what a user could do differently), fairness (sensitivity to protected attributes), and debugging (what levers flip the decision).

        Formula (goal):
        -   Find a perturbation $\Delta X$ such that the model output flips: $$ \text{Model}(X + \Delta X) = Y' \ne Y $$
        -   With minimal change: minimize $||\Delta X||$ subject to the flip. In this notebook, we simulate a simple, illustrative counterfactual.
        """)

        st.subheader("Counterfactual Explanation")

        # Ensure index is within bounds
        max_index_cf = len(st.session_state['df_llm_data']) - 1
        if max_index_cf < 0:
            st.info("No data available for counterfactual explanation.")
            return

        cf_idx = st.number_input(
            "Select Record for Counterfactuals", 
            min_value=0, 
            max_value=max_index_cf, 
            value=0,
            help="Choose an index to generate a counterfactual explanation for its prompt and output."
        )
        
        original_prompt = st.session_state['df_llm_data'].loc[cf_idx, 'prompt']
        original_output = st.session_state['df_llm_data'].loc[cf_idx, 'llm_output']
        current_model_accuracy = float(st.session_state['df_llm_data'].loc[cf_idx, 'model_accuracy'])

        cf_result = generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy)

        st.write('**Original Prompt:**', cf_result['original_prompt'])
        st.write('**Original Output:**', cf_result['original_output'])
        st.write('\n**Counterfactual Prompt:**', cf_result['counterfactual_prompt'])
        st.write('**Counterfactual Output:**', cf_result['counterfactual_output'])

        st.markdown("""
        ## 3.18 Interpreting the Counterfactual Explanation
        The counterfactual demonstrates how a small prompt change could plausibly lead to a different outcome. In practice, this:
        -   Helps determine levers for decision recourse (what a user could change).
        -   Reveals sensitivity to specific terms or conditions for fairness auditing.
        -   Supports root-cause analysis by contrasting original vs. perturbed inputs and outcomes.
        """)
    else:
        st.info("Please generate or upload data on the 'Data Generation & Validation' page to proceed with XAI techniques.")

