
import streamlit as st
import pandas as pd
import numpy as np
import random

# Reproducibility (will be set once, perhaps in session_state or app initialization)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

@st.cache_data
def generate_saliency_data(llm_outputs):
    """Generates synthetic token-level saliency scores for LLM output strings."""
    data = []
    for idx, output in enumerate(llm_outputs):
        tokens = str(output).split() # Ensure output is string
        if not tokens:
            # Handle empty strings to avoid errors later
            data.append((idx, "", 0.0))
            continue
        for token in tokens:
            saliency_score = np.random.rand()
            data.append((idx, token, saliency_score))
    return pd.DataFrame(data, columns=['output_index', 'token', 'saliency_score'])

def visualize_saliency_map(llm_output, token_scores_df, output_index, threshold=0.5):
    """
    Highlights tokens in an LLM output string based on saliency scores using HTML.
    Tokens with scores at or above the specified threshold are highlighted.
    """
    if not isinstance(llm_output, str):
        return "Invalid LLM output provided."
    
    tokens = llm_output.split()
    
    # Filter saliency scores for the specific output_index
    scores_for_output = token_scores_df[token_scores_df['output_index'] == output_index]
    
    # Create a dictionary for quick lookup of scores
    token_score_map = {row['token']: row['saliency_score'] for _, row in scores_for_output.iterrows()}

    highlighted_tokens = []
    for token in tokens:
        # Get the score for the current token, default to 0 if not found
        score = token_score_map.get(token, 0.0)
        if score >= threshold:
            highlighted_tokens.append(f"<span style=\"background-color: yellow;\">{token}</span>")
        else:
            highlighted_tokens.append(token)
            
    highlighted_text = " ".join(highlighted_tokens)
    return highlighted_text # Return raw HTML string for st.markdown

def run_page2(filtered_df):
    st.header("Page 2: XAI Concepts & Saliency Map")

    st.markdown(r"""
    ## 3.11 Interpretability vs. Transparency
    -   **Interpretability**: understanding why the model produced a specific output for a given input; focuses on input–output relationships.
    -   **Transparency**: understanding how the model works internally (architecture, training data, parameters). For large LLMs, this is often infeasible in practice.

    Business impact: When full transparency is not possible, strong interpretability (clear rationales, salient inputs, counterfactuals) supports audits, incident response, and policy compliance without exposing proprietary internals. In safety monitoring, interpretability is crucial for quickly identifying the root causes of unsafe AI behaviors.
    """)

    st.markdown(r"""
    ## 3.12 Introduction to XAI Techniques (Saliency and Counterfactuals)

    -   **Saliency Maps**: Highlight which input tokens most influenced an output. Conceptually, token importance can be related to local sensitivity of the output with respect to an input token:
        $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
        Higher values indicate stronger influence on the model’s decision, helping reviewers quickly locate decisive words. For safety monitoring, saliency can pinpoint critical parts of an LLM's response that indicate risk.

    -   **Counterfactual Explanations**: Show how minimal changes to an input could change the model’s output. Formally, we seek a small perturbation $\Delta X$ such that:
        $$ \text{Model}(X + \Delta X) = Y' \ne Y, \quad \text{with minimal } ||\Delta X|| $$
        This supports “what-if” analysis, recourse options, and fairness assessments. Counterfactuals are explored further in Page 3.

    Business value: Saliency sharpens local interpretability for specific outputs; counterfactuals reveal levers that alter outcomes—both are essential for audits, compliance, and user trust, especially in safety-critical agentic systems.
    """)

    st.markdown(r"""
    ## 3.7 Generating Synthetic Saliency Data — Context & Business Value
    We simulate token-level saliency scores to conceptually indicate which words in an LLM output were most influential. While these scores are synthetic (random), they let us:
    -   Demonstrate how token attribution can be visualized and audited.
    -   Prototype UI/UX for explanation overlays without accessing model internals.
    -   Train reviewers on how to interpret local explanations.

    Approach: Split each output on whitespace to form tokens and assign each token a random score in $[0,1]$. Higher scores indicate higher conceptual importance.
    """)

    if not filtered_df.empty:
        with st.spinner("Generating synthetic saliency data..."):
            # Generate saliency data based on the full (unfiltered) df_llm_data to maintain consistent indexing
            if 'df_llm_data' in st.session_state and not st.session_state['df_llm_data'].empty:
                st.session_state['df_saliency'] = generate_saliency_data(st.session_state['df_llm_data']['llm_output'].reset_index(drop=True))
            else:
                st.warning("Please generate LLM interaction data in Page 1 first.")
                return

        st.markdown("### Initial Inspection of Saliency Data")
        st.write(f"Shape of saliency data: {st.session_state['df_saliency'].shape}")
        st.dataframe(st.session_state['df_saliency'].head(10))
        
        st.markdown(r"""
        ## 3.8 Interpretation: What the Saliency Table Shows
        -   Each row corresponds to a token from one of the LLM outputs and its synthetic saliency score in $[0,1]$.
        -   Higher scores suggest a stronger conceptual contribution of that token to the output.
        -   We will use these scores to visually highlight important tokens in a sample output, mimicking a saliency map overlay often used in review tools for safety and compliance.
        """)

        st.markdown("---")
        st.markdown(r"""
        ## 3.13 Applying XAI Technique: Saliency Map Simulation — Context & Business Value
        Saliency maps help reviewers quickly identify which words in an output likely drove the model’s behavior. This accelerates root-cause analysis in incident reviews and supports explainability audits without revealing proprietary internals, which is critical for safety monitoring of agentic systems.

        Formula (conceptual sensitivity):
        -   Token importance: $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
        -   In this application, we simulate $S(x_i)$ with random values in $[0,1]$ and highlight tokens with scores above a threshold $\tau$.
        """)

        st.subheader("Saliency Map Visualization")
        
        # Get available indices from the original dataframe, not filtered_df, for consistent selection
        if not st.session_state['df_llm_data'].empty:
            sample_indices = st.session_state['df_llm_data'].index.tolist()
            if not sample_indices:
                st.warning("No LLM outputs available for saliency mapping.")
                return
            
            # Create a mapping from display index (0-based for user) to actual dataframe index
            display_to_df_index = {i: idx for i, idx in enumerate(sample_indices)}
            
            selected_display_index = st.number_input(
                "Select Output Index for Saliency Map", 
                min_value=0, 
                max_value=len(sample_indices)-1, 
                value=0,
                help="Choose an index from the dataset to visualize its saliency map."
            )
            sample_index_in_df = display_to_df_index[selected_display_index]
            
            saliency_threshold = st.slider(
                r"Saliency Highlight Threshold ($\tau$)", 
                0.0, 1.0, 0.7, 0.05,
                help="Tokens with saliency scores at or above this threshold will be highlighted."
            )

            # Retrieve the original output using the dataframe's index
            sample_output = st.session_state['df_llm_data'].loc[sample_index_in_df, 'llm_output']
            
            if not st.session_state['df_saliency'].empty:
                html_vis = visualize_saliency_map(sample_output, st.session_state['df_saliency'], sample_index_in_df, threshold=saliency_threshold)
                st.markdown(html_vis, unsafe_allow_html=True)
            else:
                st.info("Saliency data is not available. Generate data in Page 1 first.")
        else:
            st.info("Please generate LLM interaction data in Page 1 to simulate saliency maps.")

        st.markdown(r"""
        ## 3.15 Interpreting the Saliency Map
        The highlighted words represent tokens with saliency scores above the chosen threshold ($\tau = 0.7$). In a real system, these would indicate which parts of the text most influenced the model’s output. This helps reviewers, especially in safety monitoring:
        -   Quickly spot decisive phrases related to safety risks.
        -   Compare highlighted rationale with policy or business rules for compliance.
        -   Identify unexpected drivers of model behavior that may indicate vulnerabilities or unsafe tendencies, requiring mitigation or re-training.
        """)
    else:
        st.info("Please go to 'Page 1: Introduction & Data Setup' and generate/upload data to proceed with XAI concepts and saliency mapping.")

