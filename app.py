
import streamlit as st
import pandas as pd
import numpy as np
import random
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Reproducibility (will be set once, perhaps in session_state or app initialization)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
## Agentic AI for Safety Monitoring: An Explainable AI Lab

In this lab, we explore the critical role of Explainable AI (XAI) in understanding and monitoring Large Language Models (LLMs) within agentic systems, particularly for safety. As AI agents become more autonomous, ensuring their decisions are transparent, justifiable, and safe is paramount. This application provides a hands-on, end-to-end walkthrough using synthetic data to demonstrate how XAI concepts and techniques can be applied in practice.

We will cover:
- **Core XAI Concepts**: Distinguishing between interpretability and transparency, and their importance for risk management, trust, and AI governance.
- **Synthetic Data Generation**: Creating realistic datasets of LLM interactions and associated XAI signals at scale to simulate real-world scenarios.
- **XAI Technique Simulation**: Interactive demonstrations of cornerstone techniques like saliency maps (to understand token importance) and counterfactual explanations (to explore "what-if" scenarios).
- **Trend & Trade-off Analysis**: Visualizing how explanation quality and faithfulness evolve over time and their relationship with model accuracy.
- **Practical Filtering**: Applying filters based on XAI metrics and model confidence to focus human review efforts on high-value or high-risk cases.
- **AI Security & Defenses**: Understanding how XAI supports identifying and mitigating AI-security threats in agentic systems by providing insights into model behavior.

### Why XAI for Agentic AI Safety Monitoring?
Agentic AI systems, by their nature, can make complex decisions with limited human oversight. This autonomy necessitates robust monitoring to ensure they operate within safety guidelines, ethical boundaries, and regulatory compliance. XAI provides the tools to "look inside" these black-box LLMs, offering insights into their reasoning and potential vulnerabilities.

Key benefits include:
- **Enhanced Trust**: Stakeholders can trust systems they understand.
- **Improved Governance**: Meeting regulatory requirements for explainability.
- **Faster Incident Response**: Quickly pinpointing the cause of unsafe or undesirable behaviors.
- **Bias Detection & Mitigation**: Identifying and addressing potential biases in decision-making.
- **Robustness against Attacks**: Understanding how models might be perturbed by adversarial inputs.

This lab aims to equip you with a foundational understanding and practical examples to integrate XAI into your AI safety monitoring strategies.
""")
st.divider()

# Initialize session state for data if not already present
if 'df_llm_data' not in st.session_state:
    st.session_state['df_llm_data'] = pd.DataFrame()
if 'df_saliency' not in st.session_state:
    st.session_state['df_saliency'] = pd.DataFrame()

# Global Filtering Controls in Sidebar
st.sidebar.markdown("## Global Filters")
if not st.session_state['df_llm_data'].empty:
    min_quality = st.sidebar.slider("Min Explanation Quality Score", 0.0, 1.0, 0.0, 0.01, help="Filter to show only records with explanation quality scores above this value.")
    min_confidence = st.sidebar.slider("Min Model Confidence", 0.0, 1.0, 0.0, 0.01, help="Filter to show only records with model confidence scores above this value.")

    xai_techniques = st.session_state['df_llm_data']['xai_technique'].unique().tolist()
    # Ensure "All" is not an explicit option in multi-select, just default to all selected
    selected_techniques = st.sidebar.multiselect("Filter by XAI Technique", xai_techniques, default=xai_techniques, help="Select which XAI techniques to include in the analysis and visualizations.")

    # Apply global filters
    filtered_df = st.session_state['df_llm_data'][
        (st.session_state['df_llm_data']['explanation_quality_score'] >= min_quality) &
        (st.session_state['df_llm_data']['model_confidence'] >= min_confidence) &
        (st.session_state['df_llm_data']['xai_technique'].isin(selected_techniques))
    ].copy()
else:
    filtered_df = pd.DataFrame() # Empty if no data yet

# Navigation
page = st.sidebar.selectbox(label="Navigation", options=["Page 1: Introduction & Data Setup", "Page 2: XAI Concepts & Saliency Map", "Page 3: Counterfactuals & Trend Analysis"])

if page == "Page 1: Introduction & Data Setup":
    from application_pages.page1 import run_page1
    run_page1(filtered_df) # Pass filtered_df, though Page 1 primarily generates it
elif page == "Page 2: XAI Concepts & Saliency Map":
    from application_pages.page2 import run_page2
    run_page2(filtered_df)
elif page == "Page 3: Counterfactuals & Trend Analysis":
    from application_pages.page3 import run_page3
    run_page3(filtered_df)
