import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
# Your code starts here
st.markdown("""
This Streamlit application, 'XAI for LLMs', provides an interactive exploration of Explainable AI (XAI) concepts and techniques applied to Large Language Models (LLMs), leveraging synthetic data to simulate real-world scenarios. Users can generate synthetic LLM interaction data, explore core XAI concepts like interpretability vs. transparency, simulate saliency maps and counterfactual explanations, and analyze key visualizations related to explanation quality, faithfulness, and token influence. The application aims to enhance understanding of how to make LLM decisions more transparent and interpretable.
""")

page = st.sidebar.selectbox(label="Navigation", options=["XAI for LLMs"])
if page == "XAI for LLMs":
    from application_pages.xai_for_llms import main
    main()
# Your code ends here