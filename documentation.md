id: 690bb764a9dde09b88eee1b1_documentation
summary: Agentic AI for Safety Monitoring Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Explainable AI for Safety Monitoring with Streamlit

## 1. Introduction to Agentic AI, XAI, and QuLab Application Overview
Duration: 0:08

Welcome to the QuLab codelab on "Agentic AI for Safety Monitoring"! In this hands-on guide, we will delve into the critical domain of Explainable AI (XAI) for Large Language Models (LLMs), focusing on its practical application in safety monitoring. This application is designed to provide developers with a comprehensive, end-to-end understanding of XAI concepts through a practical Streamlit interface, utilizing synthetic data and intuitive visualizations.

<aside class="positive">
This codelab is designed to be highly practical. By the end, you will not only understand XAI concepts but also see how they can be implemented and visualized in a production-like Streamlit application.
</aside>

### Why is this application important?

As Agentic AI systems become more prevalent, understanding and monitoring their behavior becomes paramount, especially in sensitive domains like safety. This application addresses the challenge of making LLM decisions transparent and interpretable, which is crucial for:
*   **Reducing Operational Risk**: Enabling model audits and accelerating root-cause analysis in incidents.
*   **Improving User Trust**: Providing clear communication and ensuring compliance.
*   **Enhanced Governance**: Supporting policy enforcement and regulatory requirements.

### Key Concepts Explained:

*   **Interpretability vs. Transparency**: Differentiating between understanding *why* a model made a specific decision (interpretability) and understanding *how* the model works internally (transparency).
*   **Synthetic Data Generation**: Learning how to create realistic mock datasets for prototyping and testing XAI workflows without relying on live, potentially expensive, or sensitive LLM inferences.
*   **Saliency Maps**: Visualizing which parts of an LLM's input or output are most influential in its decision-making.
*   **Counterfactual Explanations**: Understanding how minimal changes to an input could alter a model's output, crucial for "what-if" analysis and recourse.
*   **Faithfulness**: Assessing how well an explanation truly reflects the model's internal reasoning.
*   **Trend & Trade-off Analysis**: Monitoring XAI metrics over time and understanding the balance between explanation quality and model accuracy.

### Key Formulae Guiding Our Understanding:

Throughout this codelab, we will conceptually touch upon foundational XAI formulae:

*   **Saliency importance (conceptual)**: $ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $ captures how sensitive the model output $Y$ is to small changes in token $x_i$. High $|\partial Y/\partial x_i|$ implies stronger influence, guiding auditors to critical tokens.
*   **Counterfactual minimal change**: Find $X' = X + \Delta X$ such that $\text{Model}(X') = Y' \ne Y$ and $||\Delta X||$ is minimal. This supports "what-if" analysis for decision recourse and fairness reviews.
*   **Accuracy (simulated binary)**: $ A = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i] $ Even when simplified, plotting $A$ against explanation quality $Q_{exp}$ surfaces potential trade-offs.

By following this codelab, you will gain a robust understanding of these concepts and their practical implications in building safer, more trustworthy Agentic AI systems.

### Learning Goals:

*   Understand key XAI concepts for LLMs, including interpretability vs. transparency, and their relevance for risk, trust, and governance.
*   Generate realistic synthetic datasets of LLM interactions to explore XAI signals at scale.
*   Simulate and visualize cornerstone XAI techniques: saliency maps and counterfactuals.
*   Analyze faithfulness over time and understand the trade-off between explanation quality and model accuracy.
*   Apply practical filtering mechanisms (e.g., by explanation verbosity and model confidence) to focus reviews on high-value cases.
*   Design testing and validation for adaptive systems and apply explainability frameworks to LLMs.
*   Understand the importance of XAI for AI-security threats and implementing defenses in agentic systems.

## 2. Setting Up and Running the Streamlit Application
Duration: 0:05

Before diving into the functionalities, let's get the Streamlit application up and running on your local machine.

### Application Architecture Overview

The application is structured into a main `app.py` file and three modular pages located in the `application_pages` directory.

*   `app.py`: This is the entry point of the application. It sets up the Streamlit page configuration, displays the main title, and acts as a router. Based on the user's selection in the sidebar navigation, it dynamically loads and runs the corresponding page module.
*   `application_pages/page1.py`: Handles data generation (synthetic or uploaded) and initial data validation and summary statistics.
*   `application_pages/page2.py`: Focuses on demonstrating core XAI techniques: saliency map visualization and counterfactual explanation generation.
*   `application_pages/page3.py`: Provides trend and trade-off analysis, including monitoring faithfulness over time and exploring the relationship between explanation quality and model accuracy, along with filtering capabilities.

This modular design helps keep the code organized and makes it easier to extend or modify specific functionalities. The `st.session_state` object is extensively used to persist data (like the generated LLM interaction dataframe) across different pages and reruns of the application, ensuring a seamless user experience.

### Prerequisites

Ensure you have Python installed (version 3.8 or higher is recommended).

### Installation Steps

1.  **Create a project directory**:
    ```bash
    mkdir qulab_xai_app
    cd qulab_xai_app
    ```

2.  **Create the `application_pages` directory**:
    ```bash
    mkdir application_pages
    ```

3.  **Create `app.py`**: Save the provided `app.py` content into a file named `app.py` in your `qulab_xai_app` directory.

4.  **Create `application_pages/page1.py`**: Save the provided `application_pages/page1.py` content into a file named `page1.py` inside the `application_pages` directory.

5.  **Create `application_pages/page2.py`**: Save the provided `application_pages/page2.py` content into a file named `page2.py` inside the `application_pages` directory.

6.  **Create `application_pages/page3.py`**: Save the provided `application_pages/page3.py` content into a file named `page3.py` inside the `application_pages` directory.

7.  **Install necessary Python packages**:
    It's recommended to create a virtual environment first:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    Then, install the dependencies:
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn
    ```

### Running the Application

Once all files are in place and dependencies are installed, run the Streamlit application from your `qulab_xai_app` directory:

```bash
streamlit run app.py
```

This command will open the application in your default web browser. You should see the "QuLab" title and a sidebar with navigation options.

## 3. Data Generation & Validation
Duration: 0:20

This step focuses on the "Data Generation & Validation" page of the application, which serves as the foundation for our XAI explorations. We will generate synthetic LLM interaction data and perform essential data validation.

### Navigating to the Page

In the Streamlit application sidebar, ensure "Data Generation & Validation" is selected in the "Navigation" dropdown.

### Data and Inputs Overview

The application utilizes fully synthetic data to simulate LLM interactions and their associated XAI metrics.

*   **Why Synthetic Data?**: It allows for rapid, reproducible prototyping of XAI concepts and monitoring workflows without the complexities, costs, and data sensitivity of real LLMs. It's an excellent way to design reporting dashboards and reviewer training programs.
*   **Assumptions**: All scores are normalized to $[0, 1]$. Timestamps are uniformly spread to facilitate trend analysis. XAI technique labels are categorical for stratified monitoring.

### 3.1 Introduction to Explainable AI (XAI)
This section of the page reiterates the core definitions of XAI:
*   **Interpretability**: Understanding *why* a specific decision was made.
*   **Transparency**: Understanding *how* the model works internally.

For LLMs, practical interpretability through techniques like saliency and counterfactuals is key when full transparency is unfeasible.

### 3.3 Overview of Synthetic Data Generation & 3.4 Generating Core Synthetic LLM Interaction Data

Here, the application explains the importance and methodology behind generating synthetic data. This data includes `prompt`, `llm_output`, `model_confidence`, `model_accuracy`, `explanation_quality_score`, `faithfulness_metric`, and `xai_technique`.

You have two options for data input:

1.  **Generate Synthetic Data**: This is the default and recommended option for this codelab.
    *   Use the "Number of Synthetic LLM Interactions" slider to select the desired number of data points (e.g., 500).
    *   Click the "Generate Data" button.
    *   Observe the `df_llm_data` being generated and displayed. This data is stored in `st.session_state['df_llm_data']` for use across other pages.

2.  **Upload Custom Data (CSV)**: If you have your own CSV dataset with similar columns, you can upload it here.

Let's look at the core function for generating this data:

```python
# application_pages/page1.py
@st.cache_data
def generate_llm_data(num_samples):
    """Generates a synthetic dataset simulating LLM interactions."""
    # ... (input validation code) ...
    
    data = {
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(num_samples)],
        'prompt': [f"Prompt {i+1}" for i in range(num_samples)],
        'llm_output': [f"Output {i+1}" for i in range(num_samples)],
        'true_label': [random.choice(['A', 'B', 'C']) for _ in range(num_samples)],
        'model_confidence': [random.uniform(0.5, 1.0) for _ in range(num_samples)],
        'model_accuracy': [random.choice([0, 1]) for _ in range(num_samples)],
        'explanation_quality_score': [random.uniform(0, 1) for _ in range(num_samples)],
        'faithfulness_metric': [random.uniform(0, 1) for _ in range(num_samples)],
        'xai_technique': [random.choice(['LIME', 'SHAP', 'GradCAM']) for _ in range(num_samples)],
    }
    return pd.DataFrame(data, columns=columns)
```
<aside class="positive">
The `st.cache_data` decorator ensures that this data generation function runs only once for a given set of inputs, speeding up subsequent reruns of the application. This is a best practice for expensive computations in Streamlit.
</aside>

### 3.6 Explanation of the Synthetic Dataset

After generating the data, the application displays a sample and its `.info()` output.
The dataset columns are:
*   `timestamp`: For trend analysis.
*   `prompt`: LLM input.
*   `llm_output`: LLM response.
*   `true_label`: A synthetic target label.
*   `model_confidence`: Simulated model certainty $[0,1]$.
*   `model_accuracy`: Simulated correctness $\{0,1\}$.
*   `explanation_quality_score`: Proxy for explanation coherence $[0,1]$.
*   `faithfulness_metric`: Conceptual alignment of explanation with model behavior $[0,1]$.
*   `xai_technique`: Type of XAI method (e.g., LIME, SHAP).

### Optional: Quick Sanity Check Metric (Accuracy)

This section provides a simple diagnostic check.
*   Adjust the "Heuristic Confidence Threshold for Accuracy" slider.
*   The application calculates how often a heuristic prediction (confidence $\ge$ threshold) matches the simulated `model_accuracy`. This demonstrates how thresholds can be set for human review workflows.

### 3.7 Generating Synthetic Saliency Data

Saliency maps are crucial for understanding which tokens influenced an LLM's output. This section generates synthetic token-level saliency scores for a small sample of LLM outputs.

*   Click the "Generate Saliency Data" button.
*   Observe the `df_saliency` being generated and displayed. This data is also stored in `st.session_state['df_saliency']`.

The `generate_saliency_data` function splits output into tokens and assigns a random saliency score:

```python
# application_pages/page1.py
@st.cache_data
def generate_saliency_data(llm_outputs):
    """Generates synthetic token-level saliency scores for LLM output strings."""
    data = []
    for idx, output in enumerate(llm_outputs):
        tokens = output.split()
        for token in tokens:
            saliency_score = np.random.rand() # Assigns a random score
            data.append((idx, token, saliency_score))
    return pd.DataFrame(data, columns=['output_index', 'token', 'saliency_score'])
```

### 3.8 Interpretation: What the Saliency Table Shows

The `df_saliency` table lists tokens from sample outputs along with their synthetic saliency scores. Higher scores conceptually indicate greater importance. This raw data will be used in the next step to visualize saliency maps.

### 3.9 Data Validation and Summary Statistics

Finally, the page performs basic validation and provides summary statistics for the main `df_llm_data`.

*   It checks for expected columns, data types, and missing values.
*   It displays descriptive statistics for numeric columns and value counts for categorical columns.

```python
# application_pages/page1.py
def validate_and_summarize_data(dataframe: pd.DataFrame):
    """
    Performs basic validation and summarizes the dataset.
    - Checks presence of expected columns
    - Validates data types for critical numeric fields
    - Checks missing values in critical fields
    - Prints descriptive statistics and categorical distributions (adapted for Streamlit)
    """
    st.subheader("Data Validation & Summary Statistics")
    # ... (column presence, type, missing value checks) ...
    st.markdown("\n**Descriptive statistics for numeric columns:**")
    st.dataframe(dataframe.select_dtypes(include='number').describe().T)
    # ... (value counts for categorical columns) ...
```

<aside class="negative">
If you upload custom data, ensure it contains the expected columns (`timestamp`, `prompt`, `llm_output`, `model_confidence`, `model_accuracy`, `explanation_quality_score`, `faithfulness_metric`, `xai_technique`) to avoid warnings or errors in subsequent pages.
</aside>

## 4. XAI Techniques: Saliency Maps & Counterfactuals
Duration: 0:15

This step explores the core XAI techniques: Saliency Maps and Counterfactual Explanations, using the synthetic data generated in the previous step.

### Navigating to the Page

In the Streamlit application sidebar, select "XAI Techniques" from the "Navigation" dropdown.

### 3.11 Interpretability vs. Transparency

This section re-emphasizes the distinction between interpretability (understanding model decisions) and transparency (understanding model internals). For large LLMs, interpretability techniques become invaluable for audits and compliance when full transparency is not feasible.

### 3.12 Introduction to XAI Techniques (Saliency and Counterfactuals)

This section provides a brief refresher on saliency and counterfactuals, including their conceptual formulae:

*   **Saliency Maps**: Identify influential input tokens.
    $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$
    A higher absolute derivative indicates a stronger influence of token $x_i$ on the model output $Y$.
*   **Counterfactual Explanations**: Show minimal input changes that flip the output.
    $$ \text{Model}(X + \Delta X) = Y' \ne Y, \quad \text{with minimal } ||\Delta X|| $$

### 3.13 Applying XAI Technique: Saliency Map Simulation

Here, we visualize the synthetic saliency scores generated earlier.

*   **Select Output for Saliency Map**: Use the number input to choose an index from the generated LLM interactions. Remember that saliency data was generated for the first 10 outputs in `page1.py`, so indices 0-9 will show meaningful highlights.
*   **Saliency Highlight Threshold $(\tau)$**: Adjust this slider to change the sensitivity of the highlighting. Tokens with a saliency score equal to or above this threshold will be highlighted in yellow.

The `visualize_saliency_map` function takes an LLM output, token scores, and a threshold to produce HTML with highlighted words:

```python
# application_pages/page2.py
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
```

### 3.15 Interpreting the Saliency Map

Observe how changing the threshold impacts the highlighted words. In a real-world scenario, these highlights would quickly guide reviewers to the most influential parts of an LLM's response, aiding in quality control, policy compliance, and debugging.

### 3.16 Applying XAI Technique: Counterfactual Explanation

Counterfactuals help answer "what if" questions, essential for understanding model behavior sensitivity.

*   **Select Record for Counterfactuals**: Choose an index from the dataset.
*   The application will display the original prompt and output, along with a simulated counterfactual prompt and its corresponding output.

The `generate_counterfactual_explanation` function provides a simple simulation:

```python
# application_pages/page2.py
def generate_counterfactual_explanation(original_prompt, original_output, current_model_accuracy):
    """
    Generates a counterfactual explanation by proposing a modified input prompt
    and its resulting output.
    """
    # Simulate a minimal modification to the original prompt
    counterfactual_prompt = original_prompt + " (What if incorrect?)"

    # Simulate generating a counterfactual output based on accuracy
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
```

### 3.18 Interpreting the Counterfactual Explanation

The generated counterfactuals illustrate how a minor change in input could lead to a different model outcome. This is vital for:
*   **Recourse**: Advising users on how to modify their input to achieve a desired outcome.
*   **Fairness**: Identifying if small changes in sensitive attributes lead to disproportionate outcome changes.
*   **Debugging**: Pinpointing levers that influence model decisions.

## 5. Trend & Trade-off Analysis
Duration: 0:15

This final step moves beyond individual explanations to aggregate analysis, focusing on monitoring XAI metrics over time and understanding trade-offs.

### Navigating to the Page

In the Streamlit application sidebar, select "Trend & Trade-off Analysis" from the "Navigation" dropdown.

### 3.19 Core Visual: Faithfulness Metric Over Time

Monitoring faithfulness over time is crucial for detecting concept drift in explanation quality. A decline in faithfulness could indicate that explanations are no longer accurately reflecting the model's behavior, requiring intervention.

*   **Filtering Controls**: This page introduces interactive filters:
    *   **Minimum Explanation Quality Score**: Filter records based on their explanation quality.
    *   **Minimum Model Confidence**: Filter records based on the model's confidence.
    *   **Filter by XAI Technique**: Select specific XAI techniques to include in the analysis.

    Adjust these filters and observe how the plots change. This demonstrates how human reviewers can focus their attention on high-risk or high-value cases.

The `plot_faithfulness_trend` function visualizes the faithfulness metric over time, stratified by XAI technique:

```python
# application_pages/page3.py
def plot_faithfulness_trend(dataframe, x_axis, y_axis, hue_column, title):
    """
    Generates a line plot showing trends of a specified metric over time for different categories.
    """
    # ... (validation and data type conversion) ...

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dataframe, x=x_axis, y=y_axis, hue=hue_column, ax=ax)
    # ... (plot styling and display) ...
    st.pyplot(fig) # Display in Streamlit
    plt.close(fig) # Close figure to free memory
```
The line plot shows the trend of `faithfulness_metric` (y-axis) against `timestamp` (x-axis), with different lines representing different `xai_technique` values. This helps in spotting performance degradation or improvements across different explanation methods.

### Explanation Quality vs. Model Accuracy Relationship

This scatter plot visualizes the relationship between the `explanation_quality_score` and `model_accuracy`. This is a critical trade-off to monitor:
*   Are high-quality explanations consistently associated with accurate model predictions?
*   Are there specific XAI techniques that perform better in terms of explanation quality for accurate predictions?

```python
# application_pages/page3.py
def plot_explanation_accuracy_relationship(dataframe):
    # ... (validation) ...
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dataframe, x='model_accuracy', y='explanation_quality_score', hue='xai_technique', ax=ax, alpha=0.6)
    ax.set_title("Explanation Quality vs. Model Accuracy")
    ax.set_xlabel("Simulated Model Accuracy (0=Incorrect, 1=Correct)")
    ax.set_ylabel("Explanation Quality Score")
    ax.legend(title="XAI Technique")
    st.pyplot(fig)
    plt.close(fig)
```
The plot will show clusters of points, indicating whether, for example, a model's incorrect predictions often come with low-quality explanations, which could be a signal for review.

### Average Faithfulness by XAI Technique

This bar plot provides an aggregate view of the average faithfulness metric for each simulated XAI technique. This allows for a quick comparison of the overall performance of different explanation methods in terms of how well they align with the model's behavior.

```python
# application_pages/page3.py
def plot_average_faithfulness_by_technique(dataframe):
    # ... (validation) ...
    avg_faithfulness = dataframe.groupby('xai_technique', observed=False)['faithfulness_metric'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=avg_faithfulness, x='xai_technique', y='faithfulness_metric', ax=ax)
    ax.set_title("Average Faithfulness by XAI Technique")
    ax.set_xlabel("XAI Technique")
    ax.set_ylabel("Average Faithfulness Metric")
    st.pyplot(fig)
    plt.close(fig)
```
This plot quickly reveals which explanation techniques, on average, are more "faithful" to the model's behavior in your simulated environment.

## 6. Summary and Next Steps
Duration: 0:05

Congratulations! You have successfully completed the QuLab codelab on "Agentic AI for Safety Monitoring" with Explainable AI for LLMs.

### What you've learned:

*   **Core XAI Concepts**: You now have a practical understanding of interpretability, transparency, saliency, and counterfactuals, and their importance for LLMs in safety monitoring.
*   **Synthetic Data for Prototyping**: You've seen how to generate and use synthetic data to simulate LLM interactions and XAI metrics, enabling rapid development of monitoring and governance workflows.
*   **Applying XAI Techniques**: You've interactively explored how saliency maps highlight important tokens and how counterfactuals reveal model sensitivity to input changes.
*   **Monitoring and Analysis**: You've learned to analyze XAI metrics like faithfulness over time, assess trade-offs between explanation quality and model accuracy, and use filtering to focus review efforts.
*   **Streamlit Development**: You've interacted with a Streamlit application, gaining insights into how interactive dashboards can be built to visualize complex AI concepts.

### Next Steps and Further Exploration:

1.  **Integrate with Real LLMs**: Adapt the XAI techniques demonstrated here with actual LLM inferences (e.g., using OpenAI, Hugging Face, or custom models) and real-world explanation libraries (e.g., LIME, SHAP, Captum).
2.  **Advanced XAI Metrics**: Explore more sophisticated XAI evaluation metrics, such as complexity, stability, or causality of explanations.
3.  **Enhanced Visualizations**: Experiment with more interactive visualization libraries (e.g., Plotly, Altair) for dynamic dashboards, allowing deeper data exploration.
4.  **Anomaly Detection**: Implement anomaly detection algorithms on the XAI metrics (e.g., sudden drops in faithfulness or explanation quality) to automatically flag incidents for review.
5.  **Human-in-the-Loop Feedback**: Design mechanisms within the Streamlit app for human reviewers to provide feedback on explanations, which can then be used to improve the XAI models or the underlying LLM.
6.  **Ethical AI Considerations**: Deepen your understanding of how XAI directly supports ethical AI principles like fairness, accountability, and transparency in agentic systems.

Thank you for participating in this codelab! We hope this guide empowers you to build more interpretable, trustworthy, and safe Agentic AI applications.
