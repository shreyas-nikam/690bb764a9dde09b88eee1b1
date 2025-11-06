id: 690bb764a9dde09b88eee1b1_user_guide
summary: Agentic AI for Safety Monitoring User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Agentic AI for Safety Monitoring: An XAI Codelab

## 1. Introduction and Data Preparation
Duration: 08:00

Welcome to the "Agentic AI for Safety Monitoring" codelab! In this guide, we will explore the critical role of Explainable AI (XAI) for Large Language Models (LLMs) in ensuring safety, building trust, and meeting governance requirements for AI systems. This application provides a practical, end-to-end introduction using synthetic data and intuitive visualizations.

<aside class="positive">
<b>Important Context:</b> As AI systems become more autonomous ("agentic"), understanding their decisions is paramount. This codelab will equip you with the knowledge to interpret LLM behaviors, which is vital for monitoring, auditing, and maintaining secure and responsible AI applications.
</aside>

### Learning Goals:
By the end of this codelab, you will be able to:
*   Understand fundamental XAI concepts for LLMs, including the distinction between interpretability and transparency, and their importance for risk management, trust, and AI governance.
*   Generate and work with realistic synthetic datasets that simulate LLM interactions and XAI signals.
*   Explore and visualize cornerstone XAI techniques: saliency maps and counterfactual explanations.
*   Analyze how explanation quality (faithfulness) changes over time and understand the trade-offs with model accuracy.
*   Apply practical filtering mechanisms (e.g., by explanation verbosity and model confidence) to focus on critical cases for human review.
*   Grasp the significance of XAI in addressing AI security threats and developing defenses for agentic systems.

### Key Concepts and Formulas:
The application will illustrate these concepts:

*   **Saliency Importance (conceptual)**: This metric helps us understand which parts of an LLM's input or output were most influential in its decision. Conceptually, it captures how sensitive the model output $Y$ is to small changes in an input token $x_i$. A high value indicates a stronger influence, guiding auditors to critical tokens.
    $$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$

*   **Counterfactual Minimal Change**: This technique helps answer "what if" questions. It aims to find the smallest possible change to the input ($X + \Delta X$) that would cause the model to produce a different output ($Y' \ne Y$). This is invaluable for "what-if" analysis, understanding recourse options, and fairness reviews.
    $$ \text{Find } X' = X + \Delta X \text{ such that } \text{Model}(X') = Y' \ne Y \text{ and } ||\Delta X|| \text{ is minimal.} $$

*   **Accuracy (simulated binary)**: In this context, accuracy is simplified to illustrate how a model's correctness ($A$) might be tracked and compared against other metrics, like explanation quality ($Q_{exp}$).
    $$ A = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i] $$

### Application Navigation:
On the left sidebar, you'll find the navigation menu. Ensure "Data Generation & Validation" is selected to begin.

The application starts on the "Data Generation & Validation" page, providing an executive summary and an overview of the synthetic data we'll be using. This synthetic data allows us to safely and efficiently explore XAI concepts without the complexities or costs of real LLM interactions.

### Generate Synthetic LLM Interaction Data:

1.  **Locate the "Generate Synthetic Data" section.** You'll see an option to choose your data source.
2.  **Select "Generate Synthetic Data"** (this is usually the default).
3.  **Adjust the "Number of Synthetic LLM Interactions" slider.** For this codelab, you can keep the default value of 500 or choose another value. This controls how many simulated LLM interactions will be in our dataset.
4.  **Click the "Generate Data" button.**
    <aside class="positive">
    <b>Tip:</b> Watch for the "Synthetic data generated!" success message. The application will automatically use this generated data for subsequent steps.
    </aside>

### Understanding the Generated Data:
After generation, you'll see an "Initial Data Inspection" section. The table shows the first few rows of our synthetic dataset. This dataset mimics real-world telemetry and includes fields like:

*   `timestamp`: When the interaction occurred.
*   `prompt`: The input given to the LLM.
*   `llm_output`: The model's response.
*   `model_confidence`: A score (0-1) indicating the model's certainty.
*   `model_accuracy`: A binary flag (0 or 1) indicating if the model's output was correct.
*   `explanation_quality_score`: A score (0-1) indicating how good the explanation for the LLM's output is.
*   `faithfulness_metric`: A score (0-1) reflecting how well the explanation aligns with the model's actual behavior.
*   `xai_technique`: The type of XAI method used (e.g., LIME, SHAP, GradCAM).

### Optional: Quick Sanity Check Metric (Accuracy):
This section provides a simple diagnostic check.

1.  **Adjust the "Heuristic Confidence Threshold for Accuracy" slider.** This threshold determines when we *predict* the model's output to be correct based purely on its confidence.
2.  **Observe the "Heuristic vs. simulated accuracy agreement" value.** This shows how often our simple confidence-based prediction matches the simulated `model_accuracy`. A higher value suggests confidence is a reasonable proxy for correctness.

### Generate Synthetic Saliency Data:

1.  **Click the "Generate Saliency Data" button.** This will create synthetic token-level saliency scores for a sample of the LLM outputs.
2.  **Observe the "Initial Saliency Data Inspection" section.** This table shows tokens from the LLM outputs and their corresponding saliency scores. Higher scores conceptually mean that token was more influential. We'll use this in the next step to visualize "saliency maps."

### Data Validation and Summary Statistics:
Finally, the application performs basic data validation and provides summary statistics for the generated data. This helps ensure data quality and gives you a quick overview of numerical distributions and categorical counts. Review the "Descriptive statistics" and "Value counts" sections.

## 2. Exploring XAI Techniques: Saliency Maps & Counterfactuals
Duration: 07:00

Now that we have our synthetic data, let's explore two powerful XAI techniques: Saliency Maps and Counterfactual Explanations.

1.  **Navigate to the "XAI Techniques" page** using the sidebar.

### Interpretability vs. Transparency:
The page starts by reiterating the difference between Interpretability and Transparency. For large LLMs, full transparency (understanding internal workings) is often impossible. Therefore, we rely on interpretability techniques like saliency and counterfactuals to understand *why* a specific decision was made, which is crucial for audits, incident response, and compliance.

### Introduction to XAI Techniques (Saliency and Counterfactuals):
This section provides a conceptual overview of the two techniques we'll be using:

*   **Saliency Maps**: These highlight the most influential input tokens that drove a model's output. Think of it as shining a spotlight on the words that mattered most. The conceptual formula for token importance is $ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $.

*   **Counterfactual Explanations**: These demonstrate how minimal changes to an input could flip the model's output. It answers the question, "What if I had said something slightly different, how would the LLM's response change?" The goal is to find a minimal perturbation $\Delta X$ such that the model output flips: $ \text{Model}(X + \Delta X) = Y' \ne Y $.

### Applying XAI Technique: Saliency Map Simulation:

1.  **Locate the "Saliency Map Visualization" section.**
2.  **Select an "Output for Saliency Map" using the number input.** Try different indices (from 0 to 9, as saliency data was generated for the first 10 outputs) to see various examples.
3.  **Adjust the "Saliency Highlight Threshold $(\tau)$" slider.** Tokens with a saliency score equal to or above this threshold will be highlighted in yellow.
    <aside class="positive">
    <b>Experiment:</b> Try adjusting the threshold up and down. A lower threshold will highlight more words, while a higher threshold will only highlight the most "salient" (important) ones.
    </aside>
4.  **Observe the highlighted LLM output.** The highlighted words are those that our synthetic saliency scores identified as conceptually important to the output.

### Interpreting the Saliency Map:
The highlighted words help reviewers quickly identify decisive phrases, compare the LLM's "rationale" with policy rules, and potentially spot unexpected drivers of behavior that might need further investigation.

### Applying XAI Technique: Counterfactual Explanation:

1.  **Locate the "Counterfactual Explanation" section.**
2.  **Select a "Record for Counterfactuals" using the number input.** You can choose any index from your generated dataset.
3.  **Observe the "Original Prompt," "Original Output," "Counterfactual Prompt," and "Counterfactual Output."** The "Counterfactual Prompt" will show a minimal modification to the original, and the "Counterfactual Output" will demonstrate a plausible different response given that modified prompt.

### Interpreting the Counterfactual Explanation:
This example shows how a small change in the input (the prompt) could lead to a different model outcome. In a real-world scenario, this helps:
*   Users understand what they might need to change to get a desired outcome (recourse).
*   Auditors assess the model's sensitivity to specific terms or conditions, which is crucial for fairness and bias detection.
*   Debugging efforts by contrasting original vs. perturbed inputs and outcomes.

## 3. Trend & Trade-off Analysis
Duration: 05:00

In this final step, we will analyze trends in explanation quality and explore potential trade-offs, which are crucial for long-term monitoring and governance of LLM-powered systems.

1.  **Navigate to the "Trend & Trade-off Analysis" page** using the sidebar.

### Core Visual: Faithfulness Metric Over Time:
This section focuses on monitoring explanation faithfulness, which indicates how well explanations truly reflect the model's behavior. Drift in faithfulness can signal issues requiring investigation.

### Filtering Controls:
Before viewing the plots, you can refine the data used for analysis:

1.  **Adjust the "Minimum Explanation Quality Score" slider.** This filters records to include only explanations deemed to be of a certain quality or higher.
2.  **Adjust the "Minimum Model Confidence" slider.** This focuses the analysis on records where the model was relatively confident in its prediction.
3.  **Select "Filter by XAI Technique" options.** You can choose to include or exclude specific synthetic XAI techniques (LIME, SHAP, GradCAM) to compare their performance or focus on a subset.

<aside class="positive">
<b>Business Value:</b> These filters are critical in production systems. They allow governance teams to prioritize human review on high-risk cases (e.g., low confidence, low explanation quality) or to compare the efficacy of different XAI methods.
</aside>

### Core Visualizations:

Now, let's interpret the plots:

1.  **Faithfulness Metric over Time by XAI Technique:**
    *   **Purpose:** This line chart visualizes how the `faithfulness_metric` (a measure of how well an explanation reflects the model's true behavior) evolves over time, broken down by the `xai_technique` used.
    *   **Interpretation:** In a real scenario, you would look for dips or spikes in faithfulness. A decreasing trend might indicate that your explanations are becoming less reliable as the model or data changes, signaling a need for recalibration or investigation. Different colored lines allow you to compare the stability of various XAI techniques.

2.  **Explanation Quality vs. Model Accuracy:**
    *   **Purpose:** This scatter plot explores the relationship between `explanation_quality_score` and `model_accuracy`. Each point represents an LLM interaction, colored by its XAI technique.
    *   **Interpretation:** This plot helps identify trade-offs. For instance, are highly accurate model outputs consistently associated with high-quality explanations? Or do we see good explanations even when the model is wrong (which could be misleading)? This is vital for understanding when and where explanations are most trustworthy.

3.  **Average Faithfulness by XAI Technique:**
    *   **Purpose:** This bar chart provides an aggregated view of the average `faithfulness_metric` for each `xai_technique`.
    *   **Interpretation:** You can quickly compare which XAI techniques (in our synthetic scenario) tend to provide more faithful explanations on average. This can inform decisions about which explanation techniques to prioritize or investigate further in a production environment.

<aside class="positive">
<b>Final Takeaway:</b> These trend and trade-off analyses are the backbone of AI governance and safety monitoring. They enable continuous oversight, helping organizations detect issues early, build more robust and transparent AI systems, and ultimately foster greater trust in agentic AI.
</aside>

This concludes our codelab on Agentic AI for Safety Monitoring using Explainable AI for LLMs. You've now gained practical insight into generating synthetic data, applying core XAI techniques, and analyzing their trends and trade-offs.
