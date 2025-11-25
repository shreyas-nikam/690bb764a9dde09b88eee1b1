id: 690bb764a9dde09b88eee1b1_user_guide
summary: Agentic AI for Safety Monitoring User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Codelab: Exploring Explainable AI (XAI) for LLMs with the QuLab Streamlit App

## Step 1: Why this app matters and what you’ll learn
Duration: 02:00

Explainable AI (XAI) is critical for building trust in AI systems, especially with Large Language Models (LLMs) whose decisions can be opaque. This app provides a safe, hands-on way to understand how XAI concepts apply to LLM behavior using synthetic data. You will explore interpretability vs transparency, experiment with saliency maps and counterfactual explanations, and analyze how explanation quality relates to model performance.

By the end of this codelab, you will:
- Understand the difference between interpretability (why a decision) and transparency (how the model works internally).
- Explore XAI techniques: saliency maps and counterfactual explanations.
- Read and interpret key visualizations: faithfulness over time, explanation quality vs accuracy, and token influence heatmaps.
- Use interactive controls to filter by explanation verbosity and model confidence, mirroring real-world analysis workflows.

<aside class="positive">
This app uses synthetic data so you can safely explore XAI concepts without needing to run expensive LLMs. It’s ideal for learning and rapid experimentation.
</aside>

<aside class="negative">
Synthetic data is randomized. Do not infer real-world model behavior from the mock trends. Use it to learn concepts and workflows.
</aside>


## Step 2: Launch the app and get oriented
Duration: 01:30

You can run the app locally if you have Streamlit installed:

```console
streamlit run app.py
```

Interface overview:
- Sidebar
  - App logo and title.
  - Navigation drop-down: select “XAI for LLMs”.
  - “Number of synthetic LLM samples” slider (100–2000).
  - “Generate/Update Data” button.
- Main area
  - Conceptual sections (Overview, Core Concepts).
  - Data display and validation.
  - XAI simulations (Saliency Map, Counterfactuals).
  - Visualizations and interactive filters.
  - Conclusion and references.

Tip: On first run, the app usually auto-generates initial data so you can start exploring immediately.


## Step 3: Generate synthetic LLM interactions
Duration: 03:00

Use the sidebar to set “Number of synthetic LLM samples.” Click “Generate/Update Data.” The app will:
- Create a dataset of LLM interactions, including prompts, outputs, and XAI-related metrics.
- Generate token-level saliency scores for a subset of outputs for demonstration.

You’ll see a preview in “4. Data Generation & Inspection.”

What each column represents:
- timestamp: When the interaction occurred.
- prompt: The input text to the LLM.
- llm_output: The model’s generated response.
- true_label: A simple ground-truth class (Positive/Negative/Neutral).
- model_confidence: Model’s self-reported confidence in its prediction, in [0, 1].
- model_accuracy: Whether the model’s prediction matches the ground truth (1 or 0).
- explanation_quality_score: A proxy for explanation quality, often denoted as $Q_{exp}$.
- faithfulness_metric: How well the explanation reflects the model’s true behavior.
- xai_technique: A label for the explanatory method (e.g., Saliency Map, Counterfactual, LIME).

<aside class="positive">
For smoother interaction and fast plotting, try 500–1000 samples.
</aside>


## Step 4: Inspect and validate the data
Duration: 03:00

- Under “4. Data Generation & Inspection,” review the dataset preview to understand the structure.
- Click “Run Data Validation” to check:
  - Presence and types of key columns.
  - Missing values in critical fields.
  - Numerical and categorical summaries.

How to read the output:
- Success messages confirm healthy data.
- Warnings highlight unexpected types or missing fields (useful for sanity checks).
- Numerical summary offers ranges and distributions for metrics like confidence and faithfulness.
- Categorical summaries show class distribution and technique usage.

<aside class="negative">
If you see “DataFrame is empty,” return to the sidebar and click “Generate/Update Data.” If plots or sections appear blank, verify data has been generated.
</aside>


## Step 5: Core concepts: Interpretability vs Transparency
Duration: 02:00

- Interpretability: Understanding why a model made a specific decision.
- Transparency: Understanding how the model works internally (architecture, parameters, training).

For LLMs, full transparency is rarely feasible due to scale and complexity, so interpretability methods are crucial. The app demonstrates two widely used interpretability techniques.

A common formalization for saliency (importance of token $x_i$ for output $Y$):
$$ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $$


## Step 6: Explore saliency maps (Which words mattered?)
Duration: 03:30

Saliency maps highlight tokens deemed influential for the model’s output.

How to use:
- Find “Saliency Map Visualization.”
- Choose “Select sample index for Saliency Map” to pick a data point.
- Adjust “Saliency Highlight Threshold.” Tokens with scores above this value are highlighted.

How to interpret:
- Highlighted words are considered more influential for that output under the synthetic scoring.
- Lower thresholds reveal more tokens; higher thresholds emphasize only the most influential ones.

<aside class="positive">
Start with a threshold near 0.7 and experiment. Compare how changing it affects the highlighted tokens.
</aside>

<aside class="negative">
Saliency here is synthetic and tokenization is simplistic. In real systems, duplicated words, punctuation, and subword tokenization can change how saliency is computed and displayed.
</aside>

<aside class="negative">
Saliency is generated for a subset of samples to stay fast. If nothing highlights for a high index, try selecting a smaller sample index or regenerate data.
</aside>


## Step 7: Explore counterfactual explanations (What if…?)
Duration: 03:00

Counterfactuals answer: “What minimal change to the input would have produced a different output?”

How to use:
- In “Counterfactual Explanation Simulation,” review the “Original Prompt/Output.”
- Compare them with the “Counterfactual Prompt/Output” shown below.

Concept:
- A counterfactual seeks an input $X'$ close to $X$ so the model changes its output from $Y$ to some $Y' \ne Y$:
  $$ \text{Model}(X + \Delta X) = Y', \quad \text{with minimal } \Delta X $$

What to take away:
- Small changes in the prompt may lead to different outcomes.
- Useful for understanding decision boundaries, fairness assessments, and identifying brittle behaviors.

<aside class="positive">
Use counterfactuals to probe “edge cases” and assess how robust model behavior is to small input changes.
</aside>


## Step 8: Visualize faithfulness and explanation–performance relationships
Duration: 04:00

1) Faithfulness over time
- Section: “Faithfulness Metric over Time (Trend Plot)”
- What you see: A line plot by timestamp, colored by XAI technique.
- Goal: Observe consistency over time. In real systems, stable and high faithfulness is desirable.

2) Explanation Quality vs Model Accuracy
- Section: “Explanation Quality Score vs. Model Accuracy (Relationship Plot)”
- What you see: A scatter plot comparing $Q_{exp}$ to accuracy.
- Goal: Spot correlations or trade-offs. Some tasks may show that maximizing accuracy does not always yield the best explanations.

3) Aggregated Influence of Top N Tokens (Heatmap)
- Use “Number of Top Tokens for Heatmap” to select how many tokens to display (e.g., 5–20).
- What you see: Tokens ranked by mean saliency across outputs.
- Goal: Identify generally influential tokens and how their influence aggregates across the dataset.

<aside class="negative">
Remember, all trends here are synthetic. In real data, use these plots to audit stability, detect drifts, and assess which explanation techniques align with model behavior.
</aside>


## Step 9: Interactive parameter simulation (Filters that shape your analysis)
Duration: 03:30

1) Explanation Verbosity Threshold ($V_{exp}$)
- Section: “Explanation Verbosity Threshold ($V_{exp}$)”
- What it does: Filters records using explanation_quality_score as a proxy for verbosity, with $V_{exp} \in [0, 1]$.
- Use cases: Focus on the most detailed or highest-quality explanations.

2) Model Confidence Threshold
- Section: “Model Confidence Threshold”
- What it does: Filters records by model_confidence.
- Use cases: Prioritize high-confidence predictions for safety-critical reviews or compare with low-confidence cases to study failure modes.

<aside class="positive">
Combine a high $V_{exp}$ with a high confidence threshold to study the most reliable, well-explained cases first. Then relax thresholds to explore edge cases.
</aside>

<aside class="negative">
If your filtered dataset is empty, lower the thresholds or regenerate data with more samples.
</aside>


## Step 10: Practical analysis workflows you can try
Duration: 02:30

- Quick audit
  - Generate data at 500–1000 samples.
  - Run validation and scan categorical distributions.
  - Review faithfulness trend for stability signals.

- Explanation deep-dive
  - Choose a sample index and examine its saliency map at different thresholds.
  - Review the counterfactual to understand sensitivity to prompt changes.

- Performance vs explainability check
  - Inspect the Explanation Quality vs Accuracy scatter.
  - Use filters to isolate high-accuracy, high-$Q_{exp}$ regions.

- Token influence triage
  - Use the heatmap to identify consistently influential tokens.
  - Form hypotheses about why certain tokens matter and verify by adjusting sample index or thresholds.


## Step 11: Troubleshooting and FAQs
Duration: 02:30

- Nothing shows in the main area:
  - Click “Generate/Update Data” in the sidebar.
- Saliency map shows no highlights:
  - Lower the threshold, or pick an earlier sample index where saliency was generated.
- Heatmap is empty:
  - Increase “Number of Top Tokens for Heatmap” or regenerate data.
- Plots feel slow:
  - Reduce the number of synthetic samples (e.g., ≤ 500).
- Validation shows warnings:
  - With synthetic data, these can happen due to randomized typing or edge cases; regenerate data to continue.
- Resetting the app:
  - Stop and rerun, or refresh your browser tab. If needed, clear Streamlit’s cache from the menu.


## Step 12: Wrap-up and next steps
Duration: 01:30

You explored how XAI concepts manifest for LLMs in a controlled, synthetic environment. You practiced:
- Distinguishing interpretability from transparency.
- Using saliency maps and counterfactuals to make sense of model behavior.
- Reading faithfulness trends, quality–accuracy relationships, and token influence heatmaps.
- Applying filters for explanation verbosity and confidence to focus analyses.

Next steps:
- Connect to a real LLM and compute actual saliency or explanation metrics.
- Add more techniques (e.g., SHAP, LIME on embeddings or classification proxies).
- Log and export results for audit trails in production.
- Consider ethical and safety implications, especially when explanations inform high-stakes decisions.

References:
- [1] Unit 5: Explainable and Trustworthy AI, Provided Resource Document (interpretability vs transparency, saliency maps, counterfactual explanations, faithfulness metrics, and explainability–performance trade-offs).
