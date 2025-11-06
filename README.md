# QuLab: Agentic AI for Safety Monitoring with Explainable LLMs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/your-repo/main/app.py)

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Agentic AI for Safety Monitoring with Explainable LLMs** is a Streamlit-based educational lab project designed to provide a practical, end-to-end introduction to Explainable AI (XAI) for Large Language Models (LLMs) in the context of safety monitoring.

This application leverages synthetic data and lightweight visualizations to demonstrate core XAI concepts and their practical applications. Users will be guided through understanding interpretability vs. transparency, generating realistic synthetic LLM interaction data, simulating cornerstone XAI techniques like saliency maps and counterfactuals, and analyzing faithfulness and trade-offs over time. The project emphasizes the business value of XAI in reducing operational risk, accelerating root-cause analysis, and improving user trust in complex AI systems.

### Learning Goals:
- Understand key XAI concepts for LLMs, including interpretability vs. transparency, and their relevance for risk, trust, and governance.
- Generate realistic synthetic datasets of LLM interactions to explore XAI signals at scale.
- Simulate and visualize cornerstone XAI techniques: saliency maps and counterfactuals.
- Analyze faithfulness over time and understand the trade-off between explanation quality and model accuracy.
- Apply practical filtering mechanisms (e.g., by explanation verbosity and model confidence) to focus reviews on high-value cases.
- Design testing and validation for adaptive systems and apply explainability frameworks to LLMs.
- Understand the importance of XAI for AI-security threats and implementing defenses in agentic systems.

### Key Formulae:
- **Saliency importance (conceptual)**: $ S(x_i) = \left| \frac{\partial Y}{\partial x_i} \right| $ captures how sensitive the model output $Y$ is to small changes in token $x_i$. High $|\partial Y/\partial x_i|$ implies stronger influence, guiding auditors to critical tokens.
- **Counterfactual minimal change**: Find $X' = X + \Delta X$ such that $\text{Model}(X') = Y' \ne Y$ and $||\Delta X||$ is minimal. This supports "what-if" analysis for decision recourse and fairness reviews.
- **Accuracy (simulated binary)**: $ A = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i] $ Even when simplified, plotting $A$ against explanation quality $Q_{exp}$ surfaces potential trade-offs.

## Features

This application offers the following key features:

*   **Synthetic LLM Interaction Data Generation**: Generate a customizable dataset simulating LLM prompts, outputs, model confidence, accuracy, explanation quality, and faithfulness metrics.
*   **Custom Data Upload**: Option to upload your own LLM interaction data via CSV.
*   **Comprehensive Data Validation & Summary**: Perform basic validation checks and display descriptive statistics for the generated or uploaded dataset.
*   **Interactive Saliency Map Visualization**: Simulate and visualize token-level saliency scores for LLM outputs, highlighting influential words based on a user-defined threshold.
*   **Simulated Counterfactual Explanations**: Demonstrate how minimal changes to an input could lead to a different model output, aiding "what-if" analysis.
*   **Explanation Faithfulness Trend Analysis**: Visualize how explanation faithfulness evolves over time, stratified by XAI technique.
*   **Trade-off Analysis**: Explore the relationship between explanation quality and simulated model accuracy.
*   **XAI Technique Performance Comparison**: Bar chart showing average faithfulness metric across different XAI techniques.
*   **Dynamic Filtering Mechanisms**: Filter data for analysis based on minimum explanation quality score, minimum model confidence, and selected XAI techniques.
*   **Modular Streamlit UI**: A clean, intuitive Streamlit interface with clear navigation between different XAI concepts and analyses.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have the following installed:

*   **Python**: Version 3.8 or higher.
*   **pip**: Python's package installer.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/qu-lab-xai-llm.git
    cd qu-lab-xai-llm
    ```
    (Replace `your-username/qu-lab-xai-llm` with the actual repository path if different).

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory of the project with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  **Ensure your virtual environment is activated** (if you created one).
2.  **Navigate to the project's root directory** in your terminal.
3.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

### Navigating the Application

The application has a sidebar for navigation:

*   **Data Generation & Validation**:
    *   Generate a synthetic dataset of LLM interactions or upload your own CSV.
    *   View initial data inspection, explanation of the synthetic dataset, and perform data validation and summary statistics.
*   **XAI Techniques**:
    *   Explore interactive visualizations of Saliency Maps to understand token importance.
    *   Generate and interpret Counterfactual Explanations to see how minimal input changes affect outputs.
*   **Trend & Trade-off Analysis**:
    *   Analyze explanation faithfulness over time and investigate the trade-off between explanation quality and model accuracy using interactive plots.
    *   Utilize filtering controls to focus on specific data subsets.

## Project Structure

```
qu-lab-xai-llm/
├── app.py                      # Main Streamlit application entry point
├── application_pages/          # Directory containing modular page logic
│   ├── __init__.py             # Makes application_pages a Python package
│   ├── page1.py                # Logic for "Data Generation & Validation"
│   ├── page2.py                # Logic for "XAI Techniques"
│   └── page3.py                # Logic for "Trend & Trade-off Analysis"
├── requirements.txt            # Python dependencies
└── README.md                   # This README file
```

## Technology Stack

*   **Frontend/UI**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Programming Language**: Python 3.8+

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3.  Make your changes and commit them with a descriptive message.
4.  Push your changes to your forked repository.
5.  Open a Pull Request to the `main` branch of the original repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if applicable, otherwise state 'Not Applicable' or 'Proprietary').

## Contact

For any questions or further information, please refer to [QuantUniversity](https://www.quantuniversity.com/).

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)
