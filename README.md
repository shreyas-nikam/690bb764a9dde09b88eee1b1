# QuLab: Agentic AI for Safety Monitoring – An Explainable AI Lab

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Agentic AI for Safety Monitoring – An Explainable AI Lab** is a Streamlit application designed to explore the critical role of Explainable AI (XAI) in understanding and monitoring Large Language Models (LLMs) within agentic systems, particularly for safety-critical applications. As AI agents gain more autonomy, ensuring their decisions are transparent, justifiable, and safe becomes paramount.

This application provides a hands-on, end-to-end walkthrough using synthetic data to demonstrate how XAI concepts and techniques can be applied in practice. It aims to equip users with a foundational understanding and practical examples to integrate XAI into their AI safety monitoring strategies.

## Features

This lab project covers a range of features focusing on XAI for LLMs in a safety monitoring context:

*   **Core XAI Concepts**: Distinguish between interpretability and transparency, highlighting their importance for risk management, trust, and AI governance.
*   **Synthetic Data Generation**: Create realistic datasets of LLM interactions and associated XAI signals at scale to simulate real-world scenarios. Includes options for customizing the number of samples or uploading your own CSV data.
*   **XAI Technique Simulation**: Interactive demonstrations of cornerstone techniques:
    *   **Saliency Maps**: Visualize which input tokens most influenced an LLM's output by highlighting "salient" words.
    *   **Counterfactual Explanations**: Explore "what-if" scenarios by simulating minimal changes to an input that would alter the LLM's output.
*   **Trend & Trade-off Analysis**:
    *   Visualize how explanation quality and faithfulness evolve over time.
    *   Analyze the relationship between explanation quality and model accuracy.
    *   Compare average faithfulness across different XAI techniques.
*   **Practical Filtering**: Apply global filters based on XAI metrics (Min Explanation Quality Score) and model confidence (Min Model Confidence) to focus human review efforts on high-value or high-risk cases.
*   **Data Validation & Summary Statistics**: Perform essential data quality checks and display descriptive statistics for generated or uploaded datasets.
*   **User-Friendly Interface**: Built with Streamlit for an interactive and intuitive experience, incorporating color-blind-friendly plots and legible fonts.
*   **Lightweight & Reproducible**: Designed to run on mid-spec laptops quickly, using only open-source libraries, ensuring fast and reproducible execution.

## Getting Started

Follow these instructions to set up and run the QuLab application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download the code):**
    ```bash
    git clone https://github.com/your-username/quLab.git
    cd quLab
    ```
    *(Note: Replace `your-username/quLab` with the actual repository path if it exists, otherwise assume the files are in a local directory named `quLab`)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root directory (`quLab/`) with the following content:

    ```
    streamlit
    pandas
    numpy
    plotly
    ```

    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once installed, you can run the Streamlit application:

1.  **Navigate to the project root directory:**
    ```bash
    cd quLab
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Access the Application:**
    Your default web browser should automatically open the application at `http://localhost:8501`. If not, copy and paste this URL into your browser.

### Basic Usage Instructions:

*   **Sidebar Navigation**: Use the "Navigation" select box in the sidebar to switch between different lab pages (Introduction & Data Setup, XAI Concepts & Saliency Map, Counterfactuals & Trend Analysis).
*   **Data Setup (Page 1)**: Start by generating synthetic data using the slider or upload your own CSV. This dataset will be used across all pages.
*   **Global Filters**: The sidebar also contains "Global Filters" (Min Explanation Quality Score, Min Model Confidence, Filter by XAI Technique). Adjust these to see how they impact the visualizations on Page 2 and Page 3.
*   **Interactive Visualizations**: Interact with plots (zoom, pan, hover) to explore the data.
*   **Read the Narratives**: Each section and visualization is accompanied by a narrative explaining the concepts, business value, and interpretation.

## Project Structure

The project is organized into modular files for clarity and maintainability:

```
quLab/
├── app.py                      # Main Streamlit application file, handles global UI and page routing.
├── application_pages/          # Directory containing individual Streamlit pages.
│   ├── page1.py                # Handles Introduction, Data Setup (generation/upload), and Data Validation.
│   ├── page2.py                # Focuses on XAI Concepts and Saliency Map visualization.
│   └── page3.py                # Demonstrates Counterfactuals and Trend Analysis (Faithfulness, Quality vs. Accuracy).
├── requirements.txt            # Lists Python dependencies.
└── README.md                   # This file.
```

## Technology Stack

*   **Frontend & Application Framework**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Interactive Visualizations**: [Plotly Express](https://plotly.com/python/plotly-express/) (part of Plotly)
*   **Programming Language**: Python

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or fix (`git checkout -b feature/your-feature-name`).
3.  **Make your changes** and commit them with descriptive messages.
4.  **Push** your branch to your forked repository.
5.  **Open a Pull Request** against the `main` branch of this repository, explaining your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
*(Note: A `LICENSE` file would typically be created at the root of the project with the MIT license text.)*

## Contact

For any questions, suggestions, or feedback, please feel free to reach out:

*   **Maintainer**: QuantUniversity Team
*   **Website**: [QuantUniversity](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com
