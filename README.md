Here's a comprehensive `README.md` file for your Streamlit application lab project, formatted for clarity and professionalism.

---

# QuLab: Robustness & Functional Validation Under Stress

## 🚀 Project Overview

**QuLab** is a specialized Streamlit application designed for AI Reliability Engineers to systematically evaluate the robustness and functional integrity of Machine Learning (ML) models, specifically focusing on market risk forecasting models, under various stress conditions. This lab project provides a structured framework to define, execute, analyze, and report on the resilience of ML models against real-world market stressors.

The application guides users through establishing a performance baseline, defining plausible stress scenarios, executing automated stress tests, visualizing performance degradation, and ultimately generating audit-ready validation reports and evidence manifests. Its core purpose is to ensure ML models maintain acceptable performance even when faced with adverse or unexpected data shifts, thereby enhancing trust and compliance in AI systems.

## ✨ Features

QuLab offers a comprehensive workflow, segmented into intuitive navigation pages:

*   **1. Baseline & Configuration:**
    *   **Model Initialization:** Automatically trains a synthetic market risk forecasting model (Random Forest Regressor) and establishes a performance baseline on clean test data.
    *   **Key Metrics:** Calculates baseline metrics including Root Mean Squared Error (RMSE), Expected Calibration Error (ECE), and Subgroup RMSE Delta (to detect bias across data segments).
    *   **Configurable Acceptance Thresholds:** Allows users to define custom percentage degradation limits for each metric, crucial for determining pass/fail status under stress.

*   **2. Scenario Builder:**
    *   **Stress Scenario Definition:** Create and manage diverse stress scenarios, including:
        *   `NOISE`: Adding random noise to features.
        *   `FEATURE_SHIFT`: Shifting feature distributions (e.g., overall increase/decrease).
        *   `MISSINGNESS`: Introducing missing values in features.
        *   `OUT_OF_DISTRIBUTION`: Scaling feature values beyond historical ranges.
    *   **Real-World Context:** Link technical stress parameters to plausible real-world market events and their expected business impacts.
    *   **Severity Levels:** Assign severity ratings (1-5) to scenarios for prioritization.
    *   **Persistence:** Save and load scenarios to/from a `stress_scenarios.json` file.

*   **3. Test Execution:**
    *   **Automated Stress Testing:** Execute all defined scenarios against the baseline model using a reproducible process.
    *   **Performance Degradation Calculation:** Quantifies the change in model performance (RMSE, ECE, Subgroup Delta) for each stressed scenario compared to the baseline.
    *   **Pass/Fail Status:** Automatically assigns a "PASS" or "FAIL" status based on the configurable acceptance thresholds.
    *   **Output Generation:** Stores raw and aggregated results in JSON files (`robustness_results.json`, `performance_degradation_report.json`).

*   **4. Results Dashboard:**
    *   **Interactive DataFrames:** View detailed performance degradation summaries.
    *   **Metric Comparison Plots:** Visualize baseline vs. stressed metric values across all scenarios.
    *   **Degradation Percentage Plots:** Graphically represent the percentage degradation for each metric, making it easy to spot failing scenarios.
    *   **Feature Distribution Shifts:** Interactively select a scenario and feature to visualize how its distribution changes under stress, aiding in root cause analysis.

*   **5. Decision & Export:**
    *   **Validation Decision Report:** Generates a summary Markdown report (`validation_decision.md`) outlining the overall validation outcome, key failures, rationale, and recommended actions.
    *   **Evidence Manifest:** Creates a JSON manifest (`evidence_manifest.json`) listing all generated output files, ensuring auditability and traceability.
    *   **Artifact Download:** Allows downloading all generated reports and data files as a single ZIP archive, streamlining evidence collection for compliance purposes.

## 🚀 Getting Started

Follow these instructions to set up and run the QuLab application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab.git
    cd quolab
    ```
    *(Replace `your-username/quolab` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    numpy>=1.25.0
    scikit-learn>=1.2.0
    matplotlib>=3.7.0
    seaborn>=0.12.0
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This command will open the QuLab application in your default web browser.

2.  **Navigate through the application workflow:**
    Follow the sidebar navigation from `1. Baseline & Configuration` to `5. Decision & Export`:

    *   **1. Baseline & Configuration:** Click "Generate Baseline Model & Metrics" to initialize the model and see baseline performance. Adjust acceptance thresholds as needed.
    *   **2. Scenario Builder:** Define new stress scenarios by selecting stress types, features, and parameters. Add descriptions, real-world events, and severity levels.
    *   **3. Test Execution:** Once scenarios are defined, click "Run All Scenarios" to execute the stress tests.
    *   **4. Results Dashboard:** Review the performance degradation, pass/fail status, and visualize metric changes and feature distribution shifts.
    *   **5. Decision & Export:** Generate the final validation decision report and the evidence manifest. Use the "Download All Artifacts as ZIP" button to collect all output files.

## 📁 Project Structure

```
quolab/
├── .venv/                         # Python virtual environment (if created)
├── app.py                         # Main Streamlit application script
├── source.py                      # Contains helper functions, constants, ML logic, and plotting functions
├── requirements.txt               # List of Python dependencies
├── README.md                      # This file
├── stress_scenarios.json          # (Generated) Stores defined stress scenarios
├── robustness_results.json        # (Generated) Raw results of stress test execution
├── performance_degradation_report.json # (Generated) Summarized degradation and status
├── validation_decision.md         # (Generated) Markdown report of the validation decision
└── evidence_manifest.json         # (Generated) JSON manifest of all output artifacts
```

### `source.py` Details

The `source.py` file is critical for the application's functionality. It contains:

*   **Constants:** `RANDOM_SEED`, `STRESS_TYPES`, `ACCEPTANCE_THRESHOLDS`.
*   **Data Generation:** `generate_synthetic_financial_data()`.
*   **Model Evaluation:** `evaluate_model_performance()`, `calculate_degradation_and_status()`.
*   **Stress Application:** `apply_noise()`, `apply_shift()`, `apply_missingness()`, `apply_out_of_distribution()`.
*   **Scenario Management:** `define_stress_scenario()`, `execute_stress_scenario()`.
*   **Reporting:** `generate_validation_decision()`, `generate_evidence_manifest()`.
*   **Plotting Functions:** `plot_metric_comparison()`, `plot_degradation_percentages()`, `plot_feature_distribution_shift()`.

## 🛠️ Technology Stack

*   **Application Framework:** [Streamlit](https://streamlit.io/)
*   **Programming Language:** Python 3.8+
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (for `RandomForestRegressor`, `train_test_split`)
*   **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **File Handling:** Standard Python libraries (`json`, `os`, `zipfile`, `io`, `uuid`, `datetime`)

## 🤝 Contributing

We welcome contributions to enhance QuLab! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices, includes appropriate tests (if applicable), and updates the documentation as necessary.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(You would typically create a `LICENSE` file in the root of your project with the MIT license text.)*

## 📧 Contact

For questions, feedback, or collaborations, please reach out:

*   **Organization:** QuantUniversity
*   **GitHub Issues:** [https://github.com/your-username/quolab/issues](https://github.com/your-username/quolab/issues)
*   **Email:** info@quantuniversity.com

---