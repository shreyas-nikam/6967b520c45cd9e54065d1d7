
# Market Stress Scenario Builder & Impact Mapper

## 1. Application Overview

The **Market Stress Scenario Builder & Impact Mapper** is a Streamlit application designed to empower AI Reliability Engineers in financial institutions. Its primary purpose is to rigorously validate the robustness of critical ML-based market risk forecasting models under various simulated market stress conditions. This application provides a structured workflow to define, execute, analyze, and document stress tests, ensuring models are fit for use in volatile environments and meet stringent audit requirements.

The application guides the user through the following high-level story flow, mirroring the workflow of an AI Reliability Engineer:

1.  **Establish Baseline**: Load a pre-trained market risk model and its associated test data. Evaluate and display the model's performance under normal, unstressed conditions to establish a crucial benchmark.
2.  **Define Stress Scenarios**: Create tailored stress scenarios by selecting stress types (e.g., noise, feature shift, missingness), configuring parameters, and linking them to plausible real-world market events and their expected business impacts.
3.  **Execute Stress Tests**: Run the defined scenarios, applying the specified perturbations to the baseline data, and re-evaluating the model's performance on these stressed datasets.
4.  **Analyze Degradation**: Quantify the model's performance degradation for each metric (e.g., RMSE, ECE, Subgroup Delta) relative to the baseline. Compare these degradations against configurable acceptance thresholds and flag any violations.
5.  **Visualize Impact**: Generate interactive charts to visually highlight performance degradation, making complex results accessible to both technical and non-technical stakeholders.
6.  **Document Decisions & Evidence**: Formally record the model's validation decision (Deploy, Restrict, or Redesign) with rationale and recommended actions. Generate a comprehensive evidence manifest to ensure traceability and auditability of all inputs and outputs.

This end-to-end workflow enables AI Reliability Engineers to build a robust model validation framework, identify critical vulnerabilities, and provide transparent, audit-ready documentation to leadership and regulatory bodies.

## 2. Code Requirements

### Import Statement

```python
from source import * # Imports all functions and global variables from source.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import io
import uuid # For generating new scenario IDs and run IDs
import datetime # For timestamps in manifest and decision
```

### `st.session_state` Design

`st.session_state` is used extensively to maintain the application's state across user interactions and page navigations, simulating a multi-page experience within a single `app.py` file.

**Initialization (at the start of `app.py`, before any page rendering):**

```python
# --- Page Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = '1. Baseline & Configuration'

# --- Model & Baseline State ---
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False # True after baseline model is loaded/trained

if 'baseline_metrics' not in st.session_state:
    st.session_state.baseline_metrics = {} # Stores global_baseline_metrics

if 'X_test' not in st.session_state:
    st.session_state.X_test = None # Stores global_X_test

if 'y_test' not in st.session_state:
    st.session_state.y_test = None # Stores global_y_test

if 'model' not in st.session_state:
    st.session_state.model = None # Stores global_model

# --- Scenario Definition State ---
if 'stress_scenarios_list' not in st.session_state:
    st.session_state.stress_scenarios_list = [] # List of scenario dictionaries

if 'acceptance_thresholds' not in st.session_state:
    # Initialize with default thresholds from source.py
    st.session_state.acceptance_thresholds = ACCEPTANCE_THRESHOLDS.copy() # Use .copy() to allow modification without altering global constant

# --- Test Execution State ---
if 'tests_executed' not in st.session_state:
    st.session_state.tests_executed = False # True after all scenarios are run

if 'raw_stressed_results' not in st.session_state:
    st.session_state.raw_stressed_results = [] # List of raw results (scenario_id, metric_name, stressed_value, etc.)

if 'performance_degradation_results' not in st.session_state:
    st.session_state.performance_degradation_results = [] # List of results with baseline, stressed, degradation_pct, status

# --- Decision & Export State ---
if 'validation_decision_data' not in st.session_state:
    st.session_state.validation_decision_data = { # Stores the decision report content
        "decision": "N/A",
        "rationale": "",
        "key_failures": [],
        "recommended_actions": [],
        "date": ""
    }

if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = str(uuid.uuid4()) # Unique ID for the current session's run

# --- File Paths (from source.py, read-only) ---
# These are used consistently for file operations and manifest generation
st.session_state.STRESS_SCENARIOS_FILE = "stress_scenarios.json"
st.session_state.ROBUSTNESS_RESULTS_FILE = "robustness_results.json"
st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE = "performance_degradation_report.json"
st.session_state.VALIDATION_DECISION_FILE = "validation_decision.md"
st.session_state.EVIDENCE_MANIFEST_FILE = "evidence_manifest.json"

if 'output_files_generated' not in st.session_state:
    st.session_state.output_files_generated = False # Flag to enable download button
```

**How Keys are Read and Updated Across Pages:**

*   **`st.session_state.page`**:
    *   **Read**: At the beginning of `app.py`, to conditionally render the content block corresponding to the selected page.
    *   **Updated**: By a `st.sidebar.selectbox` widget, which sets the new page value.
*   **`st.session_state.model_initialized`**:
    *   **Read**: Used to enable/disable buttons and conditionally display content in subsequent sections (e.g., "Scenario Builder" requires a baseline).
    *   **Updated**: Set to `True` after the "Generate Baseline Model & Metrics" button is clicked and the baseline initialization functions complete successfully.
*   **`st.session_state.baseline_metrics`, `st.session_state.X_test`, `st.session_state.y_test`, `st.session_state.model`**:
    *   **Read**:
        *   `baseline_metrics`: Displayed on "Baseline & Configuration" and used by `calculate_degradation_and_status`.
        *   `X_test`, `y_test`, `model`: Passed as arguments to `execute_stress_scenario` and for `plot_feature_distribution_shift`.
    *   **Updated**: All are populated by the `initialize_baseline_model()` function (triggered by a button on "Baseline & Configuration" page).
*   **`st.session_state.stress_scenarios_list`**:
    *   **Read**: Displayed on "Scenario Builder" and "Test Execution" pages. Iterated over as input for `execute_stress_scenario`.
    *   **Updated**: New scenarios are appended to this list when the "Add Scenario" button is clicked on the "Scenario Builder" page, using output from `define_stress_scenario`. A "Clear All Scenarios" button can empty it.
*   **`st.session_state.acceptance_thresholds`**:
    *   **Read**: Displayed on "Baseline & Configuration" page. Passed as argument to `calculate_degradation_and_status` and `generate_validation_decision`.
    *   **Updated**: Modified by `st.number_input` widgets on the "Baseline & Configuration" page.
*   **`st.session_state.tests_executed`**:
    *   **Read**: Controls visibility of content on "Results Dashboard" and "Decision & Export" pages, ensuring they are only accessible after tests have run.
    *   **Updated**: Set to `True` after the "Run All Scenarios" button is clicked and all stress tests complete successfully.
*   **`st.session_state.raw_stressed_results`**:
    *   **Read**: Used as input for `calculate_degradation_and_status`.
    *   **Updated**: Populated by the `execute_all_scenarios()` workflow (which calls `execute_stress_scenario` for each scenario).
*   **`st.session_state.performance_degradation_results`**:
    *   **Read**: Displayed as a table and used for generating plots on the "Results Dashboard" page. Passed as input to `generate_validation_decision`.
    *   **Updated**: Populated by the `execute_all_scenarios()` workflow (which calls `calculate_degradation_and_status`).
*   **`st.session_state.validation_decision_data`**:
    *   **Read**: Displayed on the "Decision & Export" page.
    *   **Updated**: Populated after `generate_validation_decision` is called.
*   **`st.session_state.current_run_id`**:
    *   **Read**: Passed as an argument to `generate_evidence_manifest`.
    *   **Updated**: Initialized once per application session with a new `uuid.uuid4()`.
*   **File Path Variables (`st.session_state.STRESS_SCENARIOS_FILE`, etc.)**:
    *   **Read**: Used by `json.dump`, `json.load`, `generate_validation_decision`, `generate_evidence_manifest`, and for the ZIP file download.
    *   **Updated**: These are global constants from `source.py` and are set once at session state initialization.
*   **`st.session_state.output_files_generated`**:
    *   **Read**: Controls the enabled state of the "Download All Artifacts" button.
    *   **Updated**: Set to `True` after `generate_validation_decision` and `generate_evidence_manifest` have successfully created their respective files.

### UI Interaction and `source.py` Function Invocation

The Streamlit app will feature a sidebar for navigation between distinct conceptual "pages." Each page will contain specific UI elements that interact with functions from `source.py` and manage `st.session_state`.

---

#### Application Layout (High-Level)

```python
# Sidebar for navigation
st.sidebar.header("Navigation")
page_selection = st.sidebar.selectbox(
    "Choose a section:",
    [
        '1. Baseline & Configuration',
        '2. Scenario Builder',
        '3. Test Execution',
        '4. Results Dashboard',
        '5. Decision & Export'
    ]
)
st.session_state.page = page_selection

# Main content area based on st.session_state.page
if st.session_state.page == '1. Baseline & Configuration':
    # Render Baseline & Configuration page
elif st.session_state.page == '2. Scenario Builder':
    # Render Scenario Builder page
# ... and so on for other pages
```

---

#### 1. Baseline & Configuration

**Purpose**: Establish the model's baseline performance and configure global acceptance thresholds.

**Markdown Blocks**:
```python
st.markdown(f"# 1. Baseline & Configuration")
st.markdown(f"As an AI Reliability Engineer, your first step is to establish a clear baseline for the ML-based market risk forecasting model. This baseline represents the model's performance under normal operating conditions and serves as the reference against which all stressed performance will be measured.")
st.markdown(f"You will also define the acceptance thresholds that determine what constitutes an 'acceptable' level of performance degradation under stress.")

st.markdown(r"## Establishing Baseline Metrics")
st.markdown(f"We'll define **Root Mean Squared Error (RMSE)** for predictive accuracy:")
st.markdown(r"$$ RMSE = \sqrt{{\frac{{1}}{{N}} \sum_{{i=1}}^{{N}} (y_i - \hat{{y}}_i)^2}} $$")
st.markdown(r"where $N$ is the number of observations, $y_i$ is the actual risk score, and $\hat{{y}}_i$ is the predicted risk score. A lower RMSE indicates better predictive accuracy.")

st.markdown(f"For **Expected Calibration Error (ECE)**, we consider the model's predicted risk scores as probabilities of a high-risk event:")
st.markdown(r"$$ ECE = \sum_{{m=1}}^{{M}} \frac{{|B_m|}}{{N}} |acc(B_m) - conf(B_m)| $$")
st.markdown(r"Here, the prediction range is divided into $M$ bins. $|B_m|$ is the number of samples in bin $m$, $N$ is the total samples, $acc(B_m)$ is the accuracy (proportion of actual high risk events) in bin $m$, and $conf(B_m)$ is the average predicted risk score (confidence) in bin $m$. A lower ECE signifies better model calibration.")

st.markdown(f"For **Subgroup Performance Deltas**, we evaluate RMSE for distinct subgroups (e.g., 'high_cap_stocks' vs. 'mid_cap_stocks') to identify potential biases. A large delta suggests disparate performance.")

st.markdown(r"## Configurable Acceptance Thresholds")
st.markdown(f"These thresholds define the maximum allowable degradation or increase for each metric before a scenario is flagged as a 'FAIL'.")
st.markdown(r"- For RMSE and ECE (where lower is better), the threshold represents the maximum acceptable percentage increase (magnitude of negative degradation). For example, a 10% threshold means an RMSE increase of up to 10% is acceptable.")
st.markdown(r"- For Subgroup RMSE Delta, the threshold is an absolute value added to the baseline delta. If the stressed delta exceeds `(baseline_delta + threshold)`, it fails.")
```

**UI Interactions and Function Calls**:

*   **"Generate Baseline Model & Metrics" Button**:
    *   **Condition**: `st.session_state.model_initialized == False`.
    *   **Action**: On click.
    *   **Calls (mimicking `source.py` initialization section)**:
        1.  `data = generate_synthetic_financial_data(num_samples=2000, random_seed=RANDOM_SEED)`
        2.  `X = data.drop('future_market_risk_score', axis=1)`
        3.  `y = data['future_market_risk_score']`
        4.  `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)`
        5.  `model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)`
        6.  `model.fit(X_train, y_train)`
        7.  `baseline_metrics, _ = evaluate_model_performance(model, X_test, y_test, subgroup_column='subgroup_flag')`
    *   **Updates `st.session_state`**:
        *   `st.session_state.model_initialized = True`
        *   `st.session_state.baseline_metrics = baseline_metrics`
        *   `st.session_state.X_test = X_test`
        *   `st.session_state.y_test = y_test`
        *   `st.session_state.model = model`
        *   `st.success("Baseline model generated and metrics established!")`
*   **Display Baseline Metrics**:
    *   **Condition**: `st.session_state.model_initialized == True`.
    *   **Read `st.session_state.baseline_metrics`**.
    *   Display using `st.dataframe(pd.DataFrame([st.session_state.baseline_metrics]).T, use_container_width=True)`.
*   **Configurable Acceptance Thresholds (Number Inputs)**:
    *   **Widgets**: `st.number_input` for each metric:
        *   `st.session_state.acceptance_thresholds['RMSE']` (label "Max % Increase in RMSE")
        *   `st.session_state.acceptance_thresholds['ECE']` (label "Max % Increase in ECE")
        *   `st.session_state.acceptance_thresholds['Subgroup_RMSE_Delta']` (label "Max Absolute Increase in Subgroup RMSE Delta")
    *   **Update `st.session_state`**: The `st.number_input` widgets will directly update `st.session_state.acceptance_thresholds` dictionary keys on user input.

---

#### 2. Scenario Builder

**Purpose**: Define and manage stress scenarios.

**Markdown Blocks**:
```python
st.markdown(f"# 2. Scenario Builder")
st.markdown(f"As an AI Reliability Engineer, you translate abstract technical concerns into concrete, business-relevant market stress scenarios. This involves defining specific stress types, parameterizing their intensity, and crucially, linking them to plausible real-world market events and their expected business impacts.")
st.markdown(f"This section allows you to craft individual stress scenarios, building a comprehensive \"Market Stress Scenario Handbook\". Each scenario is designed to be repeatable and produce a measurable performance change.")

st.markdown(f"## Current Stress Scenarios")
st.markdown(f"Review the scenarios you have already defined:")
```

**UI Interactions and Function Calls**:

*   **Condition**: `st.session_state.model_initialized == True`. (Otherwise, prompt user to initialize baseline).
*   **Display Existing Scenarios**:
    *   **Read `st.session_state.stress_scenarios_list`**.
    *   Display in a `st.dataframe` or `st.expander` for each scenario, showing `scenario_id`, `stress_type`, `description`, `real_world_event`, `severity_level`.
*   **"Clear All Scenarios" Button**:
    *   **Action**: On click.
    *   **Updates `st.session_state`**: `st.session_state.stress_scenarios_list = []`.
    *   **Confirmation**: `st.info("All scenarios cleared.")`
*   **Scenario Definition Widgets**:
    *   **Widgets**:
        *   `stress_type_selection = st.selectbox("Select Stress Type", options=STRESS_TYPES)` (Uses `STRESS_TYPES` from `source.py`).
        *   `feature_name = st.selectbox("Feature to Stress", options=st.session_state.X_test.columns.tolist() if st.session_state.X_test is not None else [])`
        *   Conditional input widgets for `parameters` based on `stress_type_selection`:
            *   **NOISE**: `std_dev_multiplier = st.number_input("Standard Deviation Multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)`
            *   **FEATURE_SHIFT**: `shift_percentage = st.number_input("Shift Percentage (e.g., 0.20 for 20% increase)", min_value=-0.5, max_value=1.0, value=0.20, step=0.01)`
            *   **MISSINGNESS**: `missing_percentage = st.number_input("Missing Percentage", min_value=0.0, max_value=1.0, value=0.30, step=0.01)`
            *   **OUT_OF_DISTRIBUTION**:
                *   `scale_factor = st.number_input("Scale Factor (e.g., 1.5 to scale beyond current max)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)`
                *   `base_value = st.number_input("Base Value (usually max of feature)", value=st.session_state.X_test[feature_name].max() if (st.session_state.X_test is not None and feature_name) else 0.0)`
        *   `description = st.text_area("Scenario Description", "Describe the stress and its technical impact.")`
        *   `real_world_event = st.text_area("Real-World Market Event", "E.g., Flash Crash, Interest Rate Shock")`
        *   `expected_business_impact = st.text_area("Expected Business Impact", "E.g., Increased RMSE, mispricing of products")`
        *   `severity_level = st.slider("Severity Level (1-5)", min_value=1, max_value=5, value=3)`
*   **"Add Scenario" Button**:
    *   **Action**: On click.
    *   **Calls**: `scenario_dict = define_stress_scenario(...)` using inputs from widgets. `uuid.uuid4()` is used to generate a unique `scenario_id`.
    *   **Updates `st.session_state`**: Appends `scenario_dict` to `st.session_state.stress_scenarios_list`.
    *   **Confirmation**: `st.success("Scenario added successfully!")`
*   **"Save Scenarios to JSON" Button**:
    *   **Action**: On click.
    *   **Calls**: `with open(st.session_state.STRESS_SCENARIOS_FILE, 'w') as f: json.dump(st.session_state.stress_scenarios_list, f, indent=4)`
    *   **Confirmation**: `st.success(f"Scenarios saved to {st.session_state.STRESS_SCENARIOS_FILE}")`

---

#### 3. Test Execution

**Purpose**: Execute all defined stress scenarios against the model.

**Markdown Blocks**:
```python
st.markdown(f"# 3. Test Execution")
st.markdown(f"With your stress scenarios formally defined, it's time to run the gauntlet. This step involves taking your clean baseline test data, applying the specified perturbations for each scenario, and then re-evaluating your market risk forecasting model on this \"stressed\" data.")
st.markdown(f"This hands-on activity directly tests the model's resilience under adverse conditions that mimic market turmoil. Deterministic scenario generation and reproducible results are paramount for auditability.")

st.markdown(f"## Scenarios to be Executed")
```

**UI Interactions and Function Calls**:

*   **Condition**: `st.session_state.model_initialized == True` and `len(st.session_state.stress_scenarios_list) > 0`. (Otherwise, prompt user to define scenarios or initialize baseline).
*   **Display Defined Scenarios**:
    *   **Read `st.session_state.stress_scenarios_list`**.
    *   Display in a `st.dataframe` for review.
*   **"Run All Scenarios" Button**:
    *   **Condition**: Button is disabled if `st.session_state.tests_executed == True` (to prevent re-running without clearing).
    *   **Action**: On click.
    *   **Progress Indicator**: `st.progress(0, text="Starting stress tests...")`.
    *   **Calls (orchestrating `source.py` functions)**:
        1.  Initialize `all_stressed_results = []`.
        2.  Loop through `st.session_state.stress_scenarios_list` with index `i`:
            *   Update `st.progress` bar.
            *   `stressed_output = execute_stress_scenario(st.session_state.model, scenario, st.session_state.X_test, st.session_state.y_test, 'subgroup_flag', RANDOM_SEED + i)` (Uses `RANDOM_SEED + i` for per-scenario reproducibility, mimicking `source.py` tests).
            *   `all_stressed_results.extend(stressed_output)`
        3.  `with open(st.session_state.ROBUSTNESS_RESULTS_FILE, 'w') as f: json.dump(all_stressed_results, f, indent=4)`
        4.  `performance_degradation_results = calculate_degradation_and_status(st.session_state.baseline_metrics, all_stressed_results, st.session_state.acceptance_thresholds)`
        5.  `with open(st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE, 'w') as f: json.dump(performance_degradation_results, f, indent=4)`
    *   **Updates `st.session_state`**:
        *   `st.session_state.raw_stressed_results = all_stressed_results`
        *   `st.session_state.performance_degradation_results = performance_degradation_results`
        *   `st.session_state.tests_executed = True`
        *   `st.session_state.output_files_generated = True` (as files `ROBUSTNESS_RESULTS_FILE` and `PERFORMANCE_DEGRADATION_REPORT_FILE` are generated).
        *   `st.success("All stress tests executed successfully!")`
*   **Results Summary (after execution)**:
    *   **Condition**: `st.session_state.tests_executed == True`.
    *   **Read `st.session_state.performance_degradation_results`**.
    *   Display a concise `st.dataframe` showing `scenario_id`, `metric_name`, `baseline_value`, `stressed_value`, `degradation_pct`, `status`.

---

#### 4. Results Dashboard

**Purpose**: Visualize performance degradation to highlight vulnerabilities.

**Markdown Blocks**:
```python
st.markdown(f"# 4. Results Dashboard")
st.markdown(f"Raw numbers in tables, while precise, can be challenging for non-technical stakeholders to digest quickly. To effectively communicate the market risk model's vulnerabilities to leadership and risk managers, clear and concise visualizations are essential.")
st.markdown(f"These charts are crucial in discussions with leadership, enabling them to grasp the relevance and severity of potential model failures at a glance.")
st.markdown(f"## Performance Degradation Summary")
st.markdown(f"This table provides a comprehensive overview of the model's performance under each stress scenario, highlighting changes in key metrics and their pass/fail status against defined thresholds.")
```

**UI Interactions and Function Calls**:

*   **Condition**: `st.session_state.tests_executed == True`. (Otherwise, prompt user to run tests).
*   **Display Performance Degradation Table**:
    *   **Read `st.session_state.performance_degradation_results`**.
    *   Convert to `pd.DataFrame` and display using `st.dataframe(..., use_container_width=True)`.
*   **Plotting**:
    *   **Read `st.session_state.performance_degradation_results`**.
    *   **Calls**:
        1.  `results_df = pd.DataFrame(st.session_state.performance_degradation_results)`
        2.  `st.markdown(f"### RMSE Performance & Degradation")`
        3.  `st.pyplot(plot_metric_comparison(results_df, 'RMSE'))`
        4.  `st.pyplot(plot_degradation_percentages(results_df, 'RMSE'))`
        5.  `st.markdown(f"### ECE Performance & Degradation")`
        6.  `st.pyplot(plot_metric_comparison(results_df, 'ECE'))`
        7.  `st.pyplot(plot_degradation_percentages(results_df, 'ECE'))`
        8.  `st.markdown(f"### Subgroup RMSE Delta Performance & Degradation")`
        9.  `st.pyplot(plot_metric_comparison(results_df, 'Subgroup_RMSE_Delta'))`
        10. `st.pyplot(plot_degradation_percentages(results_df, 'Subgroup_RMSE_Delta'))`
*   **Feature Distribution Shift Visualization**:
    *   **Markdown**:
        ```python
        st.markdown(f"## Feature Distribution Shifts")
        st.markdown(f"To understand *why* the model's performance degraded, visualizing the change in input feature distributions under stress is crucial. This helps identify scenarios where data shifts significantly, potentially pushing the model outside its trained domain.")
        ```
    *   **Widgets**:
        *   `selected_scenario_for_plot_id = st.selectbox("Select Scenario for Distribution Plot", options=[s['scenario_id'] for s in st.session_state.stress_scenarios_list], format_func=lambda x: [s['real_world_event'] for s in st.session_state.stress_scenarios_list if s['scenario_id'] == x][0])`
        *   `selected_scenario_obj = next((s for s in st.session_state.stress_scenarios_list if s['scenario_id'] == selected_scenario_for_plot_id), None)`
        *   `feature_to_plot = st.selectbox("Select Feature to Visualize", options=st.session_state.X_test.columns.tolist() if st.session_state.X_test is not None else [], key="feature_for_dist_plot")`
    *   **Action**: On selection of scenario and feature.
    *   **Calls**:
        *   **Local Helper function (`apply_stress_to_data`) within `app.py` to recreate stressed data**:
            ```python
            def apply_stress_to_data(X_baseline: pd.DataFrame, scenario: dict, random_seed: int) -> pd.DataFrame:
                """Helper to apply stress from a scenario to a DataFrame."""
                X_stressed = X_baseline.copy()
                stress_type = scenario['stress_type']
                parameters = scenario['parameters']

                # Note: Using the original RANDOM_SEED for consistency in plotting,
                # even if execute_stress_scenario used RANDOM_SEED + i.
                # This ensures the visual representation is repeatable for a given scenario.
                if stress_type == "NOISE":
                    X_stressed = apply_noise(X_stressed, parameters['feature'], parameters['std_dev_multiplier'], random_seed)
                elif stress_type == "FEATURE_SHIFT":
                    X_stressed = apply_shift(X_stressed, parameters['feature'], parameters['shift_percentage'])
                elif stress_type == "MISSINGNESS":
                    X_stressed = apply_missingness(X_stressed, parameters['feature'], parameters['missing_percentage'], random_seed)
                    # Impute missing values for stressed data as done in execute_stress_scenario
                    imputation_value = X_baseline[parameters['feature']].mean()
                    X_stressed[parameters['feature']] = X_stressed[parameters['feature']].fillna(imputation_value)
                elif stress_type == "OUT_OF_DISTRIBUTION":
                    X_stressed = apply_out_of_distribution(X_stressed, parameters['feature'], parameters['scale_factor'], parameters['base_value'], random_seed)
                return X_stressed
            ```
        *   If `selected_scenario_obj` and `feature_to_plot` are valid:
            `X_stressed_for_plot = apply_stress_to_data(st.session_state.X_test, selected_scenario_obj, RANDOM_SEED)`
            `st.pyplot(plot_feature_distribution_shift(st.session_state.X_test, X_stressed_for_plot, feature_to_plot, selected_scenario_obj['real_world_event']))`

---

#### 5. Decision & Export

**Purpose**: Formalize the validation outcome and generate audit-ready artifacts.

**Markdown Blocks**:
```python
st.markdown(f"# 5. Decision & Export")
st.markdown(f"The culmination of the stress testing process is to formalize the validation outcome and generate comprehensive, audit-ready artifacts. This involves synthesizing all findings, making a clear decision on the model's readiness for production, and creating a traceable record of the entire assessment.")
st.markdown(f"This step ensures accountability and compliance, providing transparent evidence for regulators and internal auditors.")
```

**UI Interactions and Function Calls**:

*   **Condition**: `st.session_state.tests_executed == True`. (Otherwise, prompt user to run tests).
*   **"Generate Validation Decision" Button**:
    *   **Action**: On click.
    *   **Calls**: `generate_validation_decision(st.session_state.performance_degradation_results, st.session_state.acceptance_thresholds, st.session_state.VALIDATION_DECISION_FILE)`
    *   **Updates `st.session_state`**:
        *   Reads the content of the newly generated `st.session_state.VALIDATION_DECISION_FILE` to populate `st.session_state.validation_decision_data` for display.
        *   `st.session_state.output_files_generated = True` (as `VALIDATION_DECISION_FILE` is created).
        *   `st.success("Validation Decision report generated!")`
*   **Display Validation Decision**:
    *   **Condition**: After "Generate Validation Decision" is clicked.
    *   **Read `st.session_state.validation_decision_data`**.
    *   Display using `st.markdown` to render the decision, rationale, key failures, and recommended actions.
*   **"Generate Evidence Manifest" Button**:
    *   **Condition**: `st.session_state.output_files_generated == True`.
    *   **Action**: On click.
    *   **Calls**: `generate_evidence_manifest(st.session_state.current_run_id, st.session_state.EVIDENCE_MANIFEST_FILE, [st.session_state.STRESS_SCENARIOS_FILE], [st.session_state.ROBUSTNESS_RESULTS_FILE, st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE, st.session_state.VALIDATION_DECISION_FILE])`
    *   **Updates `st.session_state`**: `st.session_state.output_files_generated = True` (confirms manifest is also generated).
    *   `st.success("Evidence Manifest generated!")`
*   **"Download All Artifacts as ZIP" Button**:
    *   **Condition**: `st.session_state.output_files_generated == True`.
    *   **Action**: On click.
    *   **Logic (in `app.py`)**:
        1.  Create an in-memory `zip_buffer = io.BytesIO()`.
        2.  Open `zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED)`.
        3.  For each relevant file (`st.session_state.STRESS_SCENARIOS_FILE`, `st.session_state.ROBUSTNESS_RESULTS_FILE`, `st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE`, `st.session_state.VALIDATION_DECISION_FILE`, `st.session_state.EVIDENCE_MANIFEST_FILE`):
            *   Read file content: `with open(filepath, 'rb') as f: data = f.read()`.
            *   Write to zip: `zip_file.writestr(os.path.basename(filepath), data)`.
        4.  Close `zip_file`.
        5.  `st.download_button(label="Download All Artifacts as ZIP", data=zip_buffer.getvalue(), file_name="validation_artifacts.zip", mime="application/zip")`.
```
