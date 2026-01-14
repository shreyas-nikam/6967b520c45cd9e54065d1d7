id: 6967b520c45cd9e54065d1d7_documentation
summary: Robustness & Functional Validation Under Stress Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Robustness & Functional Validation Under Stress

## 1. Introduction: AI Reliability Engineering in Finance
Duration: 0:10:00

Welcome to this codelab on **QuLab: Robustness & Functional Validation Under Stress**. In the rapidly evolving landscape of quantitative finance, machine learning (ML) models are increasingly deployed for critical tasks like market risk forecasting. However, the reliability and robustness of these models, especially under adverse and unprecedented market conditions, are paramount. This application serves as a comprehensive framework for **AI Reliability Engineering**, enabling developers and quantitative analysts to systematically stress-test their ML models.

### Why is this important?

*   **Financial Stability**: Unforeseen market events can cause models to fail catastrophically, leading to significant financial losses or systemic risk. Robustness testing helps identify these vulnerabilities *before* deployment.
*   **Regulatory Compliance**: Financial institutions are often required to demonstrate the stability and reliability of their models under various stress scenarios (e.g., CCAR, DFAST).
*   **Trust and Explainability**: Understanding how a model behaves under stress builds trust and provides insights into its limitations and areas for improvement.

This codelab will guide you through the functionalities of a Streamlit application designed to facilitate this validation process. You'll learn how to:
1.  **Establish Baselines**: Define normal performance metrics for your model.
2.  **Build Stress Scenarios**: Create realistic and hypothetical market stress conditions.
3.  **Execute Tests**: Apply these stresses to your data and re-evaluate model performance.
4.  **Visualize Results**: Analyze performance degradation and data shifts.
5.  **Formalize Decisions**: Generate audit-ready validation reports and artifacts.

### Core Concepts Explained

Throughout this codelab, we will focus on understanding the application's approach to measuring and reporting model robustness using key metrics:

*   **Root Mean Squared Error (RMSE)**: A measure of the model's predictive accuracy, representing the standard deviation of the residuals (prediction errors). Lower RMSE indicates better fit.
    $$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $$
    where $N$ is the number of observations, $y_i$ is the actual value, and $\hat{y}_i$ is the predicted value.

*   **Expected Calibration Error (ECE)**: Assesses how well a model's predicted probabilities align with the true probabilities. In a financial context, this means how accurately the predicted "risk score" reflects the actual likelihood of a high-risk event. A lower ECE indicates better calibration.
    $$ ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)| $$
    Here, the prediction range is divided into $M$ bins. $|B_m|$ is the number of samples in bin $m$, $N$ is the total samples, $acc(B_m)$ is the accuracy (proportion of actual high risk events) in bin $m$, and $conf(B_m)$ is the average predicted risk score (confidence) in bin $m$.

*   **Subgroup Performance Deltas**: Evaluates if the model performs consistently across different subgroups of data (e.g., small-cap vs. large-cap stocks). Disparate performance indicates potential bias or instability.

### Application Architecture Overview

The Streamlit application provides an interactive user interface to orchestrate the model validation workflow. The core logic for data generation, model training, stress application, metric calculation, and plotting resides in a `source.py` module (which you would typically implement). The application leverages Streamlit's `st.session_state` to maintain the application's state across user interactions and page navigations, ensuring a seamless experience.

Here's a high-level conceptual flow:

```
++
| 1. Baseline & Configuration        |
| - Data Generation                  |
| - Model Training                   |
| - Baseline Metric Calculation      |
| - Acceptance Threshold Definition  |
++
              |
              V
++
| 2. Scenario Builder                |
| - Define Stress Types              |
| - Parameterize Scenarios           |
| - Link to Real-World Events        |
++
              |
              V
++
| 3. Test Execution                  |
| - Apply Stress to Data             |
| - Re-evaluate Model on Stressed Data|
| - Calculate Degradation            |
++
              |
              V
++
| 4. Results Dashboard               |
| - Visualize Metric Degradation     |
| - Plot Feature Distribution Shifts |
++
              |
              V
++
| 5. Decision & Export               |
| - Formalize Validation Decision    |
| - Generate Audit Evidence Manifest |
| - Download All Artifacts (ZIP)     |
++
```

Let's dive into each step of the QuLab application.

## 2. Baseline & Configuration
Duration: 0:15:00

This is the foundational step where you establish the model's normal operating performance and define the acceptable boundaries for performance under stress. As an AI Reliability Engineer, this step ensures you have a solid reference point for all subsequent stress testing.

### Initializing the Model and Baseline Metrics

The application starts by allowing you to generate synthetic financial data, train a baseline model (a `RandomForestRegressor` in this case), and calculate its performance metrics on a test set.

```python
if not st.session_state.model_initialized:
    if st.button("Generate Baseline Model & Metrics"):
        with st.spinner("Training baseline model and calculating metrics..."):
            # Generates synthetic data for market risk forecasting
            data = generate_synthetic_financial_data(num_samples=2000, random_seed=RANDOM_SEED)
            X = data.drop('future_market_risk_score', axis=1)
            y = data['future_market_risk_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
            
            # Trains a RandomForestRegressor model
            model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
            model.fit(X_train, y_train)
            
            # Evaluates baseline performance
            baseline_metrics, _ = evaluate_model_performance(model, X_test, y_test, subgroup_column='subgroup_flag')
            
            # Stores state variables
            st.session_state.model_initialized = True
            st.session_state.baseline_metrics = baseline_metrics
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.model = model
            st.success("Baseline model generated and metrics established!")
            st.rerun()
```
<aside class="positive">
The use of `st.session_state` is crucial here. It ensures that the trained model, test data, and baseline metrics persist across Streamlit reruns and page navigations, avoiding redundant computations.
</aside>

Once the button is clicked, the application performs the following:
*   **Data Generation**: Calls `generate_synthetic_financial_data` (from `source.py`) to create a dataset suitable for market risk prediction.
*   **Train-Test Split**: Divides the data into training and testing sets.
*   **Model Training**: Initializes and trains a `RandomForestRegressor` on the training data.
*   **Baseline Evaluation**: Uses `evaluate_model_performance` (from `source.py`) to calculate the initial RMSE, ECE, and Subgroup RMSE Delta on the `X_test` and `y_test` data.
*   **State Update**: Sets `st.session_state.model_initialized` to `True` and stores the `baseline_metrics`, `X_test`, `y_test`, and `model` object in the session state.

The calculated baseline metrics are then displayed in a DataFrame:
```python
if st.session_state.model_initialized:
    st.subheader("Baseline Metrics")
    st.dataframe(pd.DataFrame([st.session_state.baseline_metrics]).T, use_container_width=True)
```
This allows for a quick review of the model's performance under ideal conditions.

### Configuring Acceptance Thresholds

After establishing the baseline, you define the "red lines" for model performance degradation. These **acceptance thresholds** determine what constitutes an acceptable vs. unacceptable change in performance under stress.

```python
st.subheader("Configure Acceptance Thresholds")
col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.acceptance_thresholds['RMSE'] = st.number_input(
        "Max % Increase in RMSE",
        min_value=0.0, max_value=100.0, 
        value=float(st.session_state.acceptance_thresholds['RMSE']),
        step=1.0
    )
with col2:
    st.session_state.acceptance_thresholds['ECE'] = st.number_input(
        "Max % Increase in ECE",
        min_value=0.0, max_value=100.0,
        value=float(st.session_state.acceptance_thresholds['ECE']),
        step=1.0
    )
with col3:
    st.session_state.acceptance_thresholds['Subgroup_RMSE_Delta'] = st.number_input(
        "Max Absolute Increase in Subgroup RMSE Delta",
        min_value=0.0, max_value=1.0,
        value=float(st.session_state.acceptance_thresholds['Subgroup_RMSE_Delta']),
        step=0.01
    )
```
*   **RMSE & ECE**: For these metrics, where lower values are better, the threshold represents the *maximum allowable percentage increase* from the baseline value. For example, a 10% threshold means if RMSE increases by more than 10%, it's a failure.
*   **Subgroup RMSE Delta**: This is an *absolute value increase* from the baseline delta. If the stressed delta exceeds `(baseline_delta + threshold)`, it fails.

<aside class="negative">
Carefully consider and justify your acceptance thresholds. Setting them too strictly might lead to models constantly failing tests, while setting them too loosely might mask critical vulnerabilities.
</aside>

These thresholds are stored in `st.session_state.acceptance_thresholds` and will be used in later steps to evaluate the pass/fail status of each stress scenario.

## 3. Scenario Builder
Duration: 0:20:00

In this step, you transition from establishing baselines to defining the "stress events" themselves. As an AI Reliability Engineer, you craft specific scenarios that simulate plausible market conditions or data quality issues that could impact your model. Each scenario is designed to be clear, repeatable, and parameterized.

### Reviewing Existing Scenarios

The application first displays any scenarios you have already defined:

```python
if st.session_state.stress_scenarios_list:
    scenarios_df = pd.DataFrame(st.session_state.stress_scenarios_list)
    st.dataframe(scenarios_df[['scenario_id', 'stress_type', 'description', 'real_world_event', 'severity_level']], use_container_width=True)
else:
    st.info("No scenarios defined yet.")

if st.button("Clear All Scenarios"):
    st.session_state.stress_scenarios_list = []
    st.info("All scenarios cleared.")
    st.rerun()
```
You can view the key attributes of each scenario, such as its ID, type, description, the real-world event it simulates, and its severity level. A "Clear All Scenarios" button allows you to reset the list.

### Defining a New Stress Scenario

The core of this page is the interface for creating new scenarios. You select a stress type and the feature to apply it to, then parameterize the intensity of the stress.

```python
st.markdown(f"## Define New Scenario")
with st.expander("Add New Stress Scenario", expanded=True):
    stress_type_selection = st.selectbox("Select Stress Type", options=STRESS_TYPES)
    feature_name = st.selectbox("Feature to Stress", options=st.session_state.X_test.columns.tolist() if st.session_state.X_test is not None else [])

    # Parameters based on stress type
    # ... (code for number_inputs based on stress_type_selection) ...

    description = st.text_area("Scenario Description", "Describe the stress and its technical impact.")
    real_world_event = st.text_area("Real-World Market Event", "E.g., Flash Crash, Interest Rate Shock")
    expected_business_impact = st.text_area("Expected Business Impact", "E.g., Increased RMSE, mispricing of products")
    severity_level = st.slider("Severity Level (1-5)", min_value=1, max_value=5, value=3)

    if st.button("Add Scenario"):
        parameters = {}
        # ... (logic to populate parameters dict based on stress_type_selection) ...
        
        scenario_dict = define_stress_scenario(
            stress_type=stress_type_selection,
            parameters=parameters,
            description=description,
            real_world_event=real_world_event,
            expected_business_impact=expected_business_impact,
            severity_level=severity_level
        )
        st.session_state.stress_scenarios_list.append(scenario_dict)
        st.success("Scenario added successfully!")
        st.rerun()
```
The `STRESS_TYPES` variable (presumably from `source.py`) defines the available stress mechanisms:
*   **NOISE**: Adds random noise to a selected feature, simulating measurement errors or minor data volatility.
*   **FEATURE_SHIFT**: Shifts the values of a feature by a certain percentage, mimicking a general market trend or a change in underlying economic factors.
*   **MISSINGNESS**: Introduces missing values into a feature, simulating data ingestion errors or incomplete market feeds.
*   **OUT_OF_DISTRIBUTION**: Scales a feature beyond its observed distribution, simulating extreme, unprecedented market movements.

For each stress type, relevant parameters are provided (e.g., `std_dev_multiplier` for NOISE, `shift_percentage` for FEATURE_SHIFT).

The `define_stress_scenario` function (from `source.py`) is responsible for structuring these inputs into a standardized dictionary format, including a unique `scenario_id`. This structured data is then appended to `st.session_state.stress_scenarios_list`.

Finally, you can save your defined scenarios to a JSON file for persistence and sharing:

```python
if st.button("Save Scenarios to JSON"):
    with open(st.session_state.STRESS_SCENARIOS_FILE, 'w') as f:
        json.dump(st.session_state.stress_scenarios_list, f, indent=4)
    st.success(f"Scenarios saved to {st.session_state.STRESS_SCENARIOS_FILE}")
```
This exports the list of scenarios to `stress_scenarios.json`.

<aside class="positive">
A well-defined set of stress scenarios is critical. Consider involving domain experts (e.g., traders, risk managers) to ensure the scenarios are realistic and cover relevant market risks.
</aside>

## 4. Test Execution
Duration: 0:25:00

This is where the rubber meets the road. After defining your baseline and stress scenarios, this step automates the process of applying those stresses to your model's test data and re-evaluating its performance. Reproducibility and systematic execution are key here.

### Overview of Scenarios to be Executed

Before running the tests, the application displays a summary of the scenarios that have been defined:

```python
st.markdown(f"## Scenarios to be Executed")
scenarios_df = pd.DataFrame(st.session_state.stress_scenarios_list)
st.dataframe(scenarios_df, use_container_width=True)
```
This provides a final review of the test plan.

### Running All Scenarios

The core functionality of this page is triggered by the "Run All Scenarios" button:

```python
if st.button("Run All Scenarios", disabled=st.session_state.tests_executed):
    progress_bar = st.progress(0, text="Starting stress tests...")
    all_stressed_results = []
    
    for i, scenario in enumerate(st.session_state.stress_scenarios_list):
        progress_bar.progress((i + 1) / len(st.session_state.stress_scenarios_list), text=f"Running Scenario: {scenario['real_world_event']}")
        # Use RANDOM_SEED + i for per-scenario reproducibility
        stressed_output = execute_stress_scenario(st.session_state.model, scenario, st.session_state.X_test, st.session_state.y_test, 'subgroup_flag', RANDOM_SEED + i)
        all_stressed_results.extend(stressed_output)
    
    # Save raw results
    with open(st.session_state.ROBUSTNESS_RESULTS_FILE, 'w') as f:
        json.dump(all_stressed_results, f, indent=4)
    
    # Calculate degradation and status
    performance_degradation_results = calculate_degradation_and_status(st.session_state.baseline_metrics, all_stressed_results, st.session_state.acceptance_thresholds)
    
    # Save degradation report
    with open(st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE, 'w') as f:
        json.dump(performance_degradation_results, f, indent=4)
    
    # Update session state
    st.session_state.raw_stressed_results = all_stressed_results
    st.session_state.performance_degradation_results = performance_degradation_results
    st.session_state.tests_executed = True
    st.session_state.output_files_generated = True
    
    progress_bar.empty()
    st.success("All stress tests executed successfully!")
    st.rerun()
```

For each defined scenario:
1.  **Apply Stress**: The `execute_stress_scenario` function (from `source.py`) takes the baseline `X_test` data and applies the perturbations defined by the current `scenario`. It then runs the `st.session_state.model` on this `X_stressed` data and re-calculates the performance metrics (RMSE, ECE, Subgroup RMSE Delta). A unique `RANDOM_SEED + i` is used to ensure per-scenario reproducibility for any stochastic stress types (like NOISE or MISSINGNESS).
2.  **Collect Results**: The stressed metrics are collected for each scenario.
3.  **Save Raw Results**: All detailed results from the stressed evaluations are saved to `robustness_results.json`.
4.  **Calculate Degradation**: The `calculate_degradation_and_status` function (from `source.py`) compares these stressed metrics against the `st.session_state.baseline_metrics` and the `st.session_state.acceptance_thresholds` to determine the percentage degradation and the pass/fail status for each metric under each scenario.
5.  **Save Degradation Report**: This summarized report is saved to `performance_degradation_report.json`.
6.  **Update Session State**: The results are stored in `st.session_state.raw_stressed_results` and `st.session_state.performance_degradation_results`, and `st.session_state.tests_executed` is set to `True`.

### Results Summary

After execution, a quick summary table of the degradation results is shown:
```python
if st.session_state.tests_executed:
    st.markdown(f"## Results Summary")
    results_df = pd.DataFrame(st.session_state.performance_degradation_results)
    st.dataframe(results_df[['scenario_id', 'metric_name', 'baseline_value', 'stressed_value', 'degradation_pct', 'status']], use_container_width=True)
```
This table gives you an immediate overview of which metrics passed or failed for each scenario.

<aside class="positive">
The use of `st.progress` provides excellent user feedback during lengthy operations, making the application feel more responsive. Storing raw and processed results in JSON files ensures data persistence and auditability.
</aside>

## 5. Results Dashboard
Duration: 0:30:00

Numerical tables are essential for precision, but effective visualization is crucial for communicating complex results to both technical and non-technical stakeholders. This dashboard allows you to visually inspect the model's performance degradation and understand the underlying data shifts caused by the stress scenarios.

### Performance Degradation Summary

The page starts by displaying the detailed performance degradation table, which was generated in the previous step.

```python
st.dataframe(pd.DataFrame(st.session_state.performance_degradation_results), use_container_width=True)
```

### Visualizing Metric Degradation

The application generates a series of plots to visualize the impact of stress on the key metrics:

*   **Metric Comparison**: Bar charts showing baseline vs. stressed values for RMSE, ECE, and Subgroup RMSE Delta across all scenarios.
*   **Degradation Percentages**: Bar charts illustrating the percentage degradation (or absolute increase for Subgroup RMSE Delta) for each metric per scenario, clearly indicating which scenarios caused significant issues.

```python
results_df = pd.DataFrame(st.session_state.performance_degradation_results)

st.markdown(f"### RMSE Performance & Degradation")
st.pyplot(plot_metric_comparison(results_df, 'RMSE'))
st.pyplot(plot_degradation_percentages(results_df, 'RMSE'))

st.markdown(f"### ECE Performance & Degradation")
st.pyplot(plot_metric_comparison(results_df, 'ECE'))
st.pyplot(plot_degradation_percentages(results_df, 'ECE'))

st.markdown(f"### Subgroup RMSE Delta Performance & Degradation")
st.pyplot(plot_metric_comparison(results_df, 'Subgroup_RMSE_Delta'))
st.pyplot(plot_degradation_percentages(results_df, 'Subgroup_RMSE_Delta', is_delta=True)) # Note: `is_delta=True` might be in source.py
```
These plots rely on functions like `plot_metric_comparison` and `plot_degradation_percentages` (from `source.py`), which typically use libraries like Matplotlib or Seaborn to generate static plots.

### Visualizing Feature Distribution Shifts

Understanding *why* a model's performance degraded often requires looking at how the input data itself changed. This section allows you to visualize the distribution shifts of specific features under a chosen stress scenario.

```python
selected_scenario_for_plot_id = st.selectbox(
    "Select Scenario for Distribution Plot", 
    options=[s['scenario_id'] for s in st.session_state.stress_scenarios_list], 
    format_func=lambda x: [s['real_world_event'] for s in st.session_state.stress_scenarios_list if s['scenario_id'] == x][0]
)

selected_scenario_obj = next((s for s in st.session_state.stress_scenarios_list if s['scenario_id'] == selected_scenario_for_plot_id), None)

feature_to_plot = st.selectbox("Select Feature to Visualize", options=st.session_state.X_test.columns.tolist() if st.session_state.X_test is not None else [], key="feature_for_dist_plot")

if selected_scenario_obj and feature_to_plot:
    # Re-applies stress to the original X_test for visualization purposes
    X_stressed_for_plot = apply_stress_to_data(st.session_state.X_test, selected_scenario_obj, RANDOM_SEED)
    st.pyplot(plot_feature_distribution_shift(st.session_state.X_test, X_stressed_for_plot, feature_to_plot, selected_scenario_obj['real_world_event']))
```
Here's how it works:
1.  You select a specific scenario from a dropdown.
2.  You select a feature that was potentially stressed in that scenario.
3.  The `apply_stress_to_data` helper function (defined in `app.py` or `source.py`) is used to re-create the stressed data for the selected scenario and feature.
4.  The `plot_feature_distribution_shift` function (from `source.py`) generates a plot (e.g., a histogram or KDE plot) comparing the distribution of the selected feature in the baseline data versus the stressed data.

<aside class="positive">
Visualizing feature distribution shifts is a powerful diagnostic tool. It helps confirm that the stress was applied as intended and provides insights into *why* the model might have failed (e.g., if the data was pushed far outside its training distribution).
</aside>

## 6. Decision & Export
Duration: 0:10:00

The final stage of the stress testing workflow involves synthesizing all findings into a formal validation decision and packaging all relevant artifacts for auditability and record-keeping. This step is critical for compliance, accountability, and clear communication regarding the model's readiness.

### Generating the Validation Decision

Based on the performance degradation results, the application can automatically generate a high-level validation decision.

```python
if st.button("Generate Validation Decision"):
    generate_validation_decision(st.session_state.performance_degradation_results, st.session_state.acceptance_thresholds, st.session_state.VALIDATION_DECISION_FILE)
    
    # Load file content for display
    if os.path.exists(st.session_state.VALIDATION_DECISION_FILE):
        with open(st.session_state.VALIDATION_DECISION_FILE, 'r') as f:
            decision_text = f.read()
            st.session_state.validation_decision_data['decision_text'] = decision_text
    
    st.session_state.output_files_generated = True
    st.success("Validation Decision report generated!")
    st.rerun()

if 'decision_text' in st.session_state.validation_decision_data:
    st.markdown(st.session_state.validation_decision_data['decision_text'])
```
The `generate_validation_decision` function (from `source.py`) takes the performance results and acceptance thresholds, then formulates a decision (e.g., "Model Passes Validation," "Model Fails Validation - High Risk") and writes it to a Markdown file (`validation_decision.md`). This Markdown content is then loaded and displayed directly in the Streamlit app.

### Generating the Evidence Manifest

To ensure full auditability, an evidence manifest is created, which lists all generated artifacts related to the current validation run.

```python
if st.session_state.output_files_generated:
    if st.button("Generate Evidence Manifest"):
        generate_evidence_manifest(
            st.session_state.current_run_id, 
            st.session_state.EVIDENCE_MANIFEST_FILE, 
            [st.session_state.STRESS_SCENARIOS_FILE], 
            [st.session_state.ROBUSTNESS_RESULTS_FILE, st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE, st.session_state.VALIDATION_DECISION_FILE]
        )
        st.success("Evidence Manifest generated!")
```
The `generate_evidence_manifest` function (from `source.py`) creates a JSON file (`evidence_manifest.json`) detailing the input files (scenarios) and output files (results, decision) for the validation run, typically including timestamps and unique run IDs.

### Downloading All Artifacts

For easy archiving and sharing, all generated reports and data files can be downloaded as a single ZIP archive.

```python
if st.session_state.output_files_generated:
    if st.button("Download All Artifacts as ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            files_to_zip = [
                st.session_state.STRESS_SCENARIOS_FILE,
                st.session_state.ROBUSTNESS_RESULTS_FILE,
                st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE,
                st.session_state.VALIDATION_DECISION_FILE,
                st.session_state.EVIDENCE_MANIFEST_FILE
            ]
            for filepath in files_to_zip:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        data = f.read()
                        zip_file.writestr(os.path.basename(filepath), data)
        
        st.download_button(
            label="Download All Artifacts as ZIP", 
            data=zip_buffer.getvalue(), 
            file_name="validation_artifacts.zip", 
            mime="application/zip"
        )
```
<button>
  [Download All Artifacts as ZIP](data:application/zip;base64,PLACEHOLDER)
</button>
<aside class="positive">
Providing a single ZIP download for all artifacts greatly simplifies the process of auditing, sharing, and archiving the validation results, enhancing the traceability of the entire process.
</aside>

This concludes the QuLab codelab. You now have a comprehensive understanding of how to use this Streamlit application for robustness and functional validation of ML models under stress, crucial for AI Reliability Engineering in quantitative finance.
