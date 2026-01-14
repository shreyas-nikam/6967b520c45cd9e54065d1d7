import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import io
import uuid
import datetime
from source import *

st.set_page_config(page_title="QuLab: Robustness & Functional Validation Under Stress", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Robustness & Functional Validation Under Stress")
st.divider()

# --- Session State Initialization ---

# --- Page Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = '1. Baseline & Configuration'

# --- Model & Baseline State ---
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False

if 'baseline_metrics' not in st.session_state:
    st.session_state.baseline_metrics = {}

if 'X_test' not in st.session_state:
    st.session_state.X_test = None

if 'y_test' not in st.session_state:
    st.session_state.y_test = None

if 'model' not in st.session_state:
    st.session_state.model = None

# --- Scenario Definition State ---
if 'stress_scenarios_list' not in st.session_state:
    st.session_state.stress_scenarios_list = []

if 'acceptance_thresholds' not in st.session_state:
    # Initialize with default thresholds from source.py
    st.session_state.acceptance_thresholds = ACCEPTANCE_THRESHOLDS.copy()

# --- Test Execution State ---
if 'tests_executed' not in st.session_state:
    st.session_state.tests_executed = False

if 'raw_stressed_results' not in st.session_state:
    st.session_state.raw_stressed_results = []

if 'performance_degradation_results' not in st.session_state:
    st.session_state.performance_degradation_results = []

# --- Decision & Export State ---
if 'validation_decision_data' not in st.session_state:
    st.session_state.validation_decision_data = {
        "decision": "N/A",
        "rationale": "",
        "key_failures": [],
        "recommended_actions": [],
        "date": ""
    }

if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = str(uuid.uuid4())

# --- File Paths (from source.py, read-only) ---
st.session_state.STRESS_SCENARIOS_FILE = "stress_scenarios.json"
st.session_state.ROBUSTNESS_RESULTS_FILE = "robustness_results.json"
st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE = "performance_degradation_report.json"
st.session_state.VALIDATION_DECISION_FILE = "validation_decision.md"
st.session_state.EVIDENCE_MANIFEST_FILE = "evidence_manifest.json"

if 'output_files_generated' not in st.session_state:
    st.session_state.output_files_generated = False


# --- Helper Functions ---

def apply_stress_to_data(X_baseline: pd.DataFrame, scenario: dict, random_seed: int) -> pd.DataFrame:
    """Helper to apply stress from a scenario to a DataFrame."""
    X_stressed = X_baseline.copy()
    stress_type = scenario['stress_type']
    parameters = scenario['parameters']

    # Note: Using the original RANDOM_SEED for consistency in plotting.
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


# --- Sidebar Navigation ---
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

# --- Page 1: Baseline & Configuration ---
if st.session_state.page == '1. Baseline & Configuration':
    st.markdown(f"# 1. Baseline & Configuration")
    st.markdown(f"As an AI Reliability Engineer, your first step is to establish a clear baseline for the ML-based market risk forecasting model. This baseline represents the model's performance under normal operating conditions and serves as the reference against which all stressed performance will be measured.")
    st.markdown(f"You will also define the acceptance thresholds that determine what constitutes an 'acceptable' level of performance degradation under stress.")

    st.markdown(r"## Establishing Baseline Metrics")
    st.markdown(f"We'll define **Root Mean Squared Error (RMSE)** for predictive accuracy:")
    st.markdown(r"$$ RMSE = √{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $$")
    st.markdown(r"where $N$ is the number of observations, $y_i$ is the actual risk score, and $\hat{y}_i$ is the predicted risk score. A lower RMSE indicates better predictive accuracy.")

    st.markdown(f"For **Expected Calibration Error (ECE)**, we consider the model's predicted risk scores as probabilities of a high-risk event:")
    st.markdown(r"$$ ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)| $$")
    st.markdown(r"Here, the prediction range is divided into $M$ bins. $|B_m|$ is the number of samples in bin $m$, $N$ is the total samples, $acc(B_m)$ is the accuracy (proportion of actual high risk events) in bin $m$, and $conf(B_m)$ is the average predicted risk score (confidence) in bin $m$. A lower ECE signifies better model calibration.")

    st.markdown(f"For **Subgroup Performance Deltas**, we evaluate RMSE for distinct subgroups (e.g., 'high_cap_stocks' vs. 'mid_cap_stocks') to identify potential biases. A large delta suggests disparate performance.")

    st.markdown(r"## Configurable Acceptance Thresholds")
    st.markdown(f"These thresholds define the maximum allowable degradation or increase for each metric before a scenario is flagged as a 'FAIL'.")
    st.markdown(r"- For RMSE and ECE (where lower is better), the threshold represents the maximum acceptable percentage increase (magnitude of negative degradation). For example, a 10% threshold means an RMSE increase of up to 10% is acceptable.")
    st.markdown(r"- For Subgroup RMSE Delta, the threshold is an absolute value added to the baseline delta. If the stressed delta exceeds `(baseline_delta + threshold)`, it fails.")

    if not st.session_state.model_initialized:
        if st.button("Generate Baseline Model & Metrics"):
            with st.spinner("Training baseline model and calculating metrics..."):
                data = generate_synthetic_financial_data(num_samples=2000, random_seed=RANDOM_SEED)
                X = data.drop('future_market_risk_score', axis=1)
                y = data['future_market_risk_score']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
                model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
                model.fit(X_train, y_train)
                baseline_metrics, _ = evaluate_model_performance(model, X_test, y_test, subgroup_column='subgroup_flag')
                
                st.session_state.model_initialized = True
                st.session_state.baseline_metrics = baseline_metrics
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model = model
                st.success("Baseline model generated and metrics established!")
                st.rerun()

    if st.session_state.model_initialized:
        st.subheader("Baseline Metrics")
        st.dataframe(pd.DataFrame([st.session_state.baseline_metrics]).T, use_container_width=True)

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

# --- Page 2: Scenario Builder ---
elif st.session_state.page == '2. Scenario Builder':
    st.markdown(f"# 2. Scenario Builder")
    st.markdown(f"As an AI Reliability Engineer, you translate abstract technical concerns into concrete, business-relevant market stress scenarios. This involves defining specific stress types, parameterizing their intensity, and crucially, linking them to plausible real-world market events and their expected business impacts.")
    st.markdown(f"This section allows you to craft individual stress scenarios, building a comprehensive \"Market Stress Scenario Handbook\". Each scenario is designed to be repeatable and produce a measurable performance change.")

    if not st.session_state.model_initialized:
        st.warning("Please initialize the baseline model in '1. Baseline & Configuration' before creating scenarios.")
    else:
        st.markdown(f"## Current Stress Scenarios")
        st.markdown(f"Review the scenarios you have already defined:")
        
        if st.session_state.stress_scenarios_list:
            scenarios_df = pd.DataFrame(st.session_state.stress_scenarios_list)
            st.dataframe(scenarios_df[['scenario_id', 'stress_type', 'description', 'real_world_event', 'severity_level']], use_container_width=True)
        else:
            st.info("No scenarios defined yet.")

        if st.button("Clear All Scenarios"):
            st.session_state.stress_scenarios_list = []
            st.info("All scenarios cleared.")
            st.rerun()

        st.markdown(f"## Define New Scenario")
        with st.expander("Add New Stress Scenario", expanded=True):
            stress_type_selection = st.selectbox("Select Stress Type", options=STRESS_TYPES)
            feature_name = st.selectbox("Feature to Stress", options=st.session_state.X_test.columns.tolist() if st.session_state.X_test is not None else [])

            # Parameters based on stress type
            std_dev_multiplier = 0.5
            shift_percentage = 0.20
            missing_percentage = 0.30
            scale_factor = 1.5
            base_value = 0.0

            if stress_type_selection == "NOISE":
                std_dev_multiplier = st.number_input("Standard Deviation Multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            elif stress_type_selection == "FEATURE_SHIFT":
                shift_percentage = st.number_input("Shift Percentage (e.g., 0.20 for 20% increase)", min_value=-0.5, max_value=1.0, value=0.20, step=0.01)
            elif stress_type_selection == "MISSINGNESS":
                missing_percentage = st.number_input("Missing Percentage", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
            elif stress_type_selection == "OUT_OF_DISTRIBUTION":
                scale_factor = st.number_input("Scale Factor (e.g., 1.5 to scale beyond current max)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
                base_value = st.number_input("Base Value (usually max of feature)", value=st.session_state.X_test[feature_name].max() if (st.session_state.X_test is not None and feature_name) else 0.0)

            description = st.text_area("Scenario Description", "Describe the stress and its technical impact.")
            real_world_event = st.text_area("Real-World Market Event", "E.g., Flash Crash, Interest Rate Shock")
            expected_business_impact = st.text_area("Expected Business Impact", "E.g., Increased RMSE, mispricing of products")
            severity_level = st.slider("Severity Level (1-5)", min_value=1, max_value=5, value=3)

            if st.button("Add Scenario"):
                parameters = {}
                if stress_type_selection == "NOISE":
                    parameters = {'feature': feature_name, 'std_dev_multiplier': std_dev_multiplier}
                elif stress_type_selection == "FEATURE_SHIFT":
                    parameters = {'feature': feature_name, 'shift_percentage': shift_percentage}
                elif stress_type_selection == "MISSINGNESS":
                    parameters = {'feature': feature_name, 'missing_percentage': missing_percentage}
                elif stress_type_selection == "OUT_OF_DISTRIBUTION":
                    parameters = {'feature': feature_name, 'scale_factor': scale_factor, 'base_value': base_value}

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

        if st.button("Save Scenarios to JSON"):
            with open(st.session_state.STRESS_SCENARIOS_FILE, 'w') as f:
                json.dump(st.session_state.stress_scenarios_list, f, indent=4)
            st.success(f"Scenarios saved to {st.session_state.STRESS_SCENARIOS_FILE}")

# --- Page 3: Test Execution ---
elif st.session_state.page == '3. Test Execution':
    st.markdown(f"# 3. Test Execution")
    st.markdown(f"With your stress scenarios formally defined, it's time to run the gauntlet. This step involves taking your clean baseline test data, applying the specified perturbations for each scenario, and then re-evaluating your market risk forecasting model on this \"stressed\" data.")
    st.markdown(f"This hands-on activity directly tests the model's resilience under adverse conditions that mimic market turmoil. Deterministic scenario generation and reproducible results are paramount for auditability.")

    if not st.session_state.model_initialized:
         st.warning("Please initialize the baseline model first.")
    elif len(st.session_state.stress_scenarios_list) == 0:
         st.warning("Please define stress scenarios in '2. Scenario Builder'.")
    else:
        st.markdown(f"## Scenarios to be Executed")
        scenarios_df = pd.DataFrame(st.session_state.stress_scenarios_list)
        st.dataframe(scenarios_df, use_container_width=True)

        if st.button("Run All Scenarios", disabled=st.session_state.tests_executed):
            progress_bar = st.progress(0, text="Starting stress tests...")
            all_stressed_results = []
            
            for i, scenario in enumerate(st.session_state.stress_scenarios_list):
                progress_bar.progress((i + 1) / len(st.session_state.stress_scenarios_list), text=f"Running Scenario: {scenario['real_world_event']}")
                # Use RANDOM_SEED + i for per-scenario reproducibility
                stressed_output = execute_stress_scenario(st.session_state.model, scenario, st.session_state.X_test, st.session_state.y_test, 'subgroup_flag', RANDOM_SEED + i)
                all_stressed_results.extend(stressed_output)
            
            with open(st.session_state.ROBUSTNESS_RESULTS_FILE, 'w') as f:
                json.dump(all_stressed_results, f, indent=4)
            
            performance_degradation_results = calculate_degradation_and_status(st.session_state.baseline_metrics, all_stressed_results, st.session_state.acceptance_thresholds)
            
            with open(st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE, 'w') as f:
                json.dump(performance_degradation_results, f, indent=4)
            
            st.session_state.raw_stressed_results = all_stressed_results
            st.session_state.performance_degradation_results = performance_degradation_results
            st.session_state.tests_executed = True
            st.session_state.output_files_generated = True
            
            progress_bar.empty()
            st.success("All stress tests executed successfully!")
            st.rerun()

        if st.session_state.tests_executed:
            st.markdown(f"## Results Summary")
            results_df = pd.DataFrame(st.session_state.performance_degradation_results)
            st.dataframe(results_df[['scenario_id', 'metric_name', 'baseline_value', 'stressed_value', 'degradation_pct', 'status']], use_container_width=True)

# --- Page 4: Results Dashboard ---
elif st.session_state.page == '4. Results Dashboard':
    st.markdown(f"# 4. Results Dashboard")
    st.markdown(f"Raw numbers in tables, while precise, can be challenging for non-technical stakeholders to digest quickly. To effectively communicate the market risk model's vulnerabilities to leadership and risk managers, clear and concise visualizations are essential.")
    st.markdown(f"These charts are crucial in discussions with leadership, enabling them to grasp the relevance and severity of potential model failures at a glance.")

    if not st.session_state.tests_executed:
        st.warning("Please execute tests in '3. Test Execution' to view results.")
    else:
        st.markdown(f"## Performance Degradation Summary")
        st.markdown(f"This table provides a comprehensive overview of the model's performance under each stress scenario, highlighting changes in key metrics and their pass/fail status against defined thresholds.")
        
        st.dataframe(pd.DataFrame(st.session_state.performance_degradation_results), use_container_width=True)

        results_df = pd.DataFrame(st.session_state.performance_degradation_results)

        st.markdown(f"### RMSE Performance & Degradation")
        st.pyplot(plot_metric_comparison(results_df, 'RMSE'))
        st.pyplot(plot_degradation_percentages(results_df, 'RMSE'))

        st.markdown(f"### ECE Performance & Degradation")
        st.pyplot(plot_metric_comparison(results_df, 'ECE'))
        st.pyplot(plot_degradation_percentages(results_df, 'ECE'))

        st.markdown(f"### Subgroup RMSE Delta Performance & Degradation")
        st.pyplot(plot_metric_comparison(results_df, 'Subgroup_RMSE_Delta'))
        st.pyplot(plot_degradation_percentages(results_df, 'Subgroup_RMSE_Delta'))

        st.markdown(f"## Feature Distribution Shifts")
        st.markdown(f"To understand *why* the model's performance degraded, visualizing the change in input feature distributions under stress is crucial. This helps identify scenarios where data shifts significantly, potentially pushing the model outside its trained domain.")

        selected_scenario_for_plot_id = st.selectbox(
            "Select Scenario for Distribution Plot", 
            options=[s['scenario_id'] for s in st.session_state.stress_scenarios_list], 
            format_func=lambda x: [s['real_world_event'] for s in st.session_state.stress_scenarios_list if s['scenario_id'] == x][0]
        )

        selected_scenario_obj = next((s for s in st.session_state.stress_scenarios_list if s['scenario_id'] == selected_scenario_for_plot_id), None)
        
        feature_to_plot = st.selectbox("Select Feature to Visualize", options=st.session_state.X_test.columns.tolist() if st.session_state.X_test is not None else [], key="feature_for_dist_plot")

        if selected_scenario_obj and feature_to_plot:
            X_stressed_for_plot = apply_stress_to_data(st.session_state.X_test, selected_scenario_obj, RANDOM_SEED)
            st.pyplot(plot_feature_distribution_shift(st.session_state.X_test, X_stressed_for_plot, feature_to_plot, selected_scenario_obj['real_world_event']))

# --- Page 5: Decision & Export ---
elif st.session_state.page == '5. Decision & Export':
    st.markdown(f"# 5. Decision & Export")
    st.markdown(f"The culmination of the stress testing process is to formalize the validation outcome and generate comprehensive, audit-ready artifacts. This involves synthesizing all findings, making a clear decision on the model's readiness for production, and creating a traceable record of the entire assessment.")
    st.markdown(f"This step ensures accountability and compliance, providing transparent evidence for regulators and internal auditors.")

    if not st.session_state.tests_executed:
        st.warning("Please execute tests in '3. Test Execution' before making a decision.")
    else:
        if st.button("Generate Validation Decision"):
            generate_validation_decision(st.session_state.performance_degradation_results, st.session_state.acceptance_thresholds, st.session_state.VALIDATION_DECISION_FILE)
            
            # Load file content for display
            if os.path.exists(st.session_state.VALIDATION_DECISION_FILE):
                with open(st.session_state.VALIDATION_DECISION_FILE, 'r') as f:
                    # Parse simplistic markdown for display structure or just read raw
                    # For this implementation, we re-use the function logic or just display the markdown content
                    decision_text = f.read()
                    # We can store the text directly or parse it. The spec says populate validation_decision_data
                    # Since the source.py generates a file, we can read the file to show it.
                    st.session_state.validation_decision_data['decision_text'] = decision_text
            
            st.session_state.output_files_generated = True
            st.success("Validation Decision report generated!")
            st.rerun()

        if 'decision_text' in st.session_state.validation_decision_data:
            st.markdown(st.session_state.validation_decision_data['decision_text'])

        if st.session_state.output_files_generated:
            if st.button("Generate Evidence Manifest"):
                generate_evidence_manifest(
                    st.session_state.current_run_id, 
                    st.session_state.EVIDENCE_MANIFEST_FILE, 
                    [st.session_state.STRESS_SCENARIOS_FILE], 
                    [st.session_state.ROBUSTNESS_RESULTS_FILE, st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE, st.session_state.VALIDATION_DECISION_FILE]
                )
                st.success("Evidence Manifest generated!")
            
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


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
