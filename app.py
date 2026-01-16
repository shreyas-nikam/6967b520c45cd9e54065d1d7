import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import io
import uuid
import datetime
import hashlib
from source import *

st.set_page_config(
    page_title="QuLab: Robustness & Functional Validation Under Stress", layout="wide")
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

# --- Create artifacts directory if it doesn't exist ---
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- File Paths (stored in artifacts directory) ---
st.session_state.STRESS_SCENARIOS_FILE = os.path.join(
    ARTIFACTS_DIR, "stress_scenarios.json")
st.session_state.ROBUSTNESS_RESULTS_FILE = os.path.join(
    ARTIFACTS_DIR, "robustness_results.json")
st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE = os.path.join(
    ARTIFACTS_DIR, "performance_degradation_report.json")
st.session_state.VALIDATION_DECISION_FILE = os.path.join(
    ARTIFACTS_DIR, "validation_decision.md")
st.session_state.EVIDENCE_MANIFEST_FILE = os.path.join(
    ARTIFACTS_DIR, "evidence_manifest.json")

if 'output_files_generated' not in st.session_state:
    st.session_state.output_files_generated = False

if 'success_message' not in st.session_state:
    st.session_state.success_message = None


# --- Helper Functions ---

def apply_stress_to_data(X_baseline: pd.DataFrame, scenario: dict, random_seed: int) -> pd.DataFrame:
    """Helper to apply stress from a scenario to a DataFrame."""
    X_stressed = X_baseline.copy()
    stress_type = scenario['stress_type']
    parameters = scenario['parameters']

    # Note: Using the original RANDOM_SEED for consistency in plotting.
    if stress_type == "NOISE":
        X_stressed = apply_noise(
            X_stressed, parameters['feature'], parameters['std_dev_multiplier'], random_seed)
    elif stress_type == "FEATURE_SHIFT":
        X_stressed = apply_shift(
            X_stressed, parameters['feature'], parameters['shift_percentage'])
    elif stress_type == "MISSINGNESS":
        X_stressed = apply_missingness(
            X_stressed, parameters['feature'], parameters['missing_percentage'], random_seed)
        # Impute missing values for stressed data as done in execute_stress_scenario
        imputation_value = X_baseline[parameters['feature']].mean()
        X_stressed[parameters['feature']] = X_stressed[parameters['feature']].fillna(
            imputation_value)
    elif stress_type == "OUT_OF_DISTRIBUTION":
        X_stressed = apply_out_of_distribution(
            X_stressed, parameters['feature'], parameters['scale_factor'], parameters['base_value'], random_seed)
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
    st.markdown(
        f"We'll define **Root Mean Squared Error (RMSE)** for predictive accuracy:")
    st.markdown(
        r"$$ RMSE = √{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $$")
    st.markdown(
        r"where $N$ is the number of observations, $y_i$ is the actual risk score, and $\hat{y}_i$ is the predicted risk score. A lower RMSE indicates better predictive accuracy.")

    st.markdown(f"For **Expected Calibration Error (ECE)**, we consider the model's predicted risk scores as probabilities of a high-risk event:")
    st.markdown(
        r"$$ ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)| $$")
    st.markdown(r"Here, the prediction range is divided into $M$ bins. $|B_m|$ is the number of samples in bin $m$, $N$ is the total samples, $acc(B_m)$ is the accuracy (proportion of actual high risk events) in bin $m$, and $conf(B_m)$ is the average predicted risk score (confidence) in bin $m$. A lower ECE signifies better model calibration.")

    st.markdown(f"For **Subgroup Performance Deltas**, we evaluate RMSE for distinct subgroups (e.g., 'high_cap_stocks' vs. 'mid_cap_stocks') to identify potential biases. A large delta suggests disparate performance.")

    st.markdown(r"## Configurable Acceptance Thresholds")
    st.markdown(f"These thresholds define the maximum allowable degradation or increase for each metric before a scenario is flagged as a 'FAIL'.")
    st.markdown(r"- For RMSE and ECE (where lower is better), the threshold represents the maximum acceptable percentage increase (magnitude of negative degradation). For example, a 10% threshold means an RMSE increase of up to 10% is acceptable.")
    st.markdown(r"- For Subgroup RMSE Delta, the threshold is an absolute value added to the baseline delta. If the stressed delta exceeds `(baseline_delta + threshold)`, it fails.")

    if not st.session_state.model_initialized:
        if st.button("Generate Baseline Model & Metrics"):
            with st.spinner("Training baseline model and calculating metrics..."):
                data = generate_synthetic_financial_data(
                    num_samples=2000, random_seed=RANDOM_SEED)
                X = data.drop('future_market_risk_score', axis=1)
                y = data['future_market_risk_score']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=RANDOM_SEED)
                model = RandomForestRegressor(
                    n_estimators=100, random_state=RANDOM_SEED)
                model.fit(X_train, y_train)
                baseline_metrics, _ = evaluate_model_performance(
                    model, X_test, y_test, subgroup_column='subgroup_flag')

                st.session_state.model_initialized = True
                st.session_state.baseline_metrics = baseline_metrics
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model = model
                st.success("Baseline model generated and metrics established!")
                st.rerun()

    if st.session_state.model_initialized:
        st.subheader("Baseline Metrics")
        st.dataframe(pd.DataFrame(
            [st.session_state.baseline_metrics]).T, use_container_width=True)

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
                min_value=0.0, max_value=100.0,
                value=float(
                    st.session_state.acceptance_thresholds['Subgroup_RMSE_Delta']),
                step=0.01
            )

# --- Page 2: Scenario Builder ---
elif st.session_state.page == '2. Scenario Builder':
    st.markdown(f"# 2. Scenario Builder")
    st.markdown(f"As an AI Reliability Engineer, you translate abstract technical concerns into concrete, business-relevant market stress scenarios. This involves defining specific stress types, parameterizing their intensity, and crucially, linking them to plausible real-world market events and their expected business impacts.")
    st.markdown(f"This section allows you to craft individual stress scenarios, building a comprehensive \"Market Stress Scenario Handbook\". Each scenario is designed to be repeatable and produce a measurable performance change.")

    if not st.session_state.model_initialized:
        st.warning(
            "Please initialize the baseline model in '1. Baseline & Configuration' before creating scenarios.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 View Scenarios", "➕ Add Scenario", "✏️ Edit Scenario", "🗑️ Delete Scenario"])

        # Display success message if present
        if st.session_state.success_message:
            st.success(st.session_state.success_message)

        # --- Tab 1: View Scenarios ---
        with tab1:
            st.markdown("## Current Stress Scenarios")

            # Clear success message when viewing scenarios
            if st.session_state.success_message:
                if st.button("✖ Clear Message", key="clear_view"):
                    st.session_state.success_message = None
                    st.rerun()

            if st.session_state.stress_scenarios_list:
                scenarios_df = pd.DataFrame(
                    st.session_state.stress_scenarios_list)
                st.dataframe(
                    scenarios_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"**Total Scenarios:** {len(st.session_state.stress_scenarios_list)}")

            else:
                st.info(
                    "No scenarios defined yet. Go to 'Add Scenario' tab to create your first scenario.")

        # --- Tab 2: Add Scenario ---
        with tab2:
            st.markdown("## Define New Stress Scenario")

            with st.form("add_scenario_form"):
                stress_type_selection = st.selectbox(
                    "Select Stress Type", options=STRESS_TYPES)
                feature_name = st.selectbox("Feature to Stress", options=st.session_state.X_test.columns.tolist(
                ) if st.session_state.X_test is not None else [])

                # Parameters based on stress type
                std_dev_multiplier = 0.5
                shift_percentage = 0.20
                missing_percentage = 0.30
                scale_factor = 1.5
                base_value = 0.0

                if stress_type_selection == "NOISE":
                    std_dev_multiplier = st.number_input(
                        "Standard Deviation Multiplier", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                elif stress_type_selection == "FEATURE_SHIFT":
                    shift_percentage = st.number_input(
                        "Shift Percentage (e.g., 0.20 for 20% increase)", min_value=-0.5, max_value=1.0, value=0.20, step=0.01)
                elif stress_type_selection == "MISSINGNESS":
                    missing_percentage = st.number_input(
                        "Missing Percentage", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
                elif stress_type_selection == "OUT_OF_DISTRIBUTION":
                    scale_factor = st.number_input(
                        "Scale Factor (e.g., 1.5 to scale beyond current max)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
                    base_value = st.number_input("Base Value (usually max of feature)", value=st.session_state.X_test[feature_name].max(
                    ) if (st.session_state.X_test is not None and feature_name) else 0.0)

                description = st.text_area(
                    "Scenario Description", placeholder="Describe the stress and its technical impact.")
                real_world_event = st.text_area(
                    "Real-World Market Event", placeholder="E.g., Flash Crash, Interest Rate Shock")
                expected_business_impact = st.text_area(
                    "Expected Business Impact", placeholder="E.g., Increased RMSE, mispricing of products")
                severity_level = st.slider(
                    "Severity Level (1-5)", min_value=1, max_value=5, value=3)

                submitted = st.form_submit_button(
                    "➕ Add Scenario", use_container_width=True)

                if submitted:
                    parameters = {}
                    if stress_type_selection == "NOISE":
                        parameters = {'feature': feature_name,
                                      'std_dev_multiplier': std_dev_multiplier}
                    elif stress_type_selection == "FEATURE_SHIFT":
                        parameters = {'feature': feature_name,
                                      'shift_percentage': shift_percentage}
                    elif stress_type_selection == "MISSINGNESS":
                        parameters = {'feature': feature_name,
                                      'missing_percentage': missing_percentage}
                    elif stress_type_selection == "OUT_OF_DISTRIBUTION":
                        parameters = {
                            'feature': feature_name, 'scale_factor': scale_factor, 'base_value': base_value}

                    scenario_dict = define_stress_scenario(
                        stress_type=stress_type_selection,
                        parameters=parameters,
                        description=description,
                        real_world_event=real_world_event,
                        expected_business_impact=expected_business_impact,
                        severity_level=severity_level
                    )
                    st.session_state.stress_scenarios_list.append(
                        scenario_dict)
                    st.session_state.success_message = f"✅ Scenario '{real_world_event}' added successfully! Total scenarios: {len(st.session_state.stress_scenarios_list)}"
                    st.rerun()

        # --- Tab 3: Edit Scenario ---
        with tab3:
            st.markdown("## Edit Existing Scenario")

            if st.session_state.stress_scenarios_list:
                scenario_to_edit = st.selectbox(
                    "Select Scenario to Edit",
                    options=range(len(st.session_state.stress_scenarios_list)),
                    format_func=lambda x: f"{st.session_state.stress_scenarios_list[x]['real_world_event']} ({st.session_state.stress_scenarios_list[x]['stress_type']})"
                )

                current_scenario = st.session_state.stress_scenarios_list[scenario_to_edit]

                with st.form("edit_scenario_form"):
                    st.info(f"Editing: {current_scenario['real_world_event']}")

                    stress_type_edit = st.selectbox(
                        "Stress Type", options=STRESS_TYPES,
                        index=STRESS_TYPES.index(current_scenario['stress_type']))

                    feature_name_edit = st.selectbox(
                        "Feature to Stress",
                        options=st.session_state.X_test.columns.tolist(),
                        index=st.session_state.X_test.columns.tolist().index(
                            current_scenario['parameters']['feature'])
                    )

                    # Parameters based on stress type
                    if stress_type_edit == "NOISE":
                        param_value = st.number_input(
                            "Standard Deviation Multiplier",
                            min_value=0.1, max_value=2.0,
                            value=float(current_scenario['parameters'].get(
                                'std_dev_multiplier', 0.5)),
                            step=0.1)
                    elif stress_type_edit == "FEATURE_SHIFT":
                        param_value = st.number_input(
                            "Shift Percentage",
                            min_value=-0.5, max_value=1.0,
                            value=float(current_scenario['parameters'].get(
                                'shift_percentage', 0.20)),
                            step=0.01)
                    elif stress_type_edit == "MISSINGNESS":
                        param_value = st.number_input(
                            "Missing Percentage",
                            min_value=0.0, max_value=1.0,
                            value=float(current_scenario['parameters'].get(
                                'missing_percentage', 0.30)),
                            step=0.01)
                    elif stress_type_edit == "OUT_OF_DISTRIBUTION":
                        scale_factor_edit = st.number_input(
                            "Scale Factor",
                            min_value=1.0, max_value=5.0,
                            value=float(current_scenario['parameters'].get(
                                'scale_factor', 1.5)),
                            step=0.1)
                        base_value_edit = st.number_input(
                            "Base Value",
                            value=float(current_scenario['parameters'].get('base_value', 0.0)))

                    description_edit = st.text_area(
                        "Scenario Description",
                        value=current_scenario.get('description', ''))
                    real_world_event_edit = st.text_area(
                        "Real-World Market Event",
                        value=current_scenario.get('real_world_event', ''))
                    expected_business_impact_edit = st.text_area(
                        "Expected Business Impact",
                        value=current_scenario.get('expected_business_impact', ''))
                    severity_level_edit = st.slider(
                        "Severity Level (1-5)",
                        min_value=1, max_value=5,
                        value=int(current_scenario.get('severity_level', 3)))

                    update_submitted = st.form_submit_button(
                        "💾 Update Scenario", use_container_width=True)

                    if update_submitted:
                        parameters = {}
                        if stress_type_edit == "NOISE":
                            parameters = {'feature': feature_name_edit,
                                          'std_dev_multiplier': param_value}
                        elif stress_type_edit == "FEATURE_SHIFT":
                            parameters = {'feature': feature_name_edit,
                                          'shift_percentage': param_value}
                        elif stress_type_edit == "MISSINGNESS":
                            parameters = {'feature': feature_name_edit,
                                          'missing_percentage': param_value}
                        elif stress_type_edit == "OUT_OF_DISTRIBUTION":
                            parameters = {
                                'feature': feature_name_edit, 'scale_factor': scale_factor_edit, 'base_value': base_value_edit}

                        updated_scenario = define_stress_scenario(
                            stress_type=stress_type_edit,
                            parameters=parameters,
                            description=description_edit,
                            real_world_event=real_world_event_edit,
                            expected_business_impact=expected_business_impact_edit,
                            severity_level=severity_level_edit
                        )
                        # Preserve the original scenario_id
                        updated_scenario['scenario_id'] = current_scenario['scenario_id']
                        st.session_state.stress_scenarios_list[scenario_to_edit] = updated_scenario
                        st.session_state.success_message = f"✅ Scenario '{real_world_event_edit}' updated successfully!"
                        st.rerun()
            else:
                st.info("No scenarios available to edit. Add a scenario first.")

        # --- Tab 4: Delete Scenario ---
        with tab4:
            st.markdown("## Delete Stress Scenario")

            if st.session_state.stress_scenarios_list:
                st.warning(
                    "⚠️ Deleting a scenario is permanent and cannot be undone.")

                scenario_to_delete = st.selectbox(
                    "Select Scenario to Delete",
                    options=range(len(st.session_state.stress_scenarios_list)),
                    format_func=lambda x: f"{st.session_state.stress_scenarios_list[x]['real_world_event']} ({st.session_state.stress_scenarios_list[x]['stress_type']})",
                    key="delete_selector"
                )

                selected_scenario = st.session_state.stress_scenarios_list[scenario_to_delete]

                # Show scenario details
                with st.expander("📄 View Scenario Details", expanded=True):
                    st.markdown(
                        f"**Real-World Event:** {selected_scenario['real_world_event']}")
                    st.markdown(
                        f"**Stress Type:** {selected_scenario['stress_type']}")
                    st.markdown(
                        f"**Description:** {selected_scenario['description']}")
                    st.markdown(
                        f"**Feature:** {selected_scenario['parameters'].get('feature', 'N/A')}")
                    st.markdown(
                        f"**Severity Level:** {selected_scenario['severity_level']}")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🗑️ Delete Selected Scenario", type="primary", use_container_width=True):
                        deleted_name = st.session_state.stress_scenarios_list[
                            scenario_to_delete]['real_world_event']
                        st.session_state.stress_scenarios_list.pop(
                            scenario_to_delete)
                        st.session_state.success_message = f"✅ Scenario '{deleted_name}' deleted successfully! Remaining scenarios: {len(st.session_state.stress_scenarios_list)}"
                        st.rerun()

                with col2:
                    if st.button("🗑️ Clear All Scenarios", type="secondary", use_container_width=True):
                        count = len(st.session_state.stress_scenarios_list)
                        st.session_state.stress_scenarios_list = []
                        st.session_state.success_message = f"✅ All {count} scenarios cleared successfully!"
                        st.rerun()
            else:
                st.info("No scenarios available to delete.")

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

        if st.button("Run All Scenarios"):
            progress_bar = st.progress(0, text="Starting stress tests...")
            all_stressed_results = []

            for i, scenario in enumerate(st.session_state.stress_scenarios_list):
                progress_bar.progress((i + 1) / len(st.session_state.stress_scenarios_list),
                                      text=f"Running Scenario: {scenario['real_world_event']}")
                # Use RANDOM_SEED + i for per-scenario reproducibility
                stressed_output = execute_stress_scenario(
                    st.session_state.model, scenario, st.session_state.X_test, st.session_state.y_test, 'subgroup_flag', RANDOM_SEED + i)
                all_stressed_results.extend(stressed_output)

            with open(st.session_state.ROBUSTNESS_RESULTS_FILE, 'w') as f:
                json.dump(all_stressed_results, f, indent=4)

            performance_degradation_results = calculate_degradation_and_status(
                st.session_state.baseline_metrics, all_stressed_results, st.session_state.acceptance_thresholds)

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
            results_df = pd.DataFrame(
                st.session_state.performance_degradation_results)
            st.dataframe(results_df[['scenario_id', 'metric_name', 'baseline_value',
                         'stressed_value', 'degradation_pct', 'status']], use_container_width=True)

# --- Page 4: Results Dashboard ---
elif st.session_state.page == '4. Results Dashboard':
    st.markdown(f"# 4. Results Dashboard")
    st.markdown(f"Comprehensive overview of stress test results with key performance indicators and visualizations for stakeholder communication.")

    if not st.session_state.tests_executed:
        st.warning("Please execute tests in '3. Test Execution' to view results.")
    else:
        results_df = pd.DataFrame(
            st.session_state.performance_degradation_results)

        # --- KPI Summary Cards ---
        st.markdown("## Executive Summary")

        total_scenarios = len(st.session_state.stress_scenarios_list)
        total_tests = len(results_df)
        passed_tests = len(results_df[results_df['status'] == 'PASS'])
        failed_tests = len(results_df[results_df['status'] == 'FAIL'])
        pass_rate = (passed_tests / total_tests *
                     100) if total_tests > 0 else 0

        # Calculate worst degradations
        rmse_worst = results_df[results_df['metric_name']
                                == 'RMSE']['degradation_pct'].min()
        ece_worst = results_df[results_df['metric_name']
                               == 'ECE']['degradation_pct'].min()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="Total Scenarios",
                value=total_scenarios,
                help="Number of stress scenarios executed"
            )

        with col2:
            st.metric(
                label="Pass Rate",
                value=f"{pass_rate:.1f}%",
                delta=f"{passed_tests}/{total_tests} tests",
                delta_color="normal" if pass_rate >= 70 else "inverse",
                help="Percentage of tests that met acceptance criteria"
            )

        with col3:
            st.metric(
                label="Failed Tests",
                value=failed_tests,
                delta="Critical" if failed_tests > total_tests * 0.3 else "Monitor",
                delta_color="inverse" if failed_tests > 0 else "off",
                help="Number of tests that exceeded thresholds"
            )

        with col4:
            st.metric(
                label="Worst RMSE Degradation",
                value=f"{rmse_worst:.1f}%",
                delta="vs baseline",
                delta_color="inverse" if rmse_worst < -10 else "normal",
                help="Maximum RMSE degradation observed"
            )

        with col5:
            st.metric(
                label="Worst ECE Degradation",
                value=f"{ece_worst:.1f}%",
                delta="vs baseline",
                delta_color="inverse" if ece_worst < -10 else "normal",
                help="Maximum ECE degradation observed"
            )

        st.divider()

        # --- Tabbed Interface for Metrics ---
        st.markdown("## Performance Analysis by Metric")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 RMSE", "📈 ECE", "📉 Subgroup Delta", "📋 Full Data"])

        with tab1:
            st.markdown("### RMSE (Root Mean Squared Error)")
            col1, col2 = st.columns(2)

            with col1:
                fig_rmse_comp = plot_metric_comparison(results_df, 'RMSE')
                st.pyplot(fig_rmse_comp)

            with col2:
                fig_rmse_deg = plot_degradation_percentages(results_df, 'RMSE')
                st.pyplot(fig_rmse_deg)

            # Summary stats for RMSE
            rmse_data = results_df[results_df['metric_name'] == 'RMSE']
            st.markdown("**RMSE Statistics:**")
            st.dataframe(
                rmse_data[['real_world_event', 'baseline_value',
                           'stressed_value', 'degradation_pct', 'status']],
                use_container_width=True,
                hide_index=True
            )

        with tab2:
            st.markdown("### ECE (Expected Calibration Error)")
            col1, col2 = st.columns(2)

            with col1:
                fig_ece_comp = plot_metric_comparison(results_df, 'ECE')
                st.pyplot(fig_ece_comp)

            with col2:
                fig_ece_deg = plot_degradation_percentages(results_df, 'ECE')
                st.pyplot(fig_ece_deg)

            # Summary stats for ECE
            ece_data = results_df[results_df['metric_name'] == 'ECE']
            st.markdown("**ECE Statistics:**")
            st.dataframe(
                ece_data[['real_world_event', 'baseline_value',
                          'stressed_value', 'degradation_pct', 'status']],
                use_container_width=True,
                hide_index=True
            )

        with tab3:
            st.markdown("### Subgroup RMSE Delta (Fairness/Bias)")
            col1, col2 = st.columns(2)

            with col1:
                fig_subgroup_comp = plot_metric_comparison(
                    results_df, 'Subgroup_RMSE_Delta')
                st.pyplot(fig_subgroup_comp)

            with col2:
                fig_subgroup_deg = plot_degradation_percentages(
                    results_df, 'Subgroup_RMSE_Delta')
                st.pyplot(fig_subgroup_deg)

            # Summary stats for Subgroup Delta
            subgroup_data = results_df[results_df['metric_name']
                                       == 'Subgroup_RMSE_Delta']
            st.markdown("**Subgroup Delta Statistics:**")
            st.dataframe(
                subgroup_data[['real_world_event', 'baseline_value',
                               'stressed_value', 'degradation_pct', 'status']],
                use_container_width=True,
                hide_index=True
            )

        with tab4:
            st.markdown("### Complete Performance Degradation Data")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            # Download option
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv,
                file_name="performance_degradation_results.csv",
                mime="text/csv"
            )

        st.divider()

        # --- Feature Distribution Analysis ---
        st.markdown("## Feature Distribution Shifts")
        st.markdown(
            "Analyze how input feature distributions change under stress to understand the root causes of performance degradation.")

        selected_scenario_for_plot_id = st.selectbox(
            "Select Stress Scenario to Visualize",
            options=[s['scenario_id']
                     for s in st.session_state.stress_scenarios_list],
            format_func=lambda x: [s['real_world_event']
                                   for s in st.session_state.stress_scenarios_list if s['scenario_id'] == x][0]
        )

        selected_scenario_obj = next(
            (s for s in st.session_state.stress_scenarios_list if s['scenario_id'] == selected_scenario_for_plot_id), None)

        if selected_scenario_obj:
            # Extract feature from scenario parameters
            feature_to_plot = selected_scenario_obj['parameters'].get(
                'feature')

            if feature_to_plot:
                # Display scenario details
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.info(
                        f"**Scenario:** {selected_scenario_obj['real_world_event']}\n\n"
                        f"**Description:** {selected_scenario_obj['description']}\n\n"
                        f"**Stress Type:** {selected_scenario_obj['stress_type']}")

                with col2:
                    st.metric(
                        label="Stressed Feature",
                        value=feature_to_plot,
                        help="The feature being modified in this scenario"
                    )

                X_stressed_for_plot = apply_stress_to_data(
                    st.session_state.X_test, selected_scenario_obj, RANDOM_SEED)
                fig_dist = plot_feature_distribution_shift(st.session_state.X_test,
                                                           X_stressed_for_plot, feature_to_plot, selected_scenario_obj['real_world_event'])
                st.pyplot(fig_dist)
            else:
                st.warning(
                    "Selected scenario does not have a feature parameter.")

# --- Page 5: Decision & Export ---
elif st.session_state.page == '5. Decision & Export':
    st.markdown(f"# 5. Decision & Export")
    st.markdown(f"The culmination of the stress testing process is to formalize the validation outcome and generate comprehensive, audit-ready artifacts. This involves synthesizing all findings, making a clear decision on the model's readiness for production, and creating a traceable record of the entire assessment.")
    st.markdown(f"This step ensures accountability and compliance, providing transparent evidence for regulators and internal auditors.")

    if not st.session_state.tests_executed:
        st.warning(
            "Please execute tests in '3. Test Execution' before making a decision.")
    else:
        if st.button("Generate Validation Decision"):
            generate_validation_decision(st.session_state.performance_degradation_results,
                                         st.session_state.acceptance_thresholds, st.session_state.VALIDATION_DECISION_FILE)

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
            with st.container(border=True):
                st.markdown(
                    st.session_state.validation_decision_data['decision_text'])

        if st.session_state.output_files_generated:

            generate_evidence_manifest(
                st.session_state.current_run_id,
                st.session_state.EVIDENCE_MANIFEST_FILE,
                [st.session_state.STRESS_SCENARIOS_FILE],
                [st.session_state.ROBUSTNESS_RESULTS_FILE, st.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE,
                    st.session_state.VALIDATION_DECISION_FILE]
            )
            st.success("Evidence Manifest generated!")

            # Generate unique filename for artifact download
            if 'artifact_download_ready' not in st.session_state:
                st.session_state.artifact_download_ready = False
                st.session_state.artifact_zip_data = None
                st.session_state.artifact_filename = None

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("📦 Generate All Artifacts as ZIP", use_container_width=True):
                    # Generate unique identifier for this download
                    unique_hash = hashlib.sha256(
                        f"{st.session_state.current_run_id}_{datetime.datetime.now().isoformat()}".encode(
                        )
                    ).hexdigest()[:12]

                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

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
                                    zip_file.writestr(
                                        os.path.basename(filepath), data)

                    # Store in session state for download
                    st.session_state.artifact_zip_data = zip_buffer.getvalue()
                    st.session_state.artifact_filename = f"validation_artifacts_{timestamp}_{unique_hash}.zip"
                    st.session_state.artifact_download_ready = True
                    st.success(
                        f"✅ Artifacts package generated! Click 'Download' to save.")
                    st.rerun()

            with col2:
                if st.session_state.artifact_download_ready and st.session_state.artifact_zip_data:
                    st.download_button(
                        label="💾 Download ZIP Package",
                        data=st.session_state.artifact_zip_data,
                        file_name=st.session_state.artifact_filename,
                        mime="application/zip",
                        use_container_width=True,
                        on_click=lambda: None  # Download happens client-side
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
