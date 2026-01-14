
import pytest
import os
import json
import pandas as pd
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

# Define dummy data and functions to mock source.py behavior for predictable test outcomes
# In a real testing environment, you would ensure source.py exists and functions correctly.
# For the purpose of these tests, we'll create simple mock returns to allow the app flow to be tested.

# Mocking constants from source.py
class MockSource:
    RANDOM_SEED = 42
    ACCEPTANCE_THRESHOLDS = {
        'RMSE': 10.0,
        'ECE': 15.0,
        'Subgroup_RMSE_Delta': 0.1
    }
    STRESS_TYPES = ["NOISE", "FEATURE_SHIFT", "MISSINGNESS", "OUT_OF_DISTRIBUTION"]

    class DummyModel:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return pd.Series([0.5] * len(X), index=X.index) # Consistent prediction for testing

    def generate_synthetic_financial_data(self, num_samples, random_seed):
        # Return a simple DataFrame for testing
        return pd.DataFrame({
            'feature_1': [10.0, 20.0, 30.0, 40.0, 50.0] * (num_samples // 5),
            'feature_2': [1.0, 2.0, 3.0, 4.0, 5.0] * (num_samples // 5),
            'subgroup_flag': ['high_cap_stocks', 'mid_cap_stocks', 'high_cap_stocks', 'mid_cap_stocks', 'high_cap_stocks'] * (num_samples // 5),
            'future_market_risk_score': [0.1, 0.2, 0.3, 0.4, 0.5] * (num_samples // 5)
        }).iloc[:num_samples] # Ensure exact num_samples

    def train_test_split(self, X, y, test_size, random_state):
        split_idx = int(len(X) * (1 - test_size))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def evaluate_model_performance(self, model, X_test, y_test, subgroup_column):
        # Return predictable baseline metrics
        return {
            'RMSE': 0.1,
            'ECE': 0.05,
            'Subgroup_RMSE_Delta': 0.02
        }, self.DummyModel().predict(X_test) # Return dummy predictions

    def define_stress_scenario(self, stress_type, parameters, description, real_world_event, expected_business_impact, severity_level):
        return {
            "scenario_id": f"scenario_{stress_type}",
            "stress_type": stress_type,
            "parameters": parameters,
            "description": description,
            "real_world_event": real_world_event,
            "expected_business_impact": expected_business_impact,
            "severity_level": severity_level
        }

    def apply_noise(self, X_baseline, feature, std_dev_multiplier, random_seed):
        return X_baseline.copy() # No actual change for test simplicity

    def apply_shift(self, X_baseline, feature, shift_percentage):
        X_shifted = X_baseline.copy()
        X_shifted[feature] = X_shifted[feature] * (1 + shift_percentage)
        return X_shifted

    def apply_missingness(self, X_baseline, feature, missing_percentage, random_seed):
        X_missing = X_baseline.copy()
        # Simulate some missing values
        if not X_missing.empty:
            X_missing.loc[X_missing.sample(frac=missing_percentage, random_state=random_seed).index, feature] = None
        return X_missing

    def apply_out_of_distribution(self, X_baseline, feature, scale_factor, base_value, random_seed):
        return X_baseline.copy() # No actual change for test simplicity

    def execute_stress_scenario(self, model, scenario, X_baseline, y_baseline, subgroup_column, random_seed):
        # Return a predictable stressed result that might pass or fail based on thresholds
        stressed_value_rmse = MockSource.ACCEPTANCE_THRESHOLDS['RMSE'] * 0.05 + MockSource.ACCEPTANCE_THRESHOLDS['RMSE'] # Slightly above threshold
        if scenario['stress_type'] == "NOISE":
            stressed_value_rmse = MockSource.ACCEPTANCE_THRESHOLDS['RMSE'] * 0.08 # Below threshold for noise to pass

        return [
            {"scenario_id": scenario['scenario_id'], "stress_type": scenario['stress_type'], "metric_name": "RMSE", "stressed_value": stressed_value_rmse},
            {"scenario_id": scenario['scenario_id'], "stress_type": scenario['stress_type'], "metric_name": "ECE", "stressed_value": 0.06}, # Pass
            {"scenario_id": scenario['scenario_id'], "stress_type": scenario['stress_type'], "metric_name": "Subgroup_RMSE_Delta", "stressed_value": 0.03} # Pass
        ]

    def calculate_degradation_and_status(self, baseline_metrics, all_stressed_results, acceptance_thresholds):
        degradation_results = []
        for res in all_stressed_results:
            metric = res['metric_name']
            baseline_val = baseline_metrics.get(metric, 0)
            stressed_val = res['stressed_value']
            
            degradation_pct = None
            if metric in ['RMSE', 'ECE']:
                degradation_pct = ((stressed_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else (float('inf') if stressed_val > 0 else 0)
                status = "FAIL" if degradation_pct > acceptance_thresholds.get(metric, 0) else "PASS"
            elif metric == 'Subgroup_RMSE_Delta':
                delta_increase = stressed_val - baseline_val
                degradation_pct = None # Not percentage for delta
                status = "FAIL" if delta_increase > acceptance_thresholds.get(metric, 0) else "PASS"

            degradation_results.append({
                'scenario_id': res['scenario_id'],
                'stress_type': res['stress_type'],
                'metric_name': metric,
                'baseline_value': baseline_val,
                'stressed_value': stressed_val,
                'degradation_pct': degradation_pct,
                'status': status
            })
        return degradation_results

    def generate_validation_decision(self, performance_degradation_results, acceptance_thresholds, output_file):
        decision_text = "# Mock Validation Decision Report\n\nOverall: PASS"
        if any(r['status'] == 'FAIL' for r in performance_degradation_results):
            decision_text = "# Mock Validation Decision Report\n\nOverall: FAIL"
        with open(output_file, 'w') as f:
            f.write(decision_text)

    def generate_evidence_manifest(self, run_id, output_file, input_files, output_files):
        manifest = {"run_id": run_id, "output_files": output_files}
        with open(output_file, 'w') as f:
            json.dump(manifest, f)

    def plot_metric_comparison(self, *args, **kwargs):
        return MagicMock() # Return a mock matplotlib figure

    def plot_degradation_percentages(self, *args, **kwargs):
        return MagicMock() # Return a mock matplotlib figure

    def plot_feature_distribution_shift(self, *args, **kwargs):
        return MagicMock() # Return a mock matplotlib figure

# Patch the 'source' module globally for all tests
mock_source_instance = MockSource()
patch_source = patch.dict('sys.modules', {'source': mock_source_instance})

@pytest.fixture(autouse=True, scope="module")
def setup_mock_source():
    patch_source.start()
    yield
    patch_source.stop()

@pytest.fixture(autouse=True)
def clean_up_files():
    # Clean up generated files before each test
    files = [
        "stress_scenarios.json",
        "robustness_results.json",
        "performance_degradation_report.json",
        "validation_decision.md",
        "evidence_manifest.json"
    ]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    yield
    # Clean up after each test as well
    for f in files:
        if os.path.exists(f):
            os.remove(f)

def test_initial_state_and_sidebar_navigation():
    at = AppTest.from_file("app.py").run()

    # Initial page check
    assert at.session_state['page'] == '1. Baseline & Configuration'
    assert "1. Baseline & Configuration" in at.markdown[0].value

    # Navigate to Scenario Builder
    at.selectbox("Choose a section:").set_value('2. Scenario Builder').run()
    assert at.session_state['page'] == '2. Scenario Builder'
    assert "2. Scenario Builder" in at.markdown[0].value

    # Navigate to Test Execution
    at.selectbox("Choose a section:").set_value('3. Test Execution').run()
    assert at.session_state['page'] == '3. Test Execution'
    assert "3. Test Execution" in at.markdown[0].value

    # Navigate to Results Dashboard
    at.selectbox("Choose a section:").set_value('4. Results Dashboard').run()
    assert at.session_state['page'] == '4. Results Dashboard'
    assert "4. Results Dashboard" in at.markdown[0].value

    # Navigate to Decision & Export
    at.selectbox("Choose a section:").set_value('5. Decision & Export').run()
    assert at.session_state['page'] == '5. Decision & Export'
    assert "5. Decision & Export" in at.markdown[0].value


def test_page1_baseline_configuration():
    at = AppTest.from_file("app.py").run()

    # Test baseline generation
    assert not at.session_state['model_initialized']
    at.button("Generate Baseline Model & Metrics").click().run()
    
    assert at.session_state['model_initialized']
    assert at.session_state['baseline_metrics']['RMSE'] == 0.1
    assert "Baseline model generated and metrics established!" in at.success[0].value
    assert at.dataframe[0].value.iloc[0, 0] == 0.1 # Check RMSE value in displayed dataframe

    # Test acceptance threshold configuration
    initial_rmse_threshold = at.session_state['acceptance_thresholds']['RMSE']
    at.number_input("Max % Increase in RMSE").set_value(15.0).run()
    assert at.session_state['acceptance_thresholds']['RMSE'] == 15.0
    
    at.number_input("Max % Increase in ECE").set_value(20.0).run()
    assert at.session_state['acceptance_thresholds']['ECE'] == 20.0

    at.number_input("Max Absolute Increase in Subgroup RMSE Delta").set_value(0.05).run()
    assert at.session_state['acceptance_thresholds']['Subgroup_RMSE_Delta'] == 0.05


def test_page2_scenario_builder():
    at = AppTest.from_file("app.py").run()

    # First, navigate to Page 1 and initialize the model
    at.selectbox("Choose a section:").set_value('1. Baseline & Configuration').run()
    at.button("Generate Baseline Model & Metrics").click().run()

    # Navigate to Page 2
    at.selectbox("Choose a section:").set_value('2. Scenario Builder').run()

    # Verify initial state (no scenarios)
    assert not at.session_state['stress_scenarios_list']
    assert "No scenarios defined yet." in at.info[0].value

    # Add NOISE scenario
    at.selectbox("Select Stress Type").set_value("NOISE").run()
    at.selectbox("Feature to Stress").set_value("feature_1").run()
    at.number_input("Standard Deviation Multiplier").set_value(0.8).run()
    at.text_area("Scenario Description").set_value("Adding noise to feature 1").run()
    at.text_area("Real-World Market Event").set_value("Market Volatility").run()
    at.text_area("Expected Business Impact").set_value("Increased RMSE").run()
    at.slider("Severity Level (1-5)").set_value(4).run()
    at.button("Add Scenario").click().run()
    
    assert len(at.session_state['stress_scenarios_list']) == 1
    assert "Scenario added successfully!" in at.success[0].value
    assert at.session_state['stress_scenarios_list'][0]['stress_type'] == 'NOISE'
    assert at.dataframe[0].value.iloc[0]['stress_type'] == 'NOISE' # Check displayed dataframe

    # Add FEATURE_SHIFT scenario
    at.selectbox("Select Stress Type").set_value("FEATURE_SHIFT").run()
    at.selectbox("Feature to Stress").set_value("feature_2").run()
    at.number_input("Shift Percentage (e.g., 0.20 for 20% increase)").set_value(0.15).run()
    at.text_area("Scenario Description").set_value("Shifting feature 2").run() # This text area will be reset by the selectbox, need to re-set.
    at.text_area("Real-World Market Event").set_value("Interest Rate Shock").run()
    at.text_area("Expected Business Impact").set_value("Model Bias").run()
    at.slider("Severity Level (1-5)", key="severity_level_widget").set_value(3).run() # Ensure unique key if re-setting is an issue
    at.button("Add Scenario", key="add_scenario_button").click().run() # Ensure unique key

    assert len(at.session_state['stress_scenarios_list']) == 2
    assert at.session_state['stress_scenarios_list'][1]['stress_type'] == 'FEATURE_SHIFT'
    assert at.dataframe[0].value.iloc[1]['stress_type'] == 'FEATURE_SHIFT'

    # Clear all scenarios
    at.button("Clear All Scenarios").click().run()
    assert not at.session_state['stress_scenarios_list']
    assert "All scenarios cleared." in at.info[0].value

    # Re-add a scenario for saving
    at.selectbox("Select Stress Type").set_value("MISSINGNESS").run()
    at.selectbox("Feature to Stress").set_value("feature_1").run()
    at.number_input("Missing Percentage").set_value(0.1).run()
    at.button("Add Scenario").click().run()
    assert len(at.session_state['stress_scenarios_list']) == 1

    # Save scenarios to JSON
    at.button("Save Scenarios to JSON").click().run()
    assert f"Scenarios saved to {at.session_state.STRESS_SCENARIOS_FILE}" in at.success[0].value
    assert os.path.exists(at.session_state.STRESS_SCENARIOS_FILE)
    with open(at.session_state.STRESS_SCENARIOS_FILE, 'r') as f:
        saved_scenarios = json.load(f)
        assert len(saved_scenarios) == 1
        assert saved_scenarios[0]['stress_type'] == 'MISSINGNESS'


def test_page3_test_execution():
    at = AppTest.from_file("app.py").run()

    # Prepare prerequisites: Initialize model and add a scenario
    at.selectbox("Choose a section:").set_value('1. Baseline & Configuration').run()
    at.button("Generate Baseline Model & Metrics").click().run()
    at.selectbox("Choose a section:").set_value('2. Scenario Builder').run()
    at.selectbox("Select Stress Type").set_value("NOISE").run()
    at.selectbox("Feature to Stress").set_value("feature_1").run()
    at.button("Add Scenario").click().run()

    # Navigate to Page 3
    at.selectbox("Choose a section:").set_value('3. Test Execution').run()

    # Verify tests not executed initially
    assert not at.session_state['tests_executed']
    assert not at.button("Run All Scenarios").disabled

    # Run all scenarios
    at.button("Run All Scenarios").click().run()
    
    assert at.session_state['tests_executed']
    assert "All stress tests executed successfully!" in at.success[0].value
    assert len(at.session_state['raw_stressed_results']) > 0
    assert len(at.session_state['performance_degradation_results']) > 0

    assert os.path.exists(at.session_state.ROBUSTNESS_RESULTS_FILE)
    assert os.path.exists(at.session_state.PERFORMANCE_DEGRADATION_REPORT_FILE)

    # Check results summary dataframe
    assert at.dataframe[0].value.shape[0] > 0 # At least one row of results


def test_page4_results_dashboard():
    at = AppTest.from_file("app.py").run()

    # Prepare prerequisites: Initialize model, add scenarios, execute tests
    at.selectbox("Choose a section:").set_value('1. Baseline & Configuration').run()
    at.button("Generate Baseline Model & Metrics").click().run()
    at.selectbox("Choose a section:").set_value('2. Scenario Builder').run()
    
    # Add a NOISE scenario (will likely pass RMSE degradation check in mock)
    at.selectbox("Select Stress Type").set_value("NOISE").run()
    at.selectbox("Feature to Stress").set_value("feature_1").run()
    at.text_area("Scenario Description").set_value("Noise test").run()
    at.text_area("Real-World Market Event").set_value("Volatility Event").run()
    at.button("Add Scenario").click().run()

    # Add a FEATURE_SHIFT scenario (will likely fail RMSE degradation check in mock)
    at.selectbox("Select Stress Type").set_value("FEATURE_SHIFT").run()
    at.selectbox("Feature to Stress").set_value("feature_2").run()
    at.number_input("Shift Percentage (e.g., 0.20 for 20% increase)").set_value(0.2).run()
    at.text_area("Scenario Description").set_value("Shift test").run()
    at.text_area("Real-World Market Event").set_value("Economic Shift").run()
    at.button("Add Scenario", key="add_shift_scenario").click().run()

    at.selectbox("Choose a section:").set_value('3. Test Execution').run()
    at.button("Run All Scenarios").click().run()

    # Navigate to Page 4
    at.selectbox("Choose a section:").set_value('4. Results Dashboard').run()

    # Verify performance degradation summary is displayed
    assert at.dataframe[0].value.shape[0] > 0
    
    # Verify plots are attempted to be generated (cannot directly check plot content)
    # Check that st.pyplot was called for each expected plot type
    assert len(at.pyplot) >= 6 # 3 metrics * 2 plots each

    # Test feature distribution shift plot selection
    # Assuming scenarios are ordered as added
    scenario_ids = [s['scenario_id'] for s in at.session_state['stress_scenarios_list']]
    assert len(scenario_ids) == 2
    
    # Select the first scenario for plotting
    at.selectbox("Select Scenario for Distribution Plot").set_value(scenario_ids[0]).run()
    at.selectbox("Select Feature to Visualize").set_value("feature_1").run()
    
    # Check that another plot (feature distribution) is attempted
    assert len(at.pyplot) >= 7


def test_page5_decision_export():
    at = AppTest.from_file("app.py").run()

    # Prepare prerequisites: Initialize model, add scenarios, execute tests
    at.selectbox("Choose a section:").set_value('1. Baseline & Configuration').run()
    at.button("Generate Baseline Model & Metrics").click().run()
    at.selectbox("Choose a section:").set_value('2. Scenario Builder').run()
    
    # Add a scenario that will result in a FAIL based on mock_source
    at.selectbox("Select Stress Type").set_value("FEATURE_SHIFT").run()
    at.selectbox("Feature to Stress").set_value("feature_1").run()
    at.text_area("Scenario Description").set_value("Shift for fail").run()
    at.text_area("Real-World Market Event").set_value("Severe Shift").run()
    at.button("Add Scenario").click().run()

    at.selectbox("Choose a section:").set_value('3. Test Execution').run()
    at.button("Run All Scenarios").click().run()

    # Navigate to Page 5
    at.selectbox("Choose a section:").set_value('5. Decision & Export').run()

    # Verify "Generate Validation Decision" button
    at.button("Generate Validation Decision").click().run()
    assert "Validation Decision report generated!" in at.success[0].value
    assert "decision_text" in at.session_state['validation_decision_data']
    assert "Overall: FAIL" in at.markdown[0].value # Based on our mock, FEATURE_SHIFT will fail RMSE

    assert os.path.exists(at.session_state.VALIDATION_DECISION_FILE)
    with open(at.session_state.VALIDATION_DECISION_FILE, 'r') as f:
        decision_content = f.read()
        assert "Overall: FAIL" in decision_content

    # Verify "Generate Evidence Manifest" button
    at.button("Generate Evidence Manifest").click().run()
    assert "Evidence Manifest generated!" in at.success[0].value
    assert os.path.exists(at.session_state.EVIDENCE_MANIFEST_FILE)
    with open(at.session_state.EVIDENCE_MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)
        assert "run_id" in manifest
        assert at.session_state.ROBUSTNESS_RESULTS_FILE in manifest['output_files']

    # Verify "Download All Artifacts as ZIP" button is present and enabled
    # We can't actually download it, but we can check for its existence
    assert at.download_button[0].label == "Download All Artifacts as ZIP"

