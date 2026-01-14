import pandas as pd
import numpy as np
import uuid
import datetime
import hashlib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define constants for file paths
STRESS_SCENARIOS_FILE = "stress_scenarios.json"
ROBUSTNESS_RESULTS_FILE = "robustness_results.json"
PERFORMANCE_DEGRADATION_REPORT_FILE = "performance_degradation_report.json"
VALIDATION_DECISION_FILE = "validation_decision.md"
EVIDENCE_MANIFEST_FILE = "evidence_manifest.json"
def generate_synthetic_financial_data(num_samples=1000, random_seed=RANDOM_SEED):
    """
    Generates synthetic financial market data.
    Features: market_index, volatility, interest_rate, oil_price, subgroup_flag.
    Target: future_market_risk_score (continuous, 0-1).
    """
    np.random.seed(random_seed)
    data = pd.DataFrame()

    data['market_index'] = np.random.normal(1000, 150, num_samples)
    data['volatility'] = np.random.normal(20, 5, num_samples)
    data['interest_rate'] = np.random.normal(3, 1, num_samples)
    data['oil_price'] = np.random.normal(70, 10, num_samples)
    data['subgroup_flag'] = np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # e.g., market segment A vs B

    # Introduce some correlations and non-linearity for a realistic risk score
    data['future_market_risk_score'] = (
        0.05 * data['volatility']
        + 0.001 * data['market_index']
        - 0.05 * data['interest_rate']
        + 0.01 * data['oil_price']
        + 0.1 * data['subgroup_flag'] # Subgroup has a small impact
        + np.random.normal(0, 0.05, num_samples)
    )

    # Scale risk score to be roughly between 0 and 1
    min_risk, max_risk = data['future_market_risk_score'].min(), data['future_market_risk_score'].max()
    # Handle the case where min_risk equals max_risk to prevent division by zero
    if max_risk == min_risk:
        data['future_market_risk_score'] = 0.5 # Assign a neutral risk score if no variation
    else:
        data['future_market_risk_score'] = (data['future_market_risk_score'] - min_risk) / (max_risk - min_risk)

    return data

def calculate_ece_from_regressor(y_true, y_pred, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE) for a regressor's output,
    treating continuous predictions as confidences for a binarized true outcome.
    """
    # Define a binary event based on y_true (e.g., actual risk score > threshold)
    risk_event_threshold = y_true.median() # Use median for a balanced binary target
    y_true_binary = (y_true > risk_event_threshold).astype(int)

    # Scale y_pred to be between 0 and 1 to interpret as 'probability of risk'
    min_pred, max_pred = y_pred.min(), y_pred.max()
    if max_pred == min_pred:
        y_pred_proba = np.full_like(y_pred, 0.5)
    else:
        y_pred_proba = (y_pred - min_pred) / (max_pred - min_pred)

    # Use sklearn's calibration_curve to get fraction of positives and mean predicted value per bin
    # We choose 'uniform' strategy to divide the prediction space into equal-width bins
    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_proba, n_bins=n_bins, strategy='uniform')

    ece = 0
    bins = np.linspace(0., 1., n_bins + 1)

    for i in range(n_bins):
        # Filter predictions within the current bin
        # Handle the last bin's upper bound inclusively
        if i == n_bins - 1:
            bin_indices = np.where((y_pred_proba >= bins[i]) & (y_pred_proba <= bins[i+1]))
        else:
            bin_indices = np.where((y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1]))

        # np.where returns a tuple of arrays (one for each dimension). For 1D, it's (array([...]),)
        # We need to extract the actual array of indices: bin_indices[0]
        if len(bin_indices[0]) > 0:
            # Accuracy in bin (proportion of actual positive events)
            # y_true_binary is a Series, so use .iloc for integer position indexing
            accuracy = y_true_binary.iloc[bin_indices[0]].mean()
            # Confidence in bin (average predicted probability)
            # y_pred_proba is a numpy array, can be indexed directly with bin_indices[0]
            confidence = y_pred_proba[bin_indices[0]].mean()
            # Proportion of samples in this bin
            proportion = len(bin_indices[0]) / len(y_true_binary)
            ece += abs(accuracy - confidence) * proportion
    return ece


def evaluate_model_performance(model, X, y, subgroup_column=None):
    """
    Evaluates the model and calculates key performance metrics: RMSE, ECE, Subgroup Delta.
    """
    y_pred = model.predict(X)

    # Overall RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Overall ECE
    ece = calculate_ece_from_regressor(y, y_pred) # y_pred is a numpy array, y is a pandas Series, which calculate_ece_from_regressor handles

    # Subgroup Performance Deltas
    subgroup_deltas = {}
    if subgroup_column and subgroup_column in X.columns:
        subgroups = X[subgroup_column].unique()
        if len(subgroups) > 1:
            subgroup_rmses = {}
            for sg_value in subgroups:
                sg_mask = (X[subgroup_column] == sg_value)
                if sg_mask.sum() > 0:
                    # Ensure consistent indexing: y and y_pred need to be aligned
                    subgroup_y = y.loc[sg_mask.index[sg_mask]] if isinstance(y, pd.Series) else y[sg_mask]
                    subgroup_y_pred = y_pred[sg_mask.values] # Apply boolean mask to numpy array y_pred
                    subgroup_rmses[f'RMSE_Subgroup_{sg_value}'] = np.sqrt(mean_squared_error(subgroup_y, subgroup_y_pred))

            # For simplicity, calculate the absolute difference between the first two subgroups found.
            sorted_subgroup_keys = sorted(subgroup_rmses.keys())
            if len(sorted_subgroup_keys) >= 2:
                delta = abs(subgroup_rmses[sorted_subgroup_keys[0]] - subgroup_rmses[sorted_subgroup_keys[1]])
                subgroup_deltas['Subgroup_RMSE_Delta'] = delta
            else:
                subgroup_deltas['Subgroup_RMSE_Delta'] = 0.0 # Not enough subgroups to compare

    metrics = {
        "RMSE": rmse,
        "ECE": ece,
    }
    metrics.update(subgroup_deltas) # Add subgroup deltas if calculated
    return metrics, y_pred


# 1. Generate Data
print("Generating synthetic financial market data...")
data = generate_synthetic_financial_data(num_samples=2000, random_seed=RANDOM_SEED)
X = data.drop('future_market_risk_score', axis=1)
y = data['future_market_risk_score']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# 3. Train a Proxy Market Risk Forecasting Model (Random Forest Regressor)
print("Training a proxy market risk forecasting model...")
model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

# 4. Evaluate Baseline Performance
print("Evaluating baseline model performance...")
baseline_metrics, y_pred_baseline = evaluate_model_performance(model, X_test, y_test, subgroup_column='subgroup_flag')

print("\n--- Baseline Model Performance ---")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.4f}")

# Store baseline data and metrics globally for later use
global_baseline_metrics = baseline_metrics
global_X_test = X_test
global_y_test = y_test
global_model = model
# Define a StressType enumeration for clarity (conceptually, not a Python Enum object here)
STRESS_TYPES = ['NOISE', 'FEATURE_SHIFT', 'MISSINGNESS', 'OUT_OF_DISTRIBUTION', 'SUBGROUP']

def define_stress_scenario(
      stress_type: str,
    description: str,
    parameters: dict,
    real_world_event: str,
    expected_business_impact: str,
    severity_level: int = 3
) -> dict:
    """
    Defines a single stress scenario with unique ID and structured metadata.
    """
    if stress_type not in STRESS_TYPES:
        raise ValueError(f"Invalid stress_type. Must be one of {STRESS_TYPES}")

    scenario = {
          "scenario_id": str(uuid.uuid4()),
        "stress_type": stress_type,
        "description": description,
        "parameters": parameters,
        "real_world_event": real_world_event,
        "expected_business_impact": expected_business_impact,
        "severity_level": severity_level, # Scale from 1 (low) to 5 (critical)
    }
    return scenario

# List to hold all defined scenarios
stress_scenarios = []

# Scenario 1: Feature Noise Injection (Gaussian)
# Simulating sudden, unpredictable market volatility spikes or data corruption during rapid data ingest.
scenario_1 = define_stress_scenario(
      stress_type="NOISE",
    description="Injects Gaussian noise into 'volatility' feature, mimicking sudden, unforecastable market volatility.",
    parameters={"feature": "volatility", "std_dev_multiplier": 0.5}, # 50% of original std dev
    real_world_event="Flash Crash / Unforeseen Geopolitical Event",
    expected_business_impact="Increased RMSE, potentially underestimating risk due to 'smoothed' predictions, leading to incorrect capital allocation.",
    severity_level=4
)
stress_scenarios.append(scenario_1)

# Scenario 2: Distributional Shift (Mean Shift)
# Simulating a sustained shift in economic indicators, like a prolonged period of high interest rates.
scenario_2 = define_stress_scenario(
      stress_type="FEATURE_SHIFT",
    description="Shifts the mean of the 'interest_rate' feature, simulating a prolonged central bank policy shift.",
    parameters={"feature": "interest_rate", "shift_percentage": 0.20}, # 20% increase
    real_world_event="Interest Rate Shock / Persistent Inflation",
    expected_business_impact="Poor calibration of risk scores as the model operates outside its learned feature distribution, leading to mispricing of financial products.",
    severity_level=5
)
stress_scenarios.append(scenario_2)

# Scenario 3: Missingness Spike
# Simulating failures in data pipelines or extreme market illiquidity leading to missing market data points.
scenario_3 = define_stress_scenario(
      stress_type="MISSINGNESS",
    description="Introduces missing values into 'oil_price' feature, simulating data pipeline failure or market illiquidity.",
    parameters={"feature": "oil_price", "missing_percentage": 0.30}, # 30% missing
    real_world_event="Supply Chain Disruption / Data Feed Outage",
    expected_business_impact="Model errors due to reliance on imputed values, leading to erroneous risk assessments for commodity-sensitive portfolios.",
    severity_level=3
)
stress_scenarios.append(scenario_3)

# Scenario 4: Out-of-Distribution Data (Synthetically Generated)
# Simulating extreme, unprecedented market conditions.
scenario_4 = define_stress_scenario(
      stress_type="OUT_OF_DISTRIBUTION",
    description="Generates synthetic 'market_index' values far outside historical range, mimicking a black swan event.",
    parameters={"feature": "market_index", "scale_factor": 1.5, "base_value": global_X_test['market_index'].max()}, # Pushing beyond max
    real_world_event="Global Financial Crisis (Black Swan Event)",
    expected_business_impact="Complete breakdown of model predictions, leading to catastrophic losses due to unmitigated exposure to extreme market movements.",
    severity_level=5
)
stress_scenarios.append(scenario_4)


# Save the scenarios to a JSON file (the "Market Stress Scenario Handbook")
with open(STRESS_SCENARIOS_FILE, 'w') as f:
    json.dump(stress_scenarios, f, indent=4)

print(f"Defined {len(stress_scenarios)} stress scenarios and saved them to {STRESS_SCENARIOS_FILE}")
print("\n--- Example Scenario Definition ---")
print(json.dumps(stress_scenarios[0], indent=4))
def apply_noise(data: pd.DataFrame, feature: str, std_dev_multiplier: float, random_state: int) -> pd.DataFrame:
    """Adds Gaussian noise to a specified feature."""
    data_copy = data.copy()
    feature_std = data_copy[feature].std()
    noise_magnitude = std_dev_multiplier * feature_std
    rng = np.random.RandomState(random_state)
    data_copy[feature] = data_copy[feature] + rng.normal(0, noise_magnitude, data_copy.shape[0])
    return data_copy


def apply_shift(data: pd.DataFrame, feature: str, shift_percentage: float) -> pd.DataFrame:
    """Shifts the mean of a specified feature by a percentage."""
    data_copy = data.copy()
    data_copy[feature] = data_copy[feature] * (1 + shift_percentage)
    return data_copy


def apply_missingness(data: pd.DataFrame, feature: str, missing_percentage: float, random_state: int) -> pd.DataFrame:
    """Introduces missing values (NaN) into a specified feature."""
    data_copy = data.copy()
    num_missing = int(len(data_copy) * missing_percentage)
    rng = np.random.RandomState(random_state)
    indices_to_na = rng.choice(data_copy.index, size=num_missing, replace=False)
    data_copy.loc[indices_to_na, feature] = np.nan
    return data_copy


def apply_out_of_distribution(data: pd.DataFrame, feature: str, scale_factor: float, base_value: float, random_state: int) -> pd.DataFrame:
    """Generates synthetic values for a feature far outside its historical range."""
    data_copy = data.copy()
    rng = np.random.RandomState(random_state)
    # Generate values significantly higher than historical max
    data_copy[feature] = base_value * (1 + rng.uniform(scale_factor - 0.1, scale_factor + 0.1, data_copy.shape[0]))
    return data_copy


def execute_stress_scenario(model, scenario: dict, X_baseline: pd.DataFrame, y_baseline: pd.Series, subgroup_column: str, random_seed: int) -> dict:
    """
    Applies a specified stress scenario to the baseline data, re-evaluates the model,
    and returns the stressed performance metrics.
    """
    X_stressed = X_baseline.copy()  # Start with a clean copy of baseline data

    # Apply the specific perturbation based on stress_type
    if scenario['stress_type'] == "NOISE":
        X_stressed = apply_noise(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['std_dev_multiplier'],
            random_seed
        )
    elif scenario['stress_type'] == "FEATURE_SHIFT":
        X_stressed = apply_shift(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['shift_percentage']
        )
    elif scenario['stress_type'] == "MISSINGNESS":
        X_stressed = apply_missingness(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['missing_percentage'],
            random_seed
        )
        # Handle missing values: Impute with the mean of the *original* baseline feature to simulate real-world imputation
        imputation_value = X_baseline[scenario['parameters']['feature']].mean()
        X_stressed[scenario['parameters']['feature']] = X_stressed[scenario['parameters']['feature']].fillna(imputation_value)
    elif scenario['stress_type'] == "OUT_OF_DISTRIBUTION":
        X_stressed = apply_out_of_distribution(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['scale_factor'],
            scenario['parameters']['base_value'],
            random_seed
        )
    # Add other stress types here if implemented

    # Re-evaluate the model on the stressed data
    stressed_metrics, _ = evaluate_model_performance(model, X_stressed, y_baseline, subgroup_column=subgroup_column)

    # Format the results for RobustnessResult schema
    results = []
    for metric_name, stressed_value in stressed_metrics.items():
        results.append({
            "scenario_id": scenario['scenario_id'],
            "metric_name": metric_name,
            "stressed_value": stressed_value,
            "description": scenario['description'],
            "real_world_event": scenario['real_world_event'],
            "expected_business_impact": scenario['expected_business_impact']
            # degradation_pct, baseline_value, threshold, status will be added later
        })
    return results


# Load defined scenarios
with open(STRESS_SCENARIOS_FILE, 'r') as f:
    loaded_scenarios = json.load(f)

all_stressed_results = []
print(f"Executing {len(loaded_scenarios)} stress scenarios...")

for i, scenario in enumerate(loaded_scenarios):
    print(f"[{i+1}/{len(loaded_scenarios)}] Executing scenario: {scenario['real_world_event']}")
    stressed_output = execute_stress_scenario(
        model=global_model,
        scenario=scenario,
        X_baseline=global_X_test,
        y_baseline=global_y_test,
        subgroup_column='subgroup_flag',
        random_seed=RANDOM_SEED + i  # Vary seed slightly for different noise patterns if applicable, but keep overall deterministic
    )
    all_stressed_results.extend(stressed_output)

# Save raw stressed results
with open(ROBUSTNESS_RESULTS_FILE, 'w') as f:
    json.dump(all_stressed_results, f, indent=4)

print(f"\nAll stress scenarios executed. Raw results saved to {ROBUSTNESS_RESULTS_FILE}")
print("\n--- Example Stressed Result Entry (first scenario, first metric) ---")
print(json.dumps(all_stressed_results[0], indent=4))
# Define configurable acceptance thresholds
ACCEPTANCE_THRESHOLDS = {
      'RMSE': 10,  # Max 10% *increase* in RMSE (i.e., degradation_pct must be >= -10)
    'ECE': 5,    # Max 5% *increase* in ECE (i.e., degradation_pct must be >= -5)
    'Subgroup_RMSE_Delta': 15 # Max absolute delta increase in subgroup RMSE (not pct)
}

def calculate_degradation_and_status(
      baseline_metrics: dict,
    stressed_results: list,
    acceptance_thresholds: dict
) -> list:
    """
    Calculates degradation percentage and determines pass/fail status for each stressed result.
    """
    final_results = []
    for res in stressed_results:
        metric_name = res['metric_name']
        stressed_value = res['stressed_value']
        baseline_value = baseline_metrics.get(metric_name)

        if baseline_value is None:
            print(f"Warning: Baseline value for {metric_name} not found. Skipping degradation calculation for this metric.")
            res.update({
                  "baseline_value": None,
                "degradation_pct": None,
                "threshold": None,
                "status": "N/A"
            })
            final_results.append(res)
            continue

        # Calculate degradation percentage using the specified formula
        # degradation_pct = (baseline - stressed) / baseline * 100
        degradation_pct = ((baseline_value - stressed_value) / baseline_value) * 100 if baseline_value != 0 else float('inf')

        # Determine pass/fail status based on metric type and thresholds
        status = "PASS"
        threshold = acceptance_thresholds.get(metric_name)

        if threshold is not None:
            if metric_name in ['RMSE', 'ECE']: # Lower is better. Negative degradation_pct means stressed value is higher (worse).
                # Model passes if the *increase* (magnitude of negative degradation_pct) is within threshold
                if degradation_pct < -threshold: # If actual stressed value is more than X% higher than baseline
                    status = "FAIL"
            elif metric_name == 'Subgroup_RMSE_Delta': # Lower is better for delta. Compare absolute values.
                # Threshold for Subgroup_RMSE_Delta is an absolute value, not a percentage increase
                # So we check if the stressed delta exceeds the baseline delta by more than the tolerance
                if stressed_value > (baseline_value + threshold): # If stressed delta is greater than baseline + tolerance
                     status = "FAIL"
                degradation_pct = ((stressed_value - baseline_value) / baseline_value) * 100 if baseline_value != 0 else float('inf') # Still calculate pct for display
            # Add other metric types (e.g., AUC where higher is better) here
            # elif metric_name == 'AUC': # Higher is better. Positive degradation_pct means stressed value is lower (worse).
            #    if degradation_pct > threshold: # If actual stressed value is more than X% lower than baseline
            #        status = "FAIL"
        else:
            status = "N/A - No Threshold" # No threshold defined for this metric

        res.update({
              "baseline_value": baseline_value,
            "stressed_value": stressed_value,
            "degradation_pct": degradation_pct,
            "threshold": threshold, # Store the defined threshold for context
            "status": status
        })
        final_results.append(res)
    return final_results

# Load raw stressed results
with open(ROBUSTNESS_RESULTS_FILE, 'r') as f:
    raw_stressed_results = json.load(f)

# Calculate degradation and status
print("Calculating performance degradation and applying acceptance thresholds...")
performance_degradation_results = calculate_degradation_and_status(
      global_baseline_metrics, raw_stressed_results, ACCEPTANCE_THRESHOLDS
)

# Store the final performance degradation report
with open(PERFORMANCE_DEGRADATION_REPORT_FILE, 'w') as f:
    json.dump(performance_degradation_results, f, indent=4)

print(f"\nPerformance degradation report generated and saved to {PERFORMANCE_DEGRADATION_REPORT_FILE}")

# Display a summary of degradation results
print("\n--- Summary of Performance Degradation Results ---")
for res in performance_degradation_results:
    if res['status'] == "FAIL":
        print(f"FAIL: Scenario {res['scenario_id'][:8]}... - Metric {res['metric_name']}: "
              f"Baseline={res['baseline_value']:.4f}, Stressed={res['stressed_value']:.4f}, "
              f"Degradation={res['degradation_pct']:.2f}%, Threshold={res['threshold']}")
    elif res['status'] == "PASS":
         print(f"PASS: Scenario {res['scenario_id'][:8]}... - Metric {res['metric_name']}: "
              f"Baseline={res['baseline_value']:.4f}, Stressed={res['stressed_value']:.4f}, "
              f"Degradation={res['degradation_pct']:.2f}%, Threshold={res['threshold']}")
import pandas as pd
import numpy as np
import uuid
import datetime
import hashlib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define constants for file paths
STRESS_SCENARIOS_FILE = "stress_scenarios.json"
ROBUSTNESS_RESULTS_FILE = "robustness_results.json"
PERFORMANCE_DEGRADATION_REPORT_FILE = "performance_degradation_report.json"
VALIDATION_DECISION_FILE = "validation_decision.md"
EVIDENCE_MANIFEST_FILE = "evidence_manifest.json"

# Define a StressType enumeration for clarity (conceptually, not a Python Enum object here)
STRESS_TYPES = ['NOISE', 'FEATURE_SHIFT', 'MISSINGNESS', 'OUT_OF_DISTRIBUTION', 'SUBGROUP']

def define_stress_scenario(
      stress_type: str,
    description: str,
    parameters: dict,
    real_world_event: str,
    expected_business_impact: str,
    severity_level: int = 3
) -> dict:
    """
    Defines a single stress scenario with unique ID and structured metadata.
    """
    if stress_type not in STRESS_TYPES:
        raise ValueError(f"Invalid stress_type. Must be one of {STRESS_TYPES}")

    scenario = {
          "scenario_id": str(uuid.uuid4()),
        "stress_type": stress_type,
        "description": description,
        "parameters": parameters,
        "real_world_event": real_world_event,
        "expected_business_impact": expected_business_impact,
        "severity_level": severity_level, # Scale from 1 (low) to 5 (critical)
    }
    return scenario

# List to hold all defined scenarios
stress_scenarios = []

# Scenario 1: Feature Noise Injection (Gaussian)
# Simulating sudden, unpredictable market volatility spikes or data corruption during rapid data ingest.
scenario_1 = define_stress_scenario(
      stress_type="NOISE",
    description="Injects Gaussian noise into 'volatility' feature, mimicking sudden, unforecastable market volatility.",
    parameters={"feature": "volatility", "std_dev_multiplier": 0.5}, # 50% of original std dev
    real_world_event="Flash Crash / Unforeseen Geopolitical Event",
    expected_business_impact="Increased RMSE, potentially underestimating risk due to 'smoothed' predictions, leading to incorrect capital allocation.",
    severity_level=4
)
stress_scenarios.append(scenario_1)

# Scenario 2: Distributional Shift (Mean Shift)
# Simulating a sustained shift in economic indicators, like a prolonged period of high interest rates.
scenario_2 = define_stress_scenario(
      stress_type="FEATURE_SHIFT",
    description="Shifts the mean of the 'interest_rate' feature, simulating a prolonged central bank policy shift.",
    parameters={"feature": "interest_rate", "shift_percentage": 0.20}, # 20% increase
    real_world_event="Interest Rate Shock / Persistent Inflation",
    expected_business_impact="Poor calibration of risk scores as the model operates outside its learned feature distribution, leading to mispricing of financial products.",
    severity_level=5
)
stress_scenarios.append(scenario_2)

# Scenario 3: Missingness Spike
# Simulating failures in data pipelines or extreme market illiquidity leading to missing market data points.
scenario_3 = define_stress_scenario(
      stress_type="MISSINGNESS",
    description="Introduces missing values into 'oil_price' feature, simulating data pipeline failure or market illiquidity.",
    parameters={"feature": "oil_price", "missing_percentage": 0.30}, # 30% missing
    real_world_event="Supply Chain Disruption / Data Feed Outage",
    expected_business_impact="Model errors due to reliance on imputed values, leading to erroneous risk assessments for commodity-sensitive portfolios.",
    severity_level=3
)
stress_scenarios.append(scenario_3)

# Scenario 4: Out-of-Distribution Data (Synthetically Generated)
# Simulating extreme, unprecedented market conditions.
scenario_4 = define_stress_scenario(
      stress_type="OUT_OF_DISTRIBUTION",
    description="Generates synthetic 'market_index' values far outside historical range, mimicking a black swan event.",
    parameters={"feature": "market_index", "scale_factor": 1.5, "base_value": 1478.966135176729}, # Pushing beyond max
    real_world_event="Global Financial Crisis (Black Swan Event)",
    expected_business_impact="Complete breakdown of model predictions, leading to catastrophic losses due to unmitigated exposure to extreme market movements.",
    severity_level=5
)
stress_scenarios.append(scenario_4)


# Save the scenarios to a JSON file (the "Market Stress Scenario Handbook")
with open(STRESS_SCENARIOS_FILE, 'w') as f:
    json.dump(stress_scenarios, f, indent=4)

print(f"Defined {len(stress_scenarios)} stress scenarios and saved them to {STRESS_SCENARIOS_FILE}")
print("\n--- Example Scenario Definition ---")
print(json.dumps(stress_scenarios[0], indent=4))

def generate_synthetic_financial_data(num_samples=1000, random_seed=RANDOM_SEED):
    """
    Generates synthetic financial market data.
    Features: market_index, volatility, interest_rate, oil_price, subgroup_flag.
    Target: future_market_risk_score (continuous, 0-1).
    """
    np.random.seed(random_seed)
    data = pd.DataFrame()

    data['market_index'] = np.random.normal(1000, 150, num_samples)
    data['volatility'] = np.random.normal(20, 5, num_samples)
    data['interest_rate'] = np.random.normal(3, 1, num_samples)
    data['oil_price'] = np.random.normal(70, 10, num_samples)
    data['subgroup_flag'] = np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # e.g., market segment A vs B

    # Introduce some correlations and non-linearity for a realistic risk score
    data['future_market_risk_score'] = (
        0.05 * data['volatility']
        + 0.001 * data['market_index']
        - 0.05 * data['interest_rate']
        + 0.01 * data['oil_price']
        + 0.1 * data['subgroup_flag'] # Subgroup has a small impact
        + np.random.normal(0, 0.05, num_samples)
    )

    # Scale risk score to be roughly between 0 and 1
    min_risk, max_risk = data['future_market_risk_score'].min(), data['future_market_risk_score'].max()
    # Handle the case where min_risk equals max_risk to prevent division by zero
    if max_risk == min_risk:
        data['future_market_risk_score'] = 0.5 # Assign a neutral risk score if no variation
    else:
        data['future_market_risk_score'] = (data['future_market_risk_score'] - min_risk) / (max_risk - min_risk)

    return data

def calculate_ece_from_regressor(y_true, y_pred, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE) for a regressor's output,
    treating continuous predictions as confidences for a binarized true outcome.
    """
    # Define a binary event based on y_true (e.g., actual risk score > threshold)
    risk_event_threshold = y_true.median() # Use median for a balanced binary target
    y_true_binary = (y_true > risk_event_threshold).astype(int)

    # Scale y_pred to be between 0 and 1 to interpret as 'probability of risk'
    min_pred, max_pred = y_pred.min(), y_pred.max()
    if max_pred == min_pred:
        y_pred_proba = np.full_like(y_pred, 0.5)
    else:
        y_pred_proba = (y_pred - min_pred) / (max_pred - min_pred)

    # Use sklearn's calibration_curve to get fraction of positives and mean predicted value per bin
    # We choose 'uniform' strategy to divide the prediction space into equal-width bins
    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_proba, n_bins=n_bins, strategy='uniform')

    ece = 0
    bins = np.linspace(0., 1., n_bins + 1)

    for i in range(n_bins):
        # Filter predictions within the current bin
        # Handle the last bin's upper bound inclusively
        if i == n_bins - 1:
            bin_indices = np.where((y_pred_proba >= bins[i]) & (y_pred_proba <= bins[i+1]))
        else:
            bin_indices = np.where((y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1]))

        # np.where returns a tuple of arrays (one for each dimension). For 1D, it's (array([...]),)
        # We need to extract the actual array of indices: bin_indices[0]
        if len(bin_indices[0]) > 0:
            # Accuracy in bin (proportion of actual positive events)
            # y_true_binary is a Series, so use .iloc for integer position indexing
            accuracy = y_true_binary.iloc[bin_indices[0]].mean()
            # Confidence in bin (average predicted probability)
            # y_pred_proba is a numpy array, can be indexed directly with bin_indices[0]
            confidence = y_pred_proba[bin_indices[0]].mean()
            # Proportion of samples in this bin
            proportion = len(bin_indices[0]) / len(y_true_binary)
            ece += abs(accuracy - confidence) * proportion
    return ece


def evaluate_model_performance(model, X, y, subgroup_column=None):
    """
    Evaluates the model and calculates key performance metrics: RMSE, ECE, Subgroup Delta.
    """
    y_pred = model.predict(X)

    # Overall RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Overall ECE
    ece = calculate_ece_from_regressor(y, y_pred) # y_pred is a numpy array, y is a pandas Series, which calculate_ece_from_regressor handles

    # Subgroup Performance Deltas
    subgroup_deltas = {}
    if subgroup_column and subgroup_column in X.columns:
        subgroups = X[subgroup_column].unique()
        if len(subgroups) > 1:
            subgroup_rmses = {}
            for sg_value in subgroups:
                sg_mask = (X[subgroup_column] == sg_value)
                if sg_mask.sum() > 0:
                    # Ensure consistent indexing: y and y_pred need to be aligned
                    subgroup_y = y.loc[sg_mask.index[sg_mask]] if isinstance(y, pd.Series) else y[sg_mask]
                    subgroup_y_pred = y_pred[sg_mask.values] # Apply boolean mask to numpy array y_pred
                    subgroup_rmses[f'RMSE_Subgroup_{sg_value}'] = np.sqrt(mean_squared_error(subgroup_y, subgroup_y_pred))

            # For simplicity, calculate the absolute difference between the first two subgroups found.
            sorted_subgroup_keys = sorted(subgroup_rmses.keys())
            if len(sorted_subgroup_keys) >= 2:
                delta = abs(subgroup_rmses[sorted_subgroup_keys[0]] - subgroup_rmses[sorted_subgroup_keys[1]])
                subgroup_deltas['Subgroup_RMSE_Delta'] = delta
            else:
                subgroup_deltas['Subgroup_RMSE_Delta'] = 0.0 # Not enough subgroups to compare

    metrics = {
        "RMSE": rmse,
        "ECE": ece,
    }
    metrics.update(subgroup_deltas) # Add subgroup deltas if calculated
    return metrics, y_pred


# 1. Generate Data
print("Generating synthetic financial market data...")
data = generate_synthetic_financial_data(num_samples=2000, random_seed=RANDOM_SEED)
X = data.drop('future_market_risk_score', axis=1)
y = data['future_market_risk_score']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# 3. Train a Proxy Market Risk Forecasting Model (Random Forest Regressor)
print("Training a proxy market risk forecasting model...")
model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

# 4. Evaluate Baseline Performance
print("Evaluating baseline model performance...")
baseline_metrics, y_pred_baseline = evaluate_model_performance(model, X_test, y_test, subgroup_column='subgroup_flag')

print("\n--- Baseline Model Performance ---")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.4f}")

# Store baseline data and metrics globally for later use
global_baseline_metrics = baseline_metrics
global_X_test = X_test
global_y_test = y_test
global_model = model

# Define configurable acceptance thresholds
ACCEPTANCE_THRESHOLDS = {
      'RMSE': 10,  # Max 10% *increase* in RMSE (i.e., degradation_pct must be >= -10)
    'ECE': 5,    # Max 5% *increase* in ECE (i.e., degradation_pct must be >= -5)
    'Subgroup_RMSE_Delta': 15 # Max absolute delta increase in subgroup RMSE (not pct)
}

def calculate_degradation_and_status(
      baseline_metrics: dict,
    stressed_results: list,
    acceptance_thresholds: dict
) -> list:
    """
    Calculates degradation percentage and determines pass/fail status for each stressed result.
    """
    final_results = []
    for res in stressed_results:
        metric_name = res['metric_name']
        stressed_value = res['stressed_value']
        baseline_value = baseline_metrics.get(metric_name)

        if baseline_value is None:
            print(f"Warning: Baseline value for {metric_name} not found. Skipping degradation calculation for this metric.")
            res.update({
                  "baseline_value": None,
                "degradation_pct": None,
                "threshold": None,
                "status": "N/A"
            })
            final_results.append(res)
            continue

        # Calculate degradation percentage using the specified formula
        # degradation_pct = (baseline - stressed) / baseline * 100
        degradation_pct = ((baseline_value - stressed_value) / baseline_value) * 100 if baseline_value != 0 else float('inf')

        # Determine pass/fail status based on metric type and thresholds
        status = "PASS"
        threshold = acceptance_thresholds.get(metric_name)

        if threshold is not None:
            if metric_name in ['RMSE', 'ECE']: # Lower is better. Negative degradation_pct means stressed value is higher (worse).
                # Model passes if the *increase* (magnitude of negative degradation_pct) is within threshold
                if degradation_pct < -threshold: # If actual stressed value is more than X% higher than baseline
                    status = "FAIL"
            elif metric_name == 'Subgroup_RMSE_Delta': # Lower is better for delta. Compare absolute values.
                # Threshold for Subgroup_RMSE_Delta is an absolute value, not a percentage increase
                # So we check if the stressed delta exceeds the baseline delta by more than the tolerance
                if stressed_value > (baseline_value + threshold): # If stressed delta is greater than baseline + tolerance
                     status = "FAIL"
                degradation_pct = ((stressed_value - baseline_value) / baseline_value) * 100 if baseline_value != 0 else float('inf') # Still calculate pct for display
            # Add other metric types (e.g., AUC where higher is better) here
            # elif metric_name == 'AUC': # Higher is better. Positive degradation_pct means stressed value is lower (worse).
            #    if degradation_pct > threshold: # If actual stressed value is more than X% lower than baseline
            #        status = "FAIL"
        else:
            status = "N/A - No Threshold" # No threshold defined for this metric

        res.update({
              "baseline_value": baseline_value,
            "stressed_value": stressed_value,
            "degradation_pct": degradation_pct,
            "threshold": threshold, # Store the defined threshold for context
            "status": status
        })
        final_results.append(res)
    return final_results


def apply_noise(data: pd.DataFrame, feature: str, std_dev_multiplier: float, random_state: int) -> pd.DataFrame:
    """Adds Gaussian noise to a specified feature."""
    data_copy = data.copy()
    feature_std = data_copy[feature].std()
    noise_magnitude = std_dev_multiplier * feature_std
    rng = np.random.RandomState(random_state)
    data_copy[feature] = data_copy[feature] + rng.normal(0, noise_magnitude, data_copy.shape[0])
    return data_copy


def apply_shift(data: pd.DataFrame, feature: str, shift_percentage: float) -> pd.DataFrame:
    """
    Shifts the mean of a specified feature by a percentage.
    """
    data_copy = data.copy()
    data_copy[feature] = data_copy[feature] * (1 + shift_percentage)
    return data_copy


def apply_missingness(data: pd.DataFrame, feature: str, missing_percentage: float, random_state: int) -> pd.DataFrame:
    """
    Introduces missing values (NaN) into a specified feature.
    """
    data_copy = data.copy()
    num_missing = int(len(data_copy) * missing_percentage)
    rng = np.random.RandomState(random_state)
    indices_to_na = rng.choice(data_copy.index, size=num_missing, replace=False)
    data_copy.loc[indices_to_na, feature] = np.nan
    return data_copy


def apply_out_of_distribution(data: pd.DataFrame, feature: str, scale_factor: float, base_value: float, random_state: int) -> pd.DataFrame:
    """
    Generates synthetic values for a feature far outside its historical range.
    """
    data_copy = data.copy()
    rng = np.random.RandomState(random_state)
    # Generate values significantly higher than historical max
    data_copy[feature] = base_value * (1 + rng.uniform(scale_factor - 0.1, scale_factor + 0.1, data_copy.shape[0]))
    return data_copy


def execute_stress_scenario(model, scenario: dict, X_baseline: pd.DataFrame, y_baseline: pd.Series, subgroup_column: str, random_seed: int) -> dict:
    """
    Applies a specified stress scenario to the baseline data, re-evaluates the model,
    and returns the stressed performance metrics.
    """
    X_stressed = X_baseline.copy()  # Start with a clean copy of baseline data

    # Apply the specific perturbation based on stress_type
    if scenario['stress_type'] == "NOISE":
        X_stressed = apply_noise(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['std_dev_multiplier'],
            random_seed
        )
    elif scenario['stress_type'] == "FEATURE_SHIFT":
        X_stressed = apply_shift(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['shift_percentage']
        )
    elif scenario['stress_type'] == "MISSINGNESS":
        X_stressed = apply_missingness(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['missing_percentage'],
            random_seed
        )
        # Handle missing values: Impute with the mean of the *original* baseline feature to simulate real-world imputation
        imputation_value = X_baseline[scenario['parameters']['feature']].mean()
        X_stressed[scenario['parameters']['feature']] = X_stressed[scenario['parameters']['feature']].fillna(imputation_value)
    elif scenario['stress_type'] == "OUT_OF_DISTRIBUTION":
        X_stressed = apply_out_of_distribution(
            X_stressed,
            scenario['parameters']['feature'],
            scenario['parameters']['scale_factor'],
            scenario['parameters']['base_value'],
            random_seed
        )
    # Add other stress types here if implemented

    # Re-evaluate the model on the stressed data
    stressed_metrics, _ = evaluate_model_performance(model, X_stressed, y_baseline, subgroup_column=subgroup_column)

    # Format the results for RobustnessResult schema
    results = []
    for metric_name, stressed_value in stressed_metrics.items():
        results.append({
            "scenario_id": scenario['scenario_id'],
            "metric_name": metric_name,
            "stressed_value": stressed_value,
            "description": scenario['description'],
            "real_world_event": scenario['real_world_event'],
            "expected_business_impact": scenario['expected_business_impact']
            # degradation_pct, baseline_value, threshold, status will be added later
        })
    return results


# Load defined scenarios
with open(STRESS_SCENARIOS_FILE, 'r') as f:
    loaded_scenarios = json.load(f)

all_stressed_results = []
print(f"Executing {len(loaded_scenarios)} stress scenarios...")

for i, scenario in enumerate(loaded_scenarios):
    print(f"[{i+1}/{len(loaded_scenarios)}] Executing scenario: {scenario['real_world_event']}")
    stressed_output = execute_stress_scenario(
        model=global_model,
        scenario=scenario,
        X_baseline=global_X_test,
        y_baseline=global_y_test,
        subgroup_column='subgroup_flag',
        random_seed=RANDOM_SEED + i  # Vary seed slightly for different noise patterns if applicable, but keep overall deterministic
    )
    all_stressed_results.extend(stressed_output)

# Save raw stressed results
with open(ROBUSTNESS_RESULTS_FILE, 'w') as f:
    json.dump(all_stressed_results, f, indent=4)

print(f"\nAll stress scenarios executed. Raw results saved to {ROBUSTNESS_RESULTS_FILE}")
print("\n--- Example Stressed Result Entry (first scenario, first metric) ---")
print(json.dumps(all_stressed_results[0], indent=4))

# Load raw stressed results
with open(ROBUSTNESS_RESULTS_FILE, 'r') as f:
    raw_stressed_results = json.load(f)

# Calculate degradation and status
print("Calculating performance degradation and applying acceptance thresholds...")
performance_degradation_results = calculate_degradation_and_status(
      global_baseline_metrics, raw_stressed_results, ACCEPTANCE_THRESHOLDS
)

# Store the final performance degradation report
with open(PERFORMANCE_DEGRADATION_REPORT_FILE, 'w') as f:
    json.dump(performance_degradation_results, f, indent=4)

print(f"\nPerformance degradation report generated and saved to {PERFORMANCE_DEGRADATION_REPORT_FILE}")

# Display a summary of degradation results
print("\n--- Summary of Performance Degradation Results ---")
for res in performance_degradation_results:
    if res['status'] == "FAIL":
        print(f"FAIL: Scenario {res['scenario_id'][:8]}... - Metric {res['metric_name']}: "
              f"Baseline={res['baseline_value']:.4f}, Stressed={res['stressed_value']:.4f}, "
              f"Degradation={res['degradation_pct']:.2f}%, Threshold={res['threshold']}")
    elif res['status'] == "PASS":
         print(f"PASS: Scenario {res['scenario_id'][:8]}... - Metric {res['metric_name']}: "
              f"Baseline={res['baseline_value']:.4f}, Stressed={res['stressed_value']:.4f}, "
              f"Degradation={res['degradation_pct']:.2f}%, Threshold={res['threshold']}")

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(performance_degradation_results)

def plot_metric_comparison(df: pd.DataFrame, metric_name: str):
    """
    Generates a bar chart comparing baseline and stressed values for a specific metric
    across all scenarios, highlighting pass/fail status.
    """
    metric_df = df[df['metric_name'] == metric_name].copy()
    if metric_df.empty:
        print(f"No data available for metric: {metric_name}")
        return

    # Prepare data for plotting: baseline is constant for a given metric
    baseline_value = metric_df['baseline_value'].iloc[0] # Assumes baseline is consistent across scenarios for a metric

    # Add a 'Baseline' row for clear comparison in the plot
    plot_data = pd.DataFrame({
          'Scenario': metric_df['real_world_event'],
        'Stressed Value': metric_df['stressed_value'],
        'Status': metric_df['status']
    })

    plt.figure(figsize=(12, 6))

    # Plot stressed values
    sns.barplot(x='Scenario', y='Stressed Value', hue='Status', data=plot_data,
                palette={'PASS': 'green', 'FAIL': 'red', 'N/A': 'gray'}, dodge=False)

    # Add baseline reference line
    plt.axhline(y=baseline_value, color='blue', linestyle='--', label='Baseline Value')

    plt.title(f'Baseline vs. Stressed {metric_name} Across Scenarios', fontsize=16)
    plt.ylabel(f'{metric_name} Value', fontsize=12)
    plt.xlabel('Real-World Event / Scenario', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Validation Status', fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_degradation_percentages(df: pd.DataFrame, metric_name: str):
    """
    Generates a bar chart showing degradation percentages for a specific metric
    across all scenarios, highlighting pass/fail status and threshold.
    """
    metric_df = df[df['metric_name'] == metric_name].copy()
    if metric_df.empty:
        print(f"No data available for metric: {metric_name}")
        return

    plt.figure(figsize=(12, 6))

    # Define a palette mapping status directly to colors
    status_palette = {'PASS': 'green', 'FAIL': 'red', 'N/A': 'gray'}

    # Use 'status' as the hue variable and the defined palette
    # Set dodge=False to ensure bars for the same x-value are not dodged/grouped.
    # Since each real_world_event has only one status, this will simply color the bars.
    sns.barplot(x='real_world_event', y='degradation_pct', hue='status', data=metric_df,
                palette=status_palette, dodge=False)

    # Add threshold lines
    threshold = metric_df['threshold'].iloc[0] if not metric_df.empty else None
    if threshold is not None:
        # For RMSE/ECE, negative degradation_pct means stressed value is higher (worse).
        # So threshold applies to the negative side.
        plt.axhline(y=-threshold, color='blue', linestyle='--', label=f'Threshold (-{threshold}%)')

    plt.title(f'{metric_name} Degradation Percentage Across Scenarios', fontsize=16)
    plt.ylabel('Degradation Percentage (%)', fontsize=12)
    plt.xlabel('Real-World Event / Scenario', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Validation Status', fontsize=10, title_fontsize=12) # Add legend for status
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_feature_distribution_shift(baseline_data: pd.DataFrame, stressed_data: pd.DataFrame, feature_name: str, scenario_description: str):
    """
    Visualizes the distribution shift for a specific feature between baseline and stressed data.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(baseline_data[feature_name], kde=True, color='blue', label='Baseline', stat='density', alpha=0.6)
    sns.histplot(stressed_data[feature_name], kde=True, color='red', label='Stressed', stat='density', alpha=0.6)
    plt.title(f'Distribution Shift for {feature_name}\nUnder Scenario: {scenario_description}', fontsize=14)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Generate plots for each key metric
print("Generating visualizations for performance degradation...")
plot_metric_comparison(results_df, 'RMSE')
plot_degradation_percentages(results_df, 'RMSE')

plot_metric_comparison(results_df, 'ECE')
plot_degradation_percentages(results_df, 'ECE')

plot_metric_comparison(results_df, 'Subgroup_RMSE_Delta')
plot_degradation_percentages(results_df, 'Subgroup_RMSE_Delta')

# Example: Visualize distribution shift for a specific scenario
# Re-run a specific scenario to get the stressed data for plotting
# Let's visualize the 'FEATURE_SHIFT' scenario on 'interest_rate'
shift_scenario = next(s for s in loaded_scenarios if s['stress_type'] == 'FEATURE_SHIFT')
X_stressed_shift = global_X_test.copy()
X_stressed_shift = apply_shift(X_stressed_shift, shift_scenario['parameters']['feature'], shift_scenario['parameters']['shift_percentage'])
plot_feature_distribution_shift(global_X_test, X_stressed_shift,
                                shift_scenario['parameters']['feature'],
                                shift_scenario['real_world_event'])

# Let's visualize the 'NOISE' scenario on 'volatility'
noise_scenario = next(s for s in loaded_scenarios if s['stress_type'] == 'NOISE')
X_stressed_noise = global_X_test.copy()
X_stressed_noise = apply_noise(X_stressed_noise, noise_scenario['parameters']['feature'],
                               noise_scenario['parameters']['std_dev_multiplier'], RANDOM_SEED)
plot_feature_distribution_shift(global_X_test, X_stressed_noise,
                                noise_scenario['parameters']['feature'],
                                noise_scenario['real_world_event'])

# Let's visualize the 'MISSINGNESS' scenario on 'oil_price'
missingness_scenario = next(s for s in loaded_scenarios if s['stress_type'] == 'MISSINGNESS')
X_stressed_missingness = global_X_test.copy()
X_stressed_missingness = apply_missingness(X_stressed_missingness, missingness_scenario['parameters']['feature'],
                                       missingness_scenario['parameters']['missing_percentage'], RANDOM_SEED)
# Impute missing values for the plot, as would happen before model inference
imputation_value = global_X_test[missingness_scenario['parameters']['feature']].mean()
X_stressed_missingness[missingness_scenario['parameters']['feature']] = X_stressed_missingness[missingness_scenario['parameters']['feature']].fillna(imputation_value)
plot_feature_distribution_shift(global_X_test, X_stressed_missingness,
                                missingness_scenario['parameters']['feature'],
                                missingness_scenario['real_world_event'])
import datetime
import hashlib
import json
import os
import uuid

# Define decision outcomes
DECISION_OUTCOMES = ['DEPLOY', 'RESTRICT', 'REDESIGN']

def generate_validation_decision(
    degradation_results: list,
    acceptance_thresholds: dict,
    output_filepath: str
) -> None:
    """
    Analyzes degradation results to determine an overall validation decision
    and generates a detailed markdown report.
    """
    total_scenarios = len(set(res['scenario_id'] for res in degradation_results))
    failed_metrics = [res for res in degradation_results if res['status'] == 'FAIL']

    decision = "DEPLOY"
    rationale = []
    key_failures = []
    recommended_actions = []

    if failed_metrics:
        decision = "REDESIGN"
        rationale.append("The model failed one or more critical robustness acceptance thresholds under stress scenarios.")
        recommended_actions.append("Initiate model redesign to address identified vulnerabilities.")

        failed_scenario_ids = set()
        for fail in failed_metrics:
            failed_scenario_ids.add(fail['scenario_id'])
            key_failures.append(
                f"- Scenario '{fail['real_world_event']}' (ID: {fail['scenario_id'][:8]}...) for metric '{fail['metric_name']}'. "
                f"Baseline: {fail['baseline_value']:.4f}, Stressed: {fail['stressed_value']:.4f}, "
                f"Degradation: {fail['degradation_pct']:.2f}%, Threshold: {fail['threshold']}%.")

        # If there are only a few minor failures, might be RESTRICT.
        # For this exercise, any fail leads to REDESIGN for simplicity given the critical model nature.
        # Heuristic for 'RESTRICT' vs 'REDESIGN'
        if len(failed_scenario_ids) < total_scenarios / 2 and len(failed_metrics) < 3:
            decision = "RESTRICT"
            rationale[-1] = "The model exhibited minor failures in some robustness acceptance thresholds under stress scenarios."
            recommended_actions[-1] = "Restrict model usage to specific, low-risk contexts. Investigate failures and plan for redesign or enhanced monitoring."

    else:
        rationale.append("The model passed all defined robustness acceptance thresholds under stress scenarios.")
        recommended_actions.append("Proceed with model deployment, ensuring continuous monitoring for concept drift and data quality.")

    # Generate Markdown content
    markdown_content = f"""# Market Risk Forecasting Model Validation Decision

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Team/User:** AI Reliability Team
**Model Name:** ML-based Market Risk Forecasting Model
**Overall Decision:** **{decision}**

## 1. Rationale

{' '.join(rationale)}

## 2. Key Failures (if any)

"""
    if key_failures:
        markdown_content += "\n".join(key_failures)
    else:
        markdown_content += "No critical failures identified against defined thresholds."

    markdown_content += f"""

## 3. Recommended Actions

* {'\n* '.join(recommended_actions)}

## 4. Acceptance Thresholds Used

"""
    for metric, threshold in acceptance_thresholds.items():
        markdown_content += f"* {metric}: {threshold}% (or absolute delta for Subgroup_RMSE_Delta)\n"

    markdown_content += f"""
## 5. Summary of Scenarios Executed

A total of {total_scenarios} stress scenarios were executed, designed to simulate various real-world market events and data disruptions.

---
*This document serves as the formal validation decision for the ML-based Market Risk Forecasting Model, providing justification and actionable recommendations based on the robustness assessment.*
"""

    with open(output_filepath, 'w') as f:
        f.write(markdown_content)
    print(f"Validation decision report generated and saved to {output_filepath}")


def generate_evidence_manifest(
    run_id: str,
    output_filepath: str,
    input_files: list,
    output_files: list,
    app_version: str = "1.0.0"
) -> None:
    """
    Generates an evidence manifest JSON file for traceability.
    Hashes file contents to ensure data integrity.
    """
    def hash_file(filepath):
        if not os.path.exists(filepath):
            return "FILE_NOT_FOUND"
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(4096)  # Read in 4KB chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    manifest = {
        "run_id": run_id,
        "generated_at": datetime.datetime.now().isoformat(),
        "team_or_user": "AI Reliability Team",
        "app_version": app_version,
        "inputs_hash": {f: hash_file(f) for f in input_files},
        "outputs_hash": {f: hash_file(f) for f in output_files},
        "artifacts": input_files + output_files
    }

    with open(output_filepath, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Evidence manifest generated and saved to {output_filepath}")


# --- Generate Validation Decision ---
print("Generating validation decision report...")
generate_validation_decision(performance_degradation_results, ACCEPTANCE_THRESHOLDS, VALIDATION_DECISION_FILE)

# --- Generate Evidence Manifest ---
current_run_id = str(uuid.uuid4())
input_artifacts = [STRESS_SCENARIOS_FILE]
output_artifacts = [ROBUSTNESS_RESULTS_FILE, PERFORMANCE_DEGRADATION_REPORT_FILE, VALIDATION_DECISION_FILE]

print("Generating evidence manifest...")
generate_evidence_manifest(current_run_id, EVIDENCE_MANIFEST_FILE, input_artifacts, output_artifacts)

print("\n--- Validation and Evidence Generation Complete ---")
print(f"All audit-ready artifacts have been created: {STRESS_SCENARIOS_FILE}, {ROBUSTNESS_RESULTS_FILE}, {PERFORMANCE_DEGRADATION_REPORT_FILE}, {VALIDATION_DECISION_FILE}, {EVIDENCE_MANIFEST_FILE}")