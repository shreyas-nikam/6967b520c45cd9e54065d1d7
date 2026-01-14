id: 6967b520c45cd9e54065d1d7_user_guide
summary: Robustness & Functional Validation Under Stress User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Validating AI Model Robustness Under Market Stress

## 1. Introduction to QuLab and Setting the Baseline
Duration: 00:05:00

Welcome to **QuLab: Robustness & Functional Validation Under Stress**! In today's rapidly evolving financial markets, Machine Learning (ML) models are at the heart of critical decisions, from predicting market risk to optimizing trading strategies. However, these models are often developed and tested under ideal conditions, making them vulnerable to unexpected shifts and stresses in real-world market environments.

As an **AI Reliability Engineer**, your crucial role is to ensure these models perform reliably, even when markets are turbulent. This application provides a comprehensive framework to rigorously test your market risk forecasting models against various stress scenarios, assess their robustness, and ensure they meet predefined performance standards.

In this first step, you will:
*   **Establish a Baseline**: Understand the model's performance under normal, unstressed conditions. This acts as your reference point.
*   **Define Key Performance Metrics**: Familiarize yourself with the metrics used to evaluate the model's predictive accuracy and reliability.
*   **Configure Acceptance Thresholds**: Set the boundaries for what constitutes an 'acceptable' level of performance degradation when the model is under stress.

<aside class="positive">
<b>Why is this important?</b> Establishing a solid baseline and clear thresholds is like setting the foundation and guardrails for a building. Without them, you can't truly measure the impact of external forces or determine if the structure is still safe.
</aside>

### Understanding Key Performance Metrics

We focus on three critical metrics for evaluating our market risk forecasting model:

1.  **Root Mean Squared Error (RMSE)** for predictive accuracy:
    This metric tells us how close our model's predictions are to the actual values. A lower RMSE indicates better predictive accuracy.
    $$ RMSE = √{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $$
    where $N$ is the number of observations, $y_i$ is the actual risk score, and $\hat{y}_i$ is the predicted risk score.

2.  **Expected Calibration Error (ECE)** for model calibration:
    Calibration measures how well a model's predicted probabilities align with actual outcomes. For instance, if a model predicts a 70% chance of a high-risk event, then a high-risk event should occur roughly 70% of the time among all instances where the model predicts 70%. A lower ECE signifies better model calibration, meaning the model's confidence in its predictions is well-aligned with reality.
    $$ ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)| $$
    Here, the prediction range is divided into $M$ bins. $|B_m|$ is the number of samples in bin $m$, $N$ is the total samples, $acc(B_m)$ is the accuracy (proportion of actual high risk events) in bin $m$, and $conf(B_m)$ is the average predicted risk score (confidence) in bin $m$.

3.  **Subgroup Performance Deltas** for identifying bias:
    This evaluates RMSE for distinct subgroups within your data (e.g., 'high_cap_stocks' vs. 'mid_cap_stocks'). We are interested in the *difference* in RMSE between these subgroups. A large delta suggests that the model might perform disparately across different segments of the market, potentially indicating bias or unfairness.

### Initializing the Model and Baseline

To begin, click the "Generate Baseline Model & Metrics" button. The application will:
*   Generate synthetic financial data.
*   Train a RandomForestRegressor model on this data.
*   Calculate the baseline RMSE, ECE, and Subgroup RMSE Delta.

```
Click the "Generate Baseline Model & Metrics" button.
```

Once the baseline is established, you will see a table displaying these metrics.

### Configuring Acceptance Thresholds

Below the baseline metrics, you'll find inputs to configure your acceptance thresholds. These thresholds define the maximum allowable degradation for each metric before a scenario is flagged as a 'FAIL'.

*   **Max % Increase in RMSE**: For RMSE and ECE (where lower is better), this threshold represents the maximum acceptable percentage increase from the baseline. For example, a 10% threshold means an RMSE increase of up to 10% is acceptable.
*   **Max % Increase in ECE**: Similar to RMSE, this is the maximum acceptable percentage increase for ECE.
*   **Max Absolute Increase in Subgroup RMSE Delta**: For Subgroup RMSE Delta, this is an absolute value. If the stressed delta exceeds `(baseline_delta + threshold)`, it fails.

Adjust these values to reflect your organization's risk tolerance.

## 2. Crafting Market Stress Scenarios
Duration: 00:07:00

Now that your baseline is set, it's time to put on your **AI Reliability Engineer** hat and define the market stress scenarios. This is where you translate abstract market concerns into concrete, measurable data perturbations. Think of this as creating your "Market Stress Scenario Handbook." Each scenario is designed to be repeatable and produce a measurable performance change in your model.

<aside class="positive">
<b>The goal here is proactive risk management.</b> By systematically defining and testing against potential market stresses, you can identify and mitigate model vulnerabilities *before* they lead to real-world financial losses or poor decisions.
</aside>

### Understanding Stress Types

The application offers several types of data perturbations, each mimicking a different kind of market stress:

*   **NOISE**: Represents increased market volatility or data measurement errors. Parameters include a `Standard Deviation Multiplier`, which dictates how much random noise is added to a selected feature.
*   **FEATURE_SHIFT**: Simulates a sustained shift in a market factor, like a sudden policy change or a new economic regime. The `Shift Percentage` determines the magnitude of this change.
*   **MISSINGNESS**: Models data feed interruptions or corrupted data. The `Missing Percentage` controls how many values for a specific feature are removed.
*   **OUT_OF_DISTRIBUTION**: Represents extreme, unprecedented market events that push feature values far beyond historical norms. `Scale Factor` and `Base Value` define how values are stretched or shifted to simulate these extremes.

### Defining a New Scenario

1.  **Select Stress Type**: Choose one of the stress types from the dropdown.
2.  **Feature to Stress**: Select which input feature of your model will be affected by this stress. This is crucial for targeting specific market factors.
3.  **Parameters**: Depending on your chosen stress type, adjust the relevant numerical parameters (e.g., `Standard Deviation Multiplier`, `Shift Percentage`).
4.  **Descriptive Fields**:
    *   **Scenario Description**: Explain the technical nature of the stress (e.g., "Adding 20% noise to volatility feature").
    *   **Real-World Market Event**: Link this technical stress to a plausible market event (e.g., "Flash Crash," "Interest Rate Shock," "Geopolitical Instability").
    *   **Expected Business Impact**: Describe the potential consequences for your business if the model fails under this scenario (e.g., "Increased RMSE leading to mispricing of products," "Delayed detection of risk," "Regulatory non-compliance").
    *   **Severity Level (1-5)**: Assign a subjective severity from 1 (minor) to 5 (critical) to help prioritize.

After filling out the details, click "Add Scenario" to add it to your list.

### Managing Scenarios

*   **Current Stress Scenarios**: View a summary of all the scenarios you've defined.
*   **Clear All Scenarios**: If you wish to start over, you can clear all defined scenarios.
*   **Save Scenarios to JSON**: It's good practice to save your defined scenarios. This will export them into a JSON file for future use and auditability.

```
<aside class="negative">
You must initialize the baseline model in "1. Baseline & Configuration" before you can define scenarios, as the application needs to know which features are available to stress.
</aside>
```

## 3. Executing Stress Tests
Duration: 00:03:00

With your stress scenarios meticulously defined, it's time for the true test: running the model through these simulated market turmoils. This step involves systematically applying each defined stress to your model's test data and then re-evaluating its performance.

<aside class="positive">
<b>Reproducibility is key.</b> The application ensures that each stress test is executed deterministically. This means if you run the same scenario again, you'll get the exact same results, which is vital for auditability and debugging.
</aside>

### How Stress Tests Work

For each scenario you defined:
1.  The original, unstressed test dataset (`X_test`) is taken as a starting point.
2.  The specific data perturbation (e.g., adding noise, shifting values, introducing missingness) defined in the scenario is applied to the selected feature(s) of this dataset, creating a new "stressed" dataset (`X_stressed`).
3.  Your trained ML model then makes predictions on this `X_stressed` data.
4.  The model's performance (RMSE, ECE, Subgroup RMSE Delta) is re-calculated using these predictions and the actual `y_test` values.
5.  Finally, the performance of the model on the stressed data is compared against its baseline performance, and the percentage degradation for each metric is calculated. This degradation is then checked against your defined acceptance thresholds to determine a 'PASS' or 'FAIL' status for each metric under that scenario.

### Running the Scenarios

1.  Review the "Scenarios to be Executed" table to ensure all desired scenarios are present.
2.  Click the "Run All Scenarios" button. A progress bar will indicate the execution status.

```
Click the "Run All Scenarios" button.
```

Once completed, the application will display a summary table of the "Results Summary," showing the baseline value, stressed value, percentage degradation, and the pass/fail status for each metric under each scenario.

<aside class="negative">
You cannot execute tests if you haven't initialized the baseline model or defined any stress scenarios. Please complete the previous steps first.
</aside>

## 4. Interpreting Results and Visualizing Vulnerabilities
Duration: 00:07:00

Raw data tables are important for precision, but understanding complex model behavior under stress often requires intuitive visualizations. This section provides a dashboard to help you effectively communicate the market risk model's vulnerabilities to both technical and non-technical stakeholders. These charts are essential for grasping the relevance and severity of potential model failures at a glance.

### Performance Degradation Summary

First, review the "Performance Degradation Summary" table. This table is a crucial overview, consolidating the results of all stress tests. It shows:
*   `scenario_id`: Unique identifier for each stress scenario.
*   `metric_name`: The performance metric being evaluated (RMSE, ECE, Subgroup RMSE Delta).
*   `baseline_value`: The model's performance on unstressed data.
*   `stressed_value`: The model's performance on data under this specific stress.
*   `degradation_pct`: The percentage change (degradation or improvement) from baseline.
*   `status`: Whether the metric passed or failed its acceptance threshold for this scenario.

### Visualizing Metric Performance and Degradation

The dashboard provides several plots to visualize the impact of stress:

*   **RMSE Performance & Degradation**:
    *   The first plot compares baseline RMSE to stressed RMSE across all scenarios. You can quickly see which scenarios lead to higher RMSE.
    *   The second plot shows the percentage degradation of RMSE for each scenario, making it easy to identify which scenarios cause the most significant drop in predictive accuracy.
*   **ECE Performance & Degradation**:
    *   Similar plots for ECE help you understand how different stresses affect the model's calibration and reliability of its probability predictions.
*   **Subgroup RMSE Delta Performance & Degradation**:
    *   These plots visualize the impact of stress on the fairness or equitable performance across different subgroups, highlighting where potential biases might emerge or worsen.

<aside class="positive">
Look for scenarios where a metric transitions from 'PASS' to 'FAIL'. These are your most critical vulnerabilities and require further investigation. High degradation percentages are also immediate red flags.
</aside>

### Feature Distribution Shifts

Understanding *why* the model's performance degraded often requires looking at the input data itself. The "Feature Distribution Shifts" section helps you visualize exactly how a chosen feature's distribution changes under a selected stress scenario, compared to its baseline distribution.

1.  **Select Scenario for Distribution Plot**: Choose one of your defined scenarios from the dropdown. The plot will show the impact of this specific scenario.
2.  **Select Feature to Visualize**: Choose the particular feature whose distribution you want to inspect.

The plot will display two histograms: one for the baseline distribution of the feature and one for its stressed distribution.

<aside class="positive">
<b>Diagnosing issues:</b> A significant shift in feature distribution (e.g., new peaks, wider spread, values outside the original range) suggests that the stress pushed the model's inputs into unfamiliar territory, which can explain poor performance. This is crucial for understanding the root cause of failures.
</aside>

## 5. Finalizing Validation and Archiving Evidence
Duration: 00:03:00

The final and arguably most critical step in the stress testing process is to formalize the validation outcome and generate comprehensive, audit-ready artifacts. This ensures that all findings are synthesized into a clear decision on the model's readiness for production, and that a traceable, immutable record of the entire assessment is created.

<aside class="positive">
This step ensures **accountability** and **compliance**. Having well-documented evidence is paramount for satisfying regulatory requirements and providing transparency to internal auditors and stakeholders.
</aside>

### Generating the Validation Decision

Click the "Generate Validation Decision" button. The application will:
*   Analyze the `performance_degradation_results` against the `acceptance_thresholds`.
*   Synthesize the overall pass/fail status.
*   Generate a Markdown report (`validation_decision.md`) detailing the decision, rationale, key failures, and recommended actions.
*   Display this report directly in the application.

```
Click the "Generate Validation Decision" button.
```

Review the generated decision. This document is a critical output that formally states whether the model is deemed robust enough for its intended purpose under market stress conditions.

### Generating the Evidence Manifest

Click the "Generate Evidence Manifest" button. The application will:
*   Create an `evidence_manifest.json` file.
*   This manifest acts as a table of contents and checksum log for all generated artifacts, including your stress scenarios, raw results, degradation reports, and the validation decision.
*   It provides a verifiable record of all files associated with this specific validation run, crucial for audit trails.

```
Click the "Generate Evidence Manifest" button.
```

### Downloading All Artifacts

Finally, to create a complete, offline record of your validation exercise, click the "Download All Artifacts as ZIP" button. This will package all the generated files into a single ZIP archive, ready for storage or sharing.

```
Click the "Download All Artifacts as ZIP" button.
```

This ZIP file contains all the necessary evidence to demonstrate the thoroughness of your model's robustness validation, providing a complete audit trail for compliance and governance.

Congratulations! You have successfully guided a market risk forecasting model through a series of rigorous stress tests using QuLab. You've established a baseline, defined critical market scenarios, executed tests, visualized the impact of stress, and generated comprehensive validation artifacts. This process is fundamental to ensuring the reliability and trustworthiness of AI models in financial applications.
