# Low-Level Design (LLD)

This document details the low-level design of the Airbnb Destination Prediction machine learning pipeline, including feature inventory, preprocessing strategies, modeling logic, and validation.

---
 
## 1. Feature Inventory

The project extracts predictions from a diverse set of engineered features derived from raw user demographic data and clickstream sessions.

### 1.1 Core Session Aggregations
- **`total_actions`**: Raw count of actions performed in the sessions.
- **`unique_actions`**, **`unique_action_types`**, **`unique_devices`**: Cardinality counts of distinct behaviors and hardware footprint.
- **`top_action`**: The most frequent action performed by the user.
- **`first_action`** / **`last_action`**: Sequential behavioral anchors.
- **`most_used_device`**: The device type predominantly used by the user.

### 1.2 Time & Duration Features
- **`session_duration`**: Total active time from the first to the last click (in seconds).
- **`avg_time_between_actions`**: Mean time difference between consecutive clicks.
- **`recency_gap`**: Distance from the global maximum session timestamp to the user's last session, catching dormancy.
- **`session_count_per_day`**: The velocity of a user's activities over the span of their interaction window.
- **`total_secs_elapsed`** / **`avg_secs_elapsed`**: Direct aggregations from the raw session durations.
- **Time of Day flags (One-hot mapped)**: `tod_morning`, `tod_afternoon`, `tod_evening`, `tod_night`.

### 1.3 Behavioral Complexity Metrics
- **`action_entropy`**: A vectorized entropy calculation ($H = -p \log_2 p$) that captures the complexity and predictability of a user's actions.
- **`actions_per_second`**: A measure of session speed and urgency.
- **`action_diversity_ratio`**: Ratio of unique actions against total actions.
- **`device_diversity_ratio`**: Number of unique devices relative to total actions.
- **`high_activity_flag`**: Boolean indicator if a user has performed > 50 actions.

---

## 2. Preprocessing Plan

The pipeline is entirely modularized using structured Python classes to handle discrete data transformation steps:

- **Data Loading & Cleaning (`DataLoader`, `DataCleaner`)**: Parsing raw files and enforcing base data schemas.
- **Session Aggregation (`SessionAggregator`)**: Grouping session event logs user-wise to output a unified flat feature vector for predictive modeling. Handling `user_id`s that completely lack session logs with default/unknown behavior mappings.
- **Feature Encoding (`FeatureEncoder`)**:
  - **Ordinal Encoding**: Applied to low-cardinality categorical features (e.g., standard demographic features), unknown values mapped dynamically to -1.
  - **Frequency Encoding**: Applied to high-cardinality columns, converting strings to floats by mapping discrete categories to their observed occurrence probability in the training set.
  - Target variables (`country_destination`) are cleanly excluded from transformation logic.
- **Null Value Treatment**:
  - *Categorical/String variables*: Explicit missing values and anomalies (e.g., `-unknown-`) are consistently imputed with `"Unknown"` or `"None"`.
  - *Age variable*: Outliers (ages < 15 or > 90) are constrained to missing (`NaN`), and subsequently all missing ages are imputed using the **training set median**.
  - *No-Session Users*: Users lacking any clickstream logs are demarcated with a dedicated `no_session` binary feature. Their session numerics are imputed with `0`.
  - *Fallback Imputation*: Any residual NaN values left over prior to pipeline finalization are defensively mapped via `.fillna(0)` for numbers and `.fillna("None")` for strings.
- **Class Imbalance Mitigation**: Implementation of **SMOTE** (Synthetic Minority Over-sampling Technique) embedded natively inside model training pipelines to upsample minority destination classes prior to gradient boosting.

---

## 3. Modeling Hypotheses & Deployment

To handle the highly non-linear nature of behavioral data and severity of the class imbalances, tree-based ensemble methods are designated as the core predictive models.

### Models
1. **LightGBM (`lightgbm.LGBMClassifier`)**:
    - **Hypothesis**: LightGBM's leaf-wise tree growth is natively faster at building decision structures without over-saturating computation, performing efficiently given heterogeneous data shapes containing sparse properties.
    - Configuration: `objective = 'multiclass'`, explicitly limiting depth constraints to generalize.
2. **XGBoost (`xgboost.XGBClassifier`)**:
    - **Hypothesis**: Functions as a robust comparison standard. XGBoost's precise level-wise formulation coupled with internal regularization functions serves to rigorously combat overfitting against noisy clickstream metrics.
    - Configuration: `objective = 'multi:softprob'`, incorporating L1 (`reg_alpha`) and L2 (`reg_lambda`) regularization.

Both models run inside an *`imblearn` Pipeline* integrating SMOTE dynamically.

---

## 4. Validation Logic

The target variable (`country_destination`) defines a multi-class categorization task spanning 12 potential destination groups with heavy skewing.

- **Cross-Validation Partitioning**: A robust 80/20 train/validation partitioned using **StratifiedSplits** (`stratify=y_encoded`) enforcing representation of rare minority destinations across folds.
- **Primary Domain Metric (NDCG@5)**: Derived specifically for recommendation quality, **Normalized Discounted Cumulative Gain at 5** tracks if the model ranked the actual converted destination favorably within the top 5 predicted likelihoods, applying logarithmic decay to lower placements.
- **Auxiliary Diagnostics Metrics**:
  - `Macro F1-Score` to directly audit if the minority destination classes are receiving correct recall rates (rather than purely optimizing against the dominant 'NDF' class).
  - `Overall Accuracy`.
