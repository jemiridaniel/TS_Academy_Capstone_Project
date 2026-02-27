# TS Academy Capstone Project | Group 9: Telco Customer Churn Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Notebook Sections](#key-notebook-sections)
- [Model Performance & Comparison](#model-performance--comparison)
- [Business Justification & Recommendations](#business-justification--recommendations)
- [Installation and Setup](#installation-and-setup)
- [Group Members](#group-members)
- [License](#license)

## Project Overview
This project focuses on building an end-to-end classification pipeline to predict customer churn for a telecommunications provider. The primary goal is to identify customers at risk of churning using supervised learning techniques, enabling targeted retention strategies.

**Chosen Track:** Classification

**Problem Statement:** Predict whether a telecom customer will churn (Yes/No).

## Dataset
**Telco Customer Churn Dataset (IBM)**
The dataset contains 7,043 customer records with 21 features covering demographic information, service subscriptions, billing details, and contract types. The target variable is `Churn` (Yes/No).

## Methodology
The project follows a standard machine learning workflow:
1.  **Data Preprocessing**: Cleaning, type conversion, handling missing values, and standardizing column names.
2.  **Exploratory Data Analysis (EDA)**: Comprehensive analysis of dataset structure, missingness, distribution of numeric and categorical features, and bivariate/multivariate relationships with the target variable.
3.  **Modeling**: Implementation and evaluation of baseline and advanced classification models, including strategies for handling class imbalance.
4.  **Evaluation**: Assessment using key classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix) and detailed error analysis.
5.  **Interpretation**: Feature importance/coefficient analysis to understand model decisions.
6.  **Business Impact**: Cost-based threshold tuning and clear business recommendations.

## Key Notebook Sections

### 1. Data Loading
Loads the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset into a pandas DataFrame.

### 2. Dataset Overview
Provides basic information such as rows, columns, data types, memory usage, and descriptive statistics for both numeric and categorical features.

### 3. Data Preprocessing
-   Standardizes column names.
-   Strips whitespace from string fields.
-   Converts `TotalCharges` to numeric, treating blanks as `NaN`.
-   Ensures correct numeric types for `tenure` and `MonthlyCharges`.
-   Validates and cleans the `SeniorCitizen` column to be 0 or 1.
-   Verifies the `Churn` target column.

### 4. Missing Values Analysis
Visualizes missing data through bar charts (% missing per column) and a heatmap-style matrix to show missing patterns.

### 5. Target Variable Analysis
Examines the class distribution of the `Churn` variable, identifying potential class imbalance.

### 6. Numeric Feature Profiling
Generates histograms, box plots, and descriptive statistics (mean, median, std, min, max, skewness, kurtosis) for all numeric features.

### 7. Categorical Feature Profiling
Provides value counts, percentage breakdowns, and bar charts for each categorical feature.

### 8. Bivariate & Multivariate Analysis
-   Computes and visualizes the correlation matrix for numeric features.
-   Presents crosstabulations of key categorical features against the `Churn` target (e.g., Contract, InternetService, PaymentMethod).

### 9. Modeling (Classification Track)
-   **Data Splitting**: Uses `train_test_split` with stratification for `X_train`, `X_test`, `y_train`, `y_test`.
-   **Preprocessing Pipeline**: A `ColumnTransformer` handles numeric features (imputation, scaling) and categorical features (imputation, one-hot encoding).
-   **Baseline Model**: Logistic Regression (without and with `class_weight='balanced'`).
-   **Advanced Model**: Random Forest Classifier (`class_weight='balanced_subsample'`).
-   **Evaluation Function**: A utility function `evaluate_model` computes Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
-   **Class Imbalance Strategy**: Compares the impact of `class_weight='balanced'` on model metrics.
-   **Threshold Tuning**: Explores the effect of different decision thresholds on model metrics (Precision, Recall, F1) and visualizes the trade-offs.
-   **Cost-based Threshold Selection**: Determines an optimal threshold by defining business costs for False Positives and False Negatives to minimize total expected cost.

### 10. Error Analysis
Quantifies False Positives and False Negatives for each model and interprets their business meaning.

### 11. Feature Importance / Coefficient Interpretation
-   For Logistic Regression, coefficients are used to identify features increasing or decreasing churn probability.
-   For Random Forest, feature importances are calculated and visualized to show the most influential features.

### 12. Business Metric
Reports the baseline churn rate from the dataset as a key performance indicator.

## Model Performance & Comparison
-   **Baseline Logistic Regression (No Imbalance Handling)**:
    -   Accuracy: ~80.6%
    -   Precision: ~65.7%
    -   Recall: ~55.9% (Struggled to identify actual churners)
-   **Balanced Logistic Regression (`class_weight='balanced'`)**:
    -   Accuracy: ~73.8%
    -   Precision: ~50.4%
    -   Recall: ~78.3% (Improved churner identification significantly)
-   **Random Forest (`balanced_subsample`):**
    -   Accuracy: ~78.3%
    -   Precision: ~61.9%
    -   Recall: ~47.3%

The `class_weight='balanced'` strategy for Logistic Regression significantly improved Recall at the cost of some Precision and overall Accuracy, which is often desirable in churn prediction to capture more at-risk customers.

## Business Justification & Recommendations

1.  **Optimal Threshold Selection**: A cost-based approach, prioritizing the reduction of False Negatives (lost customers), led to an optimal decision threshold of **0.29** for the balanced Logistic Regression model. This threshold minimizes the total financial impact of churn.
2.  **Targeted Retention Programs**: Focus retention efforts on customers exhibiting high-risk characteristics:
    -   Offer incentives to transition **Month-to-month** contract customers to longer-term plans.
    -   Investigate and address issues with **Fiber optic internet** service quality or pricing.
    -   Provide alternative payment options to customers primarily using **Electronic checks**.
3.  **Proactive Monitoring**: Implement the Balanced Logistic Regression model (with the cost-optimized threshold) to proactively identify and flag at-risk customers early in their tenure for intervention.
4.  **Financial Impact**: By adopting the cost-optimized threshold (0.29) instead of the default (0.50), the business is estimated to save approximately **₦775,000** on the test set. This saving comes from a significant reduction in False Negatives (missing fewer churning customers) despite an increase in False Positives (offering retention to some customers who would have stayed anyway).

**Business Interpretation at Best Threshold (0.29):**
-   **False Positives (FP)**: 469 customers flagged as at-risk but would stay anyway. Cost impact: 469 * ₦5,000 = ₦2,345,000
-   **False Negatives (FN)**: 25 customers predicted to stay but actually churn. Cost impact: 25 * ₦30,000 = ₦750,000
-   **Total cost**: ₦3,095,000
-   **Average cost per customer**: ₦2,196.59

## Installation and Setup
To run this notebook, you will need to install the following Python libraries


```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```

## Group Members

| Name | Email | GitHub Repository |
|------|-------|------------------|
| Jemiri Daniel Taiwo | updatedan2@gmail.com | https://github.com/jemiridaniel/TS_Academy_Capstone_Project |
| Kuram John Sokomba | Sokomba16@gmail.com |  https://github.com/SokombaGit/TS_Acedemy_Capstone_project |
| Oluba Amos Oluwasegun | amosoluba@gmail.com | https://github.com/Famous-Amos1/TS-Academy-Capstone-Project |
| Mbata Chidumaga  | mbataechidumaga@gmail.com | https://github.com/chidumaga/TS_Acedemy_Capstone_Project |
| Bethel Ya'u | sheshamkaza@gmail.com | https://github.com/She-Wins/TS_Academy_Capstone_Project |
| Nwafor Deborah | nwafordeborah41@gmail.com | https://github.com/DRDEBBIE256/TS-Academy-Capstone-Project |
| Jeremiah Yusuf | jeremiahyusuf185@gmail.com | https://github.com/Jeremy-1020/TS-Academy-Capstone-Project |
| Pyagbara Prince | princepyagbara@gmail.com | https://github.com/PrinceZorzor/TS_Academy_Capstone_Project |