# Customer Churn Prediction

Week 2 assignment for the Data Science Mentorship: turning a messy dataset into a documented, reproducible and carefully evaluated machine-learning workflow.

## Business Problem

A telecom company loses revenue every time a customer cancels, and keeping a customer is usually cheaper than winning a new one. The goal is to predict which customers are likely to churn (leave) from their account, service and billing information, so the retention team can focus offers and calls on the customers most at risk.

- **Target variable:** `Churn` (Yes = left, encoded as 1; No = stayed, encoded as 0)
- **Business value:** rank customers by churn risk and spend a limited retention budget where it matters most

## Dataset Source

**Telco Customer Churn** (7,043 customers, 21 columns), originally published by IBM as a sample dataset and shared on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The data file is not included in this repository. See [`data/README.md`](data/README.md) for download instructions and where to place the file.

## Main Cleaning Decisions

| # | Problem | Evidence | Fix |
|---|---------|----------|-----|
| 1 | `TotalCharges` stored as text | text dtype, 11 blank-string entries | Strip whitespace and convert with `pd.to_numeric` |
| 2 | Missing `TotalCharges` after conversion | 11 rows, all with `tenure == 0` | Set to 0, since these customers have not been billed yet (the mean would be misleading) |
| 3 | "No phone service" repeats `PhoneService` | 682 rows, all with `PhoneService == No` | Replace with "No" |
| 4 | "No internet service" repeated in 6 columns | 1,526 rows per column, all with `InternetService == No` | Replace with "No" |
| 5 | `SeniorCitizen` coded 0/1 while other flags are Yes/No | mixed coding | Map to "Yes" / "No" |

Also checked with no action needed: duplicate rows (0), duplicate IDs (0) and IQR outliers (0).

**Columns removed from the features:** `Churn` (it is the target) and `customerID` (a unique identifier). All data-dependent preprocessing (scaling, encoding, oversampling) is done inside a pipeline, so it is fitted only on the training folds and no information leaks from the validation folds.

## Models Used

| Role | Model |
|------|-------|
| Baseline | `DummyClassifier(strategy='most_frequent')`, predicts "No churn" for everyone |
| Model 1 | Logistic Regression |
| Model 2 | Random Forest |

Both models use the same pipeline: `StandardScaler` on the numeric columns, `OneHotEncoder` on the categorical columns, then `SMOTE` to balance the classes (applied to training folds only).

## Evaluation Metrics

- **Primary:** ROC-AUC and recall on the churn class. Missing a churner (false negative) loses the customer's future revenue, which is usually more costly than wasting a retention offer on a loyal customer (false positive).
- **Supporting:** precision (so recall is not bought with too many false alarms) and accuracy. Accuracy alone is not enough because about 73.5% of customers do not churn, so a model that always predicts "No churn" is already 73.5% accurate.
- **Method:** 5-fold stratified cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), with the same folds and metrics for all three models. Decision threshold: 0.5.

## Key Results

Mean over 5 stratified folds:

| Model | ROC-AUC | Recall | Precision | Accuracy |
|-------|---------|--------|-----------|----------|
| Baseline (most frequent) | 0.5000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic Regression | 0.8442 | 0.7972 | 0.5140 | 0.7461 |
| Random Forest | 0.8241 | 0.5816 | 0.5811 | 0.7778 |

- Logistic Regression is best on the primary metrics (ROC-AUC and recall) in all five folds, and it is the model recommended for further investigation.
- Random Forest trades recall for higher precision and accuracy.
- Both models clearly beat the baseline on ROC-AUC and recall, but the accuracy gain over the baseline is small.

## How to Run the Notebook

**Project structure**

```
week2-data-science-assignment/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── notebooks/
    └── Customer_Churn_Prediction.ipynb
```

**Locally**

1. Clone the repository and open a terminal in its folder.
2. (Optional) create and activate a virtual environment.
3. Install the packages: `pip install -r requirements.txt`
4. Download the dataset as described in [`data/README.md`](data/README.md) and save it as `data/Telco-Customer-Churn.csv`.
5. Start Jupyter (`jupyter notebook`), open `notebooks/Customer_Churn_Prediction.ipynb` and choose **Run All**.

**Google Colab**

1. Upload the notebook to Colab (or open it from GitHub).
2. Run all cells. When the data loading cell finds no file, it opens an upload prompt: select `Telco-Customer-Churn.csv`.
3. Colab already includes the required packages. If `imblearn` is missing, run `!pip install imbalanced-learn` in a cell first.

The notebook runs from start to finish with no other manual steps. Random seeds are fixed (`random_state=42`), so results are reproducible.

## Important Limitations

- Models use default hyperparameters and a fixed 0.5 threshold. Tuning, especially the threshold, could change recall and precision a lot.
- The costs of false negatives and false positives are assumed, because the data has no real figures for customer value or offer cost.
- SMOTE creates fractional values in one-hot (0/1) columns. A method for mixed data (such as SMOTENC) or class weights should be compared.
- Precision is modest (about 51% to 58%), so many retention offers would go to customers who would have stayed.
- The data is a single snapshot. `tenure` and `TotalCharges` are measured differently for churned and current customers, and the model has not been tested on a later period.
- Calibration, fairness (for example `gender` and `SeniorCitizen` as features) and external validation have not been checked.
- Results come from one cross-validation setup (seed 42).

Neither model should be considered production-ready.
