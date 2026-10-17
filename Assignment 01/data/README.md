## Dataset

**Telco Customer Churn**: one row per customer of a telecom company (7,043 rows, 21 columns). The target column is `Churn` (Yes/No). The data was originally published by IBM as a sample dataset and is widely shared on Kaggle.

## How to get it

1. Open the Kaggle dataset page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download the file `WA_Fn-UseC_-Telco-Customer-Churn.csv` (a free Kaggle account may be required).
3. Rename it to `Telco-Customer-Churn.csv`.
4. Put it in this folder so the final path is `data/Telco-Customer-Churn.csv`.

A copy of the same data is also available in IBM's public repository `IBM/telco-customer-churn-on-icp4d` (file `data/Telco-Customer-Churn.csv`).

## Notes

- The notebook looks for the file at `data/Telco-Customer-Churn.csv`, `../data/Telco-Customer-Churn.csv` or the current folder. In Google Colab, if the file is not found, the notebook opens an upload prompt.
- The file is not modified. All cleaning is done in memory inside the notebook.
- The dataset is not committed here because its redistribution terms were not checked. It contains no names or contact details, only an anonymous `customerID`.
