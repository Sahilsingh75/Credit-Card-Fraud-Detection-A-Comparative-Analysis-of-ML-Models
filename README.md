# Credit-Card-Fraud-Detection-A-Comparative-Analysis-of-ML-Models
This project focuses on detecting fraudulent credit card transactions using machine learning models. A comparative analysis of Logistic Regression, Random Forest, and XGBoost models is performed to evaluate their performance on detecting fraudulent transactions from a highly imbalanced dataset. 



## The project involves:
Exploratory Data Analysis (EDA) with visualizations to understand the data distribution.
Comparison of model performance on training and testing datasets.
Evaluation metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Dataset
The dataset used for this project is highly imbalanced and anonymized, with features transformed using Principal Component Analysis (PCA).

### Columns:
Time: Seconds elapsed between the transaction and the first transaction in the dataset.
V1 to V28: PCA-transformed features.
Amount: Transaction amount.
Class: Target variable where:
0: Non-fraudulent transaction.
1: Fraudulent transaction.


Download the dataset from Kaggle(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project folder.

## Steps Performed

1. Exploratory Data Analysis (EDA)
EDA was conducted to explore the dataset and uncover patterns. Key visualizations include:

Class Distribution: Understand the imbalance in the target variable.
Correlation Heatmap: Analyze relationships between PCA features, Amount, and Class.
Transaction Amount Distribution: Visualize the difference in amounts for fraud and non-fraud cases.

2. Model Building
Three machine learning models were trained and compared:

Logistic Regression
Random Forest
XGBoost

Each model was evaluated using the following metrics:

Accuracy: Overall correctness of the model.
Precision: How many predicted frauds were actually fraud.
Recall (Sensitivity): How many actual frauds were correctly predicted.
F1-Score: Harmonic mean of precision and recall.
ROC-AUC: Ability to distinguish between classes.

3. Comparative Analysis
Performance metrics for all models were compared to determine the best-performing model.

## Results
Metric			Logistic Regression	Random Forest	XGBoost
Accuracy		99.92%			99.96%		99.95%
Precision		82.89%			94.18%		89.01%
Recall			64.29%			82.65%		82.65%
F1-Score		72.41%			88.04%		85.71%

## Conclusion
Best Model: Based on the results, the model with the highest F1-Score and ROC-AUC is considered the most effective for detecting fraud.
While XGBoost outperformed other models in handling imbalanced data, Random Forest and Logistic Regression provided competitive results.
The dataset's class imbalance posed a challenge, highlighting the importance of using metrics like Precision, Recall, and F1-Score for evaluation.

## How to Run the Project

Install Dependencies: Ensure you have Python 3.8+ installed.
Then run:

    ```bash
      pip install -r requirements.txt
Run the Notebook: Open the Jupyter Notebook file and execute the cells:

    ```bash
    jupyter notebook creditcardfraud_LR_RF_XGB.ipynb

Clone the Repository:
    ```bash
        git clone https://github.com/Sahilsingh75/Credit-Card-Fraud-Detection-A-Comparative-Analysis-of-ML-Models.git
        cd music_genre_classification_with_gui

Dependencies
The project uses the following Python libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
Install them via requirements.txt.
