# Loan Approval Prediction System

Machine learning classifier comparing Logistic Regression, Decision Tree, and Random Forest algorithms for predicting loan approval decisions. Decision Tree achieved the highest accuracy at 80.49%.

## Overview

This project analyzes loan application data and builds predictive models to determine loan approval likelihood. The analysis includes data preprocessing, feature engineering, model comparison, and feature importance analysis.

## Model Performance

| Model | Accuracy | Status |
|-------|----------|--------|
| Decision Tree | 80.49% | Best |
| Random Forest | 78.86% | - |
| Logistic Regression | 78.05% | - |

## Dataset

Source: Kaggle Loan Prediction Dataset
- Total Applications: 614
- Features: 13 variables including income, credit history, property area
- Target: Binary classification (Approved/Rejected)

### Features
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Gender, Marital Status, Dependents
- Education, Employment Status

## Key Findings

### Feature Importance (Top 5)
Based on Random Forest analysis:

1. Credit_History: 43.76%
2. Total_inc: 14.82%
3. LoanAmount: 11.48%
4. Income_loan_ratio: 10.33%
5. ApplicantIncome: 9.64%

![Feature Importance](images/feature_importance.png)

Credit history is the dominant factor in loan approval decisions, accounting for nearly 44% of prediction importance.

## Visualizations

The project includes several data visualizations:

![Approval Status Distribution](images/approval_distribution.png)
*Distribution of approved vs rejected applications*

![Credit History Analysis](images/credit_history_analysis.png)
*Impact of credit history on loan approval*

## Technical Details

### Data Preprocessing
- Handled missing values using median/mode imputation
- Encoded categorical variables
- Created engineered features:
  - Total_inc (Applicant + Coapplicant income)
  - Income_loan_ratio
  - Loan_per_term

### Model Pipeline
All models use scikit-learn pipelines with:
1. Preprocessing step (encoding, scaling)
2. Model training
3. Evaluation with classification metrics

### Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## Technologies

- Python 3.x
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Installation

```bash
git clone https://github.com/varadshajith/loan-approval-predictor.git
cd loan-approval-predictor
pip install -r requirements.txt
```

## Usage

```python
# Load and train the best model (Decision Tree)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Model pipeline
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier(random_state=42))
])

# Train
dt_pipeline.fit(X_train, y_train)

# Predict
predictions = dt_pipeline.predict(X_test)
```

## Project Structure

```
loan-approval-predictor/
├── loan_approval_classifier.ipynb
├── train.csv
├── feature_importance.png
├── approval_distribution.png
├── credit_history_analysis.png
├── README.md
└── requirements.txt
```

## Results Analysis

Decision Tree outperformed other models due to its ability to capture non-linear relationships in the data. Credit history emerged as the most critical factor, validating real-world lending practices where past repayment behavior heavily influences approval decisions.

The model achieves reasonable accuracy (80.49%) but could be improved with:
- Larger dataset
- Additional features (debt-to-income ratio, employment length)
- Ensemble methods tuning
- Cross-validation optimization

## License

MIT License

## Contact

Varad Shajith
- GitHub: [@varadshajith](https://github.com/varadshajith)
- Email: varadshajith@gmail.com
