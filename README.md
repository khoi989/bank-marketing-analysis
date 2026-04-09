# Bank Marketing Analysis & Predictive Modeling

A data science project that performs end-to-end exploratory data analysis (EDA) and machine learning on a Portuguese bank's direct marketing campaign dataset to predict whether a client will subscribe to a term deposit.

---

## Project Overview

Banks run phone-based marketing campaigns to promote financial products. This project analyzes campaign data to uncover customer behavioral patterns and builds classification models to **predict term deposit subscription**, helping the bank optimize targeting and improve campaign efficiency.

> A detailed written report is included in [`REPORT.pdf`](./REPORT.pdf).

---

## Dataset

The dataset contains information about bank clients and their interactions with a direct marketing campaign.

| Feature | Description |
|---|---|
| `age` | Client's age |
| `job` | Type of occupation |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly account balance (EUR) |
| `housing` | Has a housing loan? |
| `loan` | Has a personal loan? |
| `contact` | Contact communication type |
| `month` | Last contact month |
| `campaign` | Number of contacts during this campaign |
| `pdays` | Days since last contact from a previous campaign |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of previous marketing campaign |
| `y` *(Target)* | Has the client subscribed to a term deposit? (`yes` / `no`) |

---

## Project Structure

```
Bank Marketing Analysis/
│
├── Bank Marketing Analysis.ipynb   # Main Jupyter Notebook
├── REPORT.pdf                      # Detailed written analysis report
└── README.md                       # Project documentation
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

- **Data Quality Check**: Verified zero null values and no duplicate records
- **Univariate Analysis**: Distribution of the target variable, categorical features, and numerical features
- **Bivariate Analysis**:
  - *Demographic attributes*: Age distribution across job types, marital status, and education level
  - *Financial status*: Average account balance segmented by occupation and marital status
  - *Loan status*: Relationship between loan types (housing, personal) and campaign outcomes

### 2. Data Preprocessing & Feature Engineering

- **Encoding**: One-hot encoding for nominal categorical variables; Label encoding for ordinal/binary variables
- **Class Imbalance Handling**: Applied **SMOTE** (Synthetic Minority Oversampling Technique) to address the significant imbalance between `yes` and `no` subscriptions in the training set
- **Feature Scaling**: StandardScaler applied to all features before model training
- **Train/Test Split**: 80% training — 20% testing

### 3. Machine Learning Models

Four classification models were built and evaluated:

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear classifier |
| **Decision Tree** | Non-linear tree-based model |
| **Random Forest** | Ensemble of decision trees (bagging) |
| **XGBoost** | Gradient-boosted ensemble model |

### 4. Model Evaluation

Models were compared using **accuracy score** and **classification report** (precision, recall, F1-score) on the held-out test set.

> **Random Forest achieved the highest accuracy** among all models evaluated.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly`, `missingno` |
| Machine Learning | `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`) |
| Boosting | `xgboost` |
| Imbalanced Data | `imbalanced-learn` (SMOTE) |

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn missingno plotly scikit-learn xgboost imbalanced-learn
```

### Run the Notebook

```bash
git clone https://github.com/<your-username>/bank-marketing-analysis.git
cd bank-marketing-analysis
jupyter notebook "Bank Marketing Analysis.ipynb"
```

---

## Key Findings

- The dataset is **highly imbalanced** — only ~11% of clients subscribed to a term deposit; SMOTE was essential to improve minority class recall.
- **Management and retired** clients hold the highest average account balances.
- Clients with **no personal or housing loans** are more likely to subscribe.
- **Random Forest** outperformed all other models, demonstrating the strength of ensemble methods for this classification task.
- Feature importance analysis revealed that **account balance, age, and campaign contact frequency** are among the strongest predictors.

---

## Contact

**Khoi** — Feel free to connect on [LinkedIn](https://linkedin.com/in/your-profile) or reach out via GitHub.

---

*This project was completed as a data science coursework assignment and presented as a personal portfolio project.*
