# Startup Founder Burnout & Failure Risk Analysis

Data science project focused on analyzing startup founder burnout, business health indicators, and startup failure risk.

The goal of this project is to explore which factors are associated with startup failure and build a machine learning model capable of predicting whether a startup is likely to fail.

## Dataset

Dataset used:

**Startup Founder Burnout and Failure Risk Dataset**
Source: Kaggle
File: `startup_founder_burnout_2026.csv`

The dataset contains **50,000 records** and **29 columns**, including founder characteristics, startup indicators, burnout metrics, financial variables, and failure risk labels.

## Project Objective

The main objective is to predict:

```text
Startup_Failure_Flag
```

Where:

```text
0 = Startup did not fail
1 = Startup failed
```

## Project Structure

```text
startup-founder-burnout-ds/
│
├── data/
│   └── raw/
│       └── startup_founder_burnout_2026.csv
│
├── models/
│   └── startup_failure_model.pkl
│
├── notebooks/
│   └── 01_startup_failure_analysis.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── model_coefficients.png
│   └── model_metrics.csv
│
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

## Exploratory Data Analysis

The analysis included:

* Dataset shape review
* Missing value validation
* Duplicate validation
* Categorical variable analysis
* Target variable distribution
* Comparison between failed and non-failed startups
* Data leakage detection
* Model evaluation and error analysis

The dataset had:

```text
Rows: 50,000
Columns: 29
Missing values: 0
Duplicate rows: 0
```

## Target Distribution

The target variable is imbalanced:

```text
Startup_Failure_Flag = 0 → 76.27%
Startup_Failure_Flag = 1 → 23.73%
```

Because of this imbalance, accuracy alone is not enough to evaluate the model. Metrics such as precision, recall, F1-score, and confusion matrix were also used.

## Data Leakage Considerations

Some variables were excluded from the model because they were too directly related to the target variable:

```text
Shutdown_Probability
Shutdown_Risk
```

These variables could cause data leakage because they already summarize the probability or risk of shutdown.

Other derived burnout variables were also excluded to avoid redundancy:

```text
Burnout_Level
Founder_Burnout_Flag
```

## Model

The first machine learning model used was:

```text
Logistic Regression
```

A pipeline was created with:

* `StandardScaler` for numerical variables
* `OneHotEncoder` for categorical variables
* `LogisticRegression` as the classifier

The model was trained using an 80/20 train-test split with stratification.

## Model Performance

The model achieved:

```text
Accuracy: 0.8697
Precision for class 1: 0.67
Recall for class 1: 0.88
F1-score for class 1: 0.76
```

The high recall for class `1` means the model was able to detect most startups that actually failed.

## Confusion Matrix

```text
[[6605, 1022],
 [ 281, 2092]]
```

Interpretation:

```text
6605 = True negatives
1022 = False positives
281  = False negatives
2092 = True positives
```

The model detected most failed startups, with relatively few false negatives.

## Key Findings

The analysis suggests that startup failure is associated with:

* Higher burnout score
* Higher founder stress
* More weekly work hours
* Less sleep
* Fewer runway months remaining
* Lower product-market fit score
* Lower monthly revenue growth
* Higher employee turnover
* Lower work-life balance
* Worse economic climate

The most influential model features included:

```text
Runway_Months_Remaining
Burnout_Score
Cofounder_Conflict_Score
Economic_Climate_Recession
Product_Market_Fit_Score
Founder_Experience_Years
```

## Visualizations

The project includes the following saved figures:

```text
reports/figures/confusion_matrix.png
reports/figures/model_coefficients.png
```

## How to Run the Project

Create and activate the Conda environment:

```bash
conda create -n riesgo python=3.11 -y
conda activate riesgo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open JupyterLab:

```bash
jupyter lab
```

Then open:

```text
notebooks/01_startup_failure_analysis.ipynb
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* JupyterLab
* Git
* GitHub
* KaggleHub

## Conclusion

This project shows how founder wellbeing, operational pressure, financial runway, product-market fit, and economic conditions can be used to estimate startup failure risk.

The model performs well as an initial baseline, especially for identifying startups that are likely to fail. Future improvements could include testing additional models, tuning thresholds, and comparing Logistic Regression with Random Forest, XGBoost, or Gradient Boosting models.
