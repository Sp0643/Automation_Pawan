Below is an example end-to-end data preprocessing workflow in Python, demonstrating how you might handle duplicates, missing values, imputation, encoding, normalization/scaling, imbalance, and multicollinearity. Keep in mind that the exact steps and order may vary depending on your dataset and business requirements.

##############################################################################
# Importing libraries
##############################################################################
import pandas as pd
import numpy as np

# For handling missing values and encoding
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# For scaling/normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# For managing imbalanced dataset
from imblearn.over_sampling import SMOTE

# For train-test split
from sklearn.model_selection import train_test_split

##############################################################################
# 1. Load your dataset
##############################################################################
df = pd.read_csv('your_data.csv')  # Replace with your actual file path

##############################################################################
# 2. Remove Duplicates
##############################################################################
print("Number of duplicates before removal:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Number of duplicates after removal:", df.duplicated().sum())

##############################################################################
# 3. Identify & Handle Missing Values
##############################################################################
# Check missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Split columns by numeric and categorical
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# Numeric imputation with median (as an example)
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical imputation with most frequent (as an example)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

##############################################################################
# 4. Encoding Categorical Variables
##############################################################################
# Option A: Label Encoding for each categorical column
# (Useful for ordinal or nominal with few categories)
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    
# Option B: One-Hot Encoding (comment out Option A and uncomment Option B if you prefer)
# df = pd.get_dummies(df, columns=cat_cols)

##############################################################################
# 5. (Optional) Split into Train/Test before Handling Imbalance & Scaling
##############################################################################
# Typically, you should split your data before applying SMOTE or scaling
# to avoid data leakage. Example below:

# Let's assume your target column is named 'target'
X = df.drop('target', axis=1)
y = df['target']

# Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

##############################################################################
# 6. Manage Imbalanced Dataset (SMOTE example)
##############################################################################
# Only oversample the training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

##############################################################################
# 7. Feature Scaling / Normalization
##############################################################################
# You can use MinMaxScaler or StandardScaler, depending on your needs.

# Example with StandardScaler
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# If you prefer MinMaxScaler, simply replace:
# scaler = MinMaxScaler()
# X_train_res_scaled = scaler.fit_transform(X_train_res)
# X_test_scaled = scaler.transform(X_test)

##############################################################################
# 8. Check and Address Multicollinearity
##############################################################################
# We'll demonstrate a correlation-based approach on the *training* set 
# after scaling/encoding. You might do it before scaling too, 
# but let's show the concept:

# Convert scaled arrays back to DataFrame for correlation analysis
X_train_res_scaled_df = pd.DataFrame(X_train_res_scaled, 
                                     columns=X_train_res.columns)

# Compute absolute correlation matrix
corr_matrix = X_train_res_scaled_df.corr().abs()

# Select upper triangle of correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation > 0.9 (example threshold)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

print("\nHighly correlated features (to potentially drop):", to_drop)

# Drop them (from train and test)
X_train_res_scaled_df.drop(to_drop, axis=1, inplace=True)
# For the test set, you need to drop the same columns
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns).drop(to_drop, axis=1)

##############################################################################
# Final: Your preprocessed data is now ready for modeling!
##############################################################################

# X_train_prepared = X_train_res_scaled_df
# y_train_prepared = y_train_res
# X_test_prepared  = X_test_scaled_df
# y_test_prepared  = y_test


---

Explanation of Key Steps

1. Remove Duplicates

Ensures each row is unique.



2. Identify & Handle Missing Values

Used SimpleImputer with different strategies for numeric vs. categorical columns.



3. Encoding Categorical Variables

Label Encoding is demonstrated.

Alternatively, One-Hot Encoding (pd.get_dummies) can be used, especially for nominal categories.



4. Train-Test Split

Perform this before SMOTE and scaling to avoid data leakage.



5. Manage Imbalance

Example uses SMOTE to oversample the minority class in the training set only.



6. Feature Scaling / Normalization

Demonstrates StandardScaler. Alternatively, MinMaxScaler can be used.



7. Multicollinearity

Uses correlation matrix to drop features with high correlation (>0.9 as an example).

There are other techniques (e.g., VIF—Variance Inflation Factor) for more robust analysis.



8. Ready for Modeling

After these steps, X_train_prepared and X_test_prepared are typically fed into your chosen machine learning algorithm (e.g., Logistic Regression, Random Forest, etc.).





---

Tip: In a real production setting, it’s best practice to encapsulate these transformations in a pipeline (sklearn.pipeline.Pipeline) so you can apply the same transformations consistently to both training and future inference data.





Below is example Python code demonstrating how to train and evaluate a variety of basic and ensemble classification models. You can adapt this to your own dataset and requirements.

> Note: This code assumes you already have X_train, X_test, y_train, and y_test from a train-test split (and any preprocessing steps) in your environment. If you don’t have them yet, see the previous data preprocessing code example to prepare your data.




---

##############################################################################
# 1. Import Libraries
##############################################################################
import numpy as np
import pandas as pd

# Basic Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensemble Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Third-party Gradient Boosting Libraries
# (Make sure you have xgboost, lightgbm, catboost installed: pip install xgboost lightgbm catboost)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# For evaluation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

##############################################################################
# 2. (Optional) Example Train-Test Split 
#    If you haven't already split your data, uncomment and adjust as needed
##############################################################################
# df = pd.read_csv('your_data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

##############################################################################
# 3. Define a Dictionary of Models
##############################################################################
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Classifier": SVC(random_state=42),

    # Ensemble models (built-in scikit-learn)
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),

    # External gradient boosting libraries
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)  # verbose=0 to suppress logs
}

##############################################################################
# 4. Train Each Model & Evaluate
##############################################################################
for model_name, model in models.items():
    print(f"====================== {model_name} ======================")
    # Train (fit) the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")


---

Explanation of Key Steps

1. Imports:

We import basic scikit-learn classifiers (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC) and ensemble models (Random Forest, AdaBoost, GradientBoosting).

We also import popular third-party gradient boosting libraries (XGBoost, LightGBM, CatBoost).



2. Model Dictionary:

We create a dictionary models that maps a model name (string) to its corresponding classifier object.

This makes it easy to iterate over all models, train, and evaluate them in a loop.



3. Model Training and Evaluation:

For each model, we call fit(X_train, y_train), then predict on X_test.

We compute the accuracy score and display a classification report (precision, recall, F1-score, and support).



4. Hyperparameters:

Default hyperparameters are used in this example. You can fine-tune them using GridSearchCV or RandomizedSearchCV for improved performance.



5. Install Requirements:

If you don’t have xgboost, lightgbm, or catboost, install them via:

pip install xgboost lightgbm catboost





---

Next Steps

Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to find the best parameters for each model.

Cross-Validation: Perform k-fold cross-validation to get a more robust estimate of model performance.

Feature Selection / Engineering: Experiment with different feature sets and transformations to improve accuracy.

Evaluation Metrics: Depending on your business goals, consider additional metrics like AUC-ROC, precision-recall curves, etc.


This code provides a starting point to quickly compare multiple models and pick the best one for your classification task.






Below is example Python code that demonstrates how to evaluate a classification model using various metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC/AUC.

You can adapt this snippet to your own data (X_test, y_test) and model (model):

##############################################################################
# 1. Import required libraries
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

##############################################################################
# 2. Generate predictions
##############################################################################
# Assume you have a trained model and test data
# model.fit(X_train, y_train)  # already trained
y_pred = model.predict(X_test)  

# If the model supports predict_proba (e.g., LogisticRegression, RandomForest, XGBoost), 
# you can get the predicted probabilities for the positive class (index 1).
# This is useful for ROC AUC calculation. 
y_prob = model.predict_proba(X_test)[:, 1]  # for binary classification

##############################################################################
# 3. Calculate basic metrics
##############################################################################
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='binary')  # For binary classification
rec = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("Accuracy:  ", acc)
print("Precision: ", prec)
print("Recall:    ", rec)
print("F1 Score:  ", f1)

##############################################################################
# 4. Confusion Matrix
##############################################################################
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Optional: Plot confusion matrix
import seaborn as sns

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

##############################################################################
# 5. Classification Report
##############################################################################
# This gives precision, recall, f1-score, and support for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

##############################################################################
# 6. ROC AUC & ROC Curve (for binary classification)
##############################################################################
# Calculate ROC AUC (requires probability estimates, not just predictions)
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:   ", roc_auc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


---

Explanation of Key Steps

1. Predictions (y_pred):

model.predict(X_test) provides the class labels (e.g., 0 or 1 in a binary classification).



2. Predicted Probabilities (y_prob):

For metrics like ROC AUC, you need the model’s probability estimates.

model.predict_proba(X_test)[:, 1] gives you the probability of the positive class (class index 1).



3. Basic Metrics:

Accuracy (accuracy_score): (TP + TN) / (TP + TN + FP + FN).

Precision (precision_score): TP / (TP + FP).

Recall (recall_score): TP / (TP + FN).

F1-score (f1_score): Harmonic mean of precision and recall.



4. Confusion Matrix:

Summarizes how many predictions were correct/incorrect for each class.

Visualizing with a heatmap helps interpret false positives and false negatives.



5. Classification Report:

Displays precision, recall, F1-score, and support (the number of true instances for each class) in a tabular format.



6. ROC AUC:

Measures the area under the Receiver Operating Characteristic curve.

Higher values (close to 1.0) indicate better discriminative power.



7. ROC Curve:

Plots the True Positive Rate (TPR) vs. False Positive Rate (FPR) at various threshold settings.





---

Handling Multi-Class Classification

If your problem is multi-class (more than 2 classes), you can still use:

classification_report with average='macro', 'weighted', or 'micro' to get aggregated metrics.

confusion_matrix to see the distribution across multiple classes.

roc_auc_score can be used with average='macro' or 'weighted' for multi-class, but you need probability estimates for each class (shape = [n_samples, n_classes]).




---

That’s it! With these metrics and plots, you can thoroughly evaluate your model’s performance and compare different classifiers.




Here's a Python script that:

Reads a dataset (CSV file) into a Pandas DataFrame.

Identifies numerical columns.

Plots normal distributions for each numerical column.

Computes and displays the skewness and kurtosis of each numerical column.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load dataset
file_path = "your_dataset.csv"  # Update this with your dataset path
df = pd.read_csv(file_path)

# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Compute skewness and kurtosis
stats_df = pd.DataFrame({
    "Column": numerical_cols,
    "Skewness": [df[col].skew() for col in numerical_cols],
    "Kurtosis": [df[col].kurtosis() for col in numerical_cols]
})

# Display statistics
import ace_tools as tools
tools.display_dataframe_to_user(name="Skewness and Kurtosis", dataframe=stats_df)

# Plot normal distribution
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=30, density=True, alpha=0.6, color='g')

    # Plot normal distribution curve
    mu, std = df[col].mean(), df[col].std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.title(f'Normal Distribution: {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

Explanation:

1. Loads the dataset – Ensure to update the file path.


2. Identifies numerical columns – Selects only numeric data for analysis.


3. Computes Skewness and Kurtosis – Displays the results in a table.


4. Plots Normal Distribution – Uses histograms overlaid with a normal distribution curve.



Let me know if you want modifications!






I'll write a Python script that compares the base model, GridSearchCV, and RandomizedSearchCV for all bagging and boosting models. This script will evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.

Here’s how the script is structured:

1. Base Models: Train and evaluate AdaBoost, GradientBoosting, RandomForest, and ExtraTrees.


2. Grid Search: Hyperparameter tuning using GridSearchCV.


3. Random Search: Hyperparameter tuning using RandomizedSearchCV.


4. Comparison Table: Store results in a DataFrame for easy comparison.




---

Python Script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Load dataset (Replace with your dataset)
# df = pd.read_csv("your_data.csv")  # Uncomment if using CSV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ExtraTrees": ExtraTreesClassifier()
}

# Hyperparameter grids
param_grids = {
    "AdaBoost": {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 1]},
    "GradientBoosting": {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 0.2]},
    "RandomForest": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]},
    "ExtraTrees": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}
}

# Random search parameters
random_search_params = {key: {k: v for k, v in param_grids[key].items()} for key in models}

# Results storage
results = []

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Base Model Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    metrics = {
        "Model": model_name,
        "Type": "Base",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }
    results.append(metrics)

    # Grid Search
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_grid_model = grid_search.best_estimator_
    y_pred_grid = best_grid_model.predict(X_test)
    y_pred_proba_grid = best_grid_model.predict_proba(X_test)[:, 1] if hasattr(best_grid_model, "predict_proba") else None

    metrics_grid = {
        "Model": model_name,
        "Type": "GridSearch",
        "Accuracy": accuracy_score(y_test, y_pred_grid),
        "Precision": precision_score(y_test, y_pred_grid),
        "Recall": recall_score(y_test, y_pred_grid),
        "F1-Score": f1_score(y_test, y_pred_grid),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba_grid) if y_pred_proba_grid is not None else None
    }
    results.append(metrics_grid)

    # Random Search
    random_search = RandomizedSearchCV(model, random_search_params[model_name], n_iter=10, cv=5, scoring="accuracy", n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_random_model = random_search.best_estimator_
    y_pred_random = best_random_model.predict(X_test)
    y_pred_proba_random = best_random_model.predict_proba(X_test)[:, 1] if hasattr(best_random_model, "predict_proba") else None

    metrics_random = {
        "Model": model_name,
        "Type": "RandomSearch",
        "Accuracy": accuracy_score(y_test, y_pred_random),
        "Precision": precision_score(y_test, y_pred_random),
        "Recall": recall_score(y_test, y_pred_random),
        "F1-Score": f1_score(y_test, y_pred_random),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba_random) if y_pred_proba_random is not None else None
    }
    results.append(metrics_random)

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Model Comparison Results", dataframe=df_results)

# Plot comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=df_results, x="Model", y="Accuracy", hue="Type")
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=45)
plt.legend(title="Model Type")
plt.show()

Explanation:

Base Model: Trains each model without hyperparameter tuning.

Grid Search: Finds the best hyperparameters using exhaustive search.

Random Search: Finds hyperparameters with a randomized approach.

Evaluation Metrics: Includes accuracy, precision, recall, F1-score, and ROC-AUC.

Comparison Plot: A barplot to visualize model performance.


This will allow you to compare the performance of bagging and boosting models efficiently. Let me know if you need modifications!





Here is the Python code for performing hypothesis testing on the loan dataset. This script includes hypothesis tests for various financial and demographic factors.

import pandas as pd
import scipy.stats as stats

# Load the dataset
file_path = "loan-test.csv"  # Make sure this file is in the same directory
df = pd.read_csv(file_path)

# Handle missing values
df_clean = df.dropna(subset=['Credit_History'])  # Remove rows where Credit_History is missing
df_clean['LoanAmount'].fillna(df_clean['LoanAmount'].median(), inplace=True)  # Fill missing LoanAmount with median

# Convert categorical columns to numeric
label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
df_clean[label_cols] = df_clean[label_cols].astype('category').apply(lambda x: x.cat.codes)

# Convert Dependents to numeric
df_clean['Dependents'].replace({'3+': 3}, inplace=True)
df_clean['Dependents'] = df_clean['Dependents'].astype(float)

# 1. Checking if Applicant Income differs based on Education Level (Graduate vs. Not Graduate)
edu_grad = df_clean[df_clean['Education'] == 1]['ApplicantIncome']
edu_non_grad = df_clean[df_clean['Education'] == 0]['ApplicantIncome']
t_stat_income, p_value_income = stats.ttest_ind(edu_grad, edu_non_grad, equal_var=False)
print(f"Applicant Income (Graduate vs Non-Graduate): t-stat={t_stat_income:.3f}, p-value={p_value_income:.5f}")

# 2. Checking if Loan Amount differs based on Property Area (Urban vs. Rural)
urban_loan = df_clean[df_clean['Property_Area'] == 2]['LoanAmount']
rural_loan = df_clean[df_clean['Property_Area'] == 0]['LoanAmount']
t_stat_loan_area, p_value_loan_area = stats.ttest_ind(urban_loan, rural_loan, equal_var=False)
print(f"Loan Amount (Urban vs Rural): t-stat={t_stat_loan_area:.3f}, p-value={p_value_loan_area:.5f}")

# 3. Checking if Coapplicant Income is significantly different for Self-Employed vs. Non-Self-Employed applicants
self_emp_co_income = df_clean[df_clean['Self_Employed'] == 1]['CoapplicantIncome']
non_self_emp_co_income = df_clean[df_clean['Self_Employed'] == 0]['CoapplicantIncome']
t_stat_co_income, p_value_co_income = stats.ttest_ind(self_emp_co_income, non_self_emp_co_income, equal_var=False)
print(f"Coapplicant Income (Self-Employed vs Non-Self-Employed): t-stat={t_stat_co_income:.3f}, p-value={p_value_co_income:.5f}")

# 4. Checking if Loan Amount Term differs based on Marital Status (Married vs. Single)
married_loan_term = df_clean[df_clean['Married'] == 1]['Loan_Amount_Term']
single_loan_term = df_clean[df_clean['Married'] == 0]['Loan_Amount_Term']

# Ensure there are variations in Loan Term
if married_loan_term.nunique() > 1 and single_loan_term.nunique() > 1:
    t_stat_loan_term, p_value_loan_term = stats.ttest_ind(married_loan_term, single_loan_term, equal_var=False)
    print(f"Loan Amount Term (Married vs Single): t-stat={t_stat_loan_term:.3f}, p-value={p_value_loan_term:.5f}")
else:
    print("Loan Amount Term (Married vs Single): No variation in loan terms, unable to conduct t-test.")

How This Works:

1. Handles missing values: Drops rows with missing Credit_History and fills missing LoanAmount with the median.


2. Encodes categorical variables: Converts them into numeric values.


3. Performs hypothesis testing:

T-test for Applicant Income: Compares income levels between graduates and non-graduates.

T-test for Loan Amount by Property Area: Compares loan amounts for urban vs. rural applicants.

T-test for Coapplicant Income: Compares income levels for self-employed vs. non-self-employed applicants.

T-test for Loan Term by Marital Status: Checks if loan terms differ between married and single applicants.



4. Handles potential issues: The script checks whether there is variation in loan terms before running the t-test.



Would you like this script in a downloadable .py file?




Hypothesis Testing Results:

1. Applicant Income (Graduate vs Non-Graduate)

t-statistic: -3.839

p-value: 0.00015

Interpretation: Since the p-value is less than 0.05, we reject the null hypothesis. This means that graduate and non-graduate applicants have significantly different income levels.



2. Loan Amount (Urban vs Rural)

t-statistic: -0.524

p-value: 0.601

Interpretation: Since the p-value is greater than 0.05, we fail to reject the null hypothesis. This suggests no significant difference in loan amounts between urban and rural applicants.



3. Coapplicant Income (Self-Employed vs Non-Self-Employed)

t-statistic: -0.485

p-value: 0.630

Interpretation: The p-value is greater than 0.05, so we fail to reject the null hypothesis. This indicates that coapplicant income is not significantly different between self-employed and non-self-employed applicants.



4. Loan Amount Term (Married vs Single)

t-statistic: NaN

p-value: NaN

Possible Issue: This suggests that the sample variance might be too low (i.e., loan terms may be identical or nearly identical in both groups). We may need to check and clean the data further to properly analyze this.




Would you like me to further investigate the Loan Amount Term issue, or apply hypothesis testing to additional variables?


