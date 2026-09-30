import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

print("\n==============================")
print("CUSTOMER CHURN PREDICTION")
print("==============================")

# NEW (Local relative path)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDataset loaded successfully!")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATA ANALYSIS
# ============================================================

print("\n==============================")
print("DATA INFORMATION")
print("==============================")

print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove customer ID because it is not useful for prediction

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)


# Convert TotalCharges to numeric

if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# Convert Churn into binary value

if "Churn" in df.columns:

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })


# Remove duplicate records

df.drop_duplicates(inplace=True)


print("\nData cleaning completed.")


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n==============================")
print("EXPLORATORY DATA ANALYSIS")
print("==============================")


# Churn distribution

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Churn",
    data=df
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()


# Churn percentage

churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn Percentage:")
print(churn_percentage)


# ============================================================
# 5. TENURE VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True
)

plt.title("Tenure vs Customer Churn")

plt.show()


# ============================================================
# 6. MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")

plt.show()


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

plt.figure(figsize=(10, 7))

sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()


# ============================================================
# 8. FEATURE / TARGET SPLIT
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Churn")


# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 12. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


print("\n==============================")
print("TRAINING LOGISTIC REGRESSION")
print("==============================")


logistic_model.fit(
    X_train,
    y_train
)


# Predictions

y_pred_lr = logistic_model.predict(X_test)

y_probability_lr = logistic_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 13. LOGISTIC REGRESSION EVALUATION
# ============================================================

print("\nLogistic Regression Results")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred_lr)
)

print(
    "Precision:",
    precision_score(y_test, y_pred_lr)
)

print(
    "Recall:",
    recall_score(y_test, y_pred_lr)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred_lr)
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_probability_lr
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_lr
    )
)


# ============================================================
# 14. RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")


random_forest_model.fit(
    X_train,
    y_train
)


# Predictions

y_pred_rf = random_forest_model.predict(
    X_test
)

y_probability_rf = random_forest_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. RANDOM FOREST EVALUATION
# ============================================================

print("\nRandom Forest Results")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred_rf)
)

print(
    "Precision:",
    precision_score(y_test, y_pred_rf)
)

print(
    "Recall:",
    recall_score(y_test, y_pred_rf)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred_rf)
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        y_probability_rf
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred_rf
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ============================================================
# 17. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability_lr,
    name="Logistic Regression"
)

RocCurveDisplay.from_predictions(
    y_test,
    y_probability_rf,
    name="Random Forest"
)

plt.title("ROC Curve Comparison")

plt.show()


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf)
    ],

    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf)
    ],

    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_probability_lr),
        roc_auc_score(y_test, y_probability_rf)
    ]
})


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results)


# ============================================================
# 19. SELECT BEST MODEL
# ============================================================

best_model = random_forest_model

print("\nBest model selected: Random Forest")


# ============================================================
# 20. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "customer_churn_model.pkl"
)

print(
    "\nModel saved as customer_churn_model.pkl"
)


# ============================================================
# 21. CUSTOMER CHURN PREDICTION FUNCTION
# ============================================================

def predict_customer(customer_data):

    customer_df = pd.DataFrame(
        [customer_data]
    )

    prediction = best_model.predict(
        customer_df
    )[0]

    probability = best_model.predict_proba(
        customer_df
    )[0][1]

    if prediction == 1:

        result = "Customer is likely to CHURN"

    else:

        result = "Customer is likely to STAY"

    print("\n==============================")
    print("CUSTOMER PREDICTION")
    print("==============================")

    print("Prediction:", result)

    print(
        "Churn Probability:",
        round(probability * 100, 2),
        "%"
    )

    return prediction, probability


# ============================================================
# 22. EXAMPLE CUSTOMER
# ============================================================

example_customer = {

    "gender": "Male",

    "SeniorCitizen": 0,

    "Partner": "Yes",

    "Dependents": "No",

    "tenure": 5,

    "PhoneService": "Yes",

    "MultipleLines": "No",

    "InternetService": "Fiber optic",

    "OnlineSecurity": "No",

    "OnlineBackup": "No",

    "DeviceProtection": "No",

    "TechSupport": "No",

    "StreamingTV": "Yes",

    "StreamingMovies": "Yes",

    "Contract": "Month-to-month",

    "PaperlessBilling": "Yes",

    "PaymentMethod": "Electronic check",

    "MonthlyCharges": 85.50,

    "TotalCharges": 427.50
}


predict_customer(
    example_customer
)


# ============================================================
# END
# ============================================================

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")