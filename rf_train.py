import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Original shape:", df.shape)

# =====================
# Basic cleaning
# =====================
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

# =====================
# Target and features
# =====================
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])

# =====================
# Best model (Gradient Boosting Classifier)
# =====================
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# =====================
# Full Pipeline
# =====================
gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", gb_model)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# Train
# =====================
gb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = gb_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== Model Evaluation (Gradient Boosting) =====")
print(f"Accuracy : {acc:.6f}")
print(f"Precision: {prec:.6f}")
print(f"Recall   : {rec:.6f}")
print(f"F1 Score : {f1:.6f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================
# Save model (IMPORTANT)
# =====================
with open("diabetes_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("\n✅ Gradient Boosting pipeline saved as diabetes_gb_pipeline.pkl")
