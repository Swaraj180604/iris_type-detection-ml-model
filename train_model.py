"""
train_model.py
Run this once to generate iris_model.joblib before deploying.
Usage: python train_model.py
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("iris_dataset.csv")

FEATURES = ["sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"]

X = df[FEATURES]
y = df["species_id"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["setosa", "versicolor", "virginica"]))

# Save
joblib.dump(model, "iris_model.joblib")
print("💾 Model saved as iris_model.joblib")
