"""
train_model.py — Run locally to regenerate iris_model.joblib
Usage: python train_model.py
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "iris_dataset.csv"))

FEATURES = ["sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"]
X = df[FEATURES]
y = df["species_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["setosa","versicolor","virginica"]))

joblib.dump(model, os.path.join(BASE_DIR, "iris_model.joblib"))
print("💾 Saved: iris_model.joblib")
