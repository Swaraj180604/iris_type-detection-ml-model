# 🌸 Iris Flower Classifier

A **Streamlit** web app that classifies Iris flowers using a **Random Forest** model saved with **joblib**.

> ⚡ **Self-healing**: If `iris_model.joblib` is missing at runtime, the app automatically trains and saves it from `iris_dataset.csv`.

---

## 📁 Project Structure

```
iris_app/
├── app.py               # Streamlit application
├── train_model.py       # Optional: retrain model locally
├── iris_model.joblib    # Pre-trained model (auto-generated if missing)
├── iris_dataset.csv     # Dataset (required)
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🚀 Deploy on Streamlit Cloud

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Iris Classifier"
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo → set **Main file path** to `app.py` → **Deploy**

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🤖 Model Details

| Property      | Value                         |
|---------------|-------------------------------|
| Algorithm     | Random Forest Classifier      |
| Estimators    | 100                           |
| Test Accuracy | 100%                          |
| Saved with    | joblib                        |
| Features      | Sepal & Petal length/width    |
| Classes       | Setosa, Versicolor, Virginica |
