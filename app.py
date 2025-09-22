# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# ----------------------------
# Page config & style
# ----------------------------
st.set_page_config(page_title="🚢 Titanic Survival Predictor",
                   page_icon="⚓",
                   layout="centered",
                   initial_sidebar_state="expanded")

PAGE_BG = """
<style>
.stApp {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0ea5a4 100%);
  color: #f8fafc;
}
.card {
  background: rgba(255,255,255,0.06);
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
h1, h2, h3 { color: #f1f5f9; }
small { color: #e2e8f0; }
</style>
"""
st.markdown(PAGE_BG, unsafe_allow_html=True)

st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict whether a passenger survives the Titanic disaster. Simple, interactive & pretty. ✨")

# ----------------------------
# Helper functions
# ----------------------------
MODEL_PATH = "titanic_model.pkl"

def build_pipeline():
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    return pipe

def train_and_save_model(df, save_path=MODEL_PATH):
    df = df.copy()
    required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = df["Survived"]

    # ✅ Correct assignment order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(pipe, save_path)
    return pipe, acc

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Setup 🔧")
model = load_model()
if model:
    st.sidebar.success("✅ Model loaded from disk")
else:
    st.sidebar.warning("No saved model found. Upload CSV to train one.")

upload_csv = st.sidebar.file_uploader("Upload Titanic CSV (with 'Survived')", type=["csv"])
if upload_csv is not None:
    try:
        df = pd.read_csv(upload_csv)
        st.sidebar.info(f"CSV loaded: {df.shape[0]} rows")
        if st.sidebar.button("Train model"):
            with st.spinner("Training model... ⚙️"):
                pipe, acc = train_and_save_model(df)
                model = pipe
            st.sidebar.success(f"Model trained & saved (accuracy: {acc:.3f})")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

uploaded_model = st.sidebar.file_uploader("Or upload model (.pkl)", type=["pkl"])
if uploaded_model is not None:
    try:
        bytes_data = uploaded_model.read()
        with open(MODEL_PATH, "wb") as f:
            f.write(bytes_data)
        model = load_model()
        st.sidebar.success("Model loaded from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Couldn't load model: {e}")

# ----------------------------
# Single Passenger Prediction
# ----------------------------
st.subheader("Predict for a single passenger 👤")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Ticket class (Pclass)", [1,2,3], index=2)
        sex = st.selectbox("Sex", ["male","female"], index=0)
        age = st.number_input("Age (years)", 0.0, 120.0, 25.0, step=0.5)
    with col2:
        sibsp = st.number_input("Siblings/Spouses (SibSp)", 0, 10, 0)
        parch = st.number_input("Parents/Children (Parch)", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 32.0, step=0.1)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)

    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict 🚀"):
    if model is None:
        st.error("No model available. Upload CSV or model file first.")
    else:
        sample = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        }])
        proba = model.predict_proba(sample)[0]
        pred = model.predict(sample)[0]
        survival_prob = float(proba[1])
        if pred == 1:
            st.success(f"🎉 Survived! (probability {survival_prob:.2%})")
        else:
            st.error(f"☹️ Did NOT survive (probability {survival_prob:.2%})")

# ----------------------------
# Batch Prediction
# ----------------------------
st.subheader("Batch prediction 📊")
batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)
        if model is None:
            st.warning("Train or upload a model first.")
        else:
            X_batch = batch_df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:,1]
            out = batch_df.copy()
            out["pred_survived"] = preds
            out["survival_prob"] = probs
            st.dataframe(out.head(15))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results", data=csv,
                               file_name="titanic_predictions.csv", mime="text/csv")
            if "Survived" in batch_df.columns:
                acc = (out["pred_survived"] == batch_df["Survived"]).mean()
                st.success(f"Accuracy: {acc:.3%}")
    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("💡 Tip: Use Kaggle Titanic `train.csv` to train model. Tweak `build_pipeline()` to experiment!")
