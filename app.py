import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="⚓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Global Model Path
# ----------------------------
MODEL_PATH = "titanic_model.pkl"

# ----------------------------
# Preprocessing + Model Pipeline
# ----------------------------
def build_pipeline():
    categorical = ["Sex", "Embarked"]
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric)
        ]
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", model)])
    return pipe

# ----------------------------
# Train and Save Model
# ----------------------------
def train_and_save_model(df, save_path=MODEL_PATH):
    df = df.copy()
    required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    df = df.dropna(subset=["Survived"])
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(pipe, save_path)
    return pipe, acc

# ----------------------------
# Load Model
# ----------------------------
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚢 Titanic Survival Prediction ⚓")
st.write("Enter passenger details to predict survival.")

# Sidebar for training
st.sidebar.header("📂 Train Model with Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    model, acc = train_and_save_model(df)
    st.sidebar.success(f"✅ Model trained with accuracy: {acc:.2f}")
else:
    model = load_model()

if model is None:
    st.warning("⚠️ Please upload a Titanic dataset (train.csv) in sidebar to train the model.")
    st.stop()

# ----------------------------
# User Inputs
# ----------------------------
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Survival 🧾"):
    # Safe feature order
    expected_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    sample = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])[expected_features]

    proba = model.predict_proba(sample)[0]
    prediction = model.predict(sample)[0]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success(f"🎉 Passenger Survived with probability {proba[1]:.2f}")
    else:
        st.error(f"☠️ Passenger Did NOT Survive with probability {proba[0]:.2f}")

