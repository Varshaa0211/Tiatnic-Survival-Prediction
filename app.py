import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os

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
# Custom CSS
# ----------------------------
page_bg = """
<style>
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    color: #f1f1f1;
}
.css-1d391kg p {
    color: #f1f1f1;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# Load / Train Model
# ----------------------------
MODEL_PATH = "titanic_model.pkl"

if not os.path.exists(MODEL_PATH):
    # Load dataset (uploaded by user)
    data = pd.read_csv("5d902bb8-76a2-4de6-a04d-39064dbcd302.csv")

    # Drop unnecessary columns
    data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Handle missing values
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

    # Split features & target
    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    # Define columns
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, MODEL_PATH)

# Load trained model
model = joblib.load(MODEL_PATH)

# ----------------------------
# Title
# ----------------------------
st.title("🚢 Titanic Survival Prediction ⚓")
st.markdown("### 💡 Enter passenger details below to predict survival probability")

# ----------------------------
# Input Form
# ----------------------------
st.subheader("👤 Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Ticket Class", [1, 2, 3])
    sex = st.selectbox("⚧ Sex", ["male", "female"])
    age = st.number_input("🎂 Age", min_value=0, max_value=100, value=25)
    sibsp = st.number_input("👨‍👩‍👦 Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("👶 Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("💵 Passenger Fare ($)", min_value=0.0, value=50.0)
    embarked = st.selectbox("🛳️ Port of Embarkation", ["S", "C", "Q"])

# ----------------------------
# Features DataFrame
# ----------------------------
features = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# ----------------------------
# Prediction
# ----------------------------
# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Survival"):
    prediction = model.predict(features.values)[0]        # FIXED
    probability = model.predict_proba(features.values)[0][1]  # FIXED

    if prediction == 1:
        st.success(f"🎉 The passenger **Survived** 🟢 (Probability: {probability:.2f})")
    else:
        st.error(f"💔 The passenger **Did Not Survive** 🔴 (Probability: {probability:.2f})")


