import streamlit as st
import pandas as pd
import joblib

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
# Custom Background (CSS)
# ----------------------------
page_bg = """
<style>
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
}
.stTextInput > div > div > input {
    color: white;
}
.stNumberInput > div > div > input {
    color: black;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# Load Model + Columns
# ----------------------------
try:
    model = joblib.load("titanic_model.pkl")
    feature_columns = joblib.load("titanic_columns.pkl")  # save during training
except:
    st.error("❌ Model or column file not found! Please ensure `titanic_model.pkl` and `titanic_columns.pkl` exist.")
    st.stop()

# ----------------------------
# Title
# ----------------------------
st.title("🚢 Titanic Survival Prediction App ⚓")
st.markdown("### Predict whether a passenger would have **survived** the Titanic tragedy 💙")

# ----------------------------
# Input Form
# ----------------------------
st.subheader("👤 Passenger Information")

pclass = st.selectbox("🎟 Ticket Class", [1, 2, 3])
sex = st.selectbox("⚧ Sex", ["male", "female"])
age = st.number_input("🎂 Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("👫 Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("👨‍👩‍👧 Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("💵 Passenger Fare ($)", min_value=0.0, value=50.0)
embarked = st.selectbox("🛳 Port of Embarkation", ["S", "C", "Q"])

# ----------------------------
# Build Features DataFrame
# ----------------------------
input_data = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}

features = pd.DataFrame([input_data])

# Align with training columns (add missing, drop extra)
for col in feature_columns:
    if col not in features_columns:
        features[col] = 0
features = features[feature_columns]

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Survival"):
    try:
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] * 100  # survival probability %

        if prediction == 1:
            st.success(f"🎉 The passenger **Survived** 🟢 (Probability: {prob:.2f}%)")
        else:
            st.error(f"💔 The passenger **Did Not Survive** 🔴 (Probability: {prob:.2f}%)")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
