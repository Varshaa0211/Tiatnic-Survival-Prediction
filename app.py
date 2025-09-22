import streamlit as st
import numpy as np
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
# Load Model
# ----------------------------
try:
    model = joblib.load("titanic_model.pkl")
except:
    st.error("❌ Model file not found! Please ensure `titanic_model.pkl` is in the same folder.")
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

pclass = st.selectbox("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Passenger Fare ($)", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

# ----------------------------
# Encode Inputs
# ----------------------------
sex_val = 0 if sex == "Male" else 1
embarked_val = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

features = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Survival"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("🎉 The passenger **Survived** 🟢")
    else:
        st.error("💔 The passenger **Did Not Survive** 🔴")
