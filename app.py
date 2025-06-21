# diabetes_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(page_title="🩺 Diabetes Risk & Prevention Advisor", layout="wide", page_icon="🩺")
load_dotenv()

# --- Load Pickle Model ---
loading_model = pickle.load(open("diabetes_model.pkl", "rb"))

# --- Categorization Helpers ---
def categorize_bmi(bmi):
    if bmi < 18.5: return "underweight"
    elif 18.5 <= bmi < 25: return "normal"
    elif 25 <= bmi < 30: return "overweight"
    else: return "obese"

def categorize_glucose(glucose):
    if glucose < 100: return "normal"
    elif 100 <= glucose < 140: return "prediabetic"
    else: return "diabetic-range"

def categorize_age(age):
    if age < 30: return "young adult"
    elif 30 <= age < 50: return "middle-aged"
    else: return "older adult"

# --- Prediction Logic ---
def diabetes_prediction(input_data):
    input_np = np.asarray(input_data).reshape(1, -1)
    prediction = loading_model.predict(input_np)
    return 1 if prediction[0] == 1 else 0

# --- Personalized Recommendations ---
def get_prevention_tips(age, bmi, glucose, prediction):
    bmi_status = categorize_bmi(bmi)
    glucose_status = categorize_glucose(glucose)
    age_group = categorize_age(age)
    is_at_risk = prediction == 1
    tips = [
        "• 💧 Stay hydrated and avoid sugary drinks to stabilize blood sugar."
    ]
    if bmi_status == "underweight":
        tips.append("• 🥑 Eat nutrient-dense meals to support healthy weight gain.")
    elif bmi_status in ["overweight", "obese"]:
        tips.append("• ⚖️ Aim for 5–10% weight loss through diet and activity.")
    if glucose_status == "prediabetic":
        tips.append("• 🥗 Prioritize fiber-rich foods like legumes and vegetables.")
    elif glucose_status == "diabetic-range":
        tips.append("• 🩺 Begin glucose monitoring and consult a healthcare provider.")
    if age_group == "young adult":
        tips.append("• 📱 Build healthy habits with apps and good sleep routines.")
    elif age_group == "older adult":
        tips.append("• 👟 Do joint-friendly exercises like tai chi or walking.")
    if is_at_risk:
        tips.append("• 📆 Schedule a check-up to assess your risk in detail.")
    else:
        tips.append("• ✔️ Maintain healthy routines and check in annually.")
    return tips

# --- Model Loader Fallback ---
@st.cache_resource
def get_model():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        df = pd.read_csv('diabetes.csv')
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open('diabetes_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model

# --- Streamlit App UI ---
def main():
    df = pd.read_csv("diabetes.csv")
    model = get_model()

    st.markdown("""
    <h1 style='text-align: center; font-family: "Segoe UI", Arial; color: #185a9d;'>🩺 Diabetes Risk & Prevention Advisor</h1>
    """, unsafe_allow_html=True)

    st.image(Image.open("img.png"), caption="Welcome To Diabetes Prediction App", use_container_width=True)

    st.markdown("""
    ### 🩺 About  
    Estimate your diabetes risk using health metrics and receive lifestyle recommendations.  
    👉 After generating a result, you can download a summary report.
    """)

    with st.sidebar:
        st.markdown("🧑‍💻 Developed by: **Yameen Munir**")
        st.markdown("**AI Enthusiast, Python Developer & Data Science Learner**")
        st.markdown("📧 yameenmunir05@gmail.com")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/yameen-munir/)")
        st.markdown("[GitHub](https://github.com/YameenMunir)")
        st.markdown("[Portfolio](https://www.datascienceportfol.io/YameenMunir)")

    st.markdown("## 🔎 Enter Your Health Information")
    Pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
    Glucose = st.number_input("Glucose Level", 0, 300, 100)
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin Level", 0, 1000, 80)
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    DPF = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
    Age = st.number_input("Age", 1, 120, 30)

    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]

    if st.button("🧠 Diabetes Test Result"):
        prediction = diabetes_prediction(input_data)
        result_text = "the person is diabetic" if prediction == 1 else "the person is not diabetic"
        st.success(f"📢 Result: {result_text}")

        report = f"""==== Diabetes Prediction Report ====
Pregnancies: {Pregnancies}
Glucose Level: {Glucose}
Blood Pressure: {BloodPressure}
Skin Thickness: {SkinThickness}
Insulin Level: {Insulin}
BMI: {BMI:.2f}
Diabetes Pedigree Function: {DPF:.2f}
Age: {Age}
Diagnosis: {result_text}
====================================
"""
        st.download_button("📄 Download Report", report, file_name="diabetes_prediction_report.txt", mime="text/plain")
        st.success("📥 Report generated successfully!")

        # Tips
        st.markdown("## 🌱 Personalized Prevention Tips")
        for tip in get_prevention_tips(Age, BMI, Glucose, prediction):
            st.markdown(f"- {tip[1:].strip() if tip.startswith('•') else tip}")

        # Feature Importance
        st.markdown("## 📊 Top Risk Contributors")
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
        st.info(f"**Insight:** The most influential factor in this prediction was **{importance_df.iloc[0]['Feature']}**.")

        # Analytics Expander
        with st.expander("📈 Explore Additional Insights"):
            st.markdown("### 🧮 Risk Group Distribution")
            counts = df['Outcome'].value_counts()
            st.bar_chart(counts.rename({0: "Non-Diabetic", 1: "Diabetic"}))
            st.info(f"Of {df.shape[0]} records, {counts[1]} are diabetic — reinforcing the need for regular screening.")

            st.markdown("### 🔍 Age vs Glucose Level")
            st.scatter_chart(df[['Age', 'Glucose']])
            st.info("Glucose levels generally increase with age — personalized care becomes essential over time.")

            st.markdown("### ⚖️ BMI Category Distribution")
            df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
            st.bar_chart(df['BMI_Category'].value_counts())
            st.info("Many individuals in the dataset fall into higher-risk BMI categories (overweight/obese).")

        # Monitoring Guide
        st.markdown("## 🧭 Health Monitoring Guide")
        st.markdown("""
- 🔬 Test fasting glucose every 3–6 months if you're at risk  
- ⚖️ Monitor BMI and waist size monthly  
- 🩸 Request annual metabolic screening (A1C, lipid panel, glucose)
""")

        # Resources
        st.markdown("## 🌐 Trusted Resources")
        st.markdown("""
- [American Diabetes Association](https://www.diabetes.org)  
- [CDC Diabetes Prevention Program](https://www.cdc.gov/diabetes/prevention/)  
- [Nutrition.gov – Diabetes](https://www.nutrition.gov/topics/diet-and-health-conditions/diabetes)
""")

if __name__ == '__main__':
    main()