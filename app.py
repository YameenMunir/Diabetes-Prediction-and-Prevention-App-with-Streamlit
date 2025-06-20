import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(
    page_title="🩺 Diabetes Risk & Prevention Advisor",
    layout="wide",
    page_icon="🩺"
)

load_dotenv()

# --- Load Model from pickle for Report Generation ---
loading_model = pickle.load(open("diabetes_model.pkl", "rb"))

# --- Prediction Helper Function for Report ---
def diabetes_prediction(input_data):
    input_np = np.asarray(input_data).reshape(1, -1)
    prediction = loading_model.predict(input_np)
    return "the person is diabetic" if prediction[0] == 1 else "the person is not diabetic"

# --- Categorization Helpers ---
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi < 25:
        return "normal"
    elif 25 <= bmi < 30:
        return "overweight"
    else:
        return "obese"

def categorize_glucose(glucose):
    if glucose < 100:
        return "normal"
    elif 100 <= glucose < 140:
        return "prediabetic"
    else:
        return "diabetic-range"

def categorize_age(age):
    if age < 30:
        return "young adult"
    elif 30 <= age < 50:
        return "middle-aged"
    else:
        return "older adult"

# --- Personalized Prevention Tips ---
def get_prevention_tips(age, bmi, glucose, prediction):
    bmi_status = categorize_bmi(bmi)
    glucose_status = categorize_glucose(glucose)
    age_group = categorize_age(age)
    is_at_risk = prediction == 1
    tips = []
    tips.append("• 💧 Stay hydrated and avoid sugary drinks to stabilize blood sugar.")
    if bmi_status == "underweight":
        tips.append("• 🥑 Eat nutrient-dense meals and snacks to support healthy weight gain.")
    elif bmi_status in ["overweight", "obese"]:
        tips.append("• ⚖️ Aim for 5–10% weight loss with a mix of activity and dietary changes.")
    if glucose_status == "prediabetic":
        tips.append("• 🥗 Cut refined carbs and focus on fiber-rich foods like lentils and greens.")
    elif glucose_status == "diabetic-range":
        tips.append("• 🩺 Book a consultation to begin glucose monitoring and diet adjustments.")
    if age_group == "young adult":
        tips.append("• 📱 Create habits early: use fitness apps and aim for 7–8 hours of sleep.")
    elif age_group == "older adult":
        tips.append("• 👟 Do joint-friendly exercises like walking, tai chi, or water aerobics.")
    if is_at_risk:
        tips.append("• 📆 Schedule a check-up soon to discuss risk and next steps.")
    else:
        tips.append("• ✔️ Maintain healthy routines and recheck your metrics every 6–12 months.")
    return tips

# --- Fallback Model Loader (used for probability and feature importance) ---
@st.cache_resource
def get_model():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        df = pd.read_csv('diabetes.csv')
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        with open('diabetes_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model

# --- Main App ---
def main():
    # Welcome banner
    img = Image.open("img.png")
    st.image(img, caption="Welcome To Diabetes Prediction App", use_container_width=True)

    st.markdown("""
    ### 🩺 About  
    This AI-powered tool helps estimate your diabetes risk based on health metrics  
    and offers personalized, practical lifestyle recommendations.

    👉 After generating your result, click **Download Report** to save a summary.
    """)
    st.markdown("---")

    # Developer credits in sidebar
    with st.sidebar:
        st.markdown("🧑‍💻 Developed by: **Yameen Munir**")
        st.markdown("**AI Enthusiast, Python Developer & Data Science Learner**")
        st.markdown("📧 Email: yameenmunir05@gmail.com")
        st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/yameen-munir/)")
        st.markdown("📄 [GitHub](https://github.com/YameenMunir)")
        st.markdown("🌐 [Portfolio](https://www.datascienceportfol.io/YameenMunir)")

    # User input fields
    st.markdown("## 🔎 Enter Your Health Information")
    Pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
    Glucose = st.number_input("Glucose Level", 0, 300, 100)
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin Level", 0, 1000, 80)
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
    Age = st.number_input("Age", 1, 120, 30)

    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    diagnosis = ""

    if st.button("🧠 Diabetes Test Result"):
        diagnosis = diabetes_prediction(input_data)
        st.success(f"📢 Result: {diagnosis}")

        # Report text
        report = f"""
==== Diabetes Prediction Report ====

Pregnancies: {Pregnancies}
Glucose Level: {Glucose}
Blood Pressure: {BloodPressure}
Skin Thickness: {SkinThickness}
Insulin Level: {Insulin}
BMI: {BMI:.2f}
Diabetes Pedigree Function: {DiabetesPedigreeFunction:.2f}
Age: {Age}

Diagnosis: {diagnosis}

====================================
"""
        # Download button
        st.download_button("📄 Download Report", data=report, file_name="diabetes_prediction_report.txt", mime="text/plain")
        st.success("📥 Report generated successfully!")

        # Personalized Recommendations
        st.markdown("## 🌱 Personalized Prevention Tips")
        model = get_model()
        reshaped = np.array(input_data).reshape(1, -1)
        prediction = model.predict(reshaped)[0]
        tips = get_prevention_tips(Age, BMI, Glucose, prediction)
        for tip in tips:
            st.markdown(f"- {tip[1:].strip() if tip.startswith('•') else tip}")

        # Feature importance
        st.markdown("## 📊 Top Risk Contributors")
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

        # Monitoring guide
        st.markdown("## 🧭 Health Monitoring Guide")
        st.markdown("""
- 🔬 Check fasting glucose every 3–6 months if you're at risk  
- ⚖️ Track BMI, waist circumference, and weight monthly  
- 🩸 Request a metabolic panel annually (glucose, A1C, lipids)
""")

        # Resources
        st.markdown("## 🌐 Trusted Resources")
        st.markdown("""
- [American Diabetes Association](https://www.diabetes.org)  
- [CDC Diabetes Prevention Program](https://www.cdc.gov/diabetes/prevention/)  
- [Nutrition.gov — Diabetes Section](https://www.nutrition.gov/topics/diet-and-health-conditions/diabetes)
""")

if __name__ == '__main__':
    main()