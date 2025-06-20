import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from dotenv import load_dotenv

# Set up page config
st.set_page_config(
    page_title="🩺 Diabetes Risk & Prevention Advisor",
    layout="wide",
    page_icon="🩺"
)

load_dotenv()

# --- Helpers for categorizing inputs ---
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

# --- Prevention Tips Generator ---
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

# --- ML Model ---
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
    st.markdown("""
        # 🩺 Diabetes Risk & Prevention Advisor
        Welcome! This AI-powered tool will:
        - Predict your risk of developing diabetes based on health metrics
        - Provide you with **personalized, practical lifestyle recommendations**
    """)
    
    st.markdown("---")

    with st.sidebar:
        st.markdown("## 🔎 Health Input")
        st.markdown("Enter your recent health metrics below:")
        pregnancies = st.number_input('Pregnancies (if applicable)', 0, 20, 0)
        glucose = st.number_input('Glucose Level (mg/dL)', 0, 300, 100)
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', 0, 200, 70)
        skin_thickness = st.number_input('Skin Thickness (mm)', 0, 100, 20)
        insulin = st.number_input('Insulin Level (mu U/ml)', 0, 1000, 80)
        bmi = st.number_input('BMI', 10.0, 60.0, 25.0)
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
        age = st.number_input('Age', 1, 120, 30)
        submit_button = st.button('🧠 Predict My Risk')

    col1, col2 = st.columns(2)

    if submit_button:
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }

        with col1:
            st.markdown("### 📋 Your Health Summary")
            st.dataframe(pd.DataFrame([input_data]))

            model = get_model()
            input_array = np.array([list(input_data.values())])
            prediction = model.predict(input_array)[0]
            prediction_proba = model.predict_proba(input_array)[0]

            st.markdown("### 📈 Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ You are at **high risk** for diabetes. Probability: {prediction_proba[1]*100:.1f}%")
            else:
                st.success(f"✅ You are at **low risk** for diabetes. Probability: {prediction_proba[0]*100:.1f}%")

            st.markdown("### 💡 Risk Contributor Chart")
            importance_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.bar_chart(importance_df.set_index('Feature'))

        with col2:
            st.markdown("### 🌱 Personalized Prevention Tips")
            st.info("Tailored just for your health profile:")
            tips = get_prevention_tips(age, bmi, glucose, prediction)
            for tip in tips:
                st.markdown(f"- {tip[1:].strip() if tip.startswith('•') else tip}")

            st.markdown("### 📅 Health Monitoring Guide")
            st.markdown("""
            - 🔬 Check your glucose every 3–6 months if you're at risk  
            - ⚖️ Track BMI and waist circumference each month  
            - 🩸 Request a full metabolic panel every 12 months  
            """)

            st.markdown("### 🌐 Trusted Resources")
            st.markdown("""
            - [American Diabetes Association](https://www.diabetes.org)  
            - [CDC Diabetes Prevention Program](https://www.cdc.gov/diabetes/prevention/)  
            - [Nutrition.gov on Diabetes](https://www.nutrition.gov/topics/diet-and-health-conditions/diabetes)  
            """)

if __name__ == '__main__':
    main()