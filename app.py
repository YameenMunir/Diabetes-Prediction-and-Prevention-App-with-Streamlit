import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(page_title="🩺 Diabetes Risk & Prevention Advisor", layout="wide", page_icon="🩺")
load_dotenv()

# --- Load Model ---
@st.cache_resource
def get_model():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        df = pd.read_csv("diabetes.csv")
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open("diabetes_model.pkl", "wb") as f:
            pickle.dump(model, f)
        return model

# --- SHAP Explainer ---
@st.cache_resource
def get_shap_explainer(_model, background_data):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(background_data)
    return explainer, shap_values

# --- Categorization Helpers ---
def categorize_bmi(bmi):
    if bmi < 18.5: return "underweight"
    elif bmi < 25: return "normal"
    elif bmi < 30: return "overweight"
    return "obese"

def categorize_glucose(glucose):
    if glucose < 100: return "normal"
    elif glucose < 140: return "prediabetic"
    return "diabetic-range"

def categorize_age(age):
    if age < 30: return "young adult"
    elif age < 50: return "middle-aged"
    return "older adult"

# --- Prediction & Tips ---
def diabetes_prediction(input_data, model):
    input_np = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_np)
    return int(prediction[0])

def get_prevention_tips(age, bmi, glucose, prediction):
    tips = ["• 💧 Stay hydrated and avoid sugary drinks to stabilize blood sugar."]
    if categorize_bmi(bmi) in ["overweight", "obese"]:
        tips.append("• ⚖️ Aim for 5–10% weight loss through diet and activity.")
    if categorize_glucose(glucose) == "prediabetic":
        tips.append("• 🥗 Prioritize fiber-rich foods like legumes and vegetables.")
    if categorize_glucose(glucose) == "diabetic-range":
        tips.append("• 🩺 Begin glucose monitoring and consult a healthcare provider.")
    if categorize_age(age) == "young adult":
        tips.append("• 📱 Build healthy habits with apps and good sleep routines.")
    elif categorize_age(age) == "older adult":
        tips.append("• 👟 Try joint-friendly exercises like tai chi or walking.")
    if prediction == 1:
        tips.append("• 📆 Schedule a check-up to assess your risk in detail.")
    else:
        tips.append("• ✔️ Maintain healthy routines and check in annually.")
    return tips

# --- Main App ---
def main():
    df = pd.read_csv("diabetes.csv")
    model = get_model()

    st.markdown("<h1 style='text-align:center;color:#185a9d;'>🩺 Diabetes Risk & Prevention Advisor</h1>", unsafe_allow_html=True)
    st.image(Image.open("img.png"), use_container_width=True)

    with st.sidebar:
        st.markdown("🧑‍💻 Developed by: **Yameen Munir**")
        st.markdown("📧 yameenmunir05@gmail.com")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/yameen-munir/)")
        st.markdown("[GitHub](https://github.com/YameenMunir)")
        st.markdown("[Portfolio](https://www.datascienceportfol.io/YameenMunir)")

    st.markdown("## 🔎 Enter Your Health Information")
    Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    Glucose = st.number_input("Glucose Level", 0, 300, 100)
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin Level", 0, 1000, 80)
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    DPF = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
    Age = st.number_input("Age", 1, 120, 30)

    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]

    if st.button("🧠 Diabetes Test Result"):
        prediction = diabetes_prediction(input_data, model)
        result_text = "the person is diabetic" if prediction == 1 else "the person is not diabetic"
        st.success(f"📢 Result: {result_text}")

        # Generate and download report
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
        st.download_button("📄 Download Report", report, "diabetes_prediction_report.txt", mime="text/plain")

        # Tips
        st.markdown("## 🌱 Personalized Prevention Tips")
        for tip in get_prevention_tips(Age, BMI, Glucose, prediction):
            st.markdown(f"- {tip[1:].strip()}")

        # Feature Importance
        st.markdown("## 📊 Top Risk Contributors")
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))
        st.info(f"**Insight:** Most impactful feature: **{importance_df.iloc[0]['Feature']}**.")

        # SHAP Explainability
        st.markdown("## 🧠 SHAP Explanation")
        background_sample = df.drop("Outcome", axis=1).sample(100, random_state=42)
        explainer, shap_vals = get_shap_explainer(model, background_sample)

        input_np = np.array(input_data).reshape(1, -1)
        shap_values_single = explainer.shap_values(input_np)
        # Robustly handle binary/multiclass and array/list outputs
        if isinstance(shap_values_single, list):
            # For binary/multiclass, use the positive class (index 1)
            shap_values_to_plot = np.array(shap_values_single[1][0]).flatten()
            expected_value_to_plot = explainer.expected_value[1]
        else:
            shap_values_to_plot = np.array(shap_values_single[0]).flatten()
            expected_value_to_plot = explainer.expected_value
        # Ensure expected_value_to_plot is a scalar
        if isinstance(expected_value_to_plot, (np.ndarray, list)):
            expected_value_to_plot = np.array(expected_value_to_plot).flatten()[0]

        # Ensure feature_names, shap_values, and data all match in length
        feature_names = list(model.feature_names_in_)
        n_features = min(len(feature_names), len(input_np[0]), len(shap_values_to_plot))
        feature_names = feature_names[:n_features]
        input_np = input_np[:, :n_features]
        shap_values_to_plot = shap_values_to_plot[:n_features]

        shap.initjs()
        plt.figure(figsize=(8, 4))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_to_plot,
            base_values=expected_value_to_plot,
            data=input_np[0],
            feature_names=feature_names
        ))
        st.pyplot(plt.gcf(), clear_figure=True)
        st.info("💡 This chart shows how each feature pushed your result higher or lower.")

        # Detailed SHAP explanation for each feature
        st.markdown("### 📝 Detailed Feature Impact Explanation")
        for fname, fval, sval in zip(feature_names, input_np[0], shap_values_to_plot):
            direction = "increased" if sval > 0 else ("decreased" if sval < 0 else "did not affect")
            st.markdown(f"- **{fname}** (value: `{fval}`) {direction} your risk by `{abs(sval):.3f}` units.")
            if sval > 0:
                st.caption(f"A higher {fname} contributed to a higher predicted risk.")
            elif sval < 0:
                st.caption(f"A lower {fname} contributed to a lower predicted risk.")
            else:
                st.caption(f"This feature had little or no effect on your risk in this prediction.")

        # Extra Insights
        with st.expander("📈 Explore More Dataset Insights"):
            st.markdown("### 🧮 Risk Group Distribution")
            st.bar_chart(df["Outcome"].value_counts().rename({0: "Non-Diabetic", 1: "Diabetic"}))

            st.markdown("### 🔍 Age vs Glucose Scatter Plot")
            st.scatter_chart(df[["Age", "Glucose"]])

            st.markdown("### ⚖️ BMI Category Breakdown")
            df["BMI_Category"] = df["BMI"].apply(categorize_bmi)
            st.bar_chart(df["BMI_Category"].value_counts())

        st.markdown("## 🧭 Health Monitoring Guide")
        st.markdown("""
- 🔬 Fasting glucose: every 3–6 months  
- ⚖️ Track BMI and waist size monthly  
- 🩸 Annual blood work (glucose, A1C, lipids)
""")

        st.markdown("## 🌐 Resources")
        st.markdown("""
- [American Diabetes Association](https://www.diabetes.org)  
- [CDC Diabetes Prevention Program](https://www.cdc.gov/diabetes/prevention/)  
- [Nutrition.gov – Diabetes](https://www.nutrition.gov/topics/diet-and-health-conditions/diabetes)
""")

if __name__ == "__main__":
    main()