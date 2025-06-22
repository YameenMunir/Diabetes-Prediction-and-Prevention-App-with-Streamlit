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
        st.markdown("👔 [LinkedIn](https://www.linkedin.com/in/yameen-munir/)")
        st.markdown("🐙 [GitHub](https://github.com/YameenMunir)")
        st.markdown("🌐 [Portfolio](https://www.datascienceportfol.io/YameenMunir)")

    st.markdown("## 🔎 Enter Your Health Information")
    Pregnancies = st.number_input("Pregnancies ℹ️", 0, 20, 0, help="Number of times you have been pregnant (0 if never).")
    Glucose = st.number_input("Glucose Level ℹ️", 0, 300, 100, help="Plasma glucose concentration after fasting (mg/dL).")
    BloodPressure = st.number_input("Blood Pressure ℹ️", 0, 200, 70, help="Diastolic blood pressure (mm Hg). Normal is around 80 mm Hg.")
    SkinThickness = st.number_input("Skin Thickness ℹ️", 0, 100, 20, help="Triceps skin fold thickness (mm). Indicates body fat.")
    Insulin = st.number_input("Insulin Level ℹ️", 0, 1000, 80, help="2-Hour serum insulin (mu U/ml). Measures insulin in blood.")
    # Height input with unit switch
    height_unit = st.radio("Select height unit:", ("cm", "ft/in"), horizontal=True, help="Choose your preferred height unit.")
    if height_unit == "cm":
        Height = st.number_input("Height (cm) ℹ️", 100, 250, 170, help="Your height in centimeters.")
    else:
        Height_ft = st.number_input("Height (feet) ℹ️", 3, 8, 5, help="Feet part of your height.")
        Height_in = st.number_input("Height (inches) ℹ️", 0, 11, 7, help="Inches part of your height.")
        Height = Height_ft * 30.48 + Height_in * 2.54  # convert to cm
        st.markdown(f"**Total Height:** {Height:.1f} cm")
    Weight = st.number_input("Weight (kg) ℹ️", 30, 200, 70, help="Your weight in kilograms.")
    # Calculate BMI
    BMI = Weight / ((Height / 100) ** 2)
    st.markdown(f"**Calculated BMI:** {BMI:.2f}")
    DPF = st.number_input("Diabetes Pedigree Function ℹ️", 0.0, 2.5, 0.5, step=0.01, help="Likelihood of diabetes based on family history.")
    Age = st.number_input("Age ℹ️", 1, 120, 30, help="Your age in years.")

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
Height (cm): {Height}\nWeight (kg): {Weight}\nBMI: {BMI:.2f}
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
        import matplotlib.pyplot as plt  # Ensure plt is available for SHAP and custom plots
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

        # Improved Detailed SHAP explanation for each feature
        st.markdown("### 📝 Detailed Feature Impact Explanation")
        # Human-friendly explanations for each feature
        feature_explanations = {
            "Pregnancies": "Number of times you have been pregnant. More pregnancies can be associated with higher diabetes risk.",
            "Glucose": "Plasma glucose concentration after fasting. High glucose levels are a strong indicator of diabetes risk.",
            "BloodPressure": "Diastolic blood pressure (mm Hg). High blood pressure is a risk factor for diabetes and related complications.",
            "SkinThickness": "Triceps skin fold thickness (mm). Indicates body fat percentage, which can affect diabetes risk.",
            "Insulin": "2-Hour serum insulin (mu U/ml). Abnormal insulin levels can signal insulin resistance or diabetes.",
            "BMI": "Body Mass Index, calculated from height and weight. Higher BMI is linked to increased diabetes risk.",
            "DiabetesPedigreeFunction": "Likelihood of diabetes based on family history. Higher values mean greater genetic risk.",
            "Age": "Your age in years. Risk of diabetes increases with age."
        }
        # Sort features by absolute impact
        impact_tuples = sorted(zip(feature_names, input_np[0], shap_values_to_plot), key=lambda x: abs(x[2]), reverse=True)
        for fname, fval, sval in impact_tuples:
            if sval > 0:
                badge = "🟥"
                direction = "increased"
                summary = f"A higher {fname} contributes to a higher predicted risk."
            elif sval < 0:
                badge = "🟩"
                direction = "decreased"
                summary = f"A lower {fname} contributes to a lower predicted risk."
            else:
                badge = "⬜"
                direction = "did not affect"
                summary = f"This feature had little or no effect on your risk in this prediction."
            explanation = feature_explanations.get(fname, "No explanation available for this feature.")
            with st.expander(f"{badge} {fname} (value: {fval})"):
                st.markdown(f"- **Impact:** {direction} your risk by `{abs(sval):.3f}` units.")
                st.info(summary)
                st.caption(f"ℹ️ {explanation}")

        # 🗂️ Explore More Dataset Insights
        st.markdown("## 🗂️ Explore More Dataset Insights")
        with st.container():
            with st.expander("🧮 Risk Group Distribution", expanded=False):
                st.bar_chart(df["Outcome"].value_counts().rename({0: "Non-Diabetic", 1: "Diabetic"}))
                st.caption("This chart shows the number of people in the dataset who are diabetic versus non-diabetic. It helps you understand the overall prevalence of diabetes in the sample.")
                st.markdown(
                    "> Imagine a room of 100 people from this dataset. This chart tells you how many of them are living with diabetes and how many are not. It gives you a sense of how common diabetes is in this group, and why prevention and awareness matter for everyone.")
                st.markdown(
                    "**Why it matters:**\n"
                    "A higher proportion of diabetics in the dataset may reflect real-world trends or specific risk factors in the studied population. If you see a large diabetic group, it highlights the importance of screening and early intervention. If the non-diabetic group is larger, it shows prevention is working, but vigilance is still needed.")

            with st.expander("📈 Average Glucose by Age Group", expanded=False):
                age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
                age_labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
                df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
                group_means = df.groupby('AgeGroup')['Glucose'].mean().reindex(age_labels)
                group_counts = df.groupby('AgeGroup')['Glucose'].count().reindex(age_labels)
                fig, ax1 = plt.subplots(figsize=(8, 4))
                color = '#185a9d'
                ax1.bar(age_labels, group_means, color=color, alpha=0.7)
                ax1.set_xlabel('Age Group (years)')
                ax1.set_ylabel('Average Glucose Level (mg/dL)', color=color)
                ax1.set_title('Average Glucose Level by Age Group')
                ax1.tick_params(axis='y', labelcolor=color)
                for i, v in enumerate(group_means):
                    if not pd.isna(v):
                        ax1.text(i, v + 2, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
                ax2 = ax1.twinx()
                line, = ax2.plot(age_labels, group_counts, color='red', marker='o', linestyle='--', alpha=0.7, label='Number of People')
                ax2.set_ylabel('Number of People', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend(loc='upper left', bbox_to_anchor=(1.18, 1))
                y_min, y_max = ax2.get_ylim()
                ax2.set_ylim(y_min, y_max * 1.15)
                st.pyplot(fig)
                st.caption("This bar chart shows the average glucose level for each age group, with a line showing the number of people in each group. It helps reveal whether older age groups tend to have higher average glucose levels.")
                st.markdown(
                    "> Each bar represents an age group, and its height shows the average glucose level for that group. The dotted line shows how many people are in each group. Look for a trend: do older age groups have higher bars? This can reveal how glucose levels change with age, highlighting the importance of regular health checks as we grow older.")
                st.markdown(
                    "**Why it matters:**\n"
                    "If you notice a trend where older age groups have higher average glucose, it suggests that aging is a risk factor for diabetes. This visualization encourages users to monitor their glucose as they age and to adopt healthy habits early to reduce risk.")

            with st.expander("⚖️ BMI Category Breakdown", expanded=False):
                df["BMI_Category"] = df["BMI"].apply(categorize_bmi)
                st.bar_chart(df["BMI_Category"].value_counts())
                st.caption("This chart displays the distribution of BMI categories (underweight, normal, overweight, obese) in the dataset. It helps you see how common each BMI group is among participants.")
                st.markdown(
                    "> Think of this as a snapshot of body types in the dataset. If most people fall into the overweight or obese categories, it signals a community-wide opportunity for healthier habits. Where do you fit in this picture?")
                st.markdown(
                    "**Why it matters:**\n"
                    "A high number of overweight or obese individuals in the dataset is a warning sign for increased diabetes risk in the community. This chart can motivate individuals and communities to focus on nutrition, exercise, and healthy living to shift the distribution toward normal BMI.")

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