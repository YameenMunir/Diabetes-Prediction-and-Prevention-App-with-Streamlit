import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Diabetes Prediction & Prevention", layout="wide")

# Load environment variables from .env file
# (Assuming you still want to keep this part for any future use)
from dotenv import load_dotenv
load_dotenv()

# Debug: Check if GEMINI_API_KEY is loaded (do not print the full key for security)
if os.getenv('GEMINI_API_KEY'):
    st.info('Gemini API key loaded from environment.')
else:
    st.error('Gemini API key NOT loaded. Please check your .env file and restart the app.')

# Initialize Hugging Face text generation pipeline
@st.cache_resource
def get_hf_generator():
    return pipeline('text-generation', model='gpt2')

# Load or train model
@st.cache_resource
def get_model():
    # Try to load pre-trained model
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # If no pre-trained model exists, train a new one
        # Load dataset
        df = pd.read_csv('diabetes.csv')  # Assuming you have this dataset
        
        # Split data
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model for future use
        with open('diabetes_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return model

# Function to get prevention tips from Hugging Face GPT-2
def get_prevention_tips(age, bmi, glucose, prediction):
    prompt = f"""
    You are a medical assistant providing diabetes prevention advice. 
    Provide 5-7 specific, actionable prevention tips for:
    
    - A {age}-year-old person
    - With BMI of {bmi}
    - Glucose level of {glucose}
    - Currently {'at risk of diabetes' if prediction == 1 else 'not at high risk of diabetes'}
    
    Recommendations should be:
    - Practical lifestyle changes
    - Dietary recommendations
    - Monitoring suggestions
    - Formatted as a bulleted list
    - Clear and concise
    - Evidence-based
    """
    try:
        generator = get_hf_generator()
        response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        # Extract only the bulleted list from the response if possible
        tips_start = response.find('-')
        if tips_start != -1:
            return response[tips_start:]
        return response
    except Exception as e:
        st.error(f"Error generating prevention tips: {str(e)}")
        return "Could not generate prevention tips. Please try again."

# Main app function
def main():
    st.title("Diabetes Risk Prediction & Prevention")
    st.write("""
    This app predicts your risk of developing diabetes based on health metrics 
    and provides personalized prevention suggestions powered by Gemini AI.
    """)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Enter Your Health Metrics")
        pregnancies = st.number_input('Number of Pregnancies (if applicable)', min_value=0, max_value=20, value=0)
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=1000, value=80)
        bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input('Age', min_value=1, max_value=120, value=30)
        
        submit_button = st.button('Predict Diabetes Risk')
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Input Summary")
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
            
            st.write(pd.DataFrame([input_data]))
            
            # Make prediction
            model = get_model()
            input_array = np.array([list(input_data.values())])
            prediction = model.predict(input_array)[0]
            prediction_proba = model.predict_proba(input_array)[0]
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"High Risk of Diabetes (Probability: {prediction_proba[1]*100:.1f}%)")
            else:
                st.success(f"Low Risk of Diabetes (Probability: {prediction_proba[0]*100:.1f}%)")
            
            # Show feature importance
            st.subheader("Key Contributing Factors")
            feature_importance = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(feature_importance.set_index('Feature'))
    
    with col2:
        if submit_button:
            st.subheader("Personalized Prevention Tips")
            with st.spinner('Generating personalized recommendations with AI...'):
                tips = get_prevention_tips(age, bmi, glucose, prediction)
                st.markdown(tips)
            
            st.subheader("Health Monitoring Suggestions")
            st.write("""
            - Check fasting glucose every 3-6 months if at risk
            - Monitor blood pressure regularly
            - Track BMI and waist circumference monthly
            - Annual comprehensive metabolic panel
            """)
            
            st.subheader("Resources")
            st.write("""
            - [American Diabetes Association](https://www.diabetes.org)
            - [CDC Diabetes Prevention Program](https://www.cdc.gov/diabetes/prevention/index.html)
            - [Nutrition.gov Diabetes Resources](https://www.nutrition.gov/topics/diet-and-health-conditions/diabetes)
            """)

if __name__ == '__main__':
    main()