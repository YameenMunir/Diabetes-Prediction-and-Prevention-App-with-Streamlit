# 🩺 Diabetes Risk & Prevention Advisor

A modern Streamlit web app that predicts your risk of developing diabetes and provides personalized, evidence-based prevention tips. No API keys required—runs fully locally!

## 🔗 Link to Application

[Link to Application](https://diabetes-prediction-and-prevention-app.streamlit.app/)

## Features
- **Diabetes risk prediction** using a trained Random Forest model
- **Personalized prevention tips** based on your health metrics (age, BMI, glucose, etc.)
- **Modern UI** with custom CSS styling
- **No cloud AI or API keys needed**
- **Health monitoring suggestions** and trusted resources

## How It Works
1. Enter your health metrics in the sidebar.
2. The app predicts your diabetes risk and probability.
3. Get tailored, actionable prevention tips and health monitoring advice.

## Setup & Run
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Diabetes-Prediction-and-Prevention-App-with-Streamlit-and-Gemini-AI
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn python-dotenv
   ```
3. **Run the app**
   ```bash
   streamlit run app.py
   ```
4. **Open in your browser**
   Go to [http://localhost:8501](http://localhost:8501)

## Files
- `app.py` — Main Streamlit app
- `diabetes.csv` — Dataset for model training
- `diabetes_model.pkl` — Pre-trained model (auto-generated if missing)
- `.env` — (Optional) For environment variables, not required for local use

## Deployment
- Do **not** commit your `.env` file or any API keys.
- The app is ready for deployment on Streamlit Cloud, Heroku, or any Python web host.

## Credits
- Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Built with [Streamlit](https://streamlit.io/)

## License
MIT License

## 🚀 Improvements
- Add user authentication for saving and tracking health data
- Integrate with wearable devices or health APIs for real-time data
- Support for additional languages and localization
- Add charts for historical trends if users return
- Use more advanced or explainable AI models
- Enable email or PDF export of results and tips
- Add a chatbot for interactive Q&A on diabetes prevention
- Improve accessibility and mobile responsiveness
- Add multi-user support with secure login and personal dashboards
- Integrate reminders/notifications for health monitoring
- Allow users to upload/export health data (CSV, PDF)
- Add interactive educational modules or quizzes about diabetes
- Enable real-time chat with healthcare professionals or AI assistant
- Support for dark mode and additional color themes
- Add voice input for accessibility
- Integrate with electronic health records (EHR) systems
- Provide localized dietary and exercise recommendations
- Add trend analysis and progress tracking over time

## UI & User Experience
- The app features a large, modern title at the very top for clear branding.
- A welcome image banner with a styled caption appears below the title.
- Each health metric input includes an information icon (ℹ️) with easy-to-read explanations.
- Sidebar credits are personalized for Yameen Munir, with updated links and no sensitive info.
- Custom CSS ensures a clean, visually appealing layout.

## SHAP Explainability & Detailed Feature Impact

The app provides a SHAP (SHapley Additive exPlanations) waterfall plot to visually explain how each health metric (feature) contributed to your diabetes risk prediction. Below the plot, a detailed explanation is provided for each feature, describing:
- The value you entered
- Whether it increased, decreased, or did not affect your risk
- A plain-language summary of the impact

This helps you understand not just the overall result, but also which factors are most important for your personal risk profile.

## 🗂️ Explore More Dataset Insights

This section of the app provides interactive, collapsible visualizations to help you understand the dataset and your risk in context:

- **🧮 Risk Group Distribution:** See how many people in the dataset are diabetic vs. non-diabetic, with an explanation of why this matters.
- **📈 Average Glucose by Age Group:** Explore how average glucose levels and group sizes change across age groups, with a combined bar/line chart and plain-language insights.
- **⚖️ BMI Category Breakdown:** View the distribution of BMI categories (underweight, normal, overweight, obese) and learn why this is important for diabetes risk.
- **🧬 Insulin Level Distribution:** Visualizes the spread of insulin levels in the dataset, highlighting common ranges and outliers. This helps users understand how their insulin value compares to the population and why abnormal insulin levels are a key risk factor for diabetes.
- **🩸 Blood Pressure by Age Group:** Shows the average blood pressure for different age groups, revealing trends and potential risk periods. This chart helps users see how blood pressure changes with age and why managing blood pressure is crucial for diabetes prevention.

Each visualization is in its own dropdown (expander) for a clean, user-friendly experience. Explanations and practical takeaways are provided for every chart.

---

## 💡 Even More Future Ideas

- Personalized risk trajectory forecasting using time-series modeling
- Integration with telemedicine platforms for direct doctor consultations
- AI-driven anomaly detection for early warning of health deterioration
- Community support forums and peer mentoring features
- Dynamic, interactive dashboards for custom data exploration
- Integration with pharmacy APIs for medication reminders and refills
- Secure blockchain-based health data storage and sharing
- Adaptive UI for users with visual impairments (e.g., screen reader optimization)
- Real-time translation for global accessibility
- Integration with national/regional diabetes registries for research and benchmarking
