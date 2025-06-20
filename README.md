# 🩺 Diabetes Risk & Prevention Advisor

A modern Streamlit web app that predicts your risk of developing diabetes and provides personalized, evidence-based prevention tips. No API keys required—runs fully locally!

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
