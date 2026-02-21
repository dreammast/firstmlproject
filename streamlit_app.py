import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Page configuration
st.set_page_config(
    page_title="Student Math Score Predictor",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-top: 30px;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 Student Math Score Predictor</h1>
    <p>Predict student math scores based on their academic profile</p>
</div>
""", unsafe_allow_html=True)

# Create form layout
col1, col2 = st.columns(2)

# Demographic Information Section
st.markdown('<h3 class="section-title">👤 Demographic Information</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox(
        "Gender",
        ["", "Male", "Female"],
        key="gender"
    )
    gender_value = "male" if gender == "Male" else ("female" if gender == "Female" else "")

with col2:
    race_ethnicity = st.selectbox(
        "Race/Ethnicity",
        ["", "Group A", "Group B", "Group C", "Group D", "Group E"],
        key="race_ethnicity"
    )
    race_value = race_ethnicity.lower() if race_ethnicity else ""

# Parent Education & Lunch Information
st.markdown('<h3 class="section-title">👨‍👩‍👧 Family Background</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    parental_education = st.selectbox(
        "Parental Level of Education",
        ["", "Some High School", "High School", "Some College", "Associate Degree", "Bachelor Degree", "Master Degree"],
        key="education"
    )
    education_mapping = {
        "Some High School": "some high school",
        "High School": "high school",
        "Some College": "some college",
        "Associate Degree": "associate degree",
        "Bachelor Degree": "bachelor degree",
        "Master Degree": "master degree"
    }
    education_value = education_mapping.get(parental_education, "")

with col2:
    lunch = st.selectbox(
        "Lunch Type",
        ["", "Standard", "Free/Reduced"],
        key="lunch"
    )
    lunch_value = lunch.lower() if lunch and lunch != "" else ""

# Test & Academic Scores
st.markdown('<h3 class="section-title">📊 Academic Performance</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    reading_score = st.number_input(
        "Reading Score (0-100)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key="reading"
    )

with col2:
    writing_score = st.number_input(
        "Writing Score (0-100)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key="writing"
    )

# Test Preparation
st.markdown('<h3 class="section-title">🎓 Test Preparation</h3>', unsafe_allow_html=True)
test_prep = st.selectbox(
    "Test Preparation Course",
    ["", "None", "Completed"],
    key="test_prep"
)
test_prep_value = test_prep.lower() if test_prep else ""

# Prediction Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🎯 Predict Math Score", use_container_width=True):
    # Validate inputs
    if not all([gender_value, race_value, education_value, lunch_value, test_prep_value]):
        st.error("⚠️ Please fill in all fields before making a prediction.")
    else:
        try:
            # Create custom data
            data = CustomData(
                gender=gender_value,
                race_ethnicity=race_value,
                parental_level_of_education=education_value,
                lunch=lunch_value,
                test_preparation_course=test_prep_value,
                reading_score=float(reading_score),
                writing_score=float(writing_score)
            )
            
            # Get data as dataframe
            pred_df = data.get_data_as_dataframe()
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(pred_df)
            
            # Display result
            st.markdown(f"""
            <div class="result-box">
                <div>✓ Prediction Completed Successfully!</div>
                <div style="font-size: 0.9rem; margin-top: 10px;">Predicted Math Score:</div>
                <div class="result-value">{prediction[0]:.2f}</div>
                <div style="font-size: 0.9rem;">Based on the input features and trained ML model</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display input summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 Input Summary")
            summary_data = {
                "Gender": gender,
                "Race/Ethnicity": race_ethnicity,
                "Parental Education": parental_education,
                "Lunch Type": lunch,
                "Test Preparation": test_prep,
                "Reading Score": reading_score,
                "Writing Score": writing_score
            }
            st.dataframe(pd.DataFrame([summary_data]).T, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
---
**Student Performance Prediction System** | Powered by Machine Learning
""")
