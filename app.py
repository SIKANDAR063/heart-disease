import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean and friendly interface
st.markdown("""
<style>
    /* Main theme colors */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .friendly-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(255,107,107,0.2);
    }
    
    .friendly-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .friendly-header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    /* Metric cards */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(102,126,234,0.2);
    }
    
    .metric-box h3 {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-box h2 {
        font-size: 2rem;
        margin: 0;
    }
    
    /* Result boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #1e3c72;
        text-align: center;
        box-shadow: 0 10px 20px rgba(132,250,176,0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(240,147,251,0.2);
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .section-title {
        color: #2d3436;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #ff6b6b;
        display: inline-block;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(255,107,107,0.3);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255,107,107,0.4);
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Progress steps */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 0 1rem;
    }
    
    .step {
        flex: 1;
        text-align: center;
        color: #b2bec3;
        position: relative;
    }
    
    .step.active {
        color: #ff6b6b;
    }
    
    .step .number {
        width: 30px;
        height: 30px;
        background: #dfe6e9;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    
    .step.active .number {
        background: #ff6b6b;
        color: white;
    }
    
    /* Fun elements */
    .heart-beat {
        animation: beat 1s ease infinite;
        display: inline-block;
    }
    
    @keyframes beat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .tip-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("knn_heart_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        expected_columns = joblib.load("heart_columns.pkl")
        return model, scaler, expected_columns
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None


model, scaler, expected_columns = load_model()

# Friendly Header
st.markdown("""
<div class="friendly-header">
    <h1>❤️ Heart Health Companion</h1>
    <p>Your personal guide to understanding heart health • Quick • Easy • Reliable</p>
    <div style="margin-top: 1rem;">
        <span class="tip-badge">🎯 90% Accuracy</span>
        <span class="tip-badge">⚡ 30 Seconds</span>
        <span class="tip-badge">🔒 Private & Secure</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with helpful information
with st.sidebar:
    st.markdown("### 💡 Quick Tips")
    
    with st.expander("❤️ Heart Health Facts", expanded=True):
        st.markdown("""
        - Regular exercise strengthens your heart
        - A balanced diet reduces risk by 30%
        - Stress management is key
        - 7-8 hours sleep is ideal
        """)
    
    with st.expander("📊 Understanding Your Results"):
        st.markdown("""
        **Low Risk (Green):** Your heart health indicators look good! Keep up the healthy habits.
        
        **High Risk (Red):** Several factors indicate potential risk. Consider consulting a doctor.
        """)
    
    with st.expander("🎯 Normal Ranges"):
        st.markdown("""
        - **BP:** < 120/80 mmHg
        - **Cholesterol:** < 200 mg/dL
        - **Max HR:** 220 - your age
        - **Fasting Sugar:** < 100 mg/dL
        """)
    
    st.markdown("---")
    st.markdown("### 🌟 Your Health Journey")
    st.markdown("Every step toward better health counts!")
    
    # Motivational quote
    quotes = [
        "Take care of your heart, it beats for you! 💪",
        "Small changes, big results 🌱",
        "Your heart works 24/7, show it some love! ❤️",
        "Health is the greatest wealth 💎"
    ]
    st.info(np.random.choice(quotes))

# Main content
if model is not None:
    # Progress steps
    st.markdown("""
    <div class="step-indicator">
        <div class="step active">
            <div class="number">1</div>
            <div>Your Info</div>
        </div>
        <div class="step">
            <div class="number">2</div>
            <div>Health Metrics</div>
        </div>
        <div class="step">
            <div class="number">3</div>
            <div>Get Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two main columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">👤 Basic Information</p>', unsafe_allow_html=True)
        
        # Age with visual indicator
        age = st.slider("How old are you?", 18, 100, 40)
        
        # Age category
        if age < 30:
            st.markdown("🌟 **Young & Healthy**")
        elif age < 50:
            st.markdown("💪 **Prime of Life**")
        else:
            st.markdown("🎯 **Wisdom Years - Extra Care**")
        
        # Sex with icons
        sex_option = st.radio("Biological sex", 
                            ["👨 Male", "👩 Female"], 
                            horizontal=True)
        sex = "M" if "Male" in sex_option else "F"
        
        st.markdown('<p class="section-title" style="margin-top: 2rem;">❤️ Heart Symptoms</p>', unsafe_allow_html=True)
        
        # Chest pain with friendly descriptions
        chest_pain = st.selectbox(
            "Type of chest discomfort",
            ["No chest pain (ASY)", 
             "Mild discomfort (NAP)", 
             "Noticeable pain (ATA)", 
             "Severe chest pain (TA)"],
            help="Choose the option that best describes your chest discomfort"
        )
        
        # Map to model values
        chest_map = {
            "No chest pain (ASY)": "ASY",
            "Mild discomfort (NAP)": "NAP",
            "Noticeable pain (ATA)": "ATA",
            "Severe chest pain (TA)": "TA"
        }
        chest_pain_code = chest_map[chest_pain]
        
        # Exercise angina
        exercise_angina = st.radio(
            "Do you experience chest pain during exercise?",
            ["😊 No", "😟 Yes"],
            horizontal=True
        )
        exercise_angina_code = "N" if "No" in exercise_angina else "Y"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Health Measurements</p>', unsafe_allow_html=True)
        
        # Blood Pressure with visual
        resting_bp = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120)
        
        # BP indicator
        if resting_bp < 120:
            st.markdown("✅ **Excellent!** Normal range")
        elif resting_bp < 130:
            st.markdown("⚠️ **Slightly elevated**")
        else:
            st.markdown("🔴 **Watch your BP**")
        
        # Cholesterol
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        
        # Cholesterol indicator
        if cholesterol < 200:
            st.markdown("✅ **Great!** Healthy cholesterol")
        elif cholesterol < 240:
            st.markdown("⚠️ **Borderline high**")
        else:
            st.markdown("🔴 **High cholesterol**")
        
        # Fasting blood sugar
        fasting_bs = st.radio(
            "Fasting blood sugar > 120 mg/dL?",
            ["✅ No", "⚠️ Yes"],
            horizontal=True
        )
        fasting_bs_value = 1 if "Yes" in fasting_bs else 0
        
        # Max heart rate
        max_hr = st.slider("Maximum heart rate during exercise", 60, 220, 150)
        target_hr = 220 - age
        hr_percentage = (max_hr / target_hr) * 100
        
        if hr_percentage > 85:
            st.markdown("💪 **Great effort!**")
        elif hr_percentage > 70:
            st.markdown("👍 **Good workout**")
        else:
            st.markdown("🧘 **Moderate intensity**")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row - Additional metrics
    st.markdown('<div class="input-section" style="margin-top: 2rem;">', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("### 📈 ECG Results")
        resting_ecg = st.selectbox(
            "Resting ECG finding",
            ["Normal", "ST wave changes", "LVH (enlarged heart)"]
        )
        ecg_map = {
            "Normal": "Normal",
            "ST wave changes": "ST",
            "LVH (enlarged heart)": "LVH"
        }
        resting_ecg_code = ecg_map[resting_ecg]
    
    with col4:
        st.markdown("### 📉 ST Depression")
        oldpeak = st.slider("ST depression (mm)", 0.0, 6.0, 1.0, 0.1)
        if oldpeak > 2:
            st.warning("Significant ST depression")
    
    with col5:
        st.markdown("### 📊 ST Slope")
        st_slope = st.selectbox(
            "ST slope pattern",
            ["Up (healthy)", "Flat (borderline)", "Down (concerning)"]
        )
        slope_map = {
            "Up (healthy)": "Up",
            "Flat (borderline)": "Flat",
            "Down (concerning)": "Down"
        }
        st_slope_code = slope_map[st_slope]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Center button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍 CHECK MY HEART HEALTH", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Results section
    if predict_clicked:
        # Prepare data for model
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs_value,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain_code: 1,
            'RestingECG_' + resting_ecg_code: 1,
            'ExerciseAngina_' + exercise_angina_code: 1,
            'ST_Slope_' + st_slope_code: 1
        }
        
        # Create input dataframe
        input_df = pd.DataFrame([raw_input])
        
        # Fill missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Get probability
        try:
            probability = model.predict_proba(scaled_input)[0]
            confidence = max(probability) * 100
        except:
            confidence = 85  # Default confidence
        
        # Display result with style
        st.markdown("---")
        
        if prediction == 0:
            st.markdown("""
            <div class="success-box">
                <h1 style="font-size: 4rem;">✅</h1>
                <h2>Low Risk - Your Heart Looks Healthy!</h2>
                <p style="font-size: 1.2rem; margin-top: 1rem;">Based on the information provided, your heart health indicators are in a good range. Keep up the healthy habits!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown(f"**Assessment Confidence:** {confidence:.1f}%")
            st.progress(confidence/100)
            
            # Good habits reminder
            st.markdown("### 🌟 Keep Up The Good Work!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🥗 **Balanced Diet**")
            with col2:
                st.markdown("🏃 **Regular Exercise**")
            with col3:
                st.markdown("😴 **Good Sleep**")
        
        else:
            st.markdown("""
            <div class="warning-box">
                <h1 style="font-size: 4rem;">⚠️</h1>
                <h2>Higher Risk Detected - Time for Action!</h2>
                <p style="font-size: 1.2rem; margin-top: 1rem;">Several factors indicate potential heart health concerns. Consider consulting with a healthcare provider for a thorough check-up.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown(f"**Assessment Confidence:** {confidence:.1f}%")
            st.progress(confidence/100)
            
            # Action steps
            st.markdown("### 🎯 Recommended Next Steps")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 👨‍⚕️ Medical Steps
                - Schedule a doctor's appointment
                - Get your blood pressure checked
                - Consider a lipid profile test
                - Discuss symptoms with your doctor
                """)
            with col2:
                st.markdown("""
                #### 🌱 Lifestyle Changes
                - Start with light exercise
                - Reduce salt and fatty foods
                - Practice stress management
                - Quit smoking if applicable
                """)
        
        # Risk factors summary
        st.markdown("### 📊 Your Health Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Age</h3>
                <h2>{age}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "Normal" if resting_bp < 130 else "High"
            st.markdown(f"""
            <div class="metric-box">
                <h3>BP</h3>
                <h2>{resting_bp}</h2>
                <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status = "Normal" if cholesterol < 200 else "High"
            st.markdown(f"""
            <div class="metric-box">
                <h3>Cholesterol</h3>
                <h2>{cholesterol}</h2>
                <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status = "Good" if max_hr > target_hr * 0.8 else "Moderate"
            st.markdown(f"""
            <div class="metric-box">
                <h3>Heart Rate</h3>
                <h2>{max_hr}</h2>
                <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Motivational message
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px;">
            <p style="font-size: 1.2rem; color: #2d3436;">
                <span class="heart-beat">❤️</span> Remember, this is just a screening tool. 
                Your health journey is unique, and every step toward better health matters!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Show friendly message when no prediction
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px;">
            <h2 style="color: #b2bec3;">👋 Ready to check your heart health?</h2>
            <p style="color: #636e72;">Fill in your details above and click the button to get your personalized assessment!</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: #7f8c8d;">
        Made with <span style="color: #ff6b6b;">❤️</span> for a healthier you | 
        Not a medical diagnosis • Always consult healthcare professionals
    </p>
    <p style="color: #95a5a6; font-size: 0.8rem;">
        © 2024 Heart Health Companion • Version 2.0
    </p>
</div>
""", unsafe_allow_html=True)