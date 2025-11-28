import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import os
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Cardiac Risk Assessment System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Blue/White/Black Corporate Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body, .main {
        font-family: 'Inter', sans-serif;
    }
    
    .header-primary {
        color: #1e3a8a !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        letter-spacing: -0.02em !important;
    }
    
    .header-secondary {
        color: #1e40af !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    
    .metric-low {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15) !important;
    }
    
    .metric-high {
        background: linear-gradient(135deg, #1e4976 0%, #2563eb 100%) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2) !important;
    }
    
    .metric-critical {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 8px 20px rgba(30, 58, 138, 0.25) !important;
    }
    
    .stMetric > label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    .stMetric > div > div {
        color: #1e3a8a !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
    }
    
    .social-links {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    
    .btn-primary-custom {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%) !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 500 !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to serve local HTML files inside Streamlit tabs
def serve_html_page(filename, height=700):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            page_html = f.read()
        components.html(page_html, height=height, scrolling=True)
    else:
        st.error(f"File not found: {filename}")

# Check model files exist before loading
required_files = ['heart_model.pkl', 'scaler.pkl', 'imputer.pkl', 'disease_info.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ Missing model files: {', '.join(missing_files)}")
    st.info("Please run `python train_model.py` first to create these files.")
    st.stop()

# Load models using cache
@st.cache_resource
def load_models():
    model = pickle.load(open('heart_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    imputer = pickle.load(open('imputer.pkl', 'rb'))
    disease_info = pickle.load(open('disease_info.pkl', 'rb'))
    return model, scaler, imputer, disease_info

model, scaler, imputer, disease_info = load_models()
disease_names = disease_info['names']
disease_descriptions = disease_info['descriptions']

# Main tabs including login and registration (embedded)
tab_login, tab_register, tab_main = st.tabs(["Login", "Register", "Cardiac Risk Assessment"])

with tab_login:
    st.markdown('<h2 class="header-secondary">User Login</h2>', unsafe_allow_html=True)
    serve_html_page("index.html")  # Login HTML file inside project folder

with tab_register:
    st.markdown('<h2 class="header-secondary">User Registration</h2>', unsafe_allow_html=True)
    serve_html_page("index1.html")  # Registration HTML file inside project folder

with tab_main:
    st.markdown('<h1 class="header-primary">Cardiac Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced machine learning model for cardiovascular risk stratification.*")

    # Split app tabs inside main tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Risk Assessment", "Risk Analytics", "Clinical Guidelines", 
        "Model Performance", "System Information"
    ])

    with tab1:
        st.markdown('<h2 class="header-secondary">Patient Risk Assessment</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Patient Demographics**")
            age = st.slider("Age (years)", 20, 90, 55, 
                            help="Patient chronological age")
            sex_options = ["Male", "Female"]
            sex = st.selectbox("Sex", sex_options)
            sex_val = sex_options.index(sex)
        with col2:
            st.markdown("**Primary Risk Factors**")
            cp_options = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
            cp = st.selectbox("Chest Pain Type", cp_options)
            cp_val = cp_options.index(cp)
            trestbps = st.slider("Resting BP (mmHg)", 90, 250, 130)

        col3, col4 = st.columns(2)
        with col3:
            chol = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)
        with col4:
            st.markdown("**Exercise Parameters**")
            thalach = st.slider("Maximum Heart Rate (bpm)", 70, 220, 150)

        with st.sidebar:
            st.markdown("**Advanced Clinical Parameters**")
            fbs_options = ["No (>120 mg/dL)", "Yes"]
            fbs = st.selectbox("Fasting Blood Sugar", fbs_options)
            fbs_val = 1 if fbs == "Yes" else 0

            restecg_options = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
            restecg = st.selectbox("Resting ECG", restecg_options)
            restecg_val = restecg_options.index(restecg)

            exang_options = ["No", "Yes"]
            exang = st.selectbox("Exercise-Induced Angina", exang_options)
            exang_val = exang_options.index(exang)

            oldpeak = st.slider("ST Depression (mm)", 0.0, 6.0, 1.0, 0.1)

            slope_options = ["Upsloping", "Flat", "Downsloping"]
            slope = st.selectbox("ST Segment Slope", slope_options)
            slope_val = slope_options.index(slope)

            ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])

            thal_options = ["Normal", "Fixed Defect", "Reversible Defect"]
            thal = st.selectbox("Thalassemia", thal_options)
            thal_val = thal_options.index(thal)

        if st.button("Generate Risk Assessment", type="primary", use_container_width=True):
            input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                                   thalach, exang_val, oldpeak, slope_val, ca, thal_val]])

            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = np.max(probabilities)

            st.markdown("---")
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown("## Primary Diagnosis")
                st.markdown(f"**{list(disease_names.values())[prediction]}**")
                st.markdown(disease_descriptions[prediction])

            with col2:
                risk_class = "LOW" if prediction == 0 else "HIGH" if prediction < 3 else "CRITICAL"
                metric_class = f"metric-{risk_class.lower()}"
                st.markdown(f"""
                <div class="{metric_class}">
                    <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>Risk Classification</div>
                    <div style='font-size: 2.2rem; font-weight: 700;'>{risk_class}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.metric("Prediction Confidence", f"{confidence:.1%}")

            st.markdown("### Probability Distribution Across Risk Categories")
            fig = px.bar(
                x=list(disease_names.values()),
                y=probabilities,
                title="Model output probabilities for each disease severity level",
                color=probabilities,
                color_continuous_scale=["#e0f2fe", "#1e40af"],
                text=probabilities.round(3)
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=450, font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<h2 class="header-secondary">Risk Factor Analytics</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Critical Risk Thresholds**")
            st.markdown("""
            | Parameter | Low Risk | High Risk | Critical |
            |-----------|----------|-----------|----------|
            | Age | <50 | 50-65 | >65 |
            | Cholesterol (mg/dL) | <200 | 200-300 | >300 |
            | Maximum Heart Rate | >160 | 120-160 | <120 |
            | ST Depression (mm) | 0 | 0.5-2 | >2 |
            """)

        with col2:
            st.markdown("**Age vs Disease Severity**")
            # Sample analytics data - replace with real if available
            age_data = pd.DataFrame({
                'Age Group': ['<45', '45-55', '55-65', '>65'],
                'Disease Prevalence': [0.12, 0.28, 0.45, 0.68]
            })
            fig_age = px.bar(age_data, x='Age Group', y='Disease Prevalence',
                             title="Disease prevalence by age group",
                             color='Disease Prevalence',
                             color_continuous_scale="Blues")
            st.plotly_chart(fig_age, use_container_width=True)

    with tab3:
        st.markdown('<h2 class="header-secondary">Clinical Guidelines</h2>', unsafe_allow_html=True)

        st.markdown("""
        **Management Protocol by Risk Stratification:**

        **LOW RISK (Level 0):** Annual screening and lifestyle maintenance

        **HIGH RISK (Levels 1-2):** Cardiology consultation and medical therapy

        **CRITICAL RISK (Levels 3-4):** Urgent cardiology care and possible intervention

        *Guidelines from ESC/ACC/AHA 2023*
        """)

    with tab4:
        st.markdown('<h2 class="header-secondary">Model Performance</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Details**")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | Algorithm | Multinomial Logistic Regression |
            | Training Samples | 734 |
            | Test Samples | 183 |
            | Classes | 5 severity levels |
            | Test Accuracy | 54.6% |
            | Features | 13 clinical parameters |
            """)

        with col2:
            st.markdown("**Confusion Matrix (Sample)**")
            # Example confusion matrix heatmap, replace with real data if available
            import plotly.graph_objects as go
            cm_data = go.Figure(data=go.Heatmap(
                z=[[45, 5, 0, 0, 0],
                   [8, 30, 10, 2, 0],
                   [3, 12, 28, 7, 2],
                   [1, 4, 9, 25, 7],
                   [0, 2, 3, 10, 20]],
                x=list(disease_names.values()),
                y=list(disease_names.values()),
                colorscale='Blues',
                texttemplate="%{z}",
                textfont={"size": 12},
                hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count: %{z}<extra></extra>"
            ))
            cm_data.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(cm_data, use_container_width=True)

    with tab5:
        st.markdown('<h2 class="header-secondary">System Information & Social Links</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Technical Architecture**

            - Model: LogisticRegression (multinomial)
            - Preprocessing: Median imputation + Standard scaling
            - Dataset: UCI Heart Disease (917 samples)
            - Deployment: Streamlit + Plotly
            - Features: 13 clinical parameters

            *For educational use only.*
            """)

        with col2:
            st.markdown("""
            <div class="social-links">
                <p style='margin: 0.5rem 0; font-weight: 600; color: #1e3a8a;'>Connect With Us</p>
                <a href="https://linkedin.com/in/yourprofile" target="_blank" style="color: #3b82f6; text-decoration: none;">LinkedIn</a><br>
                <a href="https://github.com/yourusername" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub</a><br>
                <a href="https://twitter.com/yourhandle" target="_blank" style="color: #3b82f6; text-decoration: none;">Twitter</a>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
    © 2025 Cardiac Risk Assessment System | 
    <a href='https://github.com/yourusername/heart-disease-predictor' target='_blank' style='color: #3b82f6;'>Source Code</a> | 
    For demonstration purposes only
</div>
""", unsafe_allow_html=True)
