import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

print("🔄 TRAINING HEART DISEASE MODEL...")

# Load data
df = pd.read_csv('heart_disease_uci.csv')
TARGET_COL = 'num'

# Disease descriptions (professional)
disease_names = {
    0: "No Heart Disease",
    1: "Mild Coronary Disease", 
    2: "Moderate Heart Disease",
    3: "Severe Coronary Disease",
    4: "Critical Heart Condition"
}

disease_descriptions = {
    0: """
**Diagnosis: No Heart Disease Detected**

Heart function appears normal based on clinical parameters.
**Recommendations:**
• Annual cardiovascular screening
• Maintain healthy diet and 150min/week moderate exercise
• Monitor blood pressure (<130/80 mmHg) and cholesterol
• Risk Level: LOW
    """,
    1: """
**Diagnosis: Mild Coronary Artery Disease**

Early narrowing of coronary arteries detected.
**Immediate Actions:**
• Cardiology consultation within 1 month
• Start statin therapy if LDL >100 mg/dL
• Reduce saturated fat intake <7% calories
• Smoking cessation if applicable
**Risk Level: MODERATE**
    """,
    2: """
**Diagnosis: Moderate Heart Disease**

Significant coronary artery stenosis present.
**Medical Plan:**
• Cardiology follow-up within 2 weeks
• Stress testing recommended
• Optimize blood pressure (<130/80) and diabetes control
• Supervised cardiac rehabilitation
**Risk Level: HIGH**
    """,
    3: """
**Diagnosis: Severe Coronary Disease**

Advanced multi-vessel coronary disease.
**Urgent Actions:**
• Cardiology evaluation within 48 hours
• Consider coronary angiography
• Strict medication adherence (antiplatelets, statins)
• Avoid heavy exertion
**Risk Level: VERY HIGH**
    """,
    4: """
**Diagnosis: Critical Heart Condition**

Advanced disease requiring immediate intervention.
**Emergency Protocol:**
• IMMEDIATE MEDICAL EVALUATION REQUIRED
• Urgent revascularization consideration
• Hospital admission if symptomatic
• Strict medical supervision
**Risk Level: CRITICAL**
    """
}

print(f"Dataset: {df.shape[0]} patients, {df.shape[1]} features")
print(f"Target distribution:\n{df[TARGET_COL].value_counts().sort_index()}")

# Encode categorical features
print("🔄 Encoding categorical features...")
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).fillna(1)
df['cp'] = df['cp'].map({
    'typical angina': 0, 'atypical angina': 1, 
    'non-anginal': 2, 'asymptomatic': 3
}).fillna(2)
df['restecg'] = df['restecg'].map({
    'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2
}).fillna(0)
df['exang'] = df['exang'].map({True: 1, False: 0}).fillna(0)
df['slope'] = df['slope'].map({
    'upsloping': 0, 'flat': 1, 'downsloping': 2
}).fillna(1)
df['thal'] = df['thal'].map({
    'normal': 0, 'fixed defect': 1, 'reversable defect': 2
}).fillna(0)
df['fbs'] = df['fbs'].map({True: 1, False: 0}).fillna(0)

# Clean data
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# Prepare features and target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Handle missing values
print("🔄 Imputing missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("🔄 Training Logistic Regression...")
model = LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ MODEL ACCURACY: {accuracy:.3f}")

print("\nClassification Report:")
target_names_str = [disease_names[i] for i in sorted(disease_names)]
print(classification_report(y_test, y_pred, target_names=target_names_str))

# Save everything
print("\n💾 SAVING MODEL FILES...")
pickle.dump(model, open('heart_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(imputer, open('imputer.pkl', 'wb'))
pickle.dump({'names': disease_names, 'descriptions': disease_descriptions}, 
            open('disease_info.pkl', 'wb'))

print("\n🎉 SUCCESS! Files created:")
print("- heart_model.pkl")
print("- scaler.pkl") 
print("- imputer.pkl")
print("- disease_info.pkl")
print("\n🚀 Run: streamlit run app.py")
