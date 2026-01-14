# Run this script to find and test actual high-risk patients
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv(r'C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\Filtered_PulseOximeter_RiskData.csv')

# Clean it (same as training)
df = df.dropna(subset=['Age'])
df = df[(df['SpO2'] >= 70) & (df['SpO2'] <= 100)]
df = df[(df['HeartRate'] >= 40) & (df['HeartRate'] <= 200)]

# Find patients with multiple high risks
risk_cols = ['Asthma_Exacerbation_Risk', 'COPD_Risk', 'Pneumonia_Risk', 'COVID_Like_Risk', 'Anemia_Risk']
df['total_risks'] = df[risk_cols].sum(axis=1)

# Get top 5 highest risk patients
high_risk_patients = df.nlargest(5, 'total_risks')
print("Top 5 highest risk patients:")
for i, (idx, row) in enumerate(high_risk_patients.iterrows()):
    print(f"\nPatient {i+1} (Index {idx}):")
    print(f"  Risks: {row[risk_cols].to_dict()}")
    print(f"  Age: {row['Age']:.1f}, SpO2: {row['SpO2']:.1f}, HR: {row['HeartRate']:.1f}")
    
    # Extract the 27 features in correct order
    features = [
        row['Age'], row['SpO2'], row['HeartRate'], row['PerfusionIndex'],
        row['Temperature'], row['PulseVariability'], row['Cough'],
        row['Shortness_of_Breath'], row['Chest_Pain'], row['Fatigue'],
        row['Dizziness'], row['Nausea'], row['Confusion'], row['Sore_Throat'],
        row['Runny_Nose'], row['Age_Group'], row['HR_Zone'], row['SpO2_Zone'],
        row['Oxygenation_Score'], row['Perfusion_Score'], row['Temperature_Score'],
        row['Respiratory_Symptom_Cluster'], row['Systemic_Symptom_Cluster'],
        row['Cardiac_Symptom_Cluster'], row['Age_SpO2_Interaction'],
        row['HR_Temp_Interaction'], row['PI_Age_Interaction']
    ]

    print(f"  Features for API: {features[:]}...")  # Show all features