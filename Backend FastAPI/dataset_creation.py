# ===============================================
# Synthetic Dataset for 10 Disease Risk Prediction
# Using Oximeter Data + Symptoms
# ===============================================

import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Set Dataset Size
# -----------------------------
num_samples = 8000  # number of patients

# -----------------------------
# 2️⃣ Generate Demographics
# -----------------------------
np.random.seed(42)
age = np.random.randint(1, 90, size=num_samples)
gender = np.random.choice([0, 1, 2], size=num_samples, p=[0.48, 0.48, 0.04])  # 0=Male,1=Female,2=Other

# -----------------------------
# 3️⃣ Generate Vitals (Oximeter + other)
# -----------------------------
SpO2 = np.random.normal(loc=97, scale=2.5, size=num_samples)  # normal 95-100%
SpO2 = np.clip(SpO2, 80, 100)  # realistic range

HeartRate = np.random.normal(loc=80, scale=15, size=num_samples)
HeartRate = np.clip(HeartRate, 50, 160)

PerfusionIndex = np.random.normal(loc=6, scale=3, size=num_samples)
PerfusionIndex = np.clip(PerfusionIndex, 0.5, 20)

Temperature = np.random.normal(loc=36.8, scale=0.7, size=num_samples)
Temperature = np.clip(Temperature, 35, 40)

PulseVariability = np.random.normal(loc=1, scale=0.5, size=num_samples)
PulseVariability = np.clip(PulseVariability, 0.2, 3)

# -----------------------------
# 4️⃣ Generate Symptoms (Binary)
# -----------------------------
def generate_symptom(prob):
    return np.random.binomial(1, prob, num_samples)

Cough = generate_symptom(0.25)
Shortness_of_Breath = generate_symptom(0.15)
Chest_Pain = generate_symptom(0.10)
Fatigue = generate_symptom(0.20)
Dizziness = generate_symptom(0.10)
Nausea = generate_symptom(0.08)
Confusion = generate_symptom(0.05)
Sore_Throat = generate_symptom(0.12)
Runny_Nose = generate_symptom(0.15)

# -----------------------------
# 5️⃣ Create Risk Labels (Binary, Clinically Plausible)
# -----------------------------
Hypoxia_Risk = (SpO2 < 94).astype(int)
Cardiac_Stress_Risk = (HeartRate > 110).astype(int)
Respiratory_Infection_Risk = ((Cough==1) & (SpO2<96)).astype(int)
Sepsis_Risk = ((SpO2<92) & (HeartRate>100) & (Temperature>38)).astype(int)
Asthma_Exacerbation_Risk = ((Shortness_of_Breath==1) & (Cough==1)).astype(int)
COPD_Risk = ((SpO2<95) & (Shortness_of_Breath==1)).astype(int)
Pneumonia_Risk = ((SpO2<94) & (Cough==1) & (Chest_Pain==1)).astype(int)
COVID_Like_Risk = ((SpO2<96) & (Cough==1) & (Fatigue==1)).astype(int)
Anemia_Risk = ((PerfusionIndex<3) & (Dizziness==1) & (Fatigue==1)).astype(int)
Emergency_Risk = ((Hypoxia_Risk + Cardiac_Stress_Risk + Sepsis_Risk) >=2).astype(int)

# -----------------------------
# 6️⃣ Build DataFrame
# -----------------------------
df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "SpO2": SpO2,
    "HeartRate": HeartRate,
    "PerfusionIndex": PerfusionIndex,
    "Temperature": Temperature,
    "PulseVariability": PulseVariability,
    "Cough": Cough,
    "Shortness_of_Breath": Shortness_of_Breath,
    "Chest_Pain": Chest_Pain,
    "Fatigue": Fatigue,
    "Dizziness": Dizziness,
    "Nausea": Nausea,
    "Confusion": Confusion,
    "Sore_Throat": Sore_Throat,
    "Runny_Nose": Runny_Nose,
    "Hypoxia_Risk": Hypoxia_Risk,
    "Cardiac_Stress_Risk": Cardiac_Stress_Risk,
    "Respiratory_Infection_Risk": Respiratory_Infection_Risk,
    "Sepsis_Risk": Sepsis_Risk,
    "Asthma_Exacerbation_Risk": Asthma_Exacerbation_Risk,
    "COPD_Risk": COPD_Risk,
    "Pneumonia_Risk": Pneumonia_Risk,
    "COVID_Like_Risk": COVID_Like_Risk,
    "Anemia_Risk": Anemia_Risk,
    "Emergency_Risk": Emergency_Risk
})

# -----------------------------
# 7️⃣ Save to CSV
# -----------------------------
df.to_csv("Synthetic_PulseOximeter_RiskData.csv", index=False)
print("Dataset created: Synthetic_PulseOximeter_RiskData.csv")
print("Shape:", df.shape)
print(df.head())
