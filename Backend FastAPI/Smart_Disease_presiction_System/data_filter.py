import pandas as pd

df = pd.read_csv(r"C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\Filtered_PulseOximeter_RiskData.csv")

print(df.shape)

dataset =df.drop(columns=[
    "Hypoxia_Risk",
    "Cardiac_Stress_Risk",
    "Respiratory_Infection_Risk",
    "Sepsis_Risk",
    "Emergency_Risk"
])

dataset.to_csv("Filtered_PulseOximeter_RiskData.csv", index=False)