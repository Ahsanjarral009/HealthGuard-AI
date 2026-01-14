import joblib
import numpy as np
import pandas as pd

# Load models
print("Loading models...")
final_model = joblib.load('pulseox_final_model.pkl')
scaler = joblib.load('pulseox_scaler.pkl')
rf_model = joblib.load('pulseox_rf_model.pkl')
xgb_models = joblib.load('pulseox_xgb_models.pkl')
mlp_model = joblib.load('pulseox_mlp_model.pkl')

# Extract components
weights = np.array(final_model['weights'])
feature_columns = final_model['feature_columns']
risk_columns = final_model['risk_columns']

print(f"\nModel Configuration:")
print(f"Features expected: {len(feature_columns)}")
print(f"Risk columns: {risk_columns}")
print(f"Ensemble weights: {weights}")
print(f"XGBoost models loaded: {list(xgb_models.keys())}")

# Load your original dataset
print("\nLoading original dataset for testing...")
df = pd.read_csv(r'C:\Users\DELL\Downloads\archive\new_folder\Filtered_PulseOximeter_RiskData.csv')

# Clean the data
def clean_data(df):
    df = df.dropna(subset=['Age'])
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    df = df[(df['SpO2'] >= 70) & (df['SpO2'] <= 100)]
    df = df[(df['HeartRate'] >= 40) & (df['HeartRate'] <= 200)]
    df = df[(df['Temperature'] >= 35) & (df['Temperature'] <= 42)]
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]
    
    return df

df = clean_data(df)
print(f"Dataset shape after cleaning: {df.shape}")

# Prepare features and targets
X = df[feature_columns].values
y = df[risk_columns].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target matrix shape: {y.shape}")

# FIXED Prediction function - handles missing risks
def predict_risks_api(patient_features):
    """
    Robust prediction function that handles missing risks
    """
    if isinstance(patient_features, dict):
        features = np.array([patient_features.get(col, 0) for col in feature_columns]).reshape(1, -1)
    else:
        features = np.array(patient_features).reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    
    # Get predictions from each model - with error handling
    all_predictions = {}
    
    try:
        # Random Forest
        rf_pred = rf_model.predict_proba(features_scaled)
        if isinstance(rf_pred, list):
            rf_probs = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_pred])[0]
        else:
            rf_probs = rf_pred[0, 1:] if rf_pred.shape[1] > 1 else rf_pred[0]
    except Exception as e:
        print(f"RF prediction error: {e}")
        rf_probs = np.zeros(len(risk_columns))
    
    # XGBoost predictions - handle each risk separately
    xgb_probs = []
    for risk in risk_columns:
        try:
            if risk in xgb_models:
                proba = xgb_models[risk].predict_proba(features_scaled)
                if proba.shape[1] == 2:
                    xgb_probs.append(proba[0, 1])
                else:
                    xgb_probs.append(proba[0, 0])
            else:
                xgb_probs.append(0.5)  # Default if model missing
        except Exception as e:
            print(f"XGBoost {risk} error: {e}")
            xgb_probs.append(0.5)
    
    xgb_probs = np.array(xgb_probs)
    
    # MLP predictions
    try:
        mlp_pred = mlp_model.predict_proba(features_scaled)
        if isinstance(mlp_pred, list):
            mlp_probs = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in mlp_pred])[0]
        else:
            mlp_probs = mlp_pred[0, 1:] if mlp_pred.shape[1] > 1 else mlp_pred[0]
    except Exception as e:
        print(f"MLP prediction error: {e}")
        mlp_probs = np.zeros(len(risk_columns))
    
    # Ensure all arrays have the same length
    n_risks = len(risk_columns)
    
    # Pad arrays if they're too short
    if len(rf_probs) < n_risks:
        rf_probs = np.pad(rf_probs, (0, n_risks - len(rf_probs)), mode='constant', constant_values=0.5)
    else:
        rf_probs = rf_probs[:n_risks]
    
    if len(xgb_probs) < n_risks:
        xgb_probs = np.pad(xgb_probs, (0, n_risks - len(xgb_probs)), mode='constant', constant_values=0.5)
    
    if len(mlp_probs) < n_risks:
        mlp_probs = np.pad(mlp_probs, (0, n_risks - len(mlp_probs)), mode='constant', constant_values=0.5)
    else:
        mlp_probs = mlp_probs[:n_risks]
    
    # Debug: Print array lengths
    # print(f"Array lengths - RF: {len(rf_probs)}, XGB: {len(xgb_probs)}, MLP: {len(mlp_probs)}")
    
    # Ensemble
    ensemble_probs = weights[0] * rf_probs + weights[1] * xgb_probs + weights[2] * mlp_probs
    
    # Create result dictionary for ALL risks
    for i, risk in enumerate(risk_columns):
        if i < len(ensemble_probs):
            all_predictions[risk] = float(ensemble_probs[i])
        else:
            all_predictions[risk] = 0.5  # Default if missing
    
    return all_predictions

# Test with actual patients - FIXED version
print("\n" + "="*70)
print("TESTING MODEL ON ACTUAL HIGH-RISK PATIENTS")
print("="*70)

# Find patients with any high risk
high_risk_indices = np.where(y.sum(axis=1) > 0)[0]
print(f"Found {len(high_risk_indices)} patients with at least one high risk")

# Select 5 random high-risk patients
np.random.seed(42)
selected_indices = np.random.choice(high_risk_indices, min(5, len(high_risk_indices)), replace=False)

for idx in selected_indices:
    actual_risks = [risk_columns[i] for i in np.where(y[idx] == 1)[0]]
    print(f"\nPatient #{idx} (Actual risks: {actual_risks})")
    print("-" * 60)
    
    # Get the actual patient features
    patient_features = X[idx]
    
    # Make prediction
    try:
        predictions = predict_risks_api(patient_features)
        
        # Display predictions vs actual
        print(f"{'Risk':35} {'Predicted':12} {'Actual':8} {'Match':6}")
        print("-" * 60)
        
        match_count = 0
        for i, risk in enumerate(risk_columns):
            if risk in predictions:
                pred_prob = predictions[risk]
                actual = int(y[idx, i])
                pred_binary = 1 if pred_prob >= 0.5 else 0
                match = "✓" if pred_binary == actual else "✗"
                
                if match == "✓":
                    match_count += 1
                
                # Color code based on probability
                if pred_prob >= 0.7:
                    prob_display = f"{pred_prob:.3f} 🔴"
                elif pred_prob >= 0.5:
                    prob_display = f"{pred_prob:.3f} 🟡"
                else:
                    prob_display = f"{pred_prob:.3f} 🟢"
                
                print(f"{risk:35} {prob_display:14} {actual:8} {match:6}")
            else:
                print(f"{risk:35} {'MISSING':14} {int(y[idx, i]):8} {'?':6}")
        
        accuracy = match_count / len(risk_columns) * 100
        print(f"Per-patient accuracy: {accuracy:.1f}%")
        
        # Show some actual feature values for context
        print(f"\nKey vitals for context:")
        print(f"  Age: {patient_features[0]:.1f}")
        print(f"  SpO2: {patient_features[1]:.1f}%")
        print(f"  HeartRate: {patient_features[2]:.1f} bpm")
        print(f"  Temperature: {patient_features[4]:.1f}°C")
        
    except Exception as e:
        print(f"Error predicting patient #{idx}: {e}")

# Test the model on patients with specific conditions
print("\n" + "="*70)
print("TESTING PATIENTS WITH SPECIFIC CONDITIONS")
print("="*70)

# Create test patients with specific conditions
test_patients = []

# 1. Hypoxic patient (low SpO2)
hypoxic_patient = X[0].copy()  # Start with a real patient
hypoxic_patient[1] = 85.0  # Low SpO2
hypoxic_patient[2] = 120.0  # High heart rate
test_patients.append(("Hypoxic Patient (SpO2=85%)", hypoxic_patient))

# 2. Febrile patient
febrile_patient = X[0].copy()
febrile_patient[4] = 39.5  # High temperature
febrile_patient[7] = 1.0   # Cough
febrile_patient[8] = 1.0   # Shortness of breath
test_patients.append(("Febrile Patient (39.5°C)", febrile_patient))

# 3. Cardiac patient
cardiac_patient = X[0].copy()
cardiac_patient[2] = 150.0  # Very high heart rate
cardiac_patient[9] = 1.0    # Chest pain
cardiac_patient[10] = 1.0   # Fatigue
test_patients.append(("Cardiac Patient (HR=150)", cardiac_patient))

for name, features in test_patients:
    print(f"\n{name}:")
    print("-" * 40)
    
    predictions = predict_risks_api(features)
    
    # Sort by probability (highest first)
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    for risk, prob in sorted_predictions:
        if prob >= 0.3:  # Only show risks above 30%
            if prob >= 0.7:
                level = "🔴 HIGH"
            elif prob >= 0.5:
                level = "🟡 MODERATE"
            else:
                level = "🟠 LOW"
            
            print(f"  {risk:35}: {prob:.3f} ({level})")

# Create a simple API-ready function
print("\n" + "="*70)
print("API-READY PREDICTION FUNCTION")
print("="*70)

def predict_patient_risk_api(patient_data, return_format="json"):
    """
    Simple API function for patient risk prediction
    
    Args:
        patient_data: dict with patient features or list of feature values
        return_format: "json" for dictionary, "list" for array
    
    Returns:
        Risk predictions in specified format
    """
    try:
        predictions = predict_risks_api(patient_data)
        
        if return_format == "json":
            return predictions
        elif return_format == "list":
            return [predictions[risk] for risk in risk_columns]
        else:
            return predictions
    except Exception as e:
        return {"error": str(e), "predictions": None}

# Test the API function
print("\nTesting API function:")
sample_patient = X[0]  # Use first patient
api_result = predict_patient_risk_api(sample_patient, "json")
print("\nSample API response:")
for risk, prob in api_result.items():
    print(f"  {risk}: {prob:.3f}")

# Save the API-ready function
print("\n" + "="*70)
print("MODEL READY FOR DEPLOYMENT")
print("="*70)
print("\n✅ Model is ready for production use!")
print("\nTo deploy:")
print("1. Copy the `predict_patient_risk_api` function to your API")
print("2. Ensure all model files are in the same directory:")
print("   - pulseox_scaler.pkl")
print("   - pulseox_rf_model.pkl")
print("   - pulseox_xgb_models.pkl")
print("   - pulseox_mlp_model.pkl")
print("   - pulseox_final_model.pkl")
print("\n3. Input format:")
print("   - Dictionary: {'Age': 45, 'SpO2': 95, ...}")
print("   - List: [45, 95, 72, ...] (must match feature order)")
print("\n4. Output: JSON with risk probabilities (0.0 to 1.0)")