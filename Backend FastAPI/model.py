import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import json
import joblib

# Load your 25K dataset
print("Loading 25K dataset...")
df = pd.read_csv(r'C:\Users\DELL\Downloads\archive\new_folder\Filtered_PulseOximeter_RiskData.csv')
print(f"Dataset shape: {df.shape}")

# Step 1: Clean and prepare data
def clean_data(df):
    # Remove rows with critical missing values
    df = df.dropna(subset=['Age', 'SpO2', 'HeartRate', 'Temperature'])
    
    # Fill other missing values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0 and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Clean extreme values
    df = df[(df['SpO2'] >= 70) & (df['SpO2'] <= 100)]
    df = df[(df['HeartRate'] >= 30) & (df['HeartRate'] <= 220)]
    df = df[(df['Temperature'] >= 35) & (df['Temperature'] <= 42)]
    
    return df

df = clean_data(df)
print(f"After cleaning: {df.shape}")

# Step 2: Define features and targets
# Exclude risk and severity columns from features
risk_columns = [
    'Hypoxia_Risk', 'Cardiac_Stress_Risk', 'Respiratory_Infection_Risk',
    'Sepsis_Risk', 'Emergency_Risk'
]

severity_columns = [
    'Hypoxia_Risk_Severity', 'Cardiac_Stress_Risk_Severity',
    'Respiratory_Infection_Risk_Severity', 'Sepsis_Risk_Severity',
    'Emergency_Risk_Severity'
]

# Feature selection - use all except target columns
exclude_cols = risk_columns + severity_columns + [
    'Asthma_Exacerbation_Risk', 'COPD_Risk', 'Pneumonia_Risk',
    'COVID_Like_Risk', 'Anemia_Risk',
    'Asthma_Exacerbation_Risk_Severity', 'COPD_Risk_Severity',
    'Pneumonia_Risk_Severity', 'COVID_Like_Risk_Severity',
    'Anemia_Risk_Severity'
]

feature_columns = [col for col in df.columns if col not in exclude_cols]
print(f"Number of features: {len(feature_columns)}")

# Step 3: Prepare X and y
X = df[feature_columns].values
y = df[risk_columns].values

# Step 4: Split data (70-15-15 for train-val-test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y[:, 0]  # Stratify by first target
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp[:, 0]
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Models
print("\n" + "="*50)
print("TRAINING MODELS FOR MULTI-RISK PREDICTION")
print("="*50)

# Model 1: Random Forest (Multi-output)
print("\n1. Training Random Forest...")
rf_model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
)
rf_model.fit(X_train_scaled, y_train)

# Model 2: XGBoost (Individual models for each risk)
print("2. Training XGBoost models...")
xgb_models = {}
for i, risk in enumerate(risk_columns):
    print(f"   Training for {risk}...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=sum(y_train[:, i] == 0) / sum(y_train[:, i] == 1) if sum(y_train[:, i] == 1) > 0 else 1,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train[:, i])
    xgb_models[risk] = xgb_model

# Model 3: MLP (Multi-output neural network)
print("3. Training MLP...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=False
)
mlp_model.fit(X_train_scaled, y_train)

# Step 7: Evaluate on Validation Set
print("\n" + "="*50)
print("VALIDATION PERFORMANCE")
print("="*50)

def evaluate_model(model, X_data, y_true, model_name):
    if model_name == "XGBoost Ensemble":
        y_pred_proba = np.zeros_like(y_true, dtype=float)
        for i, risk in enumerate(risk_columns):
            proba = xgb_models[risk].predict_proba(X_data)
            y_pred_proba[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
    elif hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_data)
        # Handle multi-output classifier structure
        if isinstance(y_pred_proba, list):
            y_pred_proba = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in y_pred_proba])
    else:
        y_pred_proba = model.predict(X_data)
    
    # Calculate metrics for each risk
    results = {}
    for i, risk in enumerate(risk_columns):
        if i < y_pred_proba.shape[1]:  # Check if model has this output
            y_true_risk = y_true[:, i]
            y_pred = (y_pred_proba[:, i] > 0.5).astype(int)
            
            accuracy = accuracy_score(y_true_risk, y_pred)
            auc = roc_auc_score(y_true_risk, y_pred_proba[:, i]) if len(np.unique(y_true_risk)) > 1 else 0.5
            
            results[risk] = {
                'accuracy': accuracy,
                'auc': auc,
                'pos_rate': y_true_risk.mean()
            }
        else:
            results[risk] = {
                'accuracy': 0.5,
                'auc': 0.5,
                'pos_rate': y_true[:, i].mean()
            }
    
    return results, y_pred_proba

# Evaluate each model
models = {
    "Random Forest": rf_model,
    "XGBoost Ensemble": xgb_models,
    "MLP": mlp_model
}

for name, model in models.items():
    print(f"\n{name}:")
    results, _ = evaluate_model(model, X_val_scaled, y_val, name)
    for risk, metrics in results.items():
        print(f"  {risk}: Accuracy={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")

# Step 8: Ensemble Predictions (Combine all models)
print("\n" + "="*50)
print("ENSEMBLE PREDICTIONS")
print("="*50)

# Helper function for MLP predictions
def get_mlp_predictions(model, X_data):
    """Handle MLP's multi-output structure correctly"""
    preds = model.predict_proba(X_data)
    
    if isinstance(preds, list):
        # MLP returns list of arrays for each output
        n_outputs = len(preds)
        n_samples = X_data.shape[0]
        
        # Initialize array for probabilities
        mlp_probs = np.zeros((n_samples, n_outputs))
        
        for i in range(n_outputs):
            if preds[i].shape[1] == 2:  # Binary classification
                mlp_probs[:, i] = preds[i][:, 1]  # Probability of class 1
            else:
                # If only one class (edge case), use first column
                mlp_probs[:, i] = preds[i][:, 0]
        
        return mlp_probs
    else:
        # Single output model
        if preds.shape[1] == 2:
            return preds[:, 1].reshape(-1, 1)
        else:
            return preds

# Get predictions from all models
rf_preds = rf_model.predict_proba(X_val_scaled)
if isinstance(rf_preds, list):
    rf_preds = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_preds])

xgb_preds = np.zeros((X_val_scaled.shape[0], len(risk_columns)))
for i, risk in enumerate(risk_columns):
    proba = xgb_models[risk].predict_proba(X_val_scaled)
    xgb_preds[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

mlp_preds = get_mlp_predictions(mlp_model, X_val_scaled)

# Adjust MLP predictions if needed
if mlp_preds.shape[1] != len(risk_columns):
    if mlp_preds.shape[1] < len(risk_columns):
        padding = np.zeros((mlp_preds.shape[0], len(risk_columns) - mlp_preds.shape[1]))
        mlp_preds = np.hstack([mlp_preds, padding])
    else:
        mlp_preds = mlp_preds[:, :len(risk_columns)]

print(f"Shapes: RF={rf_preds.shape}, XGB={xgb_preds.shape}, MLP={mlp_preds.shape}")

# Calculate optimal ensemble weights based on validation performance
def calculate_model_accuracies(preds, true_labels, model_name):
    """Calculate accuracy for each risk and overall"""
    pred_labels = (preds > 0.5).astype(int)
    accuracies = []
    
    for i, risk in enumerate(risk_columns):
        if i < preds.shape[1]:  # Check if model has this output
            acc = accuracy_score(true_labels[:, i], pred_labels[:, i])
            accuracies.append(acc)
        else:
            accuracies.append(0.5)  # Default if missing
    
    avg_acc = np.mean(accuracies)
    
    return accuracies, avg_acc

# Calculate accuracies for each model
rf_accuracies, rf_avg = calculate_model_accuracies(rf_preds, y_val, "Random Forest")
xgb_accuracies, xgb_avg = calculate_model_accuracies(xgb_preds, y_val, "XGBoost")
mlp_accuracies, mlp_avg = calculate_model_accuracies(mlp_preds, y_val, "MLP")

print(f"\nModel Average Accuracies:")
print(f"  Random Forest: {rf_avg:.3f}")
print(f"  XGBoost: {xgb_avg:.3f}")
print(f"  MLP: {mlp_avg:.3f}")

# Calculate optimal weights (softmax of average accuracies)
model_accs = np.array([rf_avg, xgb_avg, mlp_avg])
weights = np.exp(model_accs * 10) / np.sum(np.exp(model_accs * 10))  # Scale by 10 for sharper weights

print(f"\nOptimal Ensemble Weights:")
print(f"  Random Forest: {weights[0]:.3f}")
print(f"  XGBoost: {weights[1]:.3f}")
print(f"  MLP: {weights[2]:.3f}")

# Weighted ensemble predictions
ensemble_preds = weights[0] * rf_preds + weights[1] * xgb_preds + weights[2] * mlp_preds

# Evaluate ensemble
print("\nEnsemble Performance (Validation):")
ensemble_accuracies = []
for i, risk in enumerate(risk_columns):
    y_true = y_val[:, i]
    y_pred = (ensemble_preds[:, i] > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, ensemble_preds[:, i]) if len(np.unique(y_true)) > 1 else 0.5
    ensemble_accuracies.append(accuracy)
    print(f"  {risk}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")

print(f"  Average: {np.mean(ensemble_accuracies):.3f}")

# Step 9: Final Test Evaluation
print("\n" + "="*50)
print("FINAL TEST SET PERFORMANCE")
print("="*50)

# Get test predictions
rf_test_preds = rf_model.predict_proba(X_test_scaled)
if isinstance(rf_test_preds, list):
    rf_test_preds = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_test_preds])

xgb_test_preds = np.zeros((X_test_scaled.shape[0], len(risk_columns)))
for i, risk in enumerate(risk_columns):
    proba = xgb_models[risk].predict_proba(X_test_scaled)
    xgb_test_preds[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

mlp_test_preds = get_mlp_predictions(mlp_model, X_test_scaled)

# Adjust MLP test predictions if needed
if mlp_test_preds.shape[1] != len(risk_columns):
    if mlp_test_preds.shape[1] < len(risk_columns):
        padding = np.zeros((mlp_test_preds.shape[0], len(risk_columns) - mlp_test_preds.shape[1]))
        mlp_test_preds = np.hstack([mlp_test_preds, padding])
    else:
        mlp_test_preds = mlp_test_preds[:, :len(risk_columns)]

# Ensemble test predictions
ensemble_test_preds = weights[0] * rf_test_preds + weights[1] * xgb_test_preds + weights[2] * mlp_test_preds

# Test metrics
test_results = []
print("\nTest Set Performance:")
for i, risk in enumerate(risk_columns):
    y_true = y_test[:, i]
    y_pred = (ensemble_test_preds[:, i] > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, ensemble_test_preds[:, i]) if len(np.unique(y_true)) > 1 else 0.5
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    test_results.append({
        'risk': risk,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pos_rate': y_true.mean()
    })
    
    print(f"\n{risk}:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  AUC:       {auc:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Positive Rate: {y_true.mean():.1%}")

avg_test_accuracy = np.mean([r['accuracy'] for r in test_results])
avg_test_auc = np.mean([r['auc'] for r in test_results])

print(f"\n{'='*50}")
print(f"AVERAGE TEST PERFORMANCE")
print(f"Accuracy: {avg_test_accuracy:.1%}")
print(f"AUC:      {avg_test_auc:.3f}")
print(f"{'='*50}")

# Check if target achieved
target_achieved = avg_test_accuracy >= 0.85
print(f"\n🎯 TARGET ACHIEVED: {'✅ YES' if target_achieved else '❌ NO'}")
print(f"   Target: 85-90% accuracy")
print(f"   Actual: {avg_test_accuracy*100:.1f}% accuracy")

# Step 10: Save models and create prediction function
print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

# Save all components
joblib.dump(scaler, 'risk_scaler.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_models, 'xgb_models.pkl')
joblib.dump(mlp_model, 'mlp_model.pkl')

# Save final ensemble configuration
final_model = {
    'weights': weights.tolist(),
    'feature_columns': feature_columns,
    'risk_columns': risk_columns,
    'scaler': scaler,
    'model_accuracies': {
        'rf': rf_avg,
        'xgb': xgb_avg,
        'mlp': mlp_avg,
        'ensemble': avg_test_accuracy
    }
}
joblib.dump(final_model, 'final_risk_model.pkl')

print("✅ Models saved successfully")

# Create enhanced prediction function
def predict_risks_enhanced(patient_features, return_probs=True, threshold=0.5):
    """
    Enhanced prediction function for a single patient
    Returns JSON with risk probabilities and additional insights
    """
    # Ensure correct shape
    if isinstance(patient_features, list):
        patient_features = np.array(patient_features)
    
    if patient_features.ndim == 1:
        patient_features = patient_features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(patient_features)
    
    # Get predictions from each model
    rf_pred = rf_model.predict_proba(features_scaled)
    if isinstance(rf_pred, list):
        rf_probs = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_pred])[0]
    else:
        rf_probs = rf_pred[0, 1:] if rf_pred.shape[1] > 1 else rf_pred[0]
    
    xgb_probs = np.array([
        xgb_models[risk].predict_proba(features_scaled)[0, 1] 
        if xgb_models[risk].predict_proba(features_scaled).shape[1] == 2 
        else xgb_models[risk].predict_proba(features_scaled)[0, 0]
        for risk in risk_columns
    ])
    
    mlp_probs = get_mlp_predictions(mlp_model, features_scaled)[0]
    
    # Adjust MLP probs if needed
    if len(mlp_probs) != len(risk_columns):
        if len(mlp_probs) < len(risk_columns):
            mlp_probs = np.pad(mlp_probs, (0, len(risk_columns) - len(mlp_probs)))
        else:
            mlp_probs = mlp_probs[:len(risk_columns)]
    
    # Ensemble
    ensemble_probs = weights[0] * rf_probs + weights[1] * xgb_probs + weights[2] * mlp_probs
    
    # Create enhanced result
    result = {
        'probabilities': {},
        'predictions': {},
        'confidence': {},
        'risk_levels': {}
    }
    
    for i, risk in enumerate(risk_columns):
        prob = float(ensemble_probs[i])
        pred = 1 if prob >= threshold else 0
        
        # Calculate confidence (distance from threshold)
        confidence = abs(prob - 0.5) * 2  # 0 to 1 scale
        
        # Risk level classification
        if prob < 0.3:
            risk_level = 'Low'
        elif prob < 0.7:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        result['probabilities'][risk] = prob
        result['predictions'][risk] = pred
        result['confidence'][risk] = confidence
        result['risk_levels'][risk] = risk_level
    
    # Overall emergency score (weighted average of all risks)
    emergency_weights = {
        'Hypoxia_Risk': 0.3,
        'Respiratory_Infection_Risk': 0.2,
        'Cardiac_Stress_Risk': 0.25,
        'Sepsis_Risk': 0.15,
        'Emergency_Risk': 0.1
    }
    
    overall_score = sum(
        result['probabilities'][risk] * emergency_weights[risk] 
        for risk in risk_columns
    )
    
    result['overall_emergency_score'] = float(overall_score)
    
    # Emergency triage category
    if overall_score < 0.3:
        result['triage'] = 'Green (Non-urgent)'
    elif overall_score < 0.6:
        result['triage'] = 'Yellow (Urgent)'
    elif overall_score < 0.8:
        result['triage'] = 'Orange (Emergency)'
    else:
        result['triage'] = 'Red (Critical)'
    
    return result if return_probs else result['predictions']

# Simple prediction function (original format)
def predict_risks_simple(patient_features):
    """
    Simple prediction function that returns only probabilities
    """
    result = predict_risks_enhanced(patient_features, return_probs=True)
    return result['probabilities']

# Step 11: Example predictions
print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)

# Test with first 3 patients from test set
for i in range(3):
    patient_features = X_test[i]
    prediction = predict_risks_simple(patient_features)
    print(f"\nPatient {i+1} Risk Probabilities:")
    print(json.dumps(prediction, indent=2))

# Step 12: API-ready prediction function
def predict_batch(patients_df):
    """
    Predict risks for a batch of patients
    patients_df should have the same columns as feature_columns
    """
    # Scale features
    features_scaled = scaler.transform(patients_df)
    
    # Get predictions from each model
    rf_batch_preds = rf_model.predict_proba(features_scaled)
    if isinstance(rf_batch_preds, list):
        rf_batch_preds = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_batch_preds])
    
    xgb_batch_preds = np.zeros((len(patients_df), len(risk_columns)))
    for i, risk in enumerate(risk_columns):
        proba = xgb_models[risk].predict_proba(features_scaled)
        xgb_batch_preds[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
    
    mlp_batch_preds = get_mlp_predictions(mlp_model, features_scaled)
    
    # Adjust MLP batch predictions if needed
    if mlp_batch_preds.shape[1] != len(risk_columns):
        if mlp_batch_preds.shape[1] < len(risk_columns):
            padding = np.zeros((mlp_batch_preds.shape[0], len(risk_columns) - mlp_batch_preds.shape[1]))
            mlp_batch_preds = np.hstack([mlp_batch_preds, padding])
        else:
            mlp_batch_preds = mlp_batch_preds[:, :len(risk_columns)]
    
    # Weighted ensemble
    ensemble_batch_preds = weights[0] * rf_batch_preds + weights[1] * xgb_batch_preds + weights[2] * mlp_batch_preds
    
    # Convert to list of JSON objects (simple format)
    results = []
    for i in range(len(patients_df)):
        result_dict = {
            risk_columns[j]: float(ensemble_batch_preds[i, j])
            for j in range(len(risk_columns))
        }
        results.append(result_dict)
    
    return results

# Performance summary
print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)
print(f"Dataset Size: {len(df):,} patients")
print(f"Features Used: {len(feature_columns)}")
print(f"Models: Random Forest, XGBoost, MLP Ensemble")
print(f"Ensemble Weights: RF={weights[0]:.2f}, XGB={weights[1]:.2f}, MLP={weights[2]:.2f}")
print(f"Average Test Accuracy: {avg_test_accuracy:.1%}")
print(f"Average Test AUC: {avg_test_auc:.3f}")
print(f"Goal Achieved: {'✅ YES' if target_achieved else '❌ NO'} (85-90% target)")

# Save detailed performance report
performance_report = {
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features_used': len(feature_columns)
    },
    'model_info': {
        'ensemble_weights': weights.tolist(),
        'models': ['RandomForest', 'XGBoost', 'MLP'],
        'model_accuracies': {
            'RandomForest': rf_avg,
            'XGBoost': xgb_avg,
            'MLP': mlp_avg,
            'Ensemble': avg_test_accuracy
        }
    },
    'test_performance': test_results,
    'averages': {
        'accuracy': avg_test_accuracy,
        'auc': avg_test_auc,
        'precision': np.mean([r['precision'] for r in test_results]),
        'recall': np.mean([r['recall'] for r in test_results]),
        'f1': np.mean([r['f1'] for r in test_results])
    },
    'achievement': target_achieved
}

with open('performance_report.json', 'w') as f:
    json.dump(performance_report, f, indent=2)

print("\nPerformance report saved to 'performance_report.json'")
print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print("="*50)
print(f"\n✅ Models trained on {len(X_train)} samples")
print(f"✅ Validation performance: {np.mean(ensemble_accuracies):.1%} accuracy")
print(f"✅ Test performance: {avg_test_accuracy:.1%} average accuracy")
print(f"✅ Models saved for deployment")
print(f"✅ Prediction functions ready for API integration")
print(f"✅ Target achieved: {target_achieved}")