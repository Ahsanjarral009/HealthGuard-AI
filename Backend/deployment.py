# deployment.py
import joblib
import numpy as np
import pandas as pd

class PulseOxRiskPredictor:
    """
    Production-ready risk prediction model for pulse oximetry data
    """
    
    def __init__(self, model_path='.'):
        """
        Initialize the predictor by loading all trained models
        
        Args:
            model_path: Path to directory containing model files
        """
        print("Loading Pulse Oximetry Risk Prediction Model...")
        
        try:
            # Load all model components
            self.final_model = joblib.load(r'C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\pulseox_final_model.pkl')
            self.scaler = joblib.load(r'C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\pulseox_scaler.pkl')
            self.rf_model = joblib.load(r'C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\pulseox_rf_model.pkl')
            self.xgb_models = joblib.load(r'C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\pulseox_xgb_models.pkl')
            self.mlp_model = joblib.load(r'C:\Users\DELL\Downloads\archive\Smart_Disease_presiction_System\pulseox_mlp_model.pkl')
            
            # Extract configuration
            self.weights = np.array(self.final_model['weights'])
            self.feature_columns = self.final_model['feature_columns']
            self.risk_columns = self.final_model['risk_columns']
            
            print(f"✅ Model loaded successfully!")
            print(f"   Features: {len(self.feature_columns)}")
            print(f"   Risks: {len(self.risk_columns)}")
            print(f"   Accuracy: {self.final_model['test_performance']['average_accuracy']*100:.1f}%")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def _get_mlp_predictions(self, X_data):
        """Helper function for MLP predictions"""
        preds = self.mlp_model.predict_proba(X_data)
        
        if isinstance(preds, list):
            # Multi-output MLP
            n_outputs = len(preds)
            n_samples = X_data.shape[0]
            mlp_probs = np.zeros((n_samples, n_outputs))
            
            for i in range(n_outputs):
                if preds[i].shape[1] == 2:
                    mlp_probs[:, i] = preds[i][:, 1]
                else:
                    mlp_probs[:, i] = preds[i][:, 0]
            
            return mlp_probs
        else:
            # Single output
            if preds.shape[1] == 2:
                return preds[:, 1].reshape(-1, 1)
            else:
                return preds
    
    def predict(self, patient_data, return_binary=False, threshold=0.5):
        """
        Predict risks for a patient
        
        Args:
            patient_data: dict with feature names or list of 27 feature values
            return_binary: If True, returns 0/1 predictions; if False, returns probabilities
            threshold: Probability threshold for binary classification (default: 0.5)
        
        Returns:
            dict: Risk predictions with probabilities or binary values
        """
        try:
            # Convert input to numpy array
            if isinstance(patient_data, dict):
                # Build array from dictionary
                features = np.array([patient_data.get(col, 0) for col in self.feature_columns])
            elif isinstance(patient_data, list) or isinstance(patient_data, np.ndarray):
                # Use as-is, but ensure it's the right length
                features = np.array(patient_data)
            else:
                raise ValueError("patient_data must be dict, list, or numpy array")
            
            # Reshape for single patient
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Validate feature count
            if features.shape[1] != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {features.shape[1]}")
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from each model
            # Random Forest
            rf_pred = self.rf_model.predict_proba(features_scaled)
            if isinstance(rf_pred, list):
                rf_probs = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_pred])
            else:
                rf_probs = rf_pred[:, 1:] if rf_pred.shape[1] > 1 else rf_pred
            
            # XGBoost
            xgb_probs = np.zeros((features_scaled.shape[0], len(self.risk_columns)))
            for i, risk in enumerate(self.risk_columns):
                proba = self.xgb_models[risk].predict_proba(features_scaled)
                xgb_probs[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
            
            # MLP
            mlp_probs = self._get_mlp_predictions(features_scaled)
            
            # Adjust dimensions if needed
            min_dim = min(rf_probs.shape[1], xgb_probs.shape[1], mlp_probs.shape[1], len(self.risk_columns))
            
            rf_probs = rf_probs[:, :min_dim]
            xgb_probs = xgb_probs[:, :min_dim]
            mlp_probs = mlp_probs[:, :min_dim]
            
            # Ensemble predictions
            ensemble_probs = (
                self.weights[0] * rf_probs + 
                self.weights[1] * xgb_probs + 
                self.weights[2] * mlp_probs
            )
            
            # Prepare results
            results = {}
            for i, risk in enumerate(self.risk_columns[:min_dim]):
                prob = float(ensemble_probs[0, i])
                
                if return_binary:
                    results[risk] = 1 if prob >= threshold else 0
                else:
                    results[risk] = prob
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {risk: 0.0 for risk in self.risk_columns}
    
    def predict_batch(self, patients_data, threshold=0.5):
        """
        Predict risks for multiple patients
        
        Args:
            patients_data: List of patient feature lists/arrays or DataFrame
            threshold: Probability threshold for binary classification
        
        Returns:
            list: List of prediction dictionaries for each patient
        """
        if isinstance(patients_data, pd.DataFrame):
            # Convert DataFrame to numpy array
            features = patients_data.values
        else:
            # Convert list of lists to numpy array
            features = np.array(patients_data)
        
        # Ensure proper shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from each model
        # Random Forest
        rf_pred = self.rf_model.predict_proba(features_scaled)
        if isinstance(rf_pred, list):
            rf_probs = np.column_stack([prob[:, 1] if prob.shape[1] == 2 else prob[:, 0] for prob in rf_pred])
        else:
            rf_probs = rf_pred[:, 1:] if rf_pred.shape[1] > 1 else rf_pred
        
        # XGBoost
        xgb_probs = np.zeros((features_scaled.shape[0], len(self.risk_columns)))
        for i, risk in enumerate(self.risk_columns):
            proba = self.xgb_models[risk].predict_proba(features_scaled)
            xgb_probs[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        
        # MLP
        mlp_probs = self._get_mlp_predictions(features_scaled)
        
        # Adjust dimensions
        min_dim = min(rf_probs.shape[1], xgb_probs.shape[1], mlp_probs.shape[1], len(self.risk_columns))
        
        rf_probs = rf_probs[:, :min_dim]
        xgb_probs = xgb_probs[:, :min_dim]
        mlp_probs = mlp_probs[:, :min_dim]
        
        # Ensemble
        ensemble_probs = (
            self.weights[0] * rf_probs + 
            self.weights[1] * xgb_probs + 
            self.weights[2] * mlp_probs
        )
        
        # Prepare batch results
        batch_results = []
        for patient_idx in range(features.shape[0]):
            patient_result = {}
            for i, risk in enumerate(self.risk_columns[:min_dim]):
                prob = float(ensemble_probs[patient_idx, i])
                patient_result[risk] = prob
            batch_results.append(patient_result)
        
        return batch_results
    
    def predict_with_triage(self, patient_data):
        """
        Enhanced prediction with risk levels and triage recommendations
        
        Args:
            patient_data: Patient features (dict, list, or array)
        
        Returns:
            dict: Comprehensive prediction with risk levels and recommendations
        """
        # Get basic predictions
        predictions = self.predict(patient_data, return_binary=False)
        
        result = {
            'probabilities': predictions,
            'risk_levels': {},
            'confidence_scores': {},
            'triage': {},
            'recommendations': []
        }
        
        # Calculate risk levels and confidence
        for risk, prob in predictions.items():
            # Confidence (distance from 0.5, scaled 0-1)
            confidence = abs(prob - 0.5) * 2
            
            # Risk level classification
            if prob < 0.2:
                level = 'Low'
            elif prob < 0.4:
                level = 'Medium'
            elif prob < 0.6:
                level = 'High'
            else:
                level = 'Critical'
            
            result['risk_levels'][risk] = level
            result['confidence_scores'][risk] = float(confidence)
        
        # Overall emergency score
        emergency_weights = {
            'Asthma_Exacerbation_Risk': 0.25,
            'COPD_Risk': 0.25,
            'Pneumonia_Risk': 0.20,
            'COVID_Like_Risk': 0.20,
            'Anemia_Risk': 0.10
        }
        
        overall_score = 0
        for risk, prob in predictions.items():
            weight = emergency_weights.get(risk, 0.1)
            overall_score += prob * weight
        
        result['overall_emergency_score'] = float(overall_score)
        
        # Triage category
        if overall_score < 0.3:
            triage = 'GREEN'
            category = 'Non-urgent'
            action = 'Routine follow-up'
        elif overall_score < 0.5:
            triage = 'YELLOW'
            category = 'Monitor'
            action = 'Close monitoring, outpatient evaluation'
        elif overall_score < 0.7:
            triage = 'ORANGE'
            category = 'Urgent'
            action = 'Urgent medical attention needed'
        else:
            triage = 'RED'
            category = 'Critical'
            action = 'Immediate emergency intervention'
        
        result['triage']['level'] = triage
        result['triage']['category'] = category
        result['triage']['recommended_action'] = action
        
        # Generate specific recommendations
        high_risks = [risk for risk, prob in predictions.items() if prob >= 0.6]
        
        if not high_risks and overall_score < 0.3:
            result['recommendations'].append('No immediate intervention needed')
        elif high_risks:
            for risk in high_risks:
                if 'Asthma' in risk:
                    result['recommendations'].append('Consider bronchodilator therapy and peak flow monitoring')
                elif 'COPD' in risk:
                    result['recommendations'].append('Consider oxygen therapy and bronchodilators')
                elif 'Pneumonia' in risk:
                    result['recommendations'].append('Consider antibiotics, chest X-ray, and sputum culture')
                elif 'COVID' in risk:
                    result['recommendations'].append('Consider COVID testing, isolation, and symptomatic treatment')
                elif 'Anemia' in risk:
                    result['recommendations'].append('Consider CBC test, iron studies, and supplementation')
        else:
            result['recommendations'].append('Continue monitoring vital signs')
        
        return result
    
    def get_feature_names(self):
        """Return the list of feature names expected by the model"""
        return self.feature_columns.copy()
    
    def get_risk_names(self):
        """Return the list of risk names predicted by the model"""
        return self.risk_columns.copy()
    def predict_with_clinical_rules(self, patient_data):
     """Enhanced prediction with clinical emergency rules"""
    # Get original prediction
     result = self.predict_with_triage(patient_data)
    
    # Extract key vitals
     if isinstance(patient_data, list) and len(patient_data) >= 5:
        age = patient_data[0]
        spo2 = patient_data[1]
        hr = patient_data[2]
        temp = patient_data[4]
        
        # CRITICAL EMERGENCY RULES
        emergency_flags = []
        
        # 1. Hypoxia emergency
        if spo2 < 90:
            emergency_flags.append(f"🚨 CRITICAL HYPOXIA: SpO2 {spo2:.1f}%")
            # Force high respiratory risks
            result['probabilities']['Asthma_Exacerbation_Risk'] = max(
                result['probabilities']['Asthma_Exacerbation_Risk'], 0.8)
            result['probabilities']['COPD_Risk'] = max(
                result['probabilities']['COPD_Risk'], 0.8)
            result['probabilities']['Pneumonia_Risk'] = max(
                result['probabilities']['Pneumonia_Risk'], 0.7)
        
        # 2. Tachycardia emergency
        if hr > 130:
            emergency_flags.append(f"🚨 SEVERE TACHYCARDIA: HR {hr:.1f} bpm")
            # Force high cardiac/stress risks
            result['probabilities']['COVID_Like_Risk'] = max(
                result['probabilities']['COVID_Like_Risk'], 0.6)
        
        # 3. Fever emergency
        if temp > 39 or temp < 35.5:
            emergency_flags.append(f"🚨 CRITICAL TEMPERATURE: {temp:.1f}°C")
            result['probabilities']['COVID_Like_Risk'] = max(
                result['probabilities']['COVID_Like_Risk'], 0.7)
            result['probabilities']['Pneumonia_Risk'] = max(
                result['probabilities']['Pneumonia_Risk'], 0.6)
        
        # 4. Elderly with critical vitals
        if age > 65 and (spo2 < 92 or hr > 120 or temp > 38.5):
            emergency_flags.append(f"🚨 HIGH-RISK ELDERLY PATIENT")
        
        # Update risk levels based on new probabilities
        for risk, prob in result['probabilities'].items():
            if prob < 0.2:
                result['risk_levels'][risk] = 'Low'
            elif prob < 0.4:
                result['risk_levels'][risk] = 'Medium'
            elif prob < 0.6:
                result['risk_levels'][risk] = 'High'
            else:
                result['risk_levels'][risk] = 'Critical'
        
        # OVERRIDE TRIAGE if emergency flags
        if emergency_flags:
            result['triage'] = {
                'level': 'RED',
                'category': 'Critical',
                'recommended_action': 'IMMEDIATE EMERGENCY INTERVENTION REQUIRED'
            }
            result['clinical_emergency_flags'] = emergency_flags
            result['recommendations'] = [
                "🆘 EMERGENCY: Call code team immediately",
                "Administer supplemental oxygen",
                "Prepare for emergency department transfer",
                "Monitor vital signs continuously"
            ]
            
            # Force high overall score
            result['overall_emergency_score'] = 0.85
     
     return result