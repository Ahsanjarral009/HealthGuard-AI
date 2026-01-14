# fastapi_app.py

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import field_validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db import get_connection
from db_save_prediction import save_prediction_to_db, insert_assessment
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
import asyncio
import logging
import time
from datetime import datetime
import uuid
import json
# Import your predictor
from deployment import PulseOxRiskPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== Pydantic Models =====================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features: int
    risks: List[str]
    timestamp: datetime
    version: str = "1.0.0"


class LoginRequest(BaseModel):
    email: str
    password: str

class FeatureRequest(BaseModel):
    features: Union[List[float], Dict[str, float]] = Field(
        ...,
        description="Either a list of 27 feature values or a dictionary with feature names"
    )
    detailed: bool = False
    threshold: Optional[float] = 0.5
    return_binary: bool = False
    userid: Optional[int] = None  # Allow user ID to pass

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if isinstance(v, list) and len(v) != 27:
            raise ValueError(f'Expected 27 features, got {len(v)}')
        if isinstance(v, dict) and len(v) != 27:
            raise ValueError(f'Expected 27 features in dict, got {len(v)}')
        return v


class BatchRequest(BaseModel):
    patients: List[Union[List[float], Dict[str, float]]] = Field(
        ...,
        description="List of patients, each with 27 features"
    )
    threshold: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for binary classification"
    )

class PredictionResponse(BaseModel):
    success: bool
    predictions: Dict[str, Any]
    prediction_id: Optional[str] = None
    assessment_id: Optional[int] = None
    processing_time: Optional[float] = None
    timestamp: datetime

class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    count: int
    processing_time: float
    timestamp: datetime

class ErrorResponse(BaseModel):
    success: bool
    error: str
    error_code: str
    timestamp: datetime

class FeatureInfoResponse(BaseModel):
    features: List[str]
    risks: List[str]
    feature_descriptions: Optional[Dict[str, str]] = None
    risk_descriptions: Optional[Dict[str, str]] = None

# Feature descriptions (you can expand this)
FEATURE_DESCRIPTIONS = {
    'Age': 'Patient age in years',
    'SpO2': 'Oxygen saturation percentage',
    'HeartRate': 'Heart rate in beats per minute',
    'PerfusionIndex': 'Perfusion index value',
    'Temperature': 'Body temperature in Celsius',
    'PulseVariability': 'Heart rate variability',
    'Cough': 'Presence of cough (0=No, 1=Yes)',
    'Shortness_of_Breath': 'Shortness of breath (0=No, 1=Yes)',
    'Chest_Pain': 'Chest pain (0=No, 1=Yes)',
    'Fatigue': 'Fatigue (0=No, 1=Yes)',
    'Dizziness': 'Dizziness (0=No, 1=Yes)',
    'Nausea': 'Nausea (0=No, 1=Yes)',
    'Confusion': 'Confusion (0=No, 1=Yes)',
    'Sore_Throat': 'Sore throat (0=No, 1=Yes)',
    'Runny_Nose': 'Runny nose (0=No, 1=Yes)',
    'Age_Group': 'Age group category',
    'HR_Zone': 'Heart rate zone',
    'SpO2_Zone': 'SpO2 zone',
    'Oxygenation_Score': 'Oxygenation risk score',
    'Perfusion_Score': 'Perfusion risk score',
    'Temperature_Score': 'Temperature risk score',
    'Respiratory_Symptom_Cluster': 'Respiratory symptom cluster',
    'Systemic_Symptom_Cluster': 'Systemic symptom cluster',
    'Cardiac_Symptom_Cluster': 'Cardiac symptom cluster',
    'Age_SpO2_Interaction': 'Age-SpO2 interaction term',
    'HR_Temp_Interaction': 'Heart rate-temperature interaction',
    'PI_Age_Interaction': 'Perfusion index-age interaction'
}

RISK_DESCRIPTIONS = {
    'Asthma_Exacerbation_Risk': 'Risk of asthma exacerbation',
    'COPD_Risk': 'Risk of Chronic Obstructive Pulmonary Disease',
    'Pneumonia_Risk': 'Risk of pneumonia',
    'COVID_Like_Risk': 'Risk of COVID-like respiratory infection',
    'Anemia_Risk': 'Risk of anemia'
}

# ===================== FastAPI Application =====================

app = FastAPI(
    title="Pulse Oximetry Risk Prediction API",
    description="API for predicting patient health risks from pulse oximetry data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Global Predictor Instance =====================

predictor = None
startup_time = None

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup"""
    global predictor, startup_time
    
    logger.info("Starting Pulse Oximetry Risk Prediction API...")
    startup_time = time.time()
    
    try:
        # Load the predictor
        predictor = PulseOxRiskPredictor()
        load_time = time.time() - startup_time
        
        logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"   Features: {len(predictor.get_feature_names())}")
        logger.info(f"   Risks: {len(predictor.get_risk_names())}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

# ===================== Dependency Injection =====================

async def get_predictor():
    """Dependency to get the predictor instance"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor

# ===================== Helper Functions =====================

def format_patient_features(patient_data: Union[List[float], Dict[str, float]]) -> List[float]:
    """Format patient data to list of 27 features in correct order"""
    if isinstance(patient_data, dict):
        return [patient_data.get(f, 0.0) for f in predictor.get_feature_names()]
    elif isinstance(patient_data, list):
        return patient_data
    else:
        raise ValueError("Features must be a list of 27 floats or a dict of 27 feature values")


# ===================== API Endpoints =====================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs"""
    return {"message": "Pulse Oximetry Risk Prediction API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health_check(predictor_instance: PulseOxRiskPredictor = Depends(get_predictor)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        features=len(predictor_instance.get_feature_names()),
        risks=predictor_instance.get_risk_names(),
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/features", response_model=FeatureInfoResponse)
async def get_features(predictor_instance: PulseOxRiskPredictor = Depends(get_predictor)):
    """Get feature and risk information"""
    return FeatureInfoResponse(
        features=predictor_instance.get_feature_names(),
        risks=predictor_instance.get_risk_names(),
        feature_descriptions=FEATURE_DESCRIPTIONS,
        risk_descriptions=RISK_DESCRIPTIONS
    )



## saving Data in Database


def insert_health_features(cursor, assessment_id, features: dict):
    cursor.execute("""
        INSERT INTO health_features (
            assessment_id,
            age,
            spo2,
            heart_rate,
            perfusion_index,
            temperature,
            pulse_variability,
            cough,
            shortness_of_breath,
            chest_pain,
            fatigue,
            dizziness,
            nausea,
            confusion,
            sore_throat,
            runny_nose,
            age_group,
            hr_zone,
            spo2_zone,
            oxygenation_score,
            perfusion_score,
            temperature_score,
            respiratory_symptom_cluster,
            systemic_symptom_cluster,
            cardiac_symptom_cluster,
            age_spo2_interaction,
            hr_temp_interaction,
            pi_age_interaction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        assessment_id,
        features["Age"],
        features["SpO2"],
        features["HeartRate"],
        features["PerfusionIndex"],
        features["Temperature"],
        features["PulseVariability"],
        features["Cough"],
        features["Shortness_of_Breath"],
        features["Chest_Pain"],
        features["Fatigue"],
        features["Dizziness"],
        features["Nausea"],
        features["Confusion"],
        features["Sore_Throat"],
        features["Runny_Nose"],
        features["Age_Group"],
        features["HR_Zone"],
        features["SpO2_Zone"],
        features["Oxygenation_Score"],
        features["Perfusion_Score"],
        features["Temperature_Score"],
        features["Respiratory_Symptom_Cluster"],
        features["Systemic_Symptom_Cluster"],
        features["Cardiac_Symptom_Cluster"],
        features["Age_SpO2_Interaction"],
        features["HR_Temp_Interaction"],
        features["PI_Age_Interaction"]
    ))


def save_full_assessment(
    connection,
    user_id: int,
    features: dict,
    health_status: str,
    risk_score: float
):
    cursor = connection.cursor()

    try:
        assessment_id = insert_assessment(
            cursor,
            user_id,
            health_status,
            risk_score
        )

        insert_health_features(
            cursor,
            assessment_id,
            features
        )

        connection.commit()
        return assessment_id

    except Exception as e:
        connection.rollback()
        raise e



@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: FeatureRequest,
    predictor_instance: PulseOxRiskPredictor = Depends(get_predictor),
    save: bool = True
):
    start_time = time.time()
    prediction_id = str(uuid.uuid4())[:8]
    assessment_id = None
    
    try:
        logger.info(f"Prediction {prediction_id} requested")

        features = format_patient_features(request.features)
        is_emergency, emergency_flags = check_emergency_conditions(features)

        if is_emergency:
            result = apply_emergency_rules(features, emergency_flags, predictor_instance, request.threshold)
            result["assessment_type"] = "EMERGENCY"
            result["emergency_detected"] = True
            result["clinical_emergency_flags"] = emergency_flags
        else:
            result = (
                predictor_instance.predict_with_triage(features)
                if request.detailed
                else predictor_instance.predict(features, return_binary=request.return_binary, threshold=request.threshold)
            )
            result["assessment_type"] = "STANDARD"
            result["emergency_detected"] = False

        # Ensure risk_level and risk_score are present
        if "risk_level" not in result:
            if "overall_risk_score" in result:
                score = result["overall_risk_score"]
                if score < 0.3:
                    result["risk_level"] = "Low"
                elif score < 0.6:
                    result["risk_level"] = "Medium"
                elif score < 0.8:
                    result["risk_level"] = "High"
                else:
                    result["risk_level"] = "Critical"
            else:
                result["risk_level"] = "Medium"

        if "risk_score" not in result:
            result["risk_score"] = result.get("overall_risk_score", 0.5)

        processing_time = time.time() - start_time
        timestamp = datetime.now()

        if save:
            conn = get_connection()
            assessment_id = save_full_assessment(
                connection=conn,
                user_id=request.userid,
                features=request.features,
                health_status=result["risk_level"],
                risk_score=result["risk_score"] 
            )
            conn.close()

            save_prediction_to_db(
                prediction_id=prediction_id,
                result=result,
                processing_time=processing_time,
                timestamp=timestamp,
                assessment_id=assessment_id
            )

        return PredictionResponse(
            success=True,
            predictions=result,
            prediction_id=prediction_id,
            assessment_id=assessment_id,
            processing_time=processing_time,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Prediction error {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def check_emergency_conditions(features: List[float]) -> Tuple[bool, List[str]]:
    """Check for critical emergencies"""
    if len(features) < 5:
        return False, []
    
    emergency_flags = []
    
    # Extract vitals
    age = features[0]
    spo2 = features[1]
    heart_rate = features[2]
    temperature = features[4]
    confusion = features[12] if len(features) > 12 else 0
    
    # Emergency criteria
    if spo2 < 85:
        emergency_flags.append(f"SEVERE_HYPOXIA: SpO2 {spo2:.1f}%")
    
    if heart_rate > 140 and spo2 < 90:
        emergency_flags.append(f"CRITICAL_TACHYCARDIA_HYPOXIA")
    
    if temperature > 40.5 or temperature < 34:
        emergency_flags.append(f"CRITICAL_TEMPERATURE: {temperature:.1f}°C")
    
    if confusion == 1 and (spo2 < 92 or heart_rate > 130 or temperature > 39):
        emergency_flags.append(f"ALTERED_MENTAL_STATUS")
    
    if age > 70 and (spo2 < 88 or heart_rate > 150):
        emergency_flags.append(f"HIGH_RISK_ELDERLY")
    
    return len(emergency_flags) > 0, emergency_flags

def apply_emergency_rules(features, emergency_flags, predictor, threshold=0.5):
    """Apply emergency overrides to predictions"""
    # Get base prediction
    result = predictor.predict_with_triage(features)
    
    # Extract vitals
    spo2 = features[1]
    hr = features[2]
    temp = features[4]
    age = features[0]
    
    # Emergency risk boosts
    emergency_boosts = {
        'Asthma_Exacerbation_Risk': 0,
        'COPD_Risk': 0,
        'Pneumonia_Risk': 0,
        'COVID_Like_Risk': 0,
        'Anemia_Risk': 0
    }
    
    # Apply boosts based on conditions
    if spo2 < 90:
        emergency_boosts['Asthma_Exacerbation_Risk'] += 0.3
        emergency_boosts['COPD_Risk'] += 0.3
        emergency_boosts['Pneumonia_Risk'] += 0.2
    
    if hr > 130:
        emergency_boosts['COVID_Like_Risk'] += 0.2
    
    if temp > 39 or temp < 35.5:
        emergency_boosts['COVID_Like_Risk'] += 0.2
        emergency_boosts['Pneumonia_Risk'] += 0.2
    
    if age > 65:
        emergency_boosts['Pneumonia_Risk'] += 0.1
        emergency_boosts['COVID_Like_Risk'] += 0.1
    
    # Apply boosts
    for risk, boost in emergency_boosts.items():
        if risk in result['probabilities']:
            result['probabilities'][risk] = min(1.0, result['probabilities'][risk] + boost)
    
    # Update risk levels
    for risk, prob in result['probabilities'].items():
        if prob < 0.2:
            level = 'Low'
        elif prob < 0.4:
            level = 'Medium'
        elif prob < 0.6:
            level = 'High'
        else:
            level = 'Critical'
        result['risk_levels'][risk] = level
    
    # Set emergency triage
    result['triage'] = {
        'level': 'RED',
        'category': 'Critical',
        'recommended_action': 'IMMEDIATE EMERGENCY ASSESSMENT REQUIRED'
    }
    
    # Emergency recommendations
    result['recommendations'] = [
        "EMERGENCY: Immediate clinical evaluation required",
        "Administer supplemental oxygen if SpO2 < 90%",
        "Continuous vital sign monitoring",
        "Prepare for emergency escalation"
    ]
    
    # Add specific recommendations based on flags
    
    for flag in emergency_flags:
        if "HYPOXIA" in flag:
            result['recommendations'].append("Consider urgent oxygen therapy")
        elif "TACHYCARDIA" in flag:
            result['recommendations'].append("Cardiac monitoring required")
        elif "TEMPERATURE" in flag:
            result['recommendations'].append("Temperature management needed")
    
    # Calculate emergency score
    emergency_score = 0.7  # Base emergency score
    
    if spo2 < 85:
        emergency_score += 0.15
    elif spo2 < 90:
        emergency_score += 0.1
    
    if hr > 140:
        emergency_score += 0.1
    elif hr > 120:
        emergency_score += 0.05
    
    if temp > 40 or temp < 35:
        emergency_score += 0.05
    
    result['overall_emergency_score'] = min(0.95, emergency_score)
    
    return result


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    predictor_instance: PulseOxRiskPredictor = Depends(get_predictor)
):
    """SMART batch predictions with emergency detection"""
    start_time = time.time()
    batch_id = str(uuid.uuid4())[:8]
    
    try:
        logger.info(f"Smart batch prediction request {batch_id} with {len(request.patients)} patients")
        
        # Process each patient with emergency checking
        results = []
        for i, patient in enumerate(request.patients):
            try:
                # Format features
                features = format_patient_features(patient)
                
                # Check for emergencies
                is_emergency, emergency_flags = check_emergency_conditions(features)
                
                if is_emergency:
                    # Apply emergency rules
                    result = apply_emergency_rules(
                        features, 
                        emergency_flags,
                        predictor_instance,
                        request.threshold
                    )
                    result['assessment_type'] = 'EMERGENCY'
                    result['emergency_detected'] = True
                    result['clinical_emergency_flags'] = emergency_flags
                else:
                    # Standard prediction
                    result = predictor_instance.predict_with_triage(features)
                    result['assessment_type'] = 'STANDARD'
                    result['emergency_detected'] = False
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Patient {i} prediction error: {e}")
                # Add error info but continue with batch
                results.append({
                    'error': str(e),
                    'patient_index': i,
                    'success': False
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"Smart batch prediction {batch_id} completed in {processing_time:.3f}s")
        
        return BatchPredictionResponse(
            success=True,
            predictions=results,
            count=len(results),
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/example-patient")
async def get_example_patient(predictor_instance: PulseOxRiskPredictor = Depends(get_predictor)):
    """Get an example patient with all features"""
    # Create a sample patient (you can customize this)
    example_patient = {
        'Age': 45.0,
        'SpO2': 95.0,
        'HeartRate': 85.0,
        'PerfusionIndex': 5.0,
        'Temperature': 37.0,
        'PulseVariability': 1.2,
        'Cough': 0.0,
        'Shortness_of_Breath': 0.0,
        'Chest_Pain': 0.0,
        'Fatigue': 0.0,
        'Dizziness': 0.0,
        'Nausea': 0.0,
        'Confusion': 0.0,
        'Sore_Throat': 0.0,
        'Runny_Nose': 0.0,
        'Age_Group': 2.0,
        'HR_Zone': 0.0,
        'SpO2_Zone': 2.0,
        'Oxygenation_Score': 2.0,
        'Perfusion_Score': 3.0,
        'Temperature_Score': 1.0,
        'Respiratory_Symptom_Cluster': 0.0,
        'Systemic_Symptom_Cluster': 0.0,
        'Cardiac_Symptom_Cluster': 0.0,
        'Age_SpO2_Interaction': 1.5,
        'HR_Temp_Interaction': 60.0,
        'PI_Age_Interaction': 2.5
    }
    
    # Also provide as list
    feature_names = predictor_instance.get_feature_names()
    example_list = [example_patient[name] for name in feature_names]
    
    return {
        "as_dict": example_patient,
        "as_list": example_list,
        "feature_order": feature_names
    }

@app.post("/validate-features")
async def validate_features(
    features: Union[List[float], Dict[str, float]],
    predictor_instance: PulseOxRiskPredictor = Depends(get_predictor)
):
    """Validate if features are in correct format"""
    try:
        if isinstance(features, dict):
            # Check all required features are present
            missing = []
            for feature_name in predictor_instance.get_feature_names():
                if feature_name not in features:
                    missing.append(feature_name)
            
            if missing:
                return {
                    "valid": False,
                    "missing_features": missing,
                    "message": f"Missing {len(missing)} features"
                }
            
            return {
                "valid": True,
                "message": "All features present",
                "feature_count": len(features)
            }
        else:
            # Check list length
            if len(features) != 27:
                return {
                    "valid": False,
                    "message": f"Expected 27 features, got {len(features)}",
                    "expected": 27,
                    "received": len(features)
                }
            
            return {
                "valid": True,
                "message": "Features are valid",
                "feature_count": len(features)
            }
    except Exception as e:
        return {
            "valid": False,
            "message": str(e)
        }

# ===================== Error Handlers =====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(ErrorResponse(
            success=False,
            error=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now()
        ))
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(ErrorResponse(
            success=False,
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now()
        ))
    )
#======================Database Connection Test Endpoint=====================

@app.get("/register-user")
def register_user(email: str, password: str):
    """Register a new user in the database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT COUNT(*) FROM Users WHERE email = ?", (email,))
        if cursor.fetchone()[0] > 0:
            return {"success": False, "message": "User already exists"}
        
        # Insert new user
        cursor.execute(
            "INSERT INTO Users (email, password) VALUES (?, ?)",
            (email, password)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "User registered successfully"}
    
    except Exception as e:
        logger.error(f"Database error during user registration: {e}")
        return {"success": False, "message": str(e)}

@app.post("/login-user")
def login_user(data: LoginRequest):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check credentials - fetch both userid and username
        cursor.execute(
            "SELECT user_id, username FROM Users WHERE email = ? AND passwords = ?",
            (data.email, data.password)
        )

        row = cursor.fetchone()   # fetch data BEFORE closing cursor

        if row is None:
            return {"success": False, "message": "Invalid email or password"}

        user_id = row[0]  # userid is first column
        username = row[1]  # username is second column

        return {
            "success": True,
            "message": "Login successful",
            "userid": user_id,
            "username": username
        }

    except Exception as e:
        logger.error(f"Database error during user login: {e}")
        return {"success": False, "message": "Internal server error"}

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ===================== Main Execution =====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )

  