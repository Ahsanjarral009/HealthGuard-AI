import json
from db import get_connection

def save_prediction_to_db(
    prediction_id: str,
    result: dict,
    processing_time: float,
    timestamp,
    assessment_id: int = None
):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO Predictions
        (PredictionId, EmergencyDetected, EmergencyScore,
         TriageLevel, TriageCategory, RecommendedAction,
         AssessmentType, ProcessingTime, CreatedAt, assessment_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            result.get("emergency_detected", False),
            result.get("overall_emergency_score"),
            result.get("triage", {}).get("level"),
            result.get("triage", {}).get("category"),
            result.get("triage", {}).get("recommended_action"),
            result.get("assessment_type"),
            processing_time,
            timestamp,
            assessment_id
        ))

        cursor.execute("""
        INSERT INTO PredictionDetails
        (PredictionId, PredictionJson)
        VALUES (?, ?)
        """, (
            prediction_id,
            json.dumps(result)
        ))

        conn.commit()

    finally:
        conn.close()


def insert_assessment(cursor, user_id, health_status, risk_score):
    cursor.execute("""
        INSERT INTO health_assessments (
            user_id,
            health_status,
            risk_score
        )
        OUTPUT INSERTED.assessment_id
        VALUES (?, ?, ?)
    """, (user_id, health_status, risk_score))
    assessment_id = cursor.fetchone()[0]
    return assessment_id

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
