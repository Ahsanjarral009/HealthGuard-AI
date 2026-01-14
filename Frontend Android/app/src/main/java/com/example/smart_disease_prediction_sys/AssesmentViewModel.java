package com.example.smart_disease_prediction_sys;

import android.content.Context;
import android.util.Log;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.json.JSONObject;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class AssesmentViewModel extends ViewModel {

    private static final String TAG = "AssessmentViewModel";

    // User Information
    private MutableLiveData<Integer> userid = new MutableLiveData<>(-1);
    private MutableLiveData<String> username = new MutableLiveData<>("");
    private MutableLiveData<String> email = new MutableLiveData<>("");
    private MutableLiveData<String> password = new MutableLiveData<>("");

    // Step 1: Basic Information
    private MutableLiveData<Integer> age = new MutableLiveData<>(30);
    private MutableLiveData<String> gender = new MutableLiveData<>("U"); // M, F, U

    // Step 2: Vital Signs
    private MutableLiveData<Double> spo2 = new MutableLiveData<>(98.0);
    private MutableLiveData<Integer> heartRate = new MutableLiveData<>(75);
    private MutableLiveData<Double> temperature = new MutableLiveData<>(37.0);
    private MutableLiveData<Double> perfusionIndex = new MutableLiveData<>(5.0);
    private MutableLiveData<Double> pulseVariability = new MutableLiveData<>(1.2);

    // Step 3: Symptoms
    private MutableLiveData<Boolean> cough = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> shortnessBreath = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> chestPain = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> fatigue = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> dizziness = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> nausea = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> confusion = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> soreThroat = new MutableLiveData<>(false);
    private MutableLiveData<Boolean> runnyNose = new MutableLiveData<>(false);

    private MutableLiveData<String> notes = new MutableLiveData<>("");

    // UI State
    private MutableLiveData<Integer> currentStep = new MutableLiveData<>(1);
    private MutableLiveData<Boolean> isLoading = new MutableLiveData<>(false);
    private MutableLiveData<String> errorMessage = new MutableLiveData<>("");

    // Computed values for UI feedback
    private MutableLiveData<String> ageGroup = new MutableLiveData<>("Young Adult");
    private MutableLiveData<String> hrZone = new MutableLiveData<>("Normal");
    private MutableLiveData<String> spo2Zone = new MutableLiveData<>("Normal");

    private Gson gson;

    // ====================================
    // GETTERS AND SETTERS
    // ====================================

    // User Information
    public LiveData<Integer> getUserid() { return userid; }
    public void setUserid(int userid) {
        this.userid.setValue(userid);
    }

    public LiveData<String> getUsername() { return username; }
    public void setUsername(String username) {
        this.username.setValue(username);
    }

    public LiveData<String> getEmail() { return email; }
    public void setEmail(String email) {
        this.email.setValue(email);
    }

    public LiveData<String> getPassword() { return password; }
    public void setPassword(String password) {
        this.password.setValue(password);
    }

    // Age
    public LiveData<Integer> getAge() { return age; }
    public void setAge(int age) {
        if (age < 0 || age > 120) {
            errorMessage.setValue("Age must be between 0-120");
            return;
        }
        this.age.setValue(age);
        calculateAgeGroup(age);
    }

    // Gender
    public LiveData<String> getGender() { return gender; }
    public void setGender(String gender) {
        if (gender.equals("M") || gender.equals("F") || gender.equals("U")) {
            this.gender.setValue(gender);
        } else {
            this.gender.setValue("U");
        }
    }

    // SpO2
    public LiveData<Double> getSpo2() { return spo2; }
    public void setSpo2(double spo2) {
        if (spo2 < 70 || spo2 > 100) {
            errorMessage.setValue("SpO2 must be between 70-100%");
            return;
        }
        this.spo2.setValue(spo2);
        calculateSpo2Zone(spo2);
    }

    // Heart Rate
    public LiveData<Integer> getHeartRate() { return heartRate; }
    public void setHeartRate(int heartRate) {
        if (heartRate < 40 || heartRate > 200) {
            errorMessage.setValue("Heart rate must be between 40-200 bpm");
            return;
        }
        this.heartRate.setValue(heartRate);
        calculateHrZone(heartRate);
    }

    // Temperature
    public LiveData<Double> getTemperature() { return temperature; }
    public void setTemperature(double temperature) {
        if (temperature < 35 || temperature > 42) {
            errorMessage.setValue("Temperature must be between 35-42°C");
            return;
        }
        this.temperature.setValue(temperature);
    }

    // Perfusion Index
    public LiveData<Double> getPerfusionIndex() { return perfusionIndex; }
    public void setPerfusionIndex(double perfusionIndex) {
        if (perfusionIndex < 0.02 || perfusionIndex > 20) {
            errorMessage.setValue("Perfusion index must be between 0.02-20");
            return;
        }
        this.perfusionIndex.setValue(perfusionIndex);
    }

    // Pulse Variability
    public LiveData<Double> getPulseVariability() { return pulseVariability; }
    public void setPulseVariability(double pulseVariability) {
        if (pulseVariability < 0) {
            errorMessage.setValue("Pulse variability cannot be negative");
            return;
        }
        this.pulseVariability.setValue(pulseVariability);
    }

    // Symptoms - Getters
    public LiveData<Boolean> getCough() { return cough; }
    public LiveData<Boolean> getShortnessBreath() { return shortnessBreath; }
    public LiveData<Boolean> getChestPain() { return chestPain; }
    public LiveData<Boolean> getFatigue() { return fatigue; }
    public LiveData<Boolean> getDizziness() { return dizziness; }
    public LiveData<Boolean> getNausea() { return nausea; }
    public LiveData<Boolean> getConfusion() { return confusion; }
    public LiveData<Boolean> getSoreThroat() { return soreThroat; }
    public LiveData<Boolean> getRunnyNose() { return runnyNose; }

    // Symptoms - Setters
    public void setCough(boolean value) { cough.setValue(value); }
    public void setShortnessBreath(boolean value) { shortnessBreath.setValue(value); }
    public void setChestPain(boolean value) { chestPain.setValue(value); }
    public void setFatigue(boolean value) { fatigue.setValue(value); }
    public void setDizziness(boolean value) { dizziness.setValue(value); }
    public void setNausea(boolean value) { nausea.setValue(value); }
    public void setConfusion(boolean value) { confusion.setValue(value); }
    public void setSoreThroat(boolean value) { soreThroat.setValue(value); }
    public void setRunnyNose(boolean value) { runnyNose.setValue(value); }

    // Notes
    public LiveData<String> getNotes() { return notes; }
    public void setNotes(String notes) { this.notes.setValue(notes); }

    // UI State Getters
    public LiveData<Integer> getCurrentStep() { return currentStep; }
    public LiveData<Boolean> getIsLoading() { return isLoading; }
    public LiveData<String> getErrorMessage() { return errorMessage; }

    // Computed Value Getters
    public LiveData<String> getAgeGroup() { return ageGroup; }
    public LiveData<String> getHrZone() { return hrZone; }
    public LiveData<String> getSpo2Zone() { return spo2Zone; }

    // ====================================
    // COMPUTATION METHODS
    // ====================================

    private void calculateAgeGroup(int age) {
        if (age < 18) {
            ageGroup.setValue("Child (0-17)");
        } else if (age < 40) {
            ageGroup.setValue("Young Adult (18-39)");
        } else if (age < 60) {
            ageGroup.setValue("Middle Adult (40-59)");
        } else {
            ageGroup.setValue("Senior (60+)");
        }
    }

    private void calculateHrZone(int heartRate) {
        Integer ageValue = age.getValue();
        if (ageValue == null) ageValue = 30;

        int maxHR = 220 - ageValue;
        if (heartRate < 60) {
            hrZone.setValue("Low");
        } else if (heartRate < 100) {
            hrZone.setValue("Normal");
        } else if (heartRate < maxHR * 0.7) {
            hrZone.setValue("Elevated");
        } else {
            hrZone.setValue("High");
        }
    }

    private void calculateSpo2Zone(double spo2) {
        if (spo2 >= 95) {
            spo2Zone.setValue("Normal");
        } else if (spo2 >= 90) {
            spo2Zone.setValue("Mild Hypoxia");
        } else if (spo2 >= 85) {
            spo2Zone.setValue("Moderate Hypoxia");
        } else {
            spo2Zone.setValue("Severe Hypoxia");
        }
    }

    // ====================================
    // VALIDATION METHODS
    // ====================================

    public boolean validateStep1() {
        Integer ageValue = age.getValue();
        if (ageValue == null || ageValue < 1 || ageValue > 120) {
            errorMessage.setValue("Please enter a valid age (1-120 years)");
            return false;
        }
        return true;
    }

    public boolean validateStep2() {
        // Check required fields
        Double spo2Value = spo2.getValue();
        Integer hrValue = heartRate.getValue();
        Double tempValue = temperature.getValue();

        if (spo2Value == null || spo2Value < 70 || spo2Value > 100) {
            errorMessage.setValue("Please enter valid SpO2 (70-100%)");
            return false;
        }
        if (hrValue == null || hrValue < 40 || hrValue > 200) {
            errorMessage.setValue("Please enter valid heart rate (40-200 bpm)");
            return false;
        }
        if (tempValue == null || tempValue < 35 || tempValue > 42) {
            errorMessage.setValue("Please enter valid temperature (35-42°C)");
            return false;
        }
        return true;
    }

    public boolean validateAll() {
        return validateStep1() && validateStep2();
    }

    // ====================================
    // DATA PREPARATION FOR API
    // ====================================

    public Map<String, Object> prepareApiPayloadlogin() {
        Map<String, Object> payload = new HashMap<>();
        payload.put("email", email.getValue());
        payload.put("password", password.getValue());
        return payload;
    }

    public Map<String, Object> prepareApiPayload(Context context) {
        // ===============================
        // 1. SAFE VALUE EXTRACTION
        // ===============================
        int ageVal = age.getValue() != null ? age.getValue() : 0;
        double spo2Val = spo2.getValue() != null ? spo2.getValue() : 0.0;
        int hrVal = heartRate.getValue() != null ? heartRate.getValue() : 0;
        double tempVal = temperature.getValue() != null ? temperature.getValue() : 0.0;
        double piVal = perfusionIndex.getValue() != null ? perfusionIndex.getValue() : 0.0;
        double pvVal = pulseVariability.getValue() != null ? pulseVariability.getValue() : 0.0;

        // Get userid
        int userIdVal = 0;
        if (context != null) {
            userIdVal = UserManager.getInstance(context).getUserId();
            Log.d("ViewModel", "UserID from UserManager: " + userIdVal);
        }

        boolean coughVal = Boolean.TRUE.equals(cough.getValue());
        boolean sobVal = Boolean.TRUE.equals(shortnessBreath.getValue());
        boolean chestPainVal = Boolean.TRUE.equals(chestPain.getValue());
        boolean fatigueVal = Boolean.TRUE.equals(fatigue.getValue());
        boolean dizzinessVal = Boolean.TRUE.equals(dizziness.getValue());
        boolean nauseaVal = Boolean.TRUE.equals(nausea.getValue());
        boolean confusionVal = Boolean.TRUE.equals(confusion.getValue());
        boolean soreThroatVal = Boolean.TRUE.equals(soreThroat.getValue());
        boolean runnyNoseVal = Boolean.TRUE.equals(runnyNose.getValue());

        // ===============================
        // 2. DERIVED FEATURES (API LOGIC)
        // ===============================
        // Age Group
        double ageGroupVal;
        if (ageVal <= 18) ageGroupVal = 0;
        else if (ageVal <= 40) ageGroupVal = 1;
        else if (ageVal <= 60) ageGroupVal = 2;
        else ageGroupVal = 3;

        // HR Zone
        double hrZoneVal;
        if (hrVal < 60) hrZoneVal = 0;
        else if (hrVal <= 100) hrZoneVal = 1;
        else hrZoneVal = 2;

        // SpO2 Zone
        double spo2ZoneVal;
        if (spo2Val >= 95) spo2ZoneVal = 2;
        else if (spo2Val >= 90) spo2ZoneVal = 1;
        else spo2ZoneVal = 0;

        // Scores
        double oxygenationScore = spo2ZoneVal;

        double perfusionScore;
        if (piVal >= 5) perfusionScore = 3;
        else if (piVal >= 3) perfusionScore = 2;
        else perfusionScore = 1;

        double temperatureScore;
        if (tempVal < 37.5) temperatureScore = 1;
        else if (tempVal < 38.5) temperatureScore = 2;
        else temperatureScore = 3;

        // Symptom Clusters
        double respiratoryCluster = (coughVal || sobVal) ? 1.0 : 0.0;
        double systemicCluster = (fatigueVal || dizzinessVal || nauseaVal) ? 1.0 : 0.0;
        double cardiacCluster = (chestPainVal || hrVal > 100) ? 1.0 : 0.0;

        // Interactions
        double ageSpo2Interaction = ageVal * (spo2Val / 100.0);
        double hrTempInteraction = hrVal * tempVal;
        double piAgeInteraction = piVal * (ageVal / 10.0);



        // ===============================
        // 3. BUILD features DICTIONARY (ORDERED)
        // ===============================
        Map<String, Double> features = new java.util.LinkedHashMap<>();

        features.put("Age", (double) ageVal);
        features.put("SpO2", spo2Val);
        features.put("HeartRate", (double) hrVal);
        features.put("PerfusionIndex", piVal);
        features.put("Temperature", tempVal);
        features.put("PulseVariability", pvVal);

        features.put("Cough", coughVal ? 1.0 : 0.0);
        features.put("Shortness_of_Breath", sobVal ? 1.0 : 0.0);
        features.put("Chest_Pain", chestPainVal ? 1.0 : 0.0);
        features.put("Fatigue", fatigueVal ? 1.0 : 0.0);
        features.put("Dizziness", dizzinessVal ? 1.0 : 0.0);
        features.put("Nausea", nauseaVal ? 1.0 : 0.0);
        features.put("Confusion", confusionVal ? 1.0 : 0.0);
        features.put("Sore_Throat", soreThroatVal ? 1.0 : 0.0);
        features.put("Runny_Nose", runnyNoseVal ? 1.0 : 0.0);

        features.put("Age_Group", ageGroupVal);
        features.put("HR_Zone", hrZoneVal);
        features.put("SpO2_Zone", spo2ZoneVal);
        features.put("Oxygenation_Score", oxygenationScore);
        features.put("Perfusion_Score", perfusionScore);
        features.put("Temperature_Score", temperatureScore);
        features.put("Respiratory_Symptom_Cluster", respiratoryCluster);
        features.put("Systemic_Symptom_Cluster", systemicCluster);
        features.put("Cardiac_Symptom_Cluster", cardiacCluster);
        features.put("Age_SpO2_Interaction", ageSpo2Interaction);
        features.put("HR_Temp_Interaction", hrTempInteraction);
        features.put("PI_Age_Interaction", piAgeInteraction);

        // ===============================
        // 4. FINAL API PAYLOAD
        // ===============================
        Map<String, Object> payload = new HashMap<>();
        payload.put("features", features);
        payload.put("userid", userIdVal);  // Add userid to payload
        payload.put("detailed", false);
        payload.put("return_binary", false);
        payload.put("threshold", 0.5);

        return payload;
    }



    // ====================================
    // UTILITY METHODS
    // ====================================

    public void nextStep() {
        Integer current = currentStep.getValue();
        if (current != null && current < 4) {
            currentStep.setValue(current + 1);
        }
    }

    public void previousStep() {
        Integer current = currentStep.getValue();
        if (current != null && current > 1) {
            currentStep.setValue(current - 1);
        }
    }

    public void resetAssessment() {
        // Reset assessment fields but keep user information
        age.setValue(30);
        gender.setValue("U");

        spo2.setValue(98.0);
        heartRate.setValue(75);
        temperature.setValue(37.0);
        perfusionIndex.setValue(5.0);
        pulseVariability.setValue(1.2);

        cough.setValue(false);
        shortnessBreath.setValue(false);
        chestPain.setValue(false);
        fatigue.setValue(false);
        dizziness.setValue(false);
        nausea.setValue(false);
        confusion.setValue(false);
        soreThroat.setValue(false);
        runnyNose.setValue(false);

        notes.setValue("");
        currentStep.setValue(1);
        errorMessage.setValue("");

        // Recalculate computed values
        calculateAgeGroup(30);
        calculateHrZone(75);
        calculateSpo2Zone(98.0);
    }

    public void clearUserData() {
        // Clear user information
        userid.setValue(-1);
        username.setValue("");
        email.setValue("");
        password.setValue("");
    }

    public boolean hasSymptoms() {
        return Boolean.TRUE.equals(cough.getValue()) ||
                Boolean.TRUE.equals(shortnessBreath.getValue()) ||
                Boolean.TRUE.equals(chestPain.getValue()) ||
                Boolean.TRUE.equals(fatigue.getValue()) ||
                Boolean.TRUE.equals(dizziness.getValue()) ||
                Boolean.TRUE.equals(nausea.getValue()) ||
                Boolean.TRUE.equals(confusion.getValue()) ||
                Boolean.TRUE.equals(soreThroat.getValue()) ||
                Boolean.TRUE.equals(runnyNose.getValue());
    }

    public int countSymptoms() {
        int count = 0;
        if (Boolean.TRUE.equals(cough.getValue())) count++;
        if (Boolean.TRUE.equals(shortnessBreath.getValue())) count++;
        if (Boolean.TRUE.equals(chestPain.getValue())) count++;
        if (Boolean.TRUE.equals(fatigue.getValue())) count++;
        if (Boolean.TRUE.equals(dizziness.getValue())) count++;
        if (Boolean.TRUE.equals(nausea.getValue())) count++;
        if (Boolean.TRUE.equals(confusion.getValue())) count++;
        if (Boolean.TRUE.equals(soreThroat.getValue())) count++;
        if (Boolean.TRUE.equals(runnyNose.getValue())) count++;
        return count;
    }

    // Get summary for review screen
    public Map<String, String> getAssessmentSummary() {
        Map<String, String> summary = new HashMap<>();

        summary.put("Age", age.getValue() + " years");
        summary.put("Gender", getGenderDisplay(gender.getValue()));
        summary.put("SpO2", spo2.getValue() + "%");
        summary.put("Heart Rate", heartRate.getValue() + " bpm");
        summary.put("Temperature", temperature.getValue() + "°C");
        summary.put("Symptoms", countSymptoms() + " symptoms reported");
        summary.put("Username", username.getValue() != null ? username.getValue() : "Guest");

        return summary;
    }

    private String getGenderDisplay(String genderCode) {
        switch (genderCode) {
            case "M": return "Male";
            case "F": return "Female";
            default: return "Not specified";
        }
    }

    // ====================================
    // BATCH SETTERS FOR TESTING/DEBUGGING
    // ====================================

    public void setTestDataHealthy() {
        age.setValue(35);
        gender.setValue("M");
        spo2.setValue(98.0);
        heartRate.setValue(72);
        temperature.setValue(36.8);
        perfusionIndex.setValue(4.5);
        pulseVariability.setValue(1.5);

        // No symptoms
        resetSymptoms();
    }

    public void setTestDataRespiratory() {
        age.setValue(45);
        gender.setValue("F");
        spo2.setValue(92.0);
        heartRate.setValue(88);
        temperature.setValue(38.2);

        cough.setValue(true);
        shortnessBreath.setValue(true);
        fatigue.setValue(true);
        soreThroat.setValue(true);
    }

    private void resetSymptoms() {
        cough.setValue(false);
        shortnessBreath.setValue(false);
        chestPain.setValue(false);
        fatigue.setValue(false);
        dizziness.setValue(false);
        nausea.setValue(false);
        confusion.setValue(false);
        soreThroat.setValue(false);
        runnyNose.setValue(false);
    }

    public String serializeResponse(ApiResponse response) {
        return new Gson().toJson(response);
    }

    // Check if user is logged in
    public boolean isUserLoggedIn() {
        Integer currentUserid = userid.getValue();
        return currentUserid != null && currentUserid != -1;
    }

    // Get current user info as string for display
    public String getCurrentUserInfo() {
        Integer currentUserid = userid.getValue();
        String currentUsername = username.getValue();

        if (currentUserid != null && currentUserid != -1 && currentUsername != null && !currentUsername.isEmpty()) {
            return currentUsername + " (ID: " + currentUserid + ")";
        }
        return "Guest User";
    }
}