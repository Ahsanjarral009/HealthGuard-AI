package com.example.smart_disease_prediction_sys;

import android.annotation.SuppressLint;

import com.google.gson.annotations.SerializedName;
import java.util.List;
import java.util.Map;

public class ApiResponse {

    @SerializedName("probabilities")
    private Map<String, Double> probabilities;

    @SerializedName("risk_levels")
    private Map<String, String> riskLevels;

    @SerializedName("confidence_scores")
    private Map<String, Double> confidenceScores;

    @SerializedName("triage")
    private Triage triage;

    @SerializedName("recommendations")
    private List<String> recommendations;

    @SerializedName("overall_emergency_score")
    private Double overallEmergencyScore;

    @SerializedName("assessment_type")
    private String assessmentType;

    @SerializedName("emergency_detected")
    private Boolean emergencyDetected;

    @SerializedName("clinical_emergency_flags")
    private List<String> clinicalEmergencyFlags;
    // Getter and Setter
    public List<String> getClinicalEmergencyFlags() { return clinicalEmergencyFlags; }
    public void setClinicalEmergencyFlags(List<String> clinicalEmergencyFlags) {
        this.clinicalEmergencyFlags = clinicalEmergencyFlags;
    }

    // Triage inner class
    public static class Triage {
        @SerializedName("level")
        private String level;

        @SerializedName("category")
        private String category;

        @SerializedName("recommended_action")
        private String recommendedAction;

        // Getters and Setters
        public String getLevel() { return level; }
        public void setLevel(String level) { this.level = level; }

        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }

        public String getRecommendedAction() { return recommendedAction; }
        public void setRecommendedAction(String recommendedAction) {
            this.recommendedAction = recommendedAction;
        }
    }

    // Getters and Setters
    public Map<String, Double> getProbabilities() { return probabilities; }
    public void setProbabilities(Map<String, Double> probabilities) {
        this.probabilities = probabilities;
    }

    public Map<String, String> getRiskLevels() { return riskLevels; }
    public void setRiskLevels(Map<String, String> riskLevels) {
        this.riskLevels = riskLevels;
    }

    public Map<String, Double> getConfidenceScores() { return confidenceScores; }
    public void setConfidenceScores(Map<String, Double> confidenceScores) {
        this.confidenceScores = confidenceScores;
    }

    public Triage getTriage() { return triage; }
    public void setTriage(Triage triage) { this.triage = triage; }

    public List<String> getRecommendations() { return recommendations; }
    public void setRecommendations(List<String> recommendations) {
        this.recommendations = recommendations;
    }

    public Double getOverallEmergencyScore() { return overallEmergencyScore; }
    public void setOverallEmergencyScore(Double overallEmergencyScore) {
        this.overallEmergencyScore = overallEmergencyScore;
    }

    public String getAssessmentType() { return assessmentType; }
    public void setAssessmentType(String assessmentType) {
        this.assessmentType = assessmentType;
    }

    public Boolean getEmergencyDetected() { return emergencyDetected; }
    public void setEmergencyDetected(Boolean emergencyDetected) {
        this.emergencyDetected = emergencyDetected;
    }

    // Helper methods
    public String getOverallRiskLevel() {
        if (riskLevels != null) {
            // Check if any disease has high risk
            for (Map.Entry<String, String> entry : riskLevels.entrySet()) {
                if (entry.getValue().equalsIgnoreCase("HIGH")) {
                    return "HIGH";
                }
            }
            // Check if any disease has medium risk
            for (Map.Entry<String, String> entry : riskLevels.entrySet()) {
                if (entry.getValue().equalsIgnoreCase("MEDIUM")) {
                    return "MEDIUM";
                }
            }
        }
        return "LOW";
    }

    public String getHighestRiskDisease() {
        if (probabilities == null || probabilities.isEmpty()) {
            return "No diseases detected";
        }

        String highestRiskDisease = null;
        double highestProbability = 0.0;

        for (Map.Entry<String, Double> entry : probabilities.entrySet()) {
            if (entry.getValue() > highestProbability) {
                highestProbability = entry.getValue();
                highestRiskDisease = entry.getKey();
            }
        }

        return highestRiskDisease;
    }

    public double getHighestProbability() {
        if (probabilities == null || probabilities.isEmpty()) {
            return 0.0;
        }

        double highestProbability = 0.0;
        for (Double probability : probabilities.values()) {
            if (probability > highestProbability) {
                highestProbability = probability;
            }
        }

        return highestProbability;
    }

    @SuppressLint("DefaultLocale")
    public String getFormattedProbability(String disease) {
        if (probabilities != null && probabilities.containsKey(disease)) {
            double prob = probabilities.get(disease);
            return String.format("%.1f%%", prob * 100);
        }
        return "0.0%";
    }

    public String getRiskLevelForDisease(String disease) {
        if (riskLevels != null && riskLevels.containsKey(disease)) {
            return riskLevels.get(disease);
        }
        return "LOW";
    }
}