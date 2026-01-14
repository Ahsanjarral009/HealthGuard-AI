package com.example.smart_disease_prediction_sys;



import com.google.gson.annotations.SerializedName;

public class PredictionApiResponse {
    @SerializedName("success")
    private boolean success;

    @SerializedName("predictions")
    private ApiResponse predictions;

    @SerializedName("prediction_id")
    private String predictionId;

    @SerializedName("processing_time")
    private double processingTime;

    @SerializedName("timestamp")
    private String timestamp;

    // Getters and Setters
    public boolean isSuccess() { return success; }
    public void setSuccess(boolean success) { this.success = success; }

    public ApiResponse getPredictions() { return predictions; }
    public void setPredictions(ApiResponse predictions) { this.predictions = predictions; }

    public String getPredictionId() { return predictionId; }
    public void setPredictionId(String predictionId) { this.predictionId = predictionId; }

    public double getProcessingTime() { return processingTime; }
    public void setProcessingTime(double processingTime) { this.processingTime = processingTime; }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }


}