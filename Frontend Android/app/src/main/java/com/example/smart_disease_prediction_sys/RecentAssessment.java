package com.example.smart_disease_prediction_sys;


public class RecentAssessment {
    private String date;
    private String type;
    private String result;
    private int colorRes;

    public RecentAssessment(String date, String type, String result, int colorRes) {
        this.date = date;
        this.type = type;
        this.result = result;
        this.colorRes = colorRes;
    }

    // Getters
    public String getDate() { return date; }
    public String getType() { return type; }
    public String getResult() { return result; }
    public int getColorRes() { return colorRes; }
}