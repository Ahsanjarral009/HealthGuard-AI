package com.example.smart_disease_prediction_sys;


import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;
import java.util.Map;

public interface ApiService {

    @POST("/predict")
    Call<PredictionApiResponse> predictDisease(@Body Map<String, Object> patientData);

    @POST("/predict")
    Call<Object> testPredict(@Body Map<String, Object> patientData);

    @POST("/login-user")
    Call<LoginResponse> login(@Body Map<String, Object> body);
}


    // Alternative: If you have a PatientData model class
    // @POST("/predict")
    // Call<ApiResponse> predictDisease(@Body PatientData patientData);
