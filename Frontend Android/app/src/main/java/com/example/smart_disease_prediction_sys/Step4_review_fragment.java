package com.example.smart_disease_prediction_sys;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AlphaAnimation;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.checkbox.MaterialCheckBox;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.progressindicator.CircularProgressIndicator;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.HashMap;
import java.util.Map;

import retrofit2.Call;
import retrofit2.Callback;
import com.example.smart_disease_prediction_sys.UserManager;

import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class Step4_review_fragment extends Fragment {

    private AssesmentViewModel viewModel;

    // UI Components
    private TextView tvTotalSymptoms, tvVitalStatus, tvRiskEstimate;
    private TextView tvReviewAge, tvReviewAgeGroup, tvReviewGender;
    private TextView tvReviewSpo2, tvReviewHeartRate, tvReviewTemperature, tvReviewPerfusionIndex , tvReviewPulsevariablity;
    private TextView tvReviewNotes;
    private MaterialCheckBox checkConsent;

    // Buttons
    private MaterialButton btnBack, btnSubmit;
    private MaterialButton btnEditBasicInfo, btnEditVitalSigns, btnEditSymptoms;

    // Loading indicator
    private CircularProgressIndicator loadingIndicator;

    // API Service
    private ApiService apiService;
    private Gson gson;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_step4_review_fragment, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize ViewModel
        viewModel = new ViewModelProvider(requireActivity()).get(AssesmentViewModel.class);

        // Initialize API service
        initializeApiService();

        // Initialize all views
        initializeViews(view);

        // Set up ViewModel observers
        setupViewModelObservers();

        // Set up listeners
        setupListeners();

        // Populate data from ViewModel
        populateData();

        // Update symptom list
        updateSymptomList();
    }

    private void initializeApiService() {
        gson = new GsonBuilder()
                .setPrettyPrinting()
                .create();

        // IMPORTANT: Use 10.0.2.2 for Android emulator to access localhost
        // For real device on same network: http://YOUR_COMPUTER_IP:8000/
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://10.0.2.2:8000/") // Ensure this matches your Flask server
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build();

        apiService = retrofit.create(ApiService.class);
    }

    private void initializeViews(View view) {
        // Summary section
        tvTotalSymptoms = view.findViewById(R.id.tvTotalSymptoms);
        tvVitalStatus = view.findViewById(R.id.tvVitalStatus);
        tvRiskEstimate = view.findViewById(R.id.tvRiskEstimate);

        // Basic Information
        tvReviewAge = view.findViewById(R.id.tvReviewAge);
        tvReviewAgeGroup = view.findViewById(R.id.tvReviewAgeGroup);
        tvReviewGender = view.findViewById(R.id.tvReviewGender);
        btnEditBasicInfo = view.findViewById(R.id.btnEditBasicInfo);

        // Vital Signs
        tvReviewSpo2 = view.findViewById(R.id.tvReviewSpo2);
        tvReviewHeartRate = view.findViewById(R.id.tvReviewHeartRate);
        tvReviewTemperature = view.findViewById(R.id.tvReviewTemperature);
        tvReviewPerfusionIndex = view.findViewById(R.id.tvReviewPerfusionIndex);
        tvReviewPulsevariablity = view.findViewById(R.id.tvReviewPulsevariablity);
        btnEditVitalSigns = view.findViewById(R.id.btnEditVitalSigns);

        // Symptoms
        tvReviewNotes = view.findViewById(R.id.tvReviewNotes);
        btnEditSymptoms = view.findViewById(R.id.btnEditSymptoms);

        // Privacy & Consent
        checkConsent = view.findViewById(R.id.checkConsent);

        // Navigation buttons
        btnBack = view.findViewById(R.id.btnBack);
        btnSubmit = view.findViewById(R.id.btnSubmit);

        // Loading indicator (create programmatically if not in XML)
        loadingIndicator = view.findViewById(R.id.loadingIndicator);
        if (loadingIndicator == null) {
            loadingIndicator = new CircularProgressIndicator(requireContext());
            loadingIndicator.setIndeterminate(true);
            loadingIndicator.setVisibility(View.GONE);
            ((LinearLayout) view.findViewById(R.id.layoutSymptomsList)).addView(loadingIndicator);
        }
    }

    private void setupViewModelObservers() {
        // Observe symptoms count
        viewModel.getCough().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getShortnessBreath().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getSoreThroat().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getRunnyNose().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getChestPain().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getDizziness().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getFatigue().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getNausea().observe(getViewLifecycleOwner(), value -> updateSymptomCount());
        viewModel.getConfusion().observe(getViewLifecycleOwner(), value -> updateSymptomCount());

        // Observe vital signs for status calculation
        viewModel.getSpo2().observe(getViewLifecycleOwner(), value -> calculateVitalStatus());
        viewModel.getHeartRate().observe(getViewLifecycleOwner(), value -> calculateVitalStatus());
        viewModel.getTemperature().observe(getViewLifecycleOwner(), value -> calculateVitalStatus());
    }

    private void setupListeners() {
        // Edit buttons - navigate back to specific steps
        btnEditBasicInfo.setOnClickListener(v -> navigateToStep(1));
        btnEditVitalSigns.setOnClickListener(v -> navigateToStep(2));
        btnEditSymptoms.setOnClickListener(v -> navigateToStep(3));

        // Back button
        btnBack.setOnClickListener(v -> navigateBack());

        // Submit button
        btnSubmit.setOnClickListener(v -> submitAssessment());

        // Consent checkbox validation
        checkConsent.addOnCheckedStateChangedListener(
                (checkBox, state) -> {
                    boolean isChecked = state == MaterialCheckBox.STATE_CHECKED;
                    btnSubmit.setEnabled(isChecked);
                }
        );

    }

    private void populateData() {
        // Basic Information
        Integer age = viewModel.getAge().getValue();
        if (age != null) {
            tvReviewAge.setText(age + " years");
            tvReviewAgeGroup.setText(viewModel.getAgeGroup().getValue());
        }

        String gender = viewModel.getGender().getValue();
        if (gender != null) {
            switch (gender) {
                case "M": tvReviewGender.setText("Male"); break;
                case "F": tvReviewGender.setText("Female"); break;
                default: tvReviewGender.setText("Not specified");
            }
        }

        // Vital Signs
        Double spo2 = viewModel.getSpo2().getValue();
        if (spo2 != null) {
            tvReviewSpo2.setText(String.format("%.1f%%", spo2));
        }

        Integer heartRate = viewModel.getHeartRate().getValue();
        if (heartRate != null) {
            tvReviewHeartRate.setText(heartRate + " bpm");
        }

        Double temperature = viewModel.getTemperature().getValue();
        if (temperature != null) {
            tvReviewTemperature.setText(String.format("%.1f°C", temperature));
        }

        Double perfusionIndex = viewModel.getPerfusionIndex().getValue();
        if (perfusionIndex != null) {
            tvReviewPerfusionIndex.setText(String.format("%.1f", perfusionIndex));
        }
        Double pulseVariablity = viewModel.getPulseVariability().getValue();
        if(pulseVariablity != null){
            tvReviewPulsevariablity.setText(String.format("%.1f", pulseVariablity));
        }

        // Additional Notes (if any)
        // Additional Notes
        String notes = viewModel.getNotes().getValue();
        if (notes != null && !notes.trim().isEmpty()) {
            tvReviewNotes.setText(notes);
        } else {
            tvReviewNotes.setText("No additional notes provided");
        }

        // Update counts and status
        updateSymptomCount();
        calculateVitalStatus();
        estimateRiskLevel();
    }

    private void updateSymptomList() {
        LinearLayout symptomsLayout = requireView().findViewById(R.id.layoutSymptomsList);

        // Clear existing symptoms (except the "No symptoms" text)
        for (int i = symptomsLayout.getChildCount() - 1; i >= 0; i--) {
            View child = symptomsLayout.getChildAt(i);
            if (child.getId() != R.id.tvNoSymptoms) {
                symptomsLayout.removeViewAt(i);
            }
        }

        // Get the "No symptoms" text view
        TextView tvNoSymptoms = requireView().findViewById(R.id.tvNoSymptoms);

        // Add selected symptoms
        boolean hasSymptoms = false;

        if (Boolean.TRUE.equals(viewModel.getCough().getValue())) {
            addSymptomItem(symptomsLayout, "Cough");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getShortnessBreath().getValue())) {
            addSymptomItem(symptomsLayout, "Shortness of Breath");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getSoreThroat().getValue())) {
            addSymptomItem(symptomsLayout, "Sore Throat");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getRunnyNose().getValue())) {
            addSymptomItem(symptomsLayout, "Runny Nose");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getChestPain().getValue())) {
            addSymptomItem(symptomsLayout, "Chest Pain");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getDizziness().getValue())) {
            addSymptomItem(symptomsLayout, "Dizziness");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getFatigue().getValue())) {
            addSymptomItem(symptomsLayout, "Fatigue");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getNausea().getValue())) {
            addSymptomItem(symptomsLayout, "Nausea");
            hasSymptoms = true;
        }
        if (Boolean.TRUE.equals(viewModel.getConfusion().getValue())) {
            addSymptomItem(symptomsLayout, "Confusion");
            hasSymptoms = true;
        }

        // Show/hide "No symptoms" message
        tvNoSymptoms.setVisibility(hasSymptoms ? View.GONE : View.VISIBLE);
    }

    private void addSymptomItem(LinearLayout layout, String symptom) {
        TextView textView = new TextView(requireContext());
        textView.setText("• " + symptom);
        textView.setTextColor(getResources().getColor(R.color.text_primary));
        textView.setTextSize(14);
        textView.setPadding(0, 4, 0, 4);

        // Add fade-in animation
        AlphaAnimation animation = new AlphaAnimation(0, 1);
        animation.setDuration(300);
        textView.startAnimation(animation);

        layout.addView(textView);
    }

    private void updateSymptomCount() {
        int count = viewModel.countSymptoms();
        tvTotalSymptoms.setText(String.valueOf(count));
        updateSymptomList();
    }

    private void calculateVitalStatus() {
        // Calculate overall vital status based on individual metrics
        int abnormalCount = 0;

        // Check SpO2
        Double spo2 = viewModel.getSpo2().getValue();
        if (spo2 != null && spo2 < 95) abnormalCount++;

        // Check Heart Rate
        Integer hr = viewModel.getHeartRate().getValue();
        Integer age = viewModel.getAge().getValue();
        if (hr != null && age != null) {
            if (hr < 60 || hr > 100) abnormalCount++;
        }

        // Check Temperature
        Double temp = viewModel.getTemperature().getValue();
        if (temp != null && (temp < 36.5 || temp > 37.5)) abnormalCount++;

        // Determine status
        String status;
        int colorRes;

        if (abnormalCount == 0) {
            status = "Normal";
            colorRes = R.color.success;
        } else if (abnormalCount <= 1) {
            status = "Mild";
            colorRes = R.color.warning;
        } else {
            status = "Abnormal";
            colorRes = R.color.error;
        }

        tvVitalStatus.setText(status);
        tvVitalStatus.setTextColor(getResources().getColor(colorRes));
    }

    private void estimateRiskLevel() {
        // Simple risk estimation based on symptoms and vitals
        int riskScore = 0;

        // Age factor
        Integer age = viewModel.getAge().getValue();
        if (age != null) {
            if (age > 60) riskScore += 3;
            else if (age > 40) riskScore += 1;
        }

        // Symptom count
        int symptomCount = viewModel.countSymptoms();
        riskScore += Math.min(symptomCount, 5);

        // Vital abnormalities
        Double spo2 = viewModel.getSpo2().getValue();
        if (spo2 != null && spo2 < 92) riskScore += 2;

        Double temp = viewModel.getTemperature().getValue();
        if (temp != null && temp > 38) riskScore += 2;

        // Determine risk level
        String riskLevel;
        int colorRes;

        if (riskScore <= 2) {
            riskLevel = "Low";
            colorRes = R.color.success;
        } else if (riskScore <= 5) {
            riskLevel = "Medium";
            colorRes = R.color.warning;
        } else {
            riskLevel = "High";
            colorRes = R.color.error;
        }

        tvRiskEstimate.setText(riskLevel);
        tvRiskEstimate.setTextColor(getResources().getColor(colorRes));
    }

    private void navigateToStep(int step) {
        Fragment targetFragment = null;

        switch (step) {
            case 1:
                targetFragment = new Step1BasicProfileFragment();
                break;
            case 2:
                targetFragment = new Step2VitalSignsFragment();
                break;
            case 3:
                targetFragment = new Step3SymptomsFragment();
                break;
        }

        if (targetFragment != null) {
            requireActivity().getSupportFragmentManager()
                    .beginTransaction()
                    .setCustomAnimations(
                            R.anim.slide_in_left,
                            R.anim.slide_out_right,
                            R.anim.slide_in_right,
                            R.anim.slide_out_left
                    )
                    .replace(R.id.fragmentContainer, targetFragment)
                    .addToBackStack("step4")
                    .commit();
        }
    }

    private void navigateBack() {
        requireActivity().getSupportFragmentManager().popBackStack();
    }

    private void submitAssessment() {
        // First test the connection
        testApiConnection();

        // Then show confirmation
        new MaterialAlertDialogBuilder(requireContext())
                .setTitle("Submit Assessment")
                .setMessage("Submit your assessment for analysis?")
                .setPositiveButton("Submit", (dialog, which) -> {
                    dialog.dismiss();
                    startApiCall();
                })
                .setNegativeButton("Review Again", (dialog, which) -> dialog.dismiss())
                .show();
    }

    private void startApiCall() {
        setLoadingState(true);

        Map<String, Object> payload = viewModel.prepareApiPayload(requireContext());

        if (payload.containsKey("userid")) {
            Log.d("Step4Fragment", "Payload contains userid: " + payload.get("userid"));
        } else {
            Log.e("Step4Fragment", "Payload does NOT contain userid!");
        }

        String jsonPayload = gson.toJson(payload);
        Log.d("API_PAYLOAD", "Sending payload: " + jsonPayload);

        Call<PredictionApiResponse> call = apiService.predictDisease(payload);

        call.enqueue(new Callback<PredictionApiResponse>() {
            @Override
            public void onResponse(Call<PredictionApiResponse> call, Response<PredictionApiResponse> response) {
                setLoadingState(false);

                if (response.isSuccessful() && response.body() != null) {
                    PredictionApiResponse apiResponse = response.body();
                    if (apiResponse.isSuccess() && apiResponse.getPredictions() != null) {
                        Log.d("API_SUCCESS", "Response received successfully");
                        Log.d("API_RESPONSE", "Assessment type: " + apiResponse.getPredictions().getAssessmentType());
                        Log.d("API_RESPONSE", "Emergency detected: " + apiResponse.getPredictions().getEmergencyDetected());

                        handleApiResponse(apiResponse.getPredictions());
                    } else {
                        handleApiError("API returned unsuccessful response");
                    }
                } else {
                    String errorBody = "";
                    try {
                        errorBody = response.errorBody().string();
                    } catch (Exception e) {
                        errorBody = "Could not read error body";
                    }

                    String errorMsg = "API Error: " + response.code() + " - " + response.message();
                    Log.e("API_ERROR", errorMsg);
                    Log.e("API_ERROR_BODY", errorBody);

                    handleApiError(errorMsg + "\nServer says: " + errorBody);
                }
            }

            @Override
            public void onFailure(Call<PredictionApiResponse> call, Throwable t) {
                setLoadingState(false);
                Log.e("API_FAILURE", "Network error: " + t.getMessage(), t);

                String errorMsg = "Network Error: ";
                if (t instanceof java.net.ConnectException) {
                    errorMsg += "Cannot connect to server.";
                } else if (t instanceof java.net.SocketTimeoutException) {
                    errorMsg += "Connection timeout.";
                } else {
                    errorMsg += t.getMessage();
                }

                handleApiError(errorMsg);
            }
        });
    }

    private void handleApiResponse(ApiResponse response) {
        // Log the response for debugging
        Log.d("API_RESPONSE_DEBUG", "Response data: " + new Gson().toJson(response));

        if (response == null) {
            handleApiError("No response data received");
            return;
        }

        // Navigate to results screen with the response data
        Bundle bundle = new Bundle();
        bundle.putString("api_response", viewModel.serializeResponse(response));

        ResultsFragment resultsFragment = new ResultsFragment();
        resultsFragment.setArguments(bundle);

        requireActivity().getSupportFragmentManager()
                .beginTransaction()
                .setCustomAnimations(
                        R.anim.slide_in_right,
                        R.anim.slide_out_left,
                        R.anim.slide_in_left,
                        R.anim.slide_out_right
                )
                .replace(R.id.fragmentContainer, resultsFragment)
                .commit();
    }

    private void handleApiError(String error) {
        Log.e("API_ERROR", error);

        // Show error dialog
        new MaterialAlertDialogBuilder(requireContext())
                .setTitle("Submission Failed")
                .setMessage("Unable to submit assessment. Please check your internet connection and try again.\n\nError: " + error)
                .setPositiveButton("Try Again", (dialog, which) -> {
                    dialog.dismiss();
                    startApiCall();
                })
                .setNegativeButton("Cancel", (dialog, which) -> dialog.dismiss())
                .show();
    }
    private void testApiConnection() {
        // Simple test to verify API connectivity
        Map<String, Object> testPayload = new HashMap<>();
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("Age", 30.0);
        testFeatures.put("SpO2", 98.0);
        testFeatures.put("HeartRate", 75.0);
        testFeatures.put("Temperature", 37.0);
        testFeatures.put("PerfusionIndex", 5.0);

        testPayload.put("features", testFeatures);
        testPayload.put("detailed", false);
        testPayload.put("return_binary", false);
        testPayload.put("threshold", 0.5);

        Log.d("API_TEST", "Testing connection with simple payload");

        Call<Object> testCall = apiService.testPredict(testPayload);
        testCall.enqueue(new Callback<Object>() {
            @Override
            public void onResponse(Call<Object> call, Response<Object> response) {
                if (response.isSuccessful()) {
                    Log.d("API_TEST", "Connection successful! Status: " + response.code());
                } else {
                    Log.e("API_TEST", "Connection failed. Code: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<Object> call, Throwable t) {
                Log.e("API_TEST", "Connection error: " + t.getMessage());
            }
        });
    }
    private void setLoadingState(boolean isLoading) {
        if (isLoading) {
            btnSubmit.setText("Analyzing...");
            btnSubmit.setEnabled(false);
            if (loadingIndicator != null) {
                loadingIndicator.setVisibility(View.VISIBLE);
            }
        } else {
            btnSubmit.setText("Submit Assessment");
            btnSubmit.setEnabled(true);
            if (loadingIndicator != null) {
                loadingIndicator.setVisibility(View.GONE);
            }
        }

        // Disable edit buttons while loading
        btnEditBasicInfo.setEnabled(!isLoading);
        btnEditVitalSigns.setEnabled(!isLoading);
        btnEditSymptoms.setEnabled(!isLoading);
        btnBack.setEnabled(!isLoading);
    }

    @Override
    public void onPause() {
        super.onPause();
        // Save consent state
        if (checkConsent != null) {
            // You might want to save this in SharedPreferences or ViewModel
        }
    }
}


