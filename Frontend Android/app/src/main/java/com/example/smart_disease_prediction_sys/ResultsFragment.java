package com.example.smart_disease_prediction_sys;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.progressindicator.CircularProgressIndicator;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.gson.Gson;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class ResultsFragment extends Fragment {

    private ApiResponse apiResponse;
    private LinearLayout emergencyAlertBanner;
    private LinearLayout layoutEmergencyFlags;
    private LinearLayout layoutFlagsList;
    private MaterialCardView cardOverallRisk;
    // UI Components
    private TextView tvOverallRisk, tvRiskDescription, tvEmergencyScore, tvTimestamp;
    private TextView tvTriageLevel, tvTriageCategory, tvTriageAction;
    private TextView tvAssessmentType, tvEmergencyStatus;

    // Disease risk cards
    private MaterialCardView cardAsthma, cardCOPD, cardPneumonia, cardCOVID, cardAnemia;
    private TextView tvAsthmaRisk, tvAsthmaLevel;
    private TextView tvCOPDRisk, tvCOPDLevel;
    private TextView tvPneumoniaRisk, tvPneumoniaLevel;
    private TextView tvCOVIDRisk, tvCOVIDLevel;
    private TextView tvAnemiaRisk, tvAnemiaLevel;

    // Progress indicators
    private com.google.android.material.progressindicator.LinearProgressIndicator progressOverallRisk;
    private com.google.android.material.progressindicator.CircularProgressIndicator progressAsthma;
    private com.google.android.material.progressindicator.CircularProgressIndicator progressCOPD;
    private com.google.android.material.progressindicator.CircularProgressIndicator progressPneumonia;
    private com.google.android.material.progressindicator.CircularProgressIndicator progressCOVID;
    private com.google.android.material.progressindicator.CircularProgressIndicator progressAnemia;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_results, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Get data from arguments
        Bundle args = getArguments();
        if (args != null && args.containsKey("api_response")) {
            String apiResponseJson = args.getString("api_response");
            Gson gson = new Gson();
            apiResponse = gson.fromJson(apiResponseJson, ApiResponse.class);
        } else {
            // Fallback to mock data if needed
            apiResponse = createMockResponse();
        }

        // Initialize views
        initializeViews(view);

        // Populate data
        populateData();

        // Setup listeners
        setupListeners();


    }

    private void initializeViews(View view) {
        // Header section
        tvOverallRisk = view.findViewById(R.id.tvOverallRisk);
        tvRiskDescription = view.findViewById(R.id.tvRiskDescription);
        tvEmergencyScore = view.findViewById(R.id.tvEmergencyScore);
        tvTimestamp = view.findViewById(R.id.tvTimestamp);
        progressOverallRisk = view.findViewById(R.id.progressOverallRisk);

        // Triage information
        tvTriageLevel = view.findViewById(R.id.tvTriageLevel);
        tvTriageCategory = view.findViewById(R.id.tvTriageCategory);
        tvTriageAction = view.findViewById(R.id.tvTriageAction);

        // Assessment info
        tvAssessmentType = view.findViewById(R.id.tvAssessmentType);
        tvEmergencyStatus = view.findViewById(R.id.tvEmergencyStatus);

        // Disease risk cards
        cardAsthma = view.findViewById(R.id.cardAsthma);
        cardCOPD = view.findViewById(R.id.cardCOPD);
        cardPneumonia = view.findViewById(R.id.cardPneumonia);
        cardCOVID = view.findViewById(R.id.cardCOVID);
        cardAnemia = view.findViewById(R.id.cardAnemia);

        // Disease risk values
        tvAsthmaRisk = view.findViewById(R.id.tvAsthmaRisk);
        tvAsthmaLevel = view.findViewById(R.id.tvAsthmaLevel);
        tvCOPDRisk = view.findViewById(R.id.tvCOPDRisk);
        tvCOPDLevel = view.findViewById(R.id.tvCOPDLevel);
        tvPneumoniaRisk = view.findViewById(R.id.tvPneumoniaRisk);
        tvPneumoniaLevel = view.findViewById(R.id.tvPneumoniaLevel);
        tvCOVIDRisk = view.findViewById(R.id.tvCOVIDRisk);
        tvCOVIDLevel = view.findViewById(R.id.tvCOVIDLevel);
        tvAnemiaRisk = view.findViewById(R.id.tvAnemiaRisk);
        tvAnemiaLevel = view.findViewById(R.id.tvAnemiaLevel);

        // Progress indicators
        progressAsthma = view.findViewById(R.id.progressAsthma);
        progressCOPD = view.findViewById(R.id.progressCOPD);
        progressPneumonia = view.findViewById(R.id.progressPneumonia);
        progressCOVID = view.findViewById(R.id.progressCOVID);
        progressAnemia = view.findViewById(R.id.progressAnemia);

        cardOverallRisk = view.findViewById(R.id.cardOverallRisk);
        emergencyAlertBanner = view.findViewById(R.id.emergencyAlertBanner);
        layoutEmergencyFlags = view.findViewById(R.id.layoutEmergencyFlags);
        layoutFlagsList = view.findViewById(R.id.layoutFlagsList);

    }

    private void populateData() {
        if (apiResponse == null) return;

        // Set timestamp
        String currentTime = new SimpleDateFormat("MMMM dd, h:mm a", Locale.getDefault()).format(new Date());
        tvTimestamp.setText("Assessment completed: " + currentTime);

        // Check for emergency state
        boolean isEmergency = apiResponse.getEmergencyDetected() != null && apiResponse.getEmergencyDetected();

        if (isEmergency) {
            // EMERGENCY STATE - Show red banner and urgent styling
            cardOverallRisk.setStrokeColor(getResources().getColor(R.color.error));
            cardOverallRisk.setStrokeWidth(3);
            emergencyAlertBanner.setVisibility(View.VISIBLE);

            // Show emergency flags if available
            List<String> emergencyFlags = apiResponse.getClinicalEmergencyFlags();
            if (emergencyFlags != null && !emergencyFlags.isEmpty()) {
                layoutEmergencyFlags.setVisibility(View.VISIBLE);
                layoutFlagsList.removeAllViews();

                for (String flag : emergencyFlags) {
                    TextView flagText = new TextView(requireContext());
                    flagText.setText("• " + flag);
                    flagText.setTextColor(getResources().getColor(R.color.error));
                    flagText.setTextSize(12);
                    flagText.setPadding(0, 4, 0, 4);
                    layoutFlagsList.addView(flagText);
                }
            }
        } else {
            // NORMAL STATE
            cardOverallRisk.setStrokeColor(getResources().getColor(R.color.card_stroke));
            cardOverallRisk.setStrokeWidth(1);
            emergencyAlertBanner.setVisibility(View.GONE);
            layoutEmergencyFlags.setVisibility(View.GONE);
        }

        // Overall risk level
        String overallRisk = apiResponse.getOverallRiskLevel();
        tvOverallRisk.setText(overallRisk + " RISK");

        // Set color, progress and description based on risk
        int colorRes;
        int progress;
        String description;

        // First check if it's emergency (overrides everything)
        if (isEmergency) {
            colorRes = R.color.error;
            progress = 90;
            description = "EMERGENCY: Immediate clinical evaluation required. Please seek medical attention.";
        } else {
            // Regular risk assessment
            switch (overallRisk.toUpperCase()) {
                case "HIGH":
                    colorRes = R.color.error;
                    progress = 75;
                    description = "Significant risk factors identified. Please consult a healthcare professional immediately.";
                    break;
                case "MEDIUM":
                    colorRes = R.color.warning;
                    progress = 50;
                    description = "Some risk factors detected. Monitor your symptoms closely.";
                    break;
                case "LOW":
                default:
                    colorRes = R.color.success;
                    progress = 25;
                    description = "Minimal risk factors detected. Continue monitoring your health.";
                    break;
            }
        }

        // Apply the styling
        tvOverallRisk.setTextColor(getResources().getColor(colorRes));
        tvRiskDescription.setText(description);
        progressOverallRisk.setIndicatorColor(getResources().getColor(colorRes));
        progressOverallRisk.setProgressCompat(progress, true);

        // Emergency score
        if (apiResponse.getOverallEmergencyScore() != null) {
            double emergencyScore = apiResponse.getOverallEmergencyScore() * 100;
            String scoreText = String.format(Locale.getDefault(), "Emergency Score: %.1f%%", emergencyScore);

            // Set color based on emergency score
            if (emergencyScore > 50) {
                tvEmergencyScore.setTextColor(getResources().getColor(R.color.error));
            } else if (emergencyScore > 20) {
                tvEmergencyScore.setTextColor(getResources().getColor(R.color.warning));
            } else {
                tvEmergencyScore.setTextColor(getResources().getColor(R.color.text_secondary));
            }

            tvEmergencyScore.setText(scoreText);
        }

        // Triage information
        ApiResponse.Triage triage = apiResponse.getTriage();
        if (triage != null) {
            tvTriageLevel.setText(triage.getLevel());
            tvTriageCategory.setText(triage.getCategory());
            tvTriageAction.setText(triage.getRecommendedAction());

            // Color code triage level
            switch (triage.getLevel().toUpperCase()) {
                case "RED":
                    tvTriageLevel.setTextColor(getResources().getColor(R.color.error));
                    break;
                case "YELLOW":
                    tvTriageLevel.setTextColor(getResources().getColor(R.color.warning));
                    break;
                case "GREEN":
                    tvTriageLevel.setTextColor(getResources().getColor(R.color.success));
                    break;
                default:
                    tvTriageLevel.setTextColor(getResources().getColor(R.color.text_primary));
            }
        }

        // Assessment type
        if (apiResponse.getAssessmentType() != null) {
            tvAssessmentType.setText("Assessment Type: " + apiResponse.getAssessmentType());
        }

        // Emergency status
        if (apiResponse.getEmergencyDetected() != null) {
            if (apiResponse.getEmergencyDetected()) {
                tvEmergencyStatus.setText("⚠️ EMERGENCY DETECTED");
                tvEmergencyStatus.setTextColor(getResources().getColor(R.color.error));
            } else {
                tvEmergencyStatus.setText("✓ No Emergency");
                tvEmergencyStatus.setTextColor(getResources().getColor(R.color.success));
            }
        }

        // Disease probabilities
        Map<String, Double> probabilities = apiResponse.getProbabilities();
        Map<String, String> riskLevels = apiResponse.getRiskLevels();

        if (probabilities != null && riskLevels != null) {
            // Asthma
            double asthmaProb = probabilities.getOrDefault("Asthma_Exacerbation_Risk", 0.0);
            String asthmaLevel = riskLevels.getOrDefault("Asthma_Exacerbation_Risk", "LOW");
            tvAsthmaRisk.setText(String.format(Locale.getDefault(), "%.1f%%", asthmaProb * 100));
            tvAsthmaLevel.setText("(" + asthmaLevel + ")");
            progressAsthma.setProgressCompat((int) (asthmaProb * 100), true);
            setRiskLevelColor(tvAsthmaLevel, asthmaLevel);

            // COPD
            double copdProb = probabilities.getOrDefault("COPD_Risk", 0.0);
            String copdLevel = riskLevels.getOrDefault("COPD_Risk", "LOW");
            tvCOPDRisk.setText(String.format(Locale.getDefault(), "%.1f%%", copdProb * 100));
            tvCOPDLevel.setText("(" + copdLevel + ")");
            progressCOPD.setProgressCompat((int) (copdProb * 100), true);
            setRiskLevelColor(tvCOPDLevel, copdLevel);

            // Pneumonia
            double pneumoniaProb = probabilities.getOrDefault("Pneumonia_Risk", 0.0);
            String pneumoniaLevel = riskLevels.getOrDefault("Pneumonia_Risk", "LOW");
            tvPneumoniaRisk.setText(String.format(Locale.getDefault(), "%.1f%%", pneumoniaProb * 100));
            tvPneumoniaLevel.setText("(" + pneumoniaLevel + ")");
            progressPneumonia.setProgressCompat((int) (pneumoniaProb * 100), true);
            setRiskLevelColor(tvPneumoniaLevel, pneumoniaLevel);

            // COVID-like
            double covidProb = probabilities.getOrDefault("COVID_Like_Risk", 0.0);
            String covidLevel = riskLevels.getOrDefault("COVID_Like_Risk", "LOW");
            tvCOVIDRisk.setText(String.format(Locale.getDefault(), "%.1f%%", covidProb * 100));
            tvCOVIDLevel.setText("(" + covidLevel + ")");
            progressCOVID.setProgressCompat((int) (covidProb * 100), true);
            setRiskLevelColor(tvCOVIDLevel, covidLevel);

            // Anemia
            double anemiaProb = probabilities.getOrDefault("Anemia_Risk", 0.0);
            String anemiaLevel = riskLevels.getOrDefault("Anemia_Risk", "LOW");
            tvAnemiaRisk.setText(String.format(Locale.getDefault(), "%.1f%%", anemiaProb * 100));
            tvAnemiaLevel.setText("(" + anemiaLevel + ")");
            progressAnemia.setProgressCompat((int) (anemiaProb * 100), true);
            setRiskLevelColor(tvAnemiaLevel, anemiaLevel);
        }

        // Recommendations
        List<String> recommendations = apiResponse.getRecommendations();
        if (recommendations != null && !recommendations.isEmpty()) {
            addRecommendations(recommendations);
        }
    }

    private void setRiskLevelColor(TextView textView, String level) {
        switch (level.toUpperCase()) {
            case "HIGH":
                textView.setTextColor(getResources().getColor(R.color.error));
                break;
            case "MEDIUM":
                textView.setTextColor(getResources().getColor(R.color.warning));
                break;
            case "LOW":
                textView.setTextColor(getResources().getColor(R.color.success));
                break;
            default:
                textView.setTextColor(getResources().getColor(R.color.text_secondary));
        }
    }

    private void addRecommendations(List<String> recommendations) {
        LinearLayout layout = requireView().findViewById(R.id.layoutRecommendations);
        if (layout == null) return;

        layout.removeAllViews();

        for (String recommendation : recommendations) {
            TextView textView = new TextView(requireContext());
            textView.setText("• " + recommendation);
            textView.setTextColor(getResources().getColor(R.color.text_primary));
            textView.setTextSize(14);
            textView.setPadding(32, 8, 32, 8);
            layout.addView(textView);
        }
    }

    private void setupListeners() {
        // Save Report
        requireView().findViewById(R.id.btnSaveReport).setOnClickListener(v -> {
            Toast.makeText(requireContext(), "Report saved to history", Toast.LENGTH_SHORT).show();
        });

        // Share
        requireView().findViewById(R.id.btnShare).setOnClickListener(v -> {
            new MaterialAlertDialogBuilder(requireContext())
                    .setTitle("Share Results")
                    .setMessage("Share your health assessment results with your doctor or family?")
                    .setPositiveButton("Share", (dialog, which) -> {
                        Toast.makeText(requireContext(), "Sharing feature coming soon", Toast.LENGTH_SHORT).show();
                    })
                    .setNegativeButton("Cancel", null)
                    .show();
        });

        // New Assessment
        requireView().findViewById(R.id.btnNewAssessment).setOnClickListener(v -> {
            new MaterialAlertDialogBuilder(requireContext())
                    .setTitle("Start New Assessment")
                    .setMessage("This will clear current results. Start new assessment?")
                    .setPositiveButton("Yes", (dialog, which) -> {
                        // Get ViewModel and reset
                        AssesmentViewModel viewModel = new ViewModelProvider(requireActivity()).get(AssesmentViewModel.class);
                        viewModel.resetAssessment();

                        // Go back to first step
                        requireActivity().getSupportFragmentManager()
                                .beginTransaction()
                                .replace(R.id.fragmentContainer, new Step1BasicProfileFragment())
                                .commit();
                    })
                    .setNegativeButton("No", null)
                    .show();
        });

        // Disease cards click listeners
        cardAsthma.setOnClickListener(v -> showDiseaseDetails("Asthma",
                apiResponse.getFormattedProbability("Asthma_Exacerbation_Risk"),
                apiResponse.getRiskLevelForDisease("Asthma_Exacerbation_Risk")));

        cardCOPD.setOnClickListener(v -> showDiseaseDetails("COPD",
                apiResponse.getFormattedProbability("COPD_Risk"),
                apiResponse.getRiskLevelForDisease("COPD_Risk")));

        cardPneumonia.setOnClickListener(v -> showDiseaseDetails("Pneumonia",
                apiResponse.getFormattedProbability("Pneumonia_Risk"),
                apiResponse.getRiskLevelForDisease("Pneumonia_Risk")));

        cardCOVID.setOnClickListener(v -> showDiseaseDetails("COVID-like Symptoms",
                apiResponse.getFormattedProbability("COVID_Like_Risk"),
                apiResponse.getRiskLevelForDisease("COVID_Like_Risk")));

        cardAnemia.setOnClickListener(v -> showDiseaseDetails("Anemia",
                apiResponse.getFormattedProbability("Anemia_Risk"),
                apiResponse.getRiskLevelForDisease("Anemia_Risk")));
    }

    private void showDiseaseDetails(String disease, String probability, String riskLevel) {
        String details = getDiseaseDetails(disease);

        new MaterialAlertDialogBuilder(requireContext())
                .setTitle(disease + " Assessment")
                .setMessage("Risk Level: " + riskLevel + "\n" +
                        "Probability: " + probability + "\n\n" +
                        details)
                .setPositiveButton("OK", null)
                .show();
    }

    private String getDiseaseDetails(String disease) {
        switch (disease) {
            case "Asthma":
                return "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, wheezing and shortness of breath.";
            case "COPD":
                return "Chronic Obstructive Pulmonary Disease (COPD) is a chronic inflammatory lung disease that causes obstructed airflow from the lungs. Symptoms include breathing difficulty, cough, mucus production and wheezing.";
            case "Pneumonia":
                return "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.";
            case "COVID-like Symptoms":
                return "COVID-19 is a contagious disease caused by the SARS-CoV-2 virus. Symptoms may include fever, cough, fatigue, loss of taste or smell, and difficulty breathing.";
            case "Anemia":
                return "Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. Symptoms may include fatigue, weakness, pale skin, and shortness of breath.";
            default:
                return "No additional information available.";
        }
    }

    private ApiResponse createMockResponse() {
        ApiResponse response = new ApiResponse();

        // Set probabilities
        response.setProbabilities(new java.util.HashMap<String, Double>() {{
            put("Asthma_Exacerbation_Risk", 0.0324);
            put("COPD_Risk", 0.0482);
            put("Pneumonia_Risk", 0.0113);
            put("COVID_Like_Risk", 0.0157);
            put("Anemia_Risk", 0.0046);
        }});

        // Set risk levels
        response.setRiskLevels(new java.util.HashMap<String, String>() {{
            put("Asthma_Exacerbation_Risk", "Low");
            put("COPD_Risk", "Low");
            put("Pneumonia_Risk", "Low");
            put("COVID_Like_Risk", "Low");
            put("Anemia_Risk", "Low");
        }});

        // Set confidence scores
        response.setConfidenceScores(new java.util.HashMap<String, Double>() {{
            put("Asthma_Exacerbation_Risk", 0.9352);
            put("COPD_Risk", 0.9036);
            put("Pneumonia_Risk", 0.9774);
            put("COVID_Like_Risk", 0.9686);
            put("Anemia_Risk", 0.9907);
        }});

        // Set triage
        ApiResponse.Triage triage = new ApiResponse.Triage();
        triage.setLevel("GREEN");
        triage.setCategory("Non-urgent");
        triage.setRecommendedAction("Routine follow-up");
        response.setTriage(triage);

        // Set recommendations
        response.setRecommendations(java.util.Arrays.asList(
                "No immediate intervention needed",
                "Continue monitoring your symptoms",
                "Maintain regular hydration",
                "Follow up with primary care physician in 1-2 weeks if symptoms persist"
        ));

        response.setOverallEmergencyScore(0.0260);
        response.setAssessmentType("STANDARD");
        response.setEmergencyDetected(false);

        return response;
    }
}