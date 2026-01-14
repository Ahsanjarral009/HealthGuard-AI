package com.example.smart_disease_prediction_sys;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.textfield.TextInputEditText;

public class Step3SymptomsFragment extends Fragment {

    private AssesmentViewModel viewModel;

    // Switches for all 9 symptoms
    private SwitchMaterial switchCough, switchShortnessBreath, switchSoreThroat, switchRunnyNose;
    private SwitchMaterial switchChestPain, switchDizziness;
    private SwitchMaterial switchFatigue, switchNausea, switchConfusion;

    // Card views for each symptom (optional, for visual feedback)
    private MaterialCardView cardCough, cardShortnessBreath, cardSoreThroat, cardRunnyNose;
    private MaterialCardView cardChestPain, cardDizziness;
    private MaterialCardView cardFatigue, cardNausea, cardConfusion;

    // Quick selection chips
    private Chip chipNone, chipFluLike, chipRespiratory;
    private ChipGroup quickSelectionGroup;

    // Count displays
    private TextView tvRespiratoryCount, tvCardiacCount, tvSystemicCount, tvTotalSymptoms;

    // Additional notes
    private TextInputEditText etAdditionalNotes;

    // Navigation buttons
    private MaterialButton btnBack, btnNext;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_step3_symptoms, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize ViewModel
        viewModel = new ViewModelProvider(requireActivity()).get(AssesmentViewModel.class);

        // Initialize all views
        initializeViews(view);

        // Set up ViewModel observers
        setupViewModelObservers();

        // Set up listeners
        setupListeners();

        // Set initial values from ViewModel
        setInitialValues();

        // Update symptom counts
        updateSymptomCounts();
    }

    private void initializeViews(View view) {
        // Respiratory switches
        switchCough = view.findViewById(R.id.switchCough);
        switchShortnessBreath = view.findViewById(R.id.switchShortnessBreath);
        switchSoreThroat = view.findViewById(R.id.switchSoreThroat);
        switchRunnyNose = view.findViewById(R.id.switchRunnyNose);

        // Cardiac switches
        switchChestPain = view.findViewById(R.id.switchChestPain);
        switchDizziness = view.findViewById(R.id.switchDizziness);

        // Systemic switches
        switchFatigue = view.findViewById(R.id.switchFatigue);
        switchNausea = view.findViewById(R.id.switchNausea);
        switchConfusion = view.findViewById(R.id.switchConfusion);

        // Card views (optional, for visual feedback)
        cardCough = view.findViewById(R.id.cardCough);
        cardShortnessBreath = view.findViewById(R.id.cardShortnessBreath);
        cardSoreThroat = view.findViewById(R.id.cardSoreThroat);
        cardRunnyNose = view.findViewById(R.id.cardRunnyNose);
        cardChestPain = view.findViewById(R.id.cardChestPain);
        cardDizziness = view.findViewById(R.id.cardDizziness);
        cardFatigue = view.findViewById(R.id.cardFatigue);
        cardNausea = view.findViewById(R.id.cardNausea);
        cardConfusion = view.findViewById(R.id.cardConfusion);

        // Quick selection chips
        chipNone = view.findViewById(R.id.chipNone);
        chipFluLike = view.findViewById(R.id.chipFluLike);
        chipRespiratory = view.findViewById(R.id.chipRespiratory);
        quickSelectionGroup = view.findViewById(R.id.quickSelectionGroup);

        // Count displays
        tvRespiratoryCount = view.findViewById(R.id.tvRespiratoryCount);
        tvCardiacCount = view.findViewById(R.id.tvCardiacCount);
        tvSystemicCount = view.findViewById(R.id.tvSystemicCount);
        tvTotalSymptoms = view.findViewById(R.id.tvTotalSymptoms);

        // Additional notes
        etAdditionalNotes = view.findViewById(R.id.etAdditionalNotes);

        // Navigation buttons
        btnBack = view.findViewById(R.id.btnBack);
        btnNext = view.findViewById(R.id.btnNext);
    }

    private void setupViewModelObservers() {
        // Observe Cough
        viewModel.getCough().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchCough.isChecked() != value) {
                switchCough.setChecked(value);
                updateCardAppearance(cardCough, value);
                updateSymptomCounts();
            }
        });

        // Observe Shortness of Breath
        viewModel.getShortnessBreath().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchShortnessBreath.isChecked() != value) {
                switchShortnessBreath.setChecked(value);
                updateCardAppearance(cardShortnessBreath, value);
                updateSymptomCounts();
            }
        });

        // Observe Sore Throat
        viewModel.getSoreThroat().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchSoreThroat.isChecked() != value) {
                switchSoreThroat.setChecked(value);
                updateCardAppearance(cardSoreThroat, value);
                updateSymptomCounts();
            }
        });

        // Observe Runny Nose
        viewModel.getRunnyNose().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchRunnyNose.isChecked() != value) {
                switchRunnyNose.setChecked(value);
                updateCardAppearance(cardRunnyNose, value);
                updateSymptomCounts();
            }
        });

        // Observe Chest Pain
        viewModel.getChestPain().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchChestPain.isChecked() != value) {
                switchChestPain.setChecked(value);
                updateCardAppearance(cardChestPain, value);
                updateSymptomCounts();
            }
        });

        // Observe Dizziness
        viewModel.getDizziness().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchDizziness.isChecked() != value) {
                switchDizziness.setChecked(value);
                updateCardAppearance(cardDizziness, value);
                updateSymptomCounts();
            }
        });

        // Observe Fatigue
        viewModel.getFatigue().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchFatigue.isChecked() != value) {
                switchFatigue.setChecked(value);
                updateCardAppearance(cardFatigue, value);
                updateSymptomCounts();
            }
        });

        // Observe Nausea
        viewModel.getNausea().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchNausea.isChecked() != value) {
                switchNausea.setChecked(value);
                updateCardAppearance(cardNausea, value);
                updateSymptomCounts();
            }
        });

        // Observe Confusion
        viewModel.getConfusion().observe(getViewLifecycleOwner(), value -> {
            if (value != null && switchConfusion.isChecked() != value) {
                switchConfusion.setChecked(value);
                updateCardAppearance(cardConfusion, value);
                updateSymptomCounts();
            }
        });

        // Observe Notes
        viewModel.getNotes().observe(getViewLifecycleOwner(), notes -> {
            if (notes != null && !notes.equals(etAdditionalNotes.getText().toString())) {
                etAdditionalNotes.setText(notes);
            }
        });
    }

    private void setupListeners() {
        // Set up switch listeners
        switchCough.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setCough(isChecked);
            updateCardAppearance(cardCough, isChecked);
            updateSymptomCounts();
        });

        switchShortnessBreath.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setShortnessBreath(isChecked);
            updateCardAppearance(cardShortnessBreath, isChecked);
            updateSymptomCounts();
        });

        switchSoreThroat.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setSoreThroat(isChecked);
            updateCardAppearance(cardSoreThroat, isChecked);
            updateSymptomCounts();
        });

        switchRunnyNose.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setRunnyNose(isChecked);
            updateCardAppearance(cardRunnyNose, isChecked);
            updateSymptomCounts();
        });

        switchChestPain.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setChestPain(isChecked);
            updateCardAppearance(cardChestPain, isChecked);
            updateSymptomCounts();
        });

        switchDizziness.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setDizziness(isChecked);
            updateCardAppearance(cardDizziness, isChecked);
            updateSymptomCounts();
        });

        switchFatigue.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setFatigue(isChecked);
            updateCardAppearance(cardFatigue, isChecked);
            updateSymptomCounts();
        });

        switchNausea.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setNausea(isChecked);
            updateCardAppearance(cardNausea, isChecked);
            updateSymptomCounts();
        });

        switchConfusion.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setConfusion(isChecked);
            updateCardAppearance(cardConfusion, isChecked);
            updateSymptomCounts();
        });

        // Set up quick selection chip listeners
        chipNone.setOnClickListener(v -> {
            // Clear all symptoms
            clearAllSymptoms();
            quickSelectionGroup.clearCheck();
            chipNone.setChecked(true);
        });

        chipFluLike.setOnClickListener(v -> {
            // Select flu-like symptoms: cough, sore throat, runny nose, fatigue
            clearAllSymptoms();
            switchCough.setChecked(true);
            switchSoreThroat.setChecked(true);
            switchRunnyNose.setChecked(true);
            switchFatigue.setChecked(true);
            quickSelectionGroup.clearCheck();
            chipFluLike.setChecked(true);
        });

        chipRespiratory.setOnClickListener(v -> {
            // Select respiratory symptoms: cough, shortness of breath, sore throat, runny nose
            clearAllSymptoms();
            switchCough.setChecked(true);
            switchShortnessBreath.setChecked(true);
            switchSoreThroat.setChecked(true);
            switchRunnyNose.setChecked(true);
            quickSelectionGroup.clearCheck();
            chipRespiratory.setChecked(true);
        });

        // Set up additional notes listener
        etAdditionalNotes.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}

            @Override
            public void afterTextChanged(Editable s) {
                viewModel.setNotes(s.toString());
            }
        });

        // Set up navigation buttons
        btnBack.setOnClickListener(v -> navigateBack());
        btnNext.setOnClickListener(v -> navigateNext());
    }

    private void setInitialValues() {
        // Set values from ViewModel if they exist
        Boolean coughValue = viewModel.getCough().getValue();
        if (coughValue != null) {
            switchCough.setChecked(coughValue);
            updateCardAppearance(cardCough, coughValue);
        }

        Boolean shortnessBreathValue = viewModel.getShortnessBreath().getValue();
        if (shortnessBreathValue != null) {
            switchShortnessBreath.setChecked(shortnessBreathValue);
            updateCardAppearance(cardShortnessBreath, shortnessBreathValue);
        }

        Boolean soreThroatValue = viewModel.getSoreThroat().getValue();
        if (soreThroatValue != null) {
            switchSoreThroat.setChecked(soreThroatValue);
            updateCardAppearance(cardSoreThroat, soreThroatValue);
        }

        Boolean runnyNoseValue = viewModel.getRunnyNose().getValue();
        if (runnyNoseValue != null) {
            switchRunnyNose.setChecked(runnyNoseValue);
            updateCardAppearance(cardRunnyNose, runnyNoseValue);
        }

        Boolean chestPainValue = viewModel.getChestPain().getValue();
        if (chestPainValue != null) {
            switchChestPain.setChecked(chestPainValue);
            updateCardAppearance(cardChestPain, chestPainValue);
        }

        Boolean dizzinessValue = viewModel.getDizziness().getValue();
        if (dizzinessValue != null) {
            switchDizziness.setChecked(dizzinessValue);
            updateCardAppearance(cardDizziness, dizzinessValue);
        }

        Boolean fatigueValue = viewModel.getFatigue().getValue();
        if (fatigueValue != null) {
            switchFatigue.setChecked(fatigueValue);
            updateCardAppearance(cardFatigue, fatigueValue);
        }

        Boolean nauseaValue = viewModel.getNausea().getValue();
        if (nauseaValue != null) {
            switchNausea.setChecked(nauseaValue);
            updateCardAppearance(cardNausea, nauseaValue);
        }

        Boolean confusionValue = viewModel.getConfusion().getValue();
        if (confusionValue != null) {
            switchConfusion.setChecked(confusionValue);
            updateCardAppearance(cardConfusion, confusionValue);
        }

        // Set notes
        String notesValue = viewModel.getNotes().getValue();
        if (notesValue != null) {
            etAdditionalNotes.setText(notesValue);
        }
    }

    private void updateCardAppearance(MaterialCardView card, boolean isChecked) {
        if (card != null) {
            if (isChecked) {
                card.setStrokeColor(getResources().getColor(R.color.primary));
                card.setStrokeWidth(2);
            } else {
                card.setStrokeColor(getResources().getColor(R.color.card_stroke_light));
                card.setStrokeWidth(1);
            }
        }
    }

    private void updateSymptomCounts() {
        // Count respiratory symptoms
        int respiratoryCount = 0;
        if (switchCough.isChecked()) respiratoryCount++;
        if (switchShortnessBreath.isChecked()) respiratoryCount++;
        if (switchSoreThroat.isChecked()) respiratoryCount++;
        if (switchRunnyNose.isChecked()) respiratoryCount++;
        tvRespiratoryCount.setText(respiratoryCount + "/4");

        // Count cardiac symptoms
        int cardiacCount = 0;
        if (switchChestPain.isChecked()) cardiacCount++;
        if (switchDizziness.isChecked()) cardiacCount++;
        tvCardiacCount.setText(cardiacCount + "/2");

        // Count systemic symptoms
        int systemicCount = 0;
        if (switchFatigue.isChecked()) systemicCount++;
        if (switchNausea.isChecked()) systemicCount++;
        if (switchConfusion.isChecked()) systemicCount++;
        tvSystemicCount.setText(systemicCount + "/3");

        // Count total symptoms
        int totalCount = respiratoryCount + cardiacCount + systemicCount;
        tvTotalSymptoms.setText(totalCount + " symptoms selected");

        // Update quick selection chips based on current selection
        updateQuickSelectionChips(totalCount);
    }

    private void updateQuickSelectionChips(int totalCount) {
        if (totalCount == 0) {
            chipNone.setChecked(true);
            chipFluLike.setChecked(false);
            chipRespiratory.setChecked(false);
        } else {
            chipNone.setChecked(false);
            // Check if current selection matches flu-like pattern
            boolean isFluLike = switchCough.isChecked() &&
                    switchSoreThroat.isChecked() &&
                    switchRunnyNose.isChecked() &&
                    switchFatigue.isChecked() &&
                    !switchShortnessBreath.isChecked() &&
                    !switchChestPain.isChecked() &&
                    !switchDizziness.isChecked() &&
                    !switchNausea.isChecked() &&
                    !switchConfusion.isChecked();

            // Check if current selection matches respiratory pattern
            boolean isRespiratory = switchCough.isChecked() &&
                    switchShortnessBreath.isChecked() &&
                    switchSoreThroat.isChecked() &&
                    switchRunnyNose.isChecked() &&
                    !switchChestPain.isChecked() &&
                    !switchDizziness.isChecked() &&
                    !switchFatigue.isChecked() &&
                    !switchNausea.isChecked() &&
                    !switchConfusion.isChecked();

            chipFluLike.setChecked(isFluLike);
            chipRespiratory.setChecked(isRespiratory);
        }
    }

    private void clearAllSymptoms() {
        switchCough.setChecked(false);
        switchShortnessBreath.setChecked(false);
        switchSoreThroat.setChecked(false);
        switchRunnyNose.setChecked(false);
        switchChestPain.setChecked(false);
        switchDizziness.setChecked(false);
        switchFatigue.setChecked(false);
        switchNausea.setChecked(false);
        switchConfusion.setChecked(false);
    }

    private void navigateBack() {
        // Navigate back to vital signs
        requireActivity().getSupportFragmentManager().popBackStack();
    }

    private void navigateNext() {
        // Save all data (already done through listeners)
        // Move to next step
        viewModel.nextStep();

        // Navigate to review fragment
        Step4_review_fragment nextFragment = new Step4_review_fragment();

        requireActivity().getSupportFragmentManager()
                .beginTransaction()
                .setCustomAnimations(
                        R.anim.slide_in_right,
                        R.anim.slide_out_left,
                        R.anim.slide_in_left,
                        R.anim.slide_out_right
                )
                .replace(R.id.fragmentContainer, nextFragment)
                .addToBackStack("step3")
                .commit();
    }

    @Override
    public void onPause() {
        super.onPause();
        // Save data when leaving fragment
        saveCurrentData();
    }

    private void saveCurrentData() {
        // All data is already saved through ViewModel observers
        // This is just a safety measure
        viewModel.setCough(switchCough.isChecked());
        viewModel.setShortnessBreath(switchShortnessBreath.isChecked());
        viewModel.setSoreThroat(switchSoreThroat.isChecked());
        viewModel.setRunnyNose(switchRunnyNose.isChecked());
        viewModel.setChestPain(switchChestPain.isChecked());
        viewModel.setDizziness(switchDizziness.isChecked());
        viewModel.setFatigue(switchFatigue.isChecked());
        viewModel.setNausea(switchNausea.isChecked());
        viewModel.setConfusion(switchConfusion.isChecked());

        if (etAdditionalNotes.getText() != null) {
            viewModel.setNotes(etAdditionalNotes.getText().toString());
        }
    }

    // Helper method to get symptom summary for debugging
    public String getSymptomSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Respiratory: ");
        summary.append(switchCough.isChecked() ? "Cough " : "");
        summary.append(switchShortnessBreath.isChecked() ? "ShortnessBreath " : "");
        summary.append(switchSoreThroat.isChecked() ? "SoreThroat " : "");
        summary.append(switchRunnyNose.isChecked() ? "RunnyNose " : "");

        summary.append("\nCardiac: ");
        summary.append(switchChestPain.isChecked() ? "ChestPain " : "");
        summary.append(switchDizziness.isChecked() ? "Dizziness " : "");

        summary.append("\nSystemic: ");
        summary.append(switchFatigue.isChecked() ? "Fatigue " : "");
        summary.append(switchNausea.isChecked() ? "Nausea " : "");
        summary.append(switchConfusion.isChecked() ? "Confusion " : "");

        return summary.toString();
    }
}