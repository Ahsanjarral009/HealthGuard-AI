package com.example.smart_disease_prediction_sys;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.slider.Slider;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;

public class Step2VitalSignsFragment extends Fragment {

    private AssesmentViewModel viewModel;

    // UI Components
    private Slider sliderSpo2, sliderHeartRate, sliderTemperature;
    private TextInputEditText etSpo2, etHeartRate, etTemperature;
    private TextInputEditText etPerfusionIndex, etPulseVariability;
    private TextInputLayout spo2InputLayout, hrInputLayout, tempInputLayout;
    private TextInputLayout piInputLayout, pvInputLayout;

    private View advancedHeader, advancedContent;
    private MaterialButton btnNext, btnBack, btnConnectDevice;

    // Animation
    private Animation slideDown, slideUp;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_step2_vital_signs, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize ViewModel
        viewModel = new ViewModelProvider(requireActivity()).get(AssesmentViewModel.class);

        // Initialize animations
        slideDown = AnimationUtils.loadAnimation(getContext(), R.anim.slide_down);
        slideUp = AnimationUtils.loadAnimation(getContext(), R.anim.slide_up);

        // Initialize all views
        initializeViews(view);
        //progress indicator
        LinearProgressIndicator progressIndicator =
                view.findViewById(R.id.progressIndicator);

        progressIndicator.setProgress(40);
        // Set up UI with ViewModel data
        setupViewModelObservers();

        // Set up listeners
        setupListeners();

        // Set initial values from ViewModel
        setInitialValues();
    }

    private void initializeViews(View view) {
        // Step counter
        view.findViewById(R.id.tvStepCounter).setOnClickListener(v -> {});

        // Sliders
        sliderSpo2 = view.findViewById(R.id.sliderSpo2);
        sliderHeartRate = view.findViewById(R.id.sliderHeartRate);
        sliderTemperature = view.findViewById(R.id.sliderTemperature);

        // Text inputs
        etSpo2 = view.findViewById(R.id.etSpo2);
        etHeartRate = view.findViewById(R.id.etHeartRate);
        etTemperature = view.findViewById(R.id.etTemperature);

        // Text input layouts
        spo2InputLayout = view.findViewById(R.id.spo2InputLayout);
        hrInputLayout = view.findViewById(R.id.hrInputLayout);
        tempInputLayout = view.findViewById(R.id.tempInputLayout);

        // Advanced fields
        etPerfusionIndex = view.findViewById(R.id.etPerfusionIndex);
        etPulseVariability = view.findViewById(R.id.etPulseVariability);
        piInputLayout = view.findViewById(R.id.piInputLayout);
        pvInputLayout = view.findViewById(R.id.pvInputLayout);

        // Advanced section
        advancedHeader = view.findViewById(R.id.advancedHeader);
        advancedContent = view.findViewById(R.id.advancedContent);

        // Buttons
        btnNext = view.findViewById(R.id.btnNext);
        btnBack = view.findViewById(R.id.btnBack);
        btnConnectDevice = view.findViewById(R.id.btnConnectDevice);
    }

    private void setupViewModelObservers() {
        // Observe SpO2
        viewModel.getSpo2().observe(getViewLifecycleOwner(), spo2 -> {
            if (spo2 != null) {
                // Update slider without triggering listener
                sliderSpo2.removeOnChangeListener(spo2ChangeListener);
                sliderSpo2.setValue(spo2.floatValue());
                sliderSpo2.addOnChangeListener(spo2ChangeListener);

                // Update text input
                etSpo2.removeTextChangedListener(spo2TextWatcher);
                etSpo2.setText(String.valueOf(spo2));
                etSpo2.addTextChangedListener(spo2TextWatcher);

                // Update display
                updateSpo2Display(spo2);
            }
        });

        // Observe Heart Rate
        viewModel.getHeartRate().observe(getViewLifecycleOwner(), hr -> {
            if (hr != null) {
                sliderHeartRate.removeOnChangeListener(hrChangeListener);
                sliderHeartRate.setValue(hr);
                sliderHeartRate.addOnChangeListener(hrChangeListener);

                etHeartRate.removeTextChangedListener(hrTextWatcher);
                etHeartRate.setText(String.valueOf(hr));
                etHeartRate.addTextChangedListener(hrTextWatcher);

                updateHeartRateDisplay(hr);
            }
        });


        // Observe Temperature
        viewModel.getTemperature().observe(getViewLifecycleOwner(), temp -> {
            if (temp != null) {
                sliderTemperature.removeOnChangeListener(tempChangeListener);
                sliderTemperature.setValue(temp.floatValue());
                sliderTemperature.addOnChangeListener(tempChangeListener);

                etTemperature.removeTextChangedListener(tempTextWatcher);
                etTemperature.setText(String.valueOf(temp));
                etTemperature.addTextChangedListener(tempTextWatcher);

                updateTemperatureDisplay(temp);
            }
        });

        // Observe Perfusion Index
        viewModel.getPerfusionIndex().observe(getViewLifecycleOwner(), pi -> {
            if (pi != null) {
                etPerfusionIndex.removeTextChangedListener(piTextWatcher);
                etPerfusionIndex.setText(String.valueOf(pi));
                etPerfusionIndex.addTextChangedListener(piTextWatcher);
            }
        });

        // Observe Pulse Variability
        viewModel.getPulseVariability().observe(getViewLifecycleOwner(), pv -> {
            if (pv != null) {
                etPulseVariability.removeTextChangedListener(pvTextWatcher);
                etPulseVariability.setText(String.valueOf(pv));
                etPulseVariability.addTextChangedListener(pvTextWatcher);
            }
        });

        // Observe error messages
        viewModel.getErrorMessage().observe(getViewLifecycleOwner(), error -> {
            if (error != null && !error.isEmpty()) {
                showError(error);
                viewModel.getErrorMessage(); // Clear after showing
            }
        });
    }



    private void setupListeners() {
        // SpO2 Slider
        sliderSpo2.addOnChangeListener(spo2ChangeListener);

        // Heart Rate Slider
        sliderHeartRate.addOnChangeListener(hrChangeListener);

        // Temperature Slider
        sliderTemperature.addOnChangeListener(tempChangeListener);

        // Text input listeners
        etSpo2.addTextChangedListener(spo2TextWatcher);
        etHeartRate.addTextChangedListener(hrTextWatcher);
        etTemperature.addTextChangedListener(tempTextWatcher);
        etPerfusionIndex.addTextChangedListener(piTextWatcher);
        etPulseVariability.addTextChangedListener(pvTextWatcher);

        // Advanced section toggle
        advancedHeader.setOnClickListener(v -> toggleAdvancedSection());

        // Connect device button (placeholder for future integration)
        btnConnectDevice.setOnClickListener(v -> {
            // TODO: Implement device connection
            showMessage("Device connection feature coming soon!");
        });

        // Navigation buttons
        btnBack.setOnClickListener(v -> navigateBack());
        btnNext.setOnClickListener(v -> navigateNext());
    }

    private void setInitialValues() {
        // Set initial values if ViewModel doesn't have them
        if (viewModel.getSpo2().getValue() == null) {
            viewModel.setSpo2(98.0);
        }
        if (viewModel.getHeartRate().getValue() == null) {
            viewModel.setHeartRate(75);
        }
        if (viewModel.getTemperature().getValue() == null) {
            viewModel.setTemperature(37.0);
        }
        if (viewModel.getPerfusionIndex().getValue() == null) {
            viewModel.setPerfusionIndex(5.0);
        }
        if (viewModel.getPulseVariability().getValue() == null) {
            viewModel.setPulseVariability(1.2);
        }
    }

    // Slider Listeners
    private final Slider.OnChangeListener spo2ChangeListener = new Slider.OnChangeListener() {
        @Override
        public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
            if (fromUser) {
                viewModel.setSpo2(value);
                updateSpo2Display(value);
            }
        }
    };

    private final Slider.OnChangeListener hrChangeListener = new Slider.OnChangeListener() {
        @Override
        public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
            if (fromUser) {
                viewModel.setHeartRate((int) value);
                updateHeartRateDisplay((int) value);
            }
        }
    };

    private final Slider.OnChangeListener tempChangeListener = new Slider.OnChangeListener() {
        @Override
        public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
            if (fromUser) {
                viewModel.setTemperature((double) value);
                updateTemperatureDisplay((double) value);
            }
        }
    };

    // Text Watchers
    private final TextWatcher spo2TextWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {}

        @Override
        public void afterTextChanged(Editable s) {
            try {
                if (!s.toString().isEmpty()) {
                    double value = Double.parseDouble(s.toString());
                    viewModel.setSpo2(value);
                }
            } catch (NumberFormatException e) {
                // Ignore invalid input
            }
        }
    };

    private final TextWatcher hrTextWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {}

        @Override
        public void afterTextChanged(Editable s) {
            try {
                if (!s.toString().isEmpty()) {
                    int value = Integer.parseInt(s.toString());
                    viewModel.setHeartRate(value);
                }
            } catch (NumberFormatException e) {
                // Ignore invalid input
            }
        }
    };

    private final TextWatcher tempTextWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {}

        @Override
        public void afterTextChanged(Editable s) {
            try {
                if (!s.toString().isEmpty()) {
                    double value = Double.parseDouble(s.toString());
                    viewModel.setTemperature(value);
                }
            } catch (NumberFormatException e) {
                // Ignore invalid input
            }
        }
    };

    private final TextWatcher piTextWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {}

        @Override
        public void afterTextChanged(Editable s) {
            try {
                if (!s.toString().isEmpty()) {
                    double value = Double.parseDouble(s.toString());
                    viewModel.setPerfusionIndex(value);
                }
            } catch (NumberFormatException e) {
                // Ignore invalid input
            }
        }
    };

    private final TextWatcher pvTextWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {}

        @Override
        public void afterTextChanged(Editable s) {
            try {
                if (!s.toString().isEmpty()) {
                    double value = Double.parseDouble(s.toString());
                    viewModel.setPulseVariability(value);
                }
            } catch (NumberFormatException e) {
                // Ignore invalid input
            }
        }
    };

    // UI Update Methods
    private void updateSpo2Display(double spo2) {
        // Update value display
        requireView().findViewById(R.id.tvSpo2Value).setOnClickListener(v -> {});

        // Update status
        String status;
        int statusColor;
        int statusBg;

        if (spo2 >= 95) {
            status = "Normal";
            statusColor = getResources().getColor(R.color.success);
            statusBg = R.drawable.bg_status_normal;
        } else if (spo2 >= 90) {
            status = "Mild Hypoxia";
            statusColor = getResources().getColor(R.color.warning);
            statusBg = R.drawable.bg_status_warning;
        } else if (spo2 >= 85) {
            status = "Moderate Hypoxia";
            statusColor = getResources().getColor(R.color.warning);
            statusBg = R.drawable.bg_status_warning;
        } else {
            status = "Severe Hypoxia";
            statusColor = getResources().getColor(R.color.error);
            statusBg = R.drawable.bg_status_alert;
        }

        // Apply status
        android.widget.TextView statusView = requireView().findViewById(R.id.tvSpo2Status);
        statusView.setText(status);
        statusView.setTextColor(statusColor);
        statusView.setBackgroundResource(statusBg);
    }

    private void updateHeartRateDisplay(int heartRate) {
        requireView().findViewById(R.id.tvHeartRateValue).setOnClickListener(v -> {});

        String status;
        int statusColor;
        int statusBg;

        Integer age = viewModel.getAge().getValue();
        int maxHR = 220 - (age != null ? age : 30);

        if (heartRate < 60) {
            status = "Low";
            statusColor = getResources().getColor(R.color.warning);
            statusBg = R.drawable.bg_status_warning;
        } else if (heartRate < 100) {
            status = "Normal";
            statusColor = getResources().getColor(R.color.success);
            statusBg = R.drawable.bg_status_normal;
        } else if (heartRate < maxHR * 0.7) {
            status = "Elevated";
            statusColor = getResources().getColor(R.color.warning);
            statusBg = R.drawable.bg_status_warning;
        } else {
            status = "High";
            statusColor = getResources().getColor(R.color.error);
            statusBg = R.drawable.bg_status_alert;
        }

        android.widget.TextView statusView = requireView().findViewById(R.id.tvHeartRateStatus);
        statusView.setText(status);
        statusView.setTextColor(statusColor);
        statusView.setBackgroundResource(statusBg);
    }

    private void updateTemperatureDisplay(double temperature) {
        requireView().findViewById(R.id.tvTemperatureValue).setOnClickListener(v -> {});

        String status;
        int statusColor;
        int statusBg;

        if (temperature >= 36.5 && temperature <= 37.5) {
            status = "Normal";
            statusColor = getResources().getColor(R.color.success);
            statusBg = R.drawable.bg_status_normal;
        } else if (temperature <= 38.5) {
            status = "Mild Fever";
            statusColor = getResources().getColor(R.color.warning);
            statusBg = R.drawable.bg_status_warning;
        } else if (temperature <= 39.5) {
            status = "Moderate Fever";
            statusColor = getResources().getColor(R.color.warning);
            statusBg = R.drawable.bg_status_warning;
        } else {
            status = "High Fever";
            statusColor = getResources().getColor(R.color.error);
            statusBg = R.drawable.bg_status_alert;
        }

        android.widget.TextView statusView = requireView().findViewById(R.id.tvTemperatureStatus);
        statusView.setText(status);
        statusView.setTextColor(statusColor);
        statusView.setBackgroundResource(statusBg);
    }

    private void toggleAdvancedSection() {
        if (advancedContent.getVisibility() == View.GONE) {
            // Expand
            advancedContent.setVisibility(View.VISIBLE);
            advancedContent.startAnimation(slideDown);
            requireView().findViewById(R.id.ivExpandAdvanced).setOnClickListener(v -> {});
        } else {
            // Collapse
            advancedContent.startAnimation(slideUp);
            advancedContent.setVisibility(View.GONE);
        }
    }

    private void navigateBack() {
        // Navigate back to previous fragment
        requireActivity().getSupportFragmentManager().popBackStack();
    }

    private void navigateNext() {
        // Validate inputs
        if (validateInputs()) {
            // Move to next step
            viewModel.nextStep();

            // Navigate to symptoms fragment
            Step3SymptomsFragment nextFragment = new Step3SymptomsFragment();

            requireActivity().getSupportFragmentManager()
                    .beginTransaction()
                    .setCustomAnimations(
                            R.anim.slide_in_right,
                            R.anim.slide_out_left,
                            R.anim.slide_in_left,
                            R.anim.slide_out_right
                    )
                    .replace(R.id.fragmentContainer, nextFragment)
                    .addToBackStack("step2")
                    .commit();
        }
    }

    private boolean validateInputs() {
        boolean isValid = true;

        // Validate SpO2
        String spo2Text = etSpo2.getText().toString();
        if (spo2Text.isEmpty()) {
            spo2InputLayout.setError("SpO2 is required");
            isValid = false;
        } else {
            try {
                double spo2 = Double.parseDouble(spo2Text);
                if (spo2 < 70 || spo2 > 100) {
                    spo2InputLayout.setError("SpO2 must be between 70-100%");
                    isValid = false;
                } else {
                    spo2InputLayout.setError(null);
                }
            } catch (NumberFormatException e) {
                spo2InputLayout.setError("Invalid number format");
                isValid = false;
            }
        }

        // Validate Heart Rate
        String hrText = etHeartRate.getText().toString();
        if (hrText.isEmpty()) {
            hrInputLayout.setError("Heart rate is required");
            isValid = false;
        } else {
            try {
                int hr = Integer.parseInt(hrText);
                if (hr < 40 || hr > 200) {
                    hrInputLayout.setError("Heart rate must be between 40-200 bpm");
                    isValid = false;
                } else {
                    hrInputLayout.setError(null);
                }
            } catch (NumberFormatException e) {
                hrInputLayout.setError("Invalid number format");
                isValid = false;
            }
        }

        // Validate Temperature
        String tempText = etTemperature.getText().toString();
        if (tempText.isEmpty()) {
            tempInputLayout.setError("Temperature is required");
            isValid = false;
        } else {
            try {
                double temp = Double.parseDouble(tempText);
                if (temp < 35 || temp > 42) {
                    tempInputLayout.setError("Temperature must be between 35-42°C");
                    isValid = false;
                } else {
                    tempInputLayout.setError(null);
                }
            } catch (NumberFormatException e) {
                tempInputLayout.setError("Invalid number format");
                isValid = false;
            }
        }

        // Validate optional fields (if filled)
        String piText = etPerfusionIndex.getText().toString();
        if (!piText.isEmpty()) {
            try {
                double pi = Double.parseDouble(piText);
                if (pi < 0.02 || pi > 20) {
                    piInputLayout.setError("Perfusion index must be between 0.02-20");
                    isValid = false;
                } else {
                    piInputLayout.setError(null);
                }
            } catch (NumberFormatException e) {
                piInputLayout.setError("Invalid number format");
                isValid = false;
            }
        }

        String pvText = etPulseVariability.getText().toString();
        if (!pvText.isEmpty()) {
            try {
                double pv = Double.parseDouble(pvText);
                if (pv < 0) {
                    pvInputLayout.setError("Pulse variability cannot be negative");
                    isValid = false;
                } else {
                    pvInputLayout.setError(null);
                }
            } catch (NumberFormatException e) {
                pvInputLayout.setError("Invalid number format");
                isValid = false;
            }
        }

        return isValid;
    }

    private void showError(String message) {
        android.widget.Toast.makeText(getContext(), message, android.widget.Toast.LENGTH_SHORT).show();
    }

    private void showMessage(String message) {
        android.widget.Toast.makeText(getContext(), message, android.widget.Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onPause() {
        super.onPause();
        // Save data when leaving fragment
        saveCurrentData();
    }

    private void saveCurrentData() {
        // Data is already saved via ViewModel observers
        // This is just a safety measure
        try {
            if (!etSpo2.getText().toString().isEmpty()) {
                viewModel.setSpo2(Double.parseDouble(etSpo2.getText().toString()));
            }
            if (!etHeartRate.getText().toString().isEmpty()) {
                viewModel.setHeartRate(Integer.parseInt(etHeartRate.getText().toString()));
            }
            if (!etTemperature.getText().toString().isEmpty()) {
                viewModel.setTemperature(Double.parseDouble(etTemperature.getText().toString()));
            }
            if (!etPerfusionIndex.getText().toString().isEmpty()) {
                viewModel.setPerfusionIndex(Double.parseDouble(etPerfusionIndex.getText().toString()));
            }
            if (!etPulseVariability.getText().toString().isEmpty()) {
                viewModel.setPulseVariability(Double.parseDouble(etPulseVariability.getText().toString()));
            }
        } catch (NumberFormatException e) {
            // Ignore parsing errors
        }
    }
}