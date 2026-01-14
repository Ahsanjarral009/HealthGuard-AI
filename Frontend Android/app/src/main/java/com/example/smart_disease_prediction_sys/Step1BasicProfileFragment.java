package com.example.smart_disease_prediction_sys;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;

import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.slider.Slider;

public class Step1BasicProfileFragment extends Fragment {

    private AssesmentViewModel viewModel;
    private Slider ageSlider;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_step1_basic_profile, container, false);

        viewModel = new ViewModelProvider(requireActivity()).get(AssesmentViewModel.class);

        ageSlider = view.findViewById(R.id.ageSlider);
        TextView ageValue = view.findViewById(R.id.ageValue);
        TextView ageGroup = view.findViewById(R.id.tvAgeGroup);
        MaterialButtonToggleGroup genderToggleGroup = view.findViewById(R.id.genderToggleGroup);

        // Observe age changes
        viewModel.getAge().observe(getViewLifecycleOwner(), age -> {
            if (age != null && ageSlider.getValue() != age) {
                ageSlider.setValue(age);
                ageValue.setText(age + " years");
            }
        });

        // Observe age group
        viewModel.getAgeGroup().observe(getViewLifecycleOwner(), group -> {
            if (group != null) {
                ageGroup.setText(group);
            }
        });

        // Set up age slider
        ageSlider.addOnChangeListener((slider, value, fromUser) -> {
            if (fromUser) {
                viewModel.setAge((int) value);
            }
        });

        // Set up gender selection
        viewModel.getGender().observe(getViewLifecycleOwner(), gender -> {
            if (gender != null) {
                switch (gender) {
                    case "M":
                        genderToggleGroup.check(R.id.btnMale);
                        break;
                    case "F":
                        genderToggleGroup.check(R.id.btnFemale);
                        break;
                    default:
                        genderToggleGroup.check(R.id.btnNotSpecified);
                }
            }
        });

        genderToggleGroup.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (isChecked) {
                if (checkedId == R.id.btnMale) {
                    viewModel.setGender("M");
                } else if (checkedId == R.id.btnFemale) {
                    viewModel.setGender("F");
                } else {
                    viewModel.setGender("U");
                }
            }
        });

        //progress indicator
        LinearProgressIndicator progressIndicator =
                view.findViewById(R.id.progressIndicator);

        progressIndicator.setProgress(20);


        // Next button
        view.findViewById(R.id.nextBtn).setOnClickListener(v -> {
            if (viewModel.validateStep1()) {
                navigateToStep2();
            } else {
                // Show error message from ViewModel
                viewModel.getErrorMessage().observe(getViewLifecycleOwner(), error -> {
                    if (error != null && !error.isEmpty()) {
                        Toast.makeText(getContext(), error, Toast.LENGTH_SHORT).show();
                        viewModel.getErrorMessage(); // Clear error
                    }
                });
            }
        });

        return view;
    }

    private void navigateToStep2() {
        viewModel.nextStep();
        getParentFragmentManager()
                .beginTransaction()
                .replace(R.id.fragmentContainer, new Step2VitalSignsFragment())
                .addToBackStack(null)
                .commit();
    }
}