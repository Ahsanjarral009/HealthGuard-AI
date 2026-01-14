package com.example.smart_disease_prediction_sys;

import android.os.Bundle;

import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;


import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.LinearLayout;




public class OnboardingFragment2 extends Fragment {

    public OnboardingFragment2() {
        super(R.layout.fragment_onboarding_2); // your XML layout
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // --- PULSE animation for icons ---
        int[] iconIds = {R.id.iconStep1, R.id.iconStep2, R.id.iconStep3};
        Animation pulse = AnimationUtils.loadAnimation(requireContext(), R.anim.pulse);
        for (int id : iconIds) {
            ImageView icon = view.findViewById(id);
            icon.startAnimation(pulse);
        }

        // --- STAGGERED SLIDE-IN animation for step cards ---
        int[] stepIds = {R.id.step1, R.id.step2, R.id.step3};
        for (int i = 0; i < stepIds.length; i++) {
            LinearLayout stepCard = view.findViewById(stepIds[i]);
            Animation slideIn = AnimationUtils.loadAnimation(requireContext(), R.anim.slide_in);
            slideIn.setStartOffset(i * 200); // stagger by 200ms
            stepCard.startAnimation(slideIn);
        }

        // Optional: you can also add fade-in for title or important box here
    }
}
