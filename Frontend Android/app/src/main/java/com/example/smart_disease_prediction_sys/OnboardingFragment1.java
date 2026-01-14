package com.example.smart_disease_prediction_sys;

import android.os.Bundle;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class OnboardingFragment1 extends Fragment {

    public OnboardingFragment1() {
        super(R.layout.fragment_onboadring_1);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {

        ImageView imgDoctor = view.findViewById(R.id.imgDoctor);
        Animation pulse = AnimationUtils.loadAnimation(requireContext(), R.anim.pulse);
        imgDoctor.startAnimation(pulse);
    }
}
