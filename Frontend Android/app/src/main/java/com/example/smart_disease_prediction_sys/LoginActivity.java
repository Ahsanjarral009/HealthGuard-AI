package com.example.smart_disease_prediction_sys;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.example.smart_disease_prediction_sys.UserManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.google.gson.Gson;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import java.util.Map;

public class LoginActivity extends AppCompatActivity {

    private EditText edtEmail, edtPassword;
    private Button btnLogin;

    private ApiService apiService;
    private AssesmentViewModel viewModel;
    private SharedPreferences sharedPreferences;
    private Gson gson;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        edtEmail = findViewById(R.id.edtEmail);
        edtPassword = findViewById(R.id.edtPassword);
        btnLogin = findViewById(R.id.btnLogin);

        viewModel = new ViewModelProvider(this).get(AssesmentViewModel.class);

        // Initialize SharedPreferences
        sharedPreferences = getSharedPreferences("UserPrefs", MODE_PRIVATE);

        initializeApiService();

        btnLogin.setOnClickListener(v -> loginUser());

        // Check if user is already logged in
        checkExistingLogin();
    }

    private void initializeApiService() {
        gson = new Gson();

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://10.0.2.2:8000/")
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build();

        apiService = retrofit.create(ApiService.class);
    }

    private void checkExistingLogin() {
        boolean isLoggedIn = sharedPreferences.getBoolean("IS_LOGGED_IN", false);
        if (isLoggedIn) {
            // Auto-login user
            int savedUserId = sharedPreferences.getInt("USERID", -1);
            String savedUsername = sharedPreferences.getString("USERNAME", "");

            if (savedUserId != -1) {
                viewModel.setUserid(savedUserId);
                viewModel.setUsername(savedUsername);

                Intent intent = new Intent(LoginActivity.this, MainActivity.class);
                intent.putExtra("USERID", savedUserId);
                intent.putExtra("USERNAME", savedUsername);
                startActivity(intent);
                finish();
            }
        }
    }

    private void loginUser() {
        String email = edtEmail.getText().toString().trim();
        String password = edtPassword.getText().toString().trim();

        if (email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Email and password required", Toast.LENGTH_SHORT).show();
            return;
        }

        viewModel.setEmail(email);
        viewModel.setPassword(password);

        Map<String, Object> payload = viewModel.prepareApiPayloadlogin();

        Log.d("LOGIN_PAYLOAD", new Gson().toJson(payload));

        Call<LoginResponse> call = apiService.login(payload);

        call.enqueue(new Callback<LoginResponse>() {
            @Override
            public void onResponse(Call<LoginResponse> call, Response<LoginResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    LoginResponse res = response.body();

                    if (res.isSuccess()) {
                        // Save user data to SharedPreferences

                        saveUserData(res);
                        UserManager.getInstance(LoginActivity.this).saveUser(res.getUserid(), res.getUsername());
                        // Set userid in ViewModel
                        viewModel.setUserid(res.getUserid());
                        viewModel.setUsername(res.getUsername());

                        Toast.makeText(LoginActivity.this,
                                "Welcome " + res.getUsername(),
                                Toast.LENGTH_LONG).show();

                        // Navigate to MainActivity with user data
                        Intent intent = new Intent(LoginActivity.this, MainActivity.class);
                        intent.putExtra("USERID", res.getUserid());
                        intent.putExtra("USERNAME", res.getUsername());
                        startActivity(intent);
                        finish();

                    } else {
                        Toast.makeText(LoginActivity.this,
                                res.getMessage(),
                                Toast.LENGTH_LONG).show();
                    }
                } else {
                    String errorMsg = "Invalid login credentials";
                    try {
                        if (response.errorBody() != null) {
                            errorMsg = "Error: " + response.code() + " - " + response.errorBody().string();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    Toast.makeText(LoginActivity.this, errorMsg, Toast.LENGTH_LONG).show();
                }
            }

            @Override
            public void onFailure(Call<LoginResponse> call, Throwable t) {
                Toast.makeText(LoginActivity.this,
                        "Server error: " + t.getMessage(),
                        Toast.LENGTH_LONG).show();
                Log.e("LOGIN_ERROR", "Login failed", t);
            }
        });
    }

    private void saveUserData(LoginResponse response) {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putInt("USERID", response.getUserid());
        editor.putString("USERNAME", response.getUsername());
        editor.putString("EMAIL", edtEmail.getText().toString().trim());
        editor.putBoolean("IS_LOGGED_IN", true);
        editor.apply();

        Log.d("USER_DATA", "Saved user data: ID=" + response.getUserid() +
                ", Name=" + response.getUsername());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Clear sensitive data from ViewModel
        viewModel.setEmail("");
        viewModel.setPassword("");
    }
}