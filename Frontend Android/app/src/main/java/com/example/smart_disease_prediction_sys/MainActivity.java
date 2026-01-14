package com.example.smart_disease_prediction_sys;

import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.Chip;
import com.google.android.material.snackbar.Snackbar;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    // UI Components
    private MaterialButton btnStartAssessment;
    private MaterialButton btnLogout;
    private MaterialButton btnRefreshTips;
    private TextView txtUsername;
    private TextView tvUserId;
    private TextView tvLastCheck;
    private TextView tvNoRecent;
    private Chip chipHealthStatus;
    private ImageView ivUserAvatar;
    private RecyclerView rvRecentAssessments;

    // ViewModel
    private AssesmentViewModel viewModel;

    // SharedPreferences
    private SharedPreferences sharedPreferences;
    private SharedPreferences.Editor editor;

    // Adapter for recent assessments
    private RecentAssessmentAdapter recentAdapter;

    // Constants
    private static final String TAG = "MainActivity";
    private static final String PREF_NAME = "UserPrefs";
    private static final String KEY_USER_ID = "USERID";
    private static final String KEY_USERNAME = "USERNAME";
    private static final String KEY_LAST_CHECK = "LAST_CHECK";
    private static final String KEY_IS_LOGGED_IN = "IS_LOGGED_IN";
    private static final String KEY_HEALTH_STATUS = "HEALTH_STATUS";

    @SuppressLint("CommitPrefEdits")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        // Handle window insets for edge-to-edge display
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Log.d(TAG, "MainActivity onCreate");

        // Initialize SharedPreferences FIRST
        sharedPreferences = getSharedPreferences(PREF_NAME, MODE_PRIVATE);
        editor = sharedPreferences.edit();

        // Initialize ViewModel
        viewModel = new ViewModelProvider(this).get(AssesmentViewModel.class);

        // Initialize UI Components - CORRECT ORDER
        initializeViews();          // Step 1: Find views
        setupRecyclerView();        // Step 2: Setup RecyclerView and Adapter
        initializeUI();             // Step 3: Set initial UI state (AFTER adapter is ready)

        // Load user data
        loadUserData();

        // Set up click listeners
        setupClickListeners();

        // Set up ViewModel observers
        setupViewModelObservers();
    }

    private void initializeViews() {
        // Just find views, don't set any data that depends on adapter
        txtUsername = findViewById(R.id.txtusername);
        tvUserId = findViewById(R.id.tvUserId);
        tvLastCheck = findViewById(R.id.tvLastCheck);
        tvNoRecent = findViewById(R.id.tvNoRecent);
        chipHealthStatus = findViewById(R.id.chipHealthStatus);
        ivUserAvatar = findViewById(R.id.ivUserAvatar);
        rvRecentAssessments = findViewById(R.id.rvRecentAssessments);

        btnStartAssessment = findViewById(R.id.btnStartAssessment);
        btnLogout = findViewById(R.id.btnLogout);

    }

    private void setupRecyclerView() {
        // Initialize adapter with empty list
        recentAdapter = new RecentAssessmentAdapter(new ArrayList<>());

        // Setup RecyclerView
        rvRecentAssessments.setLayoutManager(new LinearLayoutManager(
                this, LinearLayoutManager.HORIZONTAL, false
        ));
        rvRecentAssessments.setAdapter(recentAdapter);

        // Add item decoration for spacing (if you have this class)
        try {
            // Create a simple ItemDecoration if you don't have the class
            rvRecentAssessments.addItemDecoration(new RecyclerView.ItemDecoration() {
                // Add spacing between items
            });
        } catch (Exception e) {
            Log.e(TAG, "Error setting up RecyclerView decoration", e);
        }
    }

    private void initializeUI() {
        // Set initial UI states WITHOUT calling updateUIForGuest
        // We'll call it later after checking user status
        txtUsername.setText("👋 Hello, Guest!");
        tvUserId.setText("User ID: Not logged in");
        tvLastCheck.setText("Last check: Not available");
        chipHealthStatus.setText("Status: Not available");
        chipHealthStatus.setChipBackgroundColorResource(R.color.gray_light);
        // Make sure these drawables exist or use default ones
        try {
            chipHealthStatus.setChipIconResource(R.drawable.ic_info);
        } catch (Exception e) {
            // Use default if drawable doesn't exist
        }
        btnLogout.setVisibility(View.GONE);

        // Set visibility for "no recent" text
        tvNoRecent.setVisibility(View.VISIBLE);
        tvNoRecent.setText("No assessments yet");
    }

    private void loadUserData() {
        // Check if user data is passed from LoginActivity
        Intent intent = getIntent();
        int userid = intent.getIntExtra("USERID", -1);
        String username = intent.getStringExtra("USERNAME");

        if (userid != -1 && username != null && !username.isEmpty()) {
            // Data passed from LoginActivity
            handleUserData(userid, username);
        } else {
            // Check SharedPreferences for saved user data
            checkSavedUserData();
        }
    }

    private void checkSavedUserData() {
        boolean isLoggedIn = sharedPreferences.getBoolean(KEY_IS_LOGGED_IN, false);

        if (isLoggedIn) {
            int savedUserId = sharedPreferences.getInt(KEY_USER_ID, -1);
            String savedUsername = sharedPreferences.getString(KEY_USERNAME, "");

            if (savedUserId != -1 && !savedUsername.isEmpty()) {
                handleUserData(savedUserId, savedUsername);
            } else {
                // Invalid saved data, show guest mode
                updateUIForGuest();
            }
        } else {
            // Not logged in, show guest mode
            updateUIForGuest();
        }
    }

    private void handleUserData(int userid, String username) {
        // Save to ViewModel
        viewModel.setUserid(userid);
        viewModel.setUsername(username);

        // Save to SharedPreferences
        saveUserPreferences(userid, username);

        // Update UI
        updateUserUI(userid, username);

        // Load recent assessments for this user
        loadRecentAssessments(userid);

        // Update last check date
        updateLastCheckDate();

        Log.d(TAG, "User loaded: " + username + " (ID: " + userid + ")");
    }

    private void saveUserPreferences(int userid, String username) {
        editor.putInt(KEY_USER_ID, userid);
        editor.putString(KEY_USERNAME, username);
        editor.putBoolean(KEY_IS_LOGGED_IN, true);
        editor.apply();
    }

    @SuppressLint("SetTextI18n")
    private void updateUserUI(int userid, String username) {
        // Update username display
        txtUsername.setText("👋 Hello, " + username + "!");

        // Update user ID display
        tvUserId.setText("User ID: " + userid);

        // Update logout button visibility
        btnLogout.setVisibility(View.VISIBLE);

        // Load saved health status or set default
        String healthStatus = sharedPreferences.getString(KEY_HEALTH_STATUS, "Good");
        updateHealthStatus(healthStatus);

        // Load last check date
        String lastCheck = sharedPreferences.getString(KEY_LAST_CHECK, "Not available");
        tvLastCheck.setText("Last check: " + lastCheck);
    }

    private void updateUIForGuest() {
        txtUsername.setText("👋 Hello, Guest!");
        tvUserId.setText("User ID: Not logged in");
        tvLastCheck.setText("Last check: Not available");
        chipHealthStatus.setText("Status: Not available");

        // Use safe resource references
        try {
            chipHealthStatus.setChipBackgroundColorResource(R.color.gray_light);
            chipHealthStatus.setChipIconResource(R.drawable.ic_info);
        } catch (Exception e) {
            Log.e(TAG, "Error setting chip resources", e);
        }

        btnLogout.setVisibility(View.GONE);

        // SAFE: Check if adapter is initialized before using it
        if (recentAdapter != null) {
            recentAdapter.clearData();
        }

        tvNoRecent.setVisibility(View.VISIBLE);
        tvNoRecent.setText("No assessments yet. Login to see your history!");
    }

    private void updateHealthStatus(String status) {
        chipHealthStatus.setText("Status: " + status);

        // Set color based on status
        try {
            switch (status.toLowerCase()) {
                case "excellent":
                case "good":
                    chipHealthStatus.setChipBackgroundColorResource(R.color.success_light);
                    chipHealthStatus.setChipIconResource(R.drawable.check_16799048);
                    break;
                case "fair":
                case "moderate":
                    chipHealthStatus.setChipBackgroundColorResource(R.color.warning_light);
                    chipHealthStatus.setChipIconResource(R.drawable.ic_warning);
                    break;
                case "poor":
                case "critical":
                    chipHealthStatus.setChipBackgroundColorResource(R.color.error_light);
                    chipHealthStatus.setChipIconResource(R.drawable.ic_critical);
                    break;
                default:
                    chipHealthStatus.setChipBackgroundColorResource(R.color.gray_light);
                    chipHealthStatus.setChipIconResource(R.drawable.ic_info);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error updating health status", e);
        }

        // Save status to preferences
        editor.putString(KEY_HEALTH_STATUS, status);
        editor.apply();
    }

    private void updateLastCheckDate() {
        String currentDate = new SimpleDateFormat("MMM dd, yyyy", Locale.getDefault()).format(new Date());
        tvLastCheck.setText("Last check: " + currentDate);
        editor.putString(KEY_LAST_CHECK, currentDate);
        editor.apply();
    }

    private void loadRecentAssessments(int userId) {
        // Check if adapter is initialized
        if (recentAdapter == null) {
            Log.e(TAG, "Adapter not initialized when loading recent assessments");
            return;
        }

        // Clear existing data
        recentAdapter.clearData();

        // Check if we have any saved assessments in SharedPreferences
        String savedAssessments = sharedPreferences.getString("RECENT_ASSESSMENTS_" + userId, "");

        if (!savedAssessments.isEmpty()) {
            // In real app, parse saved assessments here
            // For now, show mock data
            List<RecentAssessment> recentList = new ArrayList<>();
            recentList.add(new RecentAssessment("Today", "Respiratory Check", "Normal", R.color.success));
            recentList.add(new RecentAssessment("Yesterday", "Full Assessment", "Moderate", R.color.warning));

            recentAdapter.setData(recentList);
            tvNoRecent.setVisibility(View.GONE);
        } else {
            tvNoRecent.setVisibility(View.VISIBLE);
            tvNoRecent.setText("No assessments yet. Start your first one!");
        }
    }

    private void setupClickListeners() {
        // Start Assessment Button
        if (btnStartAssessment != null) {
            btnStartAssessment.setOnClickListener(v -> {
                if (viewModel.isUserLoggedIn()) {
                    startAssessment();
                } else {
                    showLoginPrompt();
                }
            });
        }

        // Logout Button
        if (btnLogout != null) {
            btnLogout.setOnClickListener(v -> {
                showLogoutConfirmation();
            });
        }

        // Refresh Tips Button
        if (btnRefreshTips != null) {
            btnRefreshTips.setOnClickListener(v -> {
                refreshHealthTips();
            });
        }

        // User Avatar Click (optional: navigate to profile)
        if (ivUserAvatar != null) {
            ivUserAvatar.setOnClickListener(v -> {
                showUserProfile();
            });
        }
    }

    private void setupViewModelObservers() {
        // Observe user ID changes
        if (viewModel.getUserid() != null) {
            viewModel.getUserid().observe(this, userid -> {
                if (userid != null && userid != -1) {
                    // User is logged in
                    String username = viewModel.getUsername().getValue();
                    if (username != null && !username.isEmpty()) {
                        updateUserUI(userid, username);
                    }
                } else {
                    // User is not logged in
                    updateUIForGuest();
                }
            });
        }

        // Observe error messages
        if (viewModel.getErrorMessage() != null) {
            viewModel.getErrorMessage().observe(this, error -> {
                if (error != null && !error.isEmpty()) {
                    showErrorSnackbar(error);
                }
            });
        }
    }

    private void startAssessment() {
        Log.d(TAG, "Starting assessment for user: " + viewModel.getUsername().getValue());

        // Pass user info to AssessmentActivity
        Intent intent = new Intent(MainActivity.this, AssessmentActivity.class);
        Integer currentUserid = viewModel.getUserid().getValue();
        String currentUsername = viewModel.getUsername().getValue();

        if (currentUserid != null && currentUserid != -1) {
            intent.putExtra("USERID", currentUserid);
        }
        if (currentUsername != null && !currentUsername.isEmpty()) {
            intent.putExtra("USERNAME", currentUsername);
        }

        startActivity(intent);
    }

    private void showLoginPrompt() {
        new AlertDialog.Builder(this)
                .setTitle("Login Required")
                .setMessage("You need to login to start an assessment. Would you like to login now?")
                .setPositiveButton("Login", (dialog, which) -> {
                    navigateToLogin();
                })
                .setNegativeButton("Cancel", null)
                .setCancelable(true)
                .show();
    }

    private void showLogoutConfirmation() {
        new AlertDialog.Builder(this)
                .setTitle("Logout")
                .setMessage("Are you sure you want to logout?")
                .setPositiveButton("Yes, Logout", (dialog, which) -> {
                    performLogout();
                })
                .setNegativeButton("Cancel", null)
                .setCancelable(true)
                .show();
    }

    private void performLogout() {
        // Clear ViewModel user data
        viewModel.clearUserData();

        // Clear SharedPreferences
        editor.clear();
        editor.apply();

        // Show logout message
        Toast.makeText(this, "Logged out successfully", Toast.LENGTH_SHORT).show();

        // Update UI for guest
        updateUIForGuest();

        Log.d(TAG, "User logged out");
    }

    private void refreshHealthTips() {
        Toast.makeText(this, "Refreshing health tips...", Toast.LENGTH_SHORT).show();

        Snackbar.make(findViewById(android.R.id.content),
                        "Health tips refreshed!",
                        Snackbar.LENGTH_SHORT)
                .show();
    }

    private void showUserProfile() {
        if (viewModel.isUserLoggedIn()) {
            String userInfo = "Username: " + viewModel.getUsername().getValue() +
                    "\nUser ID: " + viewModel.getUserid().getValue();

            new AlertDialog.Builder(this)
                    .setTitle("User Profile")
                    .setMessage(userInfo)
                    .setPositiveButton("OK", null)
                    .show();
        } else {
            Toast.makeText(this, "Please login to view profile", Toast.LENGTH_SHORT).show();
        }
    }

    private void navigateToLogin() {
        Intent intent = new Intent(MainActivity.this, LoginActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
        finish();
    }

    private void showErrorSnackbar(String error) {
        View rootView = findViewById(android.R.id.content);
        Snackbar.make(rootView, error, Snackbar.LENGTH_LONG)
                .setAction("Dismiss", v -> {})
                .show();
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "MainActivity onResume");

        // Refresh user data when returning to activity
        if (viewModel.isUserLoggedIn()) {
            Integer userid = viewModel.getUserid().getValue();
            if (userid != null && recentAdapter != null) {
                loadRecentAssessments(userid);
            }
        }
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        new AlertDialog.Builder(this)
                .setTitle("Exit App")
                .setMessage("Are you sure you want to exit?")
                .setPositiveButton("Exit", (dialog, which) -> {
                    finishAffinity();
                })
                .setNegativeButton("Cancel", null)
                .setCancelable(true)
                .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "MainActivity onDestroy");
    }
}