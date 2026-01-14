package com.example.smart_disease_prediction_sys;


import android.content.Context;
import android.content.SharedPreferences;

public class UserManager {
    private static UserManager instance;
    private SharedPreferences sharedPreferences;

    private static final String PREFS_NAME = "UserPrefs";
    private static final String KEY_USER_ID = "USERID";
    private static final String KEY_USERNAME = "USERNAME";

    private UserManager(Context context) {
        sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    public static synchronized UserManager getInstance(Context context) {
        if (instance == null) {
            instance = new UserManager(context.getApplicationContext());
        }
        return instance;
    }

    public void saveUser(int userid, String username) {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putInt(KEY_USER_ID, userid);
        editor.putString(KEY_USERNAME, username);
        editor.putBoolean("IS_LOGGED_IN", true);
        editor.apply();
    }

    public int getUserId() {
        return sharedPreferences.getInt(KEY_USER_ID, 0);
    }

    public String getUsername() {
        return sharedPreferences.getString(KEY_USERNAME, "");
    }

    public boolean isLoggedIn() {
        return sharedPreferences.getBoolean("IS_LOGGED_IN", false);
    }

    public void logout() {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.apply();
    }
}