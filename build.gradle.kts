// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
    // Added 2026-08-04 for native push notifications -- requires
    // app/google-services.json (from the Firebase console) to be present
    // or the build fails immediately with a clear "File google-services.json
    // is missing" error. See docs/FCM_SETUP.md.
    alias(libs.plugins.google.services) apply false
}