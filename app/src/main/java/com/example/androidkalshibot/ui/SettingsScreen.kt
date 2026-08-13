package com.example.androidkalshibot.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    initialUrl: String,
    initialUser: String,
    initialPass: String,
    canCancel: Boolean,
    onSave: (String, String, String) -> Unit,
    onCancel: () -> Unit,
) {
    var url by remember { mutableStateOf(initialUrl) }
    var user by remember { mutableStateOf(initialUser) }
    var pass by remember { mutableStateOf(initialPass) }

    Scaffold(topBar = { TopAppBar(title = { Text("Server connection") }) }) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                "Enter your server's HTTPS address (from deploy/README.md, " +
                    "e.g. https://203-0-113-5.nip.io) and the login you set for it. " +
                    "This connects to the JSON API, not the web dashboard -- same " +
                    "server, same login. Stored encrypted on this device only.",
                style = MaterialTheme.typography.bodyMedium,
            )
            OutlinedTextField(
                value = url, onValueChange = { url = it },
                label = { Text("Server URL") }, modifier = Modifier.fillMaxWidth(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                singleLine = true,
            )
            OutlinedTextField(
                value = user, onValueChange = { user = it },
                label = { Text("Username") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
            )
            OutlinedTextField(
                value = pass, onValueChange = { pass = it },
                label = { Text("Password") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
            )
            Button(
                onClick = { onSave(url.trim().trimEnd('/'), user.trim(), pass) },
                enabled = url.trim().startsWith("https://"),
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Save & connect") }
            if (!url.trim().startsWith("https://") && url.isNotBlank()) {
                Text(
                    "URL must start with https:// -- this app refuses plain http on purpose.",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (canCancel) {
                TextButton(onClick = onCancel, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
            }
        }
    }
}
