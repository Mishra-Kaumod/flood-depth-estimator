package com.flood.monitor

import android.os.Bundle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.android.gms.maps.model.LatLng
import com.google.maps.android.compose.GoogleMap
import com.google.maps.android.compose.MapProperties
import com.google.maps.android.compose.Marker
import com.google.maps.android.compose.MarkerState

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                FloodMonitorScreen()
            }
        }
    }
}

@Composable
private fun FloodMonitorScreen(vm: UploadViewModel = viewModel()) {
    val context = LocalContext.current
    val pickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia(10),
        onResult = { uris ->
            vm.uploadImages(
                contentResolver = context.contentResolver,
                uris = uris,
            )
        },
    )

    var loaded by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        if (!loaded) {
            vm.loadPoints()
            loaded = true
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Bengaluru Flood Map", style = MaterialTheme.typography.headlineSmall)
        Text(vm.statusMessage, style = MaterialTheme.typography.bodySmall)

        GoogleMap(
            modifier = Modifier
                .fillMaxWidth()
                .height(360.dp),
            properties = MapProperties(isMyLocationEnabled = false),
        ) {
            vm.points.forEach { p ->
                Marker(
                    state = MarkerState(position = LatLng(p.latitude, p.longitude)),
                    title = p.location_name,
                    snippet = "Depth ${p.depth_cm} cm | Scale ${p.intensity_scale}",
                    onClick = {
                        vm.selectPoint(p)
                        false
                    },
                )
            }
        }

        Button(onClick = { vm.loadPoints() }) {
            Text("Refresh from Backend")
        }

        Button(onClick = {
            pickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) {
            Text("Upload Multiple Images")
        }

        val selected = vm.selectedPoint
        Text("Selected point JSON", style = MaterialTheme.typography.titleMedium)
        Text(
            text = if (selected == null) "{}" else """
                {
                  "camera_id": "${selected.camera_id}",
                  "location_id": ${selected.location_id?.let { "\"$it\"" } ?: "null"},
                  "location_name": "${selected.location_name}",
                  "latitude": ${selected.latitude},
                  "longitude": ${selected.longitude},
                  "depth_cm": ${selected.depth_cm},
                  "intensity_scale": ${selected.intensity_scale},
                  "image_name": "${selected.image_name}",
                  "timestamp": "${selected.timestamp}",
                  "safety_risk_assessment": "${selected.safety_risk_assessment}"
                }
            """.trimIndent(),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
