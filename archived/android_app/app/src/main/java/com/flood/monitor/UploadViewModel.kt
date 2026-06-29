package com.flood.monitor

import android.content.ContentResolver
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream

class UploadViewModel : ViewModel() {
    var points: List<MapPoint> = emptyList()
        private set
    var selectedPoint: MapPoint? = null
        private set
    var statusMessage: String = ""
        private set

    fun loadPoints() {
        viewModelScope.launch {
            runCatching { FloodApiClient.service.getMapPoints() }
                .onSuccess {
                    points = it.points
                    if (selectedPoint == null && points.isNotEmpty()) {
                        selectedPoint = points.first()
                    }
                    statusMessage = "Loaded ${it.count} flood points"
                }
                .onFailure { statusMessage = "Failed to load points: ${it.message}" }
        }
    }

    fun selectPoint(point: MapPoint) {
        selectedPoint = point
    }

    fun uploadImages(
        contentResolver: ContentResolver,
        uris: List<Uri>,
        cameraId: String? = null,
        locationId: String? = null,
    ) {
        if (uris.isEmpty()) {
            statusMessage = "No images selected"
            return
        }
        viewModelScope.launch {
            runCatching {
                val parts = uris.mapIndexed { idx, uri ->
                    val tempFile = File.createTempFile("upload_$idx", ".jpg")
                    contentResolver.openInputStream(uri).use { input ->
                        FileOutputStream(tempFile).use { output ->
                            input?.copyTo(output)
                        }
                    }
                    val body = tempFile.asRequestBody("image/jpeg".toMediaTypeOrNull())
                    MultipartBody.Part.createFormData("images", tempFile.name, body)
                }
                FloodApiClient.service.batchUpload(
                    images = parts,
                    cameraId = cameraId?.toRequestBody("text/plain".toMediaTypeOrNull()),
                    locationId = locationId?.toRequestBody("text/plain".toMediaTypeOrNull()),
                    locationName = null,
                    latitude = null,
                    longitude = null,
                    context = "android-batch-upload".toRequestBody("text/plain".toMediaTypeOrNull()),
                )
            }.onSuccess {
                statusMessage = "Uploaded ${it.count} image(s) via backend (${it.execution_mode})"
                loadPoints()
            }.onFailure {
                statusMessage = "Upload failed: ${it.message}"
            }
        }
    }
}
