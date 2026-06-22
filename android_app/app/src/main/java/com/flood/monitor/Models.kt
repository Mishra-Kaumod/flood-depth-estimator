package com.flood.monitor

data class MapPoint(
    val record_id: Int,
    val camera_id: String,
    val location_id: String?,
    val location_name: String,
    val latitude: Double,
    val longitude: Double,
    val depth_cm: Double,
    val intensity_scale: Int,
    val image_name: String,
    val timestamp: String,
    val safety_risk_assessment: String,
)

data class MapPointsResponse(
    val status: String,
    val count: Int,
    val points: List<MapPoint>,
)

data class BatchIngestItem(
    val file_name: String,
    val camera_id: String,
    val location_id: String?,
    val location_name: String,
    val latitude: Double,
    val longitude: Double,
    val execution_mode: String,
)

data class BatchIngestResponse(
    val status: String,
    val count: Int,
    val execution_mode: String,
    val items: List<BatchIngestItem>,
)
