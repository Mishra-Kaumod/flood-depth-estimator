package com.flood.monitor

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query

interface FloodApiService {
    @GET("api/v1/telemetry/map-points/")
    suspend fun getMapPoints(@Query("limit") limit: Int = 200): MapPointsResponse

    @Multipart
    @POST("api/v1/ingest/batch/")
    suspend fun batchUpload(
        @Part images: List<MultipartBody.Part>,
        @Part("camera_id") cameraId: RequestBody?,
        @Part("location_id") locationId: RequestBody?,
        @Part("location_name") locationName: RequestBody?,
        @Part("latitude") latitude: RequestBody?,
        @Part("longitude") longitude: RequestBody?,
        @Part("context") context: RequestBody?,
    ): BatchIngestResponse
}

object FloodApiClient {
    // Replace with production backend URL in release flavors.
    private const val BASE_URL = "http://10.0.2.2:8000/"

    val service: FloodApiService by lazy {
        val logger = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }
        val client = OkHttpClient.Builder().addInterceptor(logger).build()
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(MoshiConverterFactory.create())
            .build()
            .create(FloodApiService::class.java)
    }
}
