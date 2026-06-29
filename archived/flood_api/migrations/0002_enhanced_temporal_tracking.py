# flood_api/migrations/0002_enhanced_temporal_tracking.py
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('flood_api', '0001_initial'),
    ]

    operations = [
        # Create CameraLocation model
        migrations.CreateModel(
            name='CameraLocation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('camera_id', models.CharField(db_index=True, max_length=50, unique=True)),
                ('location_name', models.CharField(max_length=255)),
                ('latitude', models.FloatField(blank=True, null=True)),
                ('longitude', models.FloatField(blank=True, null=True)),
                ('description', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ['camera_id'],
            },
        ),
        
        # Add fields to FloodInundationTelemetry
        migrations.AddField(
            model_name='floodinundationtelemetry',
            name='camera',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, to='flood_api.cameralocation'),
        ),
        migrations.AddField(
            model_name='floodinundationtelemetry',
            name='detected_reference_objects',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='floodinundationtelemetry',
            name='num_reference_objects',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='floodinundationtelemetry',
            name='is_water_confirmed',
            field=models.BooleanField(default=False),
        ),
        
        # Add indexes
        migrations.AddIndex(
            model_name='floodinundationtelemetry',
            index=models.Index(fields=['camera', '-timestamp'], name='flood_api_f_camera_idx'),
        ),
        migrations.AddIndex(
            model_name='floodinundationtelemetry',
            index=models.Index(fields=['is_water_confirmed', '-timestamp'], name='flood_api_f_water_idx'),
        ),
        
        # Create TemporalFloodSequence model
        migrations.CreateModel(
            name='TemporalFloodSequence',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sequence_start', models.DateTimeField(db_index=True)),
                ('sequence_end', models.DateTimeField()),
                ('image_count', models.IntegerField(default=0)),
                ('average_depth_cm', models.FloatField(blank=True, null=True)),
                ('max_depth_cm', models.FloatField(blank=True, null=True)),
                ('min_depth_cm', models.FloatField(blank=True, null=True)),
                ('water_detected_in_images', models.IntegerField(default=0)),
                ('detected_anchor_types', models.JSONField(blank=True, default=list)),
                ('consensus_water_present', models.BooleanField(default=False)),
                ('confidence_score', models.FloatField(default=0.0)),
                ('camera', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='flood_api.cameralocation')),
                ('telemetry_records', models.ManyToManyField(to='flood_api.floodinundationtelemetry')),
            ],
            options={
                'verbose_name_plural': 'Temporal Flood Sequences',
                'ordering': ['-sequence_start'],
            },
        ),
        
        # Add indexes to TemporalFloodSequence
        migrations.AddIndex(
            model_name='temporalfloodsequence',
            index=models.Index(fields=['camera', '-sequence_start'], name='flood_api_t_camera_idx'),
        ),
    ]
