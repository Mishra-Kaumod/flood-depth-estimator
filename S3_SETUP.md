# AWS S3 Integration Guide

## Overview
The flood detection system now supports AWS S3 for reading input images/videos and writing results. This enables cloud-based processing of large datasets.

## Storage Modes

### Local Mode (DEFAULT)
```bash
python main.py image test_images/flood_image.jpg
python main.py video test_videos/flood_video.mp4
python main.py object test_images/flood_image.jpg
```

### AWS S3 Mode
```bash
python main.py image images/flood_image.jpg --storage=aws
python main.py video videos/flood_video.mp4 results.csv 2 --storage=aws
python main.py object images/flood_image.jpg objects_output.jpg --storage=aws
```

## Setup Instructions

### 1. Install boto3
```bash
pip install boto3
```

### 2. Configure AWS Credentials

#### Option A: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
export S3_BUCKET=your-bucket-name
```

#### Option B: AWS Credentials File (~/.aws/credentials)
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

#### Option C: .env File
1. Copy `.env.example` to `.env`
2. Fill in your AWS credentials:
```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
```
3. Load the .env file before running:
```bash
# On Linux/Mac
source .env

# On Windows PowerShell
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
$env:S3_BUCKET="your-bucket-name"
```

### 3. Create S3 Bucket
```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

### 4. Set Bucket Structure (Optional)
```bash
aws s3 cp images/ s3://your-bucket-name/images/ --recursive
aws s3 cp videos/ s3://your-bucket-name/videos/ --recursive
```

## Usage Examples

### Single Image Analysis
```bash
# Local
python main.py image test_images/flood_1.jpg

# S3
python main.py image flood_images/flood_1.jpg --storage=aws
```

### Video Processing
```bash
# Local (saves CSV locally)
python main.py video test_videos/flood.mp4

# S3 (downloads video, uploads results CSV)
python main.py video flood_videos/flood.mp4 results.csv 2 --storage=aws
```

### Object Detection
```bash
# Local
python main.py object test_images/flood_1.jpg output.jpg

# S3 (reads image from S3, uploads annotated image to S3)
python main.py object flood_images/flood_1.jpg annotated_output.jpg --storage=aws
```

## S3 Path Structure

### Recommended Organization
```
s3://your-bucket/
├── images/
│   ├── flood_1.jpg
│   ├── flood_2.jpg
│   └── ...
├── videos/
│   ├── flood_1.mp4
│   ├── flood_2.mp4
│   └── ...
├── results/
│   ├── video_analysis_1.csv
│   ├── video_analysis_2.csv
│   └── ...
└── annotated/
    ├── objects_detected_1.jpg
    ├── objects_detected_2.jpg
    └── ...
```

## Example Workflow

### 1. Prepare Data in S3
```bash
# Upload images
aws s3 cp ./local_images s3://my-bucket/images/ --recursive

# Upload videos
aws s3 cp ./local_videos s3://my-bucket/videos/ --recursive
```

### 2. Process with AWS Mode
```bash
# Analyze single image
python main.py image images/flood_scene.jpg --storage=aws

# Process video
python main.py video videos/flood_recording.mp4 flood_results.csv 1 --storage=aws

# Detect objects
python main.py object images/flood_scene.jpg flood_annotated.jpg --storage=aws
```

### 3. Download Results
```bash
# Download specific results
aws s3 cp s3://my-bucket/flood_results.csv . 

# Download all results
aws s3 cp s3://my-bucket/results/ ./results/ --recursive
```

## S3Handler API

### Reading Operations

```python
from modules.s3_handler import S3Handler

handler = S3Handler(bucket_name='my-bucket')

# Read image
image = handler.read_image_from_s3('images/flood_1.jpg')

# Read CSV
csv_content = handler.read_csv_from_s3('results/analysis.csv')

# List images
images = handler.list_images_in_s3(prefix='images/')
```

### Writing Operations

```python
# Write image
handler.write_image_to_s3(image, 'results/annotated_flood_1.jpg')

# Write CSV
handler.write_csv_to_s3(df, 'results/video_analysis.csv')
```

## Cost Optimization

### Tips for Reducing S3 Costs

1. **Use S3 Transfer Acceleration** (for large files)
   ```python
   s3_handler = S3Handler()
   # Speeds up uploads/downloads for large objects
   ```

2. **Organize by Region** - Keep bucket in same region as compute
   ```bash
   # Ireland region
   export AWS_REGION=eu-west-1
   ```

3. **Set Lifecycle Policies** - Auto-delete old results
   ```bash
   aws s3api put-bucket-lifecycle-configuration \
     --bucket my-bucket \
     --lifecycle-configuration file://lifecycle.json
   ```

4. **Use S3 Storage Classes** - Store old results in Glacier
   ```json
   {
     "Rules": [{
       "Filter": {"Prefix": "results/"},
       "NoncurrentVersionTransition": {
         "Days": 30,
         "StorageClass": "GLACIER"
       }
     }]
   }
   ```

## Troubleshooting

### Error: "InvalidAccessKeyId"
- Check AWS credentials are correct
- Verify credentials haven't expired

### Error: "NoSuchBucket"
- Verify bucket name is correct
- Check you have access to the bucket
- Create bucket if it doesn't exist

### Error: "Access Denied"
- Verify IAM user has S3 permissions:
  - s3:GetObject
  - s3:PutObject
  - s3:ListBucket

### Slow Download/Upload
- Check network connection
- Use S3 Transfer Acceleration
- Consider multipart uploads for videos

## IAM Policy Example

For security, create an IAM user with minimal permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::flood-analysis",
        "arn:aws:s3:::flood-analysis/*"
      ]
    }
  ]
}
```

## Performance

### Expected Times (on m5.large EC2 instance)

| Operation | Time | Notes |
|-----------|------|-------|
| Single image (local) | 2-3s | ResNet18 inference |
| Single image (S3) | 4-5s | +1-2s for S3 I/O |
| 1 min video (local) | 2-3min | ~25 fps processing |
| 1 min video (S3) | 3-4min | +download/upload time |

### Network Bandwidth

- Small image (100KB): ~100ms download
- Large video (100MB): ~10-20s download (depending on network)
- CSV result (50KB): ~50ms upload

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| AWS_ACCESS_KEY_ID | AWS access key | (required for --storage=aws) |
| AWS_SECRET_ACCESS_KEY | AWS secret key | (required for --storage=aws) |
| S3_BUCKET | S3 bucket name | flood-analysis |
| AWS_REGION | AWS region | us-east-1 |

## Next Steps

1. Set up AWS credentials
2. Create S3 bucket
3. Upload sample images
4. Test with: `python main.py image images/test.jpg --storage=aws`
5. Monitor AWS CloudWatch for usage
6. Set up cost alerts if needed
