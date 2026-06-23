# AWS S3 Quick Start

## 1. Install boto3 (AWS SDK for Python)
```bash
pip install boto3
```

## 2. Configure AWS Credentials

### Option A: Environment Variables (Recommended for CI/CD)
```bash
# Linux/Mac
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export S3_BUCKET=your-bucket-name

# Windows PowerShell
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
$env:S3_BUCKET="your-bucket-name"
```

### Option B: .env File (Recommended for Local Development)
```bash
# 1. Create .env from template
cp .env.example .env

# 2. Edit .env with your credentials
# 3. Load it before running
source .env  # Linux/Mac
```

### Option C: AWS Credentials File (~/.aws/credentials)
```ini
[default]
aws_access_key_id = your_key
aws_secret_access_key = your_secret
```

## 3. Create S3 Bucket (if not exists)
```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

## 4. Upload Sample Images to S3
```bash
aws s3 cp ./test_images s3://your-bucket-name/images/ --recursive
```

## 5. Run Flood Detection with S3

### Single Image Analysis
```bash
python main.py image images/flood_sample.jpg --storage=aws
```

### Video Processing
```bash
python main.py video videos/flood_video.mp4 results.csv 1 --storage=aws
```

### Object Detection
```bash
python main.py object images/flood_sample.jpg annotated.jpg --storage=aws
```

## 6. Download Results from S3
```bash
# Download specific file
aws s3 cp s3://your-bucket-name/results.csv .

# Download all results
aws s3 cp s3://your-bucket-name/results/ ./results/ --recursive
```

## Verify Setup
```bash
# Test local mode (should work)
python main.py image test_images/sample.jpg

# Test S3 mode (requires AWS setup)
python main.py image images/sample.jpg --storage=aws
```

## Troubleshooting

### "InvalidAccessKeyId" Error
- Check AWS_ACCESS_KEY_ID is correct
- Verify credentials haven't expired

### "NoSuchBucket" Error
- Create bucket: `aws s3 mb s3://bucket-name`
- Check bucket name matches S3_BUCKET env var

### "Access Denied" Error
- Verify IAM user has S3 permissions
- Required permissions:
  - s3:GetObject
  - s3:PutObject
  - s3:ListBucket

## Cheat Sheet

| Task | Command |
|------|---------|
| List buckets | `aws s3 ls` |
| List bucket contents | `aws s3 ls s3://bucket-name/` |
| Upload file | `aws s3 cp file.jpg s3://bucket-name/` |
| Download file | `aws s3 cp s3://bucket-name/file.jpg .` |
| Sync local to S3 | `aws s3 sync ./local s3://bucket-name/remote` |
| Sync S3 to local | `aws s3 sync s3://bucket-name/remote ./local` |

## For More Details
See [S3_SETUP.md](S3_SETUP.md) for comprehensive documentation.
