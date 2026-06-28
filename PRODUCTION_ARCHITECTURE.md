# PRODUCTION ARCHITECTURE DOCUMENTATION
# Bengaluru Flood Depth Estimator - AWS-Native Modular System

## OVERVIEW

This refactored system is structured into **four completely decoupled components**, each mapping to specialized AWS services:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. CONFIG LAYER (config/config.yaml)                           │
│     └─ Centralized hyperparameter management                    │
│     └─ Environment overrides (dev/staging/prod)                 │
│                                                                   │
│  2. DATA INGESTION & PROCESSING (src/dataset.py)                │
│     └─ PyTorch Custom Dataset with S3 support                   │
│     └─ Augmentation (training only), normalization              │
│     └─ AWS S3 → EC2 data streaming                              │
│                                                                   │
│  3. OPTIMIZED TRAINING (src/train.py)                           │
│     └─ EfficientNet-B0 + transfer learning                      │
│     └─ AdamW + MSELoss + ReduceLROnPlateau                      │
│     └─ Early stopping, checkpointing                            │
│     └─ Runs on AWS EC2 (p3.2xlarge GPU)                         │
│                                                                   │
│  4. HIGH-THROUGHPUT INFERENCE (serve.py)                        │
│     └─ LitServe dynamic batching                                │
│     └─ Completely decoupled from training                       │
│     └─ AWS ECS/Fargate (inference service)                      │
│     └─ AWS Lambda (batch processing)                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## FILE STRUCTURE

```
flood-depth-estimator/
├── config/
│   └── config.yaml                  # [LAYER 1] All hyperparameters
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # [LAYER 2] Data ingestion
│   └── train.py                     # [LAYER 3] Training engine
├── serve.py                         # [LAYER 4] Inference server
├── models/
│   └── best_flood_model.pth         # Output artifact
├── archive/                         # Legacy code (preserved)
└── README.md
```

---

## HOW THE COMPONENTS INTERLINK

### Step 1: Configuration (config/config.yaml)
- **Purpose**: Single source of truth for all parameters
- **Consumed by**:
  - `src/train.py`: Reads batch_size, epochs, optimizer settings, image_size
  - `src/dataset.py`: Reads augmentation, normalization, data paths
  - `serve.py`: Reads inference batching, LitServe config, model path
- **AWS Benefit**: Externalize config → Deploy same code to dev/staging/prod with different YAML

### Step 2: Data Ingestion (src/dataset.py)
- **Purpose**: Load images with augmentation (training) or just preprocessing (val/inference)
- **Inputs**: 
  - Local filesystem OR AWS S3 bucket
  - Image paths + depth labels
- **Outputs**: PyTorch DataLoader with batched image tensors
- **Used by**:
  - `src/train.py`: Creates train/val loaders via `create_dataloaders()`
- **AWS Integration**:
  - Local mode (dev): Read from `flood_dataset/train`, `flood_dataset/val`
  - S3 mode (prod): Read from `s3://bengaluru-flood-datasets/train/`, etc.

### Step 3: Training Engine (src/train.py)
- **Purpose**: Train EfficientNet-B0 on labeled data
- **Inputs**: DataLoaders from Layer 2
- **Outputs**: `models/best_flood_model.pth` (state_dict only)
- **Key Features**:
  - AdamW optimizer + MSELoss
  - ReduceLROnPlateau learning rate scheduler
  - Custom EarlyStopping guardrail (stops if no improvement)
  - Model checkpointing every epoch
- **Execution**:
  ```bash
  python src/train.py --config config/config.yaml --output-dir models
  ```
- **AWS Deployment**: Run on EC2 p3.2xlarge (GPU), upload `best_flood_model.pth` to S3

### Step 4: Inference Server (serve.py)
- **Purpose**: Load trained model, handle incoming image requests, batch predictions
- **Completely Decoupled**: Zero training logic—only loads weights and runs forward pass
- **Key Features**:
  - LitServe dynamic batching (accumulate up to 8 images or 0.05s timeout)
  - `decode_request()`: Parse base64 images → PIL Image objects
  - `predict()`: Batch inference on GPU/CPU
  - `encode_response()`: Return JSON results
- **Execution**:
  ```bash
  python serve.py  # Starts LitServe on 0.0.0.0:8000
  ```
- **AWS Deployment**: Container in ECS/Fargate OR Lambda function

---

## EXECUTION GUIDE FOR 10-MEMBER OPERATIONS TEAM

### LOCAL DEVELOPMENT SETUP (Laptop/Workstation)

#### Prerequisites
```bash
# Install dependencies
pip install torch torchvision pyyaml boto3 pillow tqdm litserve

# Clone repo
git clone https://github.com/Mishra-Kaumod/flood-depth-estimator.git
cd flood-depth-estimator
git checkout kaumod-configure-git-lfs
```

#### 1. Prepare Dataset (Local)
```bash
# Create dataset structure
mkdir -p flood_dataset/{train,val,test}

# Add training images to:
# - flood_dataset/train/depth{N}cm_*.jpg  (N = estimated depth)
# - flood_dataset/val/depth{N}cm_*.jpg
```

#### 2. Configure Parameters
Edit `config/config.yaml`:
```yaml
# For development
environments:
  development:
    batch_size: 8
    epochs: 2
    data.use_local: true
```

#### 3. Train Model (Local)
```bash
python src/train.py \
  --config config/config.yaml \
  --output-dir models \
  --use-s3 false

# Expected output:
# ✅ Training complete! Best model: models/best_flood_model.pth
```

#### 4. Run Inference Server (Local)
```bash
python serve.py

# Server starts at http://localhost:8000
# Ready to accept requests
```

#### 5. Test Inference (New Terminal)
```python
import requests
import base64

# Read test image
with open("test_image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "images": [{
            "id": "test_001",
            "data": img_data,
            "format": "jpg"
        }]
    }
)

print(response.json())
# Output:
# {
#   "status": "success",
#   "results": [{
#       "image_id": "test_001",
#       "prediction": {
#           "depth_cm": 35.5,
#           "confidence": 0.892,
#           "intensity": "HIGH",
#           "is_flooded": true
#       }
#   }],
#   "summary": {"total_images": 1, "successful": 1, ...}
# }
```

---

## AWS INFRASTRUCTURE DEPLOYMENT GUIDE

### Phase 1: Training Pipeline (EC2)

#### Setup EC2 Instance
```bash
# Launch p3.2xlarge instance (1x NVIDIA V100 GPU)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --security-groups flood-ml-sg \
  --region ap-south-1

# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Install CUDA + dependencies
sudo apt update && sudo apt install -y python3-pip nvidia-utils
pip install torch torchvision pyyaml boto3 pillow tqdm

# Clone repo
git clone https://github.com/Mishra-Kaumod/flood-depth-estimator.git
cd flood-depth-estimator
```

#### Upload Dataset to S3
```bash
# Create S3 bucket
aws s3 mb s3://bengaluru-flood-datasets --region ap-south-1

# Upload training data
aws s3 sync flood_dataset/train/ s3://bengaluru-flood-datasets/train/ --region ap-south-1
aws s3 sync flood_dataset/val/ s3://bengaluru-flood-datasets/val/ --region ap-south-1
```

#### Run Training on EC2
```bash
# On EC2 instance
python src/train.py \
  --config config/config.yaml \
  --output-dir models \
  --use-s3 true \
  --s3-bucket bengaluru-flood-datasets \
  --s3-region ap-south-1

# After training completes, upload model to S3
aws s3 cp models/best_flood_model.pth s3://bengaluru-flood-models/production/
```

### Phase 2: Inference Deployment (ECS/Fargate)

#### Build Docker Image
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

COPY config/ config/
COPY src/ src/
COPY serve.py .

RUN pip install pyyaml boto3 litserve

EXPOSE 8000

CMD ["python", "serve.py"]
```

#### Build & Push to ECR
```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name bengaluru-flood-inference \
  --region ap-south-1

# Build image
docker build -t bengaluru-flood-inference:latest .

# Tag for ECR
docker tag bengaluru-flood-inference:latest \
  <account-id>.dkr.ecr.ap-south-1.amazonaws.com/bengaluru-flood-inference:latest

# Push to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-south-1.amazonaws.com
docker push <account-id>.dkr.ecr.ap-south-1.amazonaws.com/bengaluru-flood-inference:latest
```

#### Create ECS Cluster & Service
```bash
# Create cluster
aws ecs create-cluster --cluster-name flood-inference-cluster --region ap-south-1

# Create task definition (see next section)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster flood-inference-cluster \
  --service-name flood-depth-predictor \
  --task-definition flood-inference-task:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"
```

#### Task Definition JSON
```json
{
  "family": "flood-inference-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "flood-inference",
      "image": "<account-id>.dkr.ecr.ap-south-1.amazonaws.com/bengaluru-flood-inference:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "s3://bengaluru-flood-models/production/best_flood_model.pth"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/aws/flood-depth-estimator",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Phase 3: Lambda Batch Processor (Optional)

```python
# lambda_handler.py
import json
import boto3
import base64
import requests

s3 = boto3.client("s3")

def lambda_handler(event, context):
    """
    Triggered by S3 upload → Batch process images via ECS inference endpoint
    """
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    
    # Download image from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    image_data = base64.b64encode(obj["Body"].read()).decode()
    
    # Call ECS inference endpoint
    response = requests.post(
        "http://<ecs-alb-dns>:8000/predict",
        json={
            "images": [{
                "id": key,
                "data": image_data,
                "format": "jpg"
            }]
        },
        timeout=30
    )
    
    # Save results to S3
    result = response.json()
    s3.put_object(
        Bucket="bengaluru-flood-results",
        Key=f"predictions/{key.replace('.jpg', '.json')}",
        Body=json.dumps(result)
    )
    
    return {"statusCode": 200, "body": "Processed successfully"}
```

---

## INFRASTRUCTURE MAPPING

```
┌──────────────────────────────────────────────────────────────────────┐
│                       AWS PRODUCTION LAYOUT                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  S3 BUCKETS                                                          │
│  ├─ bengaluru-flood-datasets        (Training data)                  │
│  │  ├─ train/                                                        │
│  │  ├─ val/                                                          │
│  │  └─ test/                                                         │
│  ├─ bengaluru-flood-models          (Model artifacts)                │
│  │  └─ production/best_flood_model.pth                               │
│  └─ bengaluru-flood-results         (Inference results)              │
│                                                                       │
│  EC2 (TRAINING)                                                      │
│  └─ p3.2xlarge Instance                                              │
│     ├─ Python 3.11 + PyTorch 2.0 + CUDA 11.8                         │
│     ├─ Pulls data from S3 (src/dataset.py)                           │
│     ├─ Runs training (src/train.py)                                  │
│     └─ Uploads best_flood_model.pth to S3                            │
│                                                                       │
│  ECS on FARGATE (INFERENCE)                                          │
│  └─ Cluster: flood-inference-cluster                                 │
│     └─ Service: flood-depth-predictor                                │
│        ├─ Task Definition: flood-inference-task                      │
│        ├─ Container Image: benjaminfieldengineering/bengaluru-flood  │
│        ├─ CPU: 2048, Memory: 4096 MB                                 │
│        ├─ Desired Count: 2 (auto-scaling up to 4)                    │
│        └─ Exposed: Port 8000 via ALB                                 │
│                                                                       │
│  LAMBDA (BATCH PROCESSING)                                           │
│  └─ Function: flood-depth-batch-processor                            │
│     ├─ Trigger: S3 upload to bengaluru-flood-datasets/incoming/      │
│     ├─ Memory: 3008 MB, Timeout: 900s                                │
│     ├─ Invokes ECS inference endpoint                                │
│     └─ Writes results to S3 bengaluru-flood-results/                 │
│                                                                       │
│  CLOUDWATCH                                                          │
│  └─ Log Group: /aws/flood-depth-estimator                            │
│     ├─ Training logs (EC2)                                           │
│     ├─ Inference logs (ECS)                                          │
│     └─ Batch processing logs (Lambda)                                │
│                                                                       │
│  NETWORK                                                             │
│  └─ VPC: flood-vpc                                                   │
│     ├─ Security Group: flood-ml-sg (EC2, ECS, Lambda)                │
│     └─ Load Balancer: flood-alb (routes to ECS Fargate)              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## OPERATIONAL WORKFLOWS

### Workflow 1: Model Retraining (Monthly)
1. Data team uploads new labeled images to S3
2. Launch EC2 p3.2xlarge instance
3. Run `src/train.py --use-s3 true`
4. Monitor training in CloudWatch
5. Validate metrics
6. Upload `best_flood_model.pth` to S3
7. Trigger rolling update of ECS service (new model deployed)

### Workflow 2: Daily Inference
1. Operations team uploads field images to S3
2. Lambda auto-triggers, calls ECS endpoint
3. Predictions saved to S3
4. BBMP Command Center queries results via API
5. Alerts generated for CRITICAL/HIGH severity zones

### Workflow 3: A/B Testing
1. Train two model versions
2. Deploy Version A (80%) and Version B (20%) in ECS
3. Monitor metrics separately in CloudWatch
4. If V2 wins, promote to 100%

---

## MONITORING & ALERTS

### CloudWatch Metrics
```
- Training/EC2:
  - train_loss (per epoch)
  - val_loss (per epoch)
  - learning_rate (scheduler step)
  
- Inference/ECS:
  - inference_latency_p50, p95, p99
  - batch_size (actual vs max)
  - throughput (images/sec)
  - error_rate
  
- Lambda:
  - invocation_count
  - error_rate
  - duration (median, max)
```

### Alerts
```
- Alert if val_loss plateaus for 5 epochs
- Alert if inference latency p95 > 500ms
- Alert if ECS service CPU > 80% for 5 min
- Alert if Lambda error rate > 5%
```

---

## COST ESTIMATES (USD/month, ap-south-1)

| Component | Type | Cost |
|-----------|------|------|
| EC2 p3.2xlarge | Training (on-demand, 720 hrs) | $1,200 |
| S3 Storage | Datasets + Models (500GB) | $12 |
| ECS Fargate | 2 tasks × 2048 CPU, 4GB RAM | $180 |
| Lambda | 10M invocations @ 3008 MB | $50 |
| CloudWatch | Logs (100 GB/month) | $50 |
| **TOTAL** | | **~$1,500/month** |

---

## KEY ARCHITECTURAL BENEFITS

✅ **Complete Modularity**: Each file can be developed/tested independently
✅ **Config-Driven**: Deploy to dev/staging/prod with single YAML change
✅ **Scalable**: Training on EC2, inference on Fargate with auto-scaling
✅ **Cost-Efficient**: Train infrequently, scale inference as needed
✅ **Observable**: CloudWatch integration from day one
✅ **Reproducible**: Version control on config + model weights
✅ **Team-Friendly**: Clear separation of concerns, documented interfaces

---

## LEGACY CODE ARCHIVAL

All original files moved to `archive/` folder (preserved for reference):
- `retrain_flood_classifier.py`
- `app/predict_image_v2.py`
- `flood_api/` (old Django endpoints)
- etc.

New production code starts fresh with Layer 1-4 architecture.
