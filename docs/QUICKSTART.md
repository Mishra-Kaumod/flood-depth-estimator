# QUICK START GUIDE - PRODUCTION ARCHITECTURE

## 5-Minute Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements-production.txt
```

### 2. Prepare Data
```bash
# Create directory structure
mkdir -p flood_dataset/{train,val,test}

# Copy images
# Format: depth{N}cm_*.jpg where N = depth in cm
# Example:
# - flood_dataset/train/depth15cm_image_001.jpg
# - flood_dataset/train/depth35cm_image_002.jpg
```

### 3. Review Configuration
```bash
# Open and review
cat config/config.yaml

# Key sections:
# - training.batch_size = 32
# - training.epochs = 20
# - training.image_size = [224, 224]
# - inference.litserve.port = 8000
```

### 4. Train Model (2-5 minutes on CPU, 30 seconds on GPU)
```bash
python src/train.py --config config/config.yaml --output-dir models

# Progress:
# Creating dataloaders...
# Building efficientnet_b0...
# Starting training for 20 epochs
# EPOCH 1/20
# Training 100%|████| ... | loss: 0.0234
# Validating 100%|████| ... | loss: 0.0156
# ✅ Best model saved to models/best_flood_model.pth
```

### 5. Start Inference Server
```bash
python serve.py

# Output:
# 🚀 Initializing FloodDepthPredictor on cpu
# ✅ Loaded model weights from models/best_flood_model.pth
# ✅ FloodDepthPredictor ready for inference
# [2025-06-28 14:52:30] Starting LitServe server on 0.0.0.0:8000
# Ready to accept requests at http://localhost:8000/predict
```

### 6. Test Inference (New Terminal)
```bash
python -c "
import requests
import base64
from PIL import Image
import io

# Create a test image
img = Image.new('RGB', (224, 224), color='blue')
buffer = io.BytesIO()
img.save(buffer, format='JPG')
img_data = base64.b64encode(buffer.getvalue()).decode()

# Send request
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'images': [{
            'id': 'test_001',
            'data': img_data,
            'format': 'jpg'
        }]
    }
)

import json
print(json.dumps(response.json(), indent=2))
"

# Expected output:
# {
#   "status": "success",
#   "timestamp": "2025-06-28T14:52:45.123456",
#   "results": [
#     {
#       "image_id": "test_001",
#       "prediction": {
#         "depth_cm": 18.5,
#         "confidence": 0.756,
#         "intensity": "MEDIUM",
#         "is_flooded": true
#       },
#       "status": "success"
#     }
#   ],
#   "summary": {
#     "total_images": 1,
#     "successful": 1,
#     "failed": 0,
#     "avg_depth_cm": 18.5
#   }
# }
```

---

## TROUBLESHOOTING

### Issue: "Config file not found"
**Solution**: Ensure you're in the project root:
```bash
pwd  # Should show: .../flood-depth-estimator
ls config/config.yaml  # Should exist
```

### Issue: "No module named 'src'"
**Solution**: Add project to Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python src/train.py --config config/config.yaml
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config.yaml:
```yaml
training:
  batch_size: 8  # Reduce from 32
```

### Issue: "Model not found for inference"
**Solution**: Train first:
```bash
python src/train.py --config config/config.yaml --output-dir models
# Then run: python serve.py
```

---

## AWS DEPLOYMENT CHECKLIST

- [ ] AWS CLI configured: `aws configure`
- [ ] IAM role with S3, ECR, ECS, EC2 permissions
- [ ] S3 buckets created: `bengaluru-flood-datasets`, `bengaluru-flood-models`
- [ ] EC2 security group with port 22 (SSH) and 8000 (inference)
- [ ] VPC and subnet configured
- [ ] ECR repository created: `bengaluru-flood-inference`
- [ ] ECS cluster created: `flood-inference-cluster`
- [ ] Application Load Balancer (ALB) configured

---

## ENVIRONMENT VARIABLES (Optional)

```bash
# For AWS S3 access
export AWS_REGION=ap-south-1
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>

# For LitServe optimization
export LITSERVE_WORKERS=4
export TORCH_NUM_THREADS=4

# For training on specific GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## MONITORING & LOGS

### Local Logs
```bash
# Training logs
tail -f logs/training.log

# Inference logs
tail -f logs/inference.log
```

### AWS CloudWatch (Production)
```bash
# View training logs
aws logs tail /aws/flood-depth-estimator/training --follow

# View inference logs
aws logs tail /aws/flood-depth-estimator/ecs --follow

# View Lambda logs
aws logs tail /aws/lambda/flood-depth-batch-processor --follow
```

---

## NEXT STEPS

1. **Read Full Architecture**: See `PRODUCTION_ARCHITECTURE.md`
2. **Deploy to AWS**: Follow AWS Deployment Guide in architecture doc
3. **Set up CI/CD**: Use GitHub Actions to auto-train on new data
4. **Monitor Performance**: CloudWatch dashboards for metrics
5. **Scale**: Add more ECS tasks, Lambda concurrency as needed

---

## SUPPORT

For issues or questions:
1. Check logs: `tail -f logs/*.log`
2. Review config: `cat config/config.yaml`
3. Test components individually:
   ```bash
   # Test dataset loading
   python -c "from src.dataset import load_config, create_dataloaders; cfg = load_config(); train, val = create_dataloaders(cfg)"
   
   # Test model building
   python -c "from src.train import build_model; import torch; model = build_model({}, torch.device('cpu')); print(model)"
   ```

---

## VERSION INFO

- Python: 3.8+
- PyTorch: 2.0+
- LitServe: 0.1+
- Config Format: YAML
- Model: EfficientNet-B0
- Loss: MSELoss
- Optimizer: AdamW
