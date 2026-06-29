import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from collections import deque

# Load Model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Buffer
decision_buffer = deque(maxlen=5)

def get_road_mask(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    return cv2.resize((mask == 15).astype(np.uint8) * 255, (frame.shape[1], frame.shape[0]))

def cv2_ensemble_estimator(current_frame, dry_ref_frame):
    global decision_buffer
    road_mask = get_road_mask(current_frame)
    masked_curr = cv2.bitwise_and(current_frame, current_frame, mask=road_mask)
    masked_ref = cv2.bitwise_and(dry_ref_frame, dry_ref_frame, mask=road_mask)
    
    gray_curr = cv2.cvtColor(masked_curr, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(masked_ref, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_ref, gray_curr)
    
    edges = cv2.Canny(gray_curr, 100, 200)
    score_smooth = 1.0 - (cv2.countNonZero(edges) / max(1, cv2.countNonZero(road_mask)))
    
    _, motion_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    score_motion = cv2.countNonZero(motion_mask) / max(1, cv2.countNonZero(road_mask))
    
    total_confidence = (score_smooth * 0.3) + (score_motion * 0.7)
    
    is_flood = total_confidence > 0.15
    decision_buffer.append(is_flood)
    
    if sum(decision_buffer) >= 4:
        return 5.0, "FLOODED"
    return 0.0, "DRY"
