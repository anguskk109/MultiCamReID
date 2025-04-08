import torch
import cv2
import numpy as np
import torchreid

def build_pcb_model():
    model = torchreid.models.build_model(
        name='pcb_p6',  # 6-part PCB
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    return model

def extract_pcb_features(model, frame):
    img = cv2.resize(frame, (128, 384))  # PCB default input size
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    # exclude top stripe(head, facial data)
    # double weight for lower stripes, which likely will not be affected by putting on gowns
    weights = np.array([0, 1, 1, 1, 2, 2])
    weights = weights / weights.sum()

    features = features.squeeze(0).cpu().numpy()
    # Weighted sum of features
    weighted_feat = np.sum(features * weights[:, None], axis=0)

    # Normalize
    feat_np = weighted_feat.astype(np.float32)
    # feat_np = features.cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(feat_np)
    if norm > 0:
        feat_np = feat_np / norm

    return feat_np

