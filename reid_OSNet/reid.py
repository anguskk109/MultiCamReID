import torch
import cv2
import numpy as np
import torchreid

'''
Available OSNet variants:

osnet_x1_0 – best accuracy

osnet_x0_75 – balance

osnet_x0_5 – lightweight

osnet_x0_25 – ultra-light for edge
'''

def build_osnet_model():
    model = torchreid.models.build_model(
        name='osnet_x1_0',  # or 'osnet_x0_75' for lighter version
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    return model


def extract_osnet_features(model, frame):
    img = cv2.resize(frame, (256, 128))  # OSNet uses (128w x 256h)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    feat_np = features.cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(feat_np)
    if norm > 0:
        feat_np = feat_np / norm

    return feat_np
