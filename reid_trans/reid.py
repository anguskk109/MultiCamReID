import torch
import cv2
import numpy as np
from yacs.config import CfgNode as CN

# Load TransReID configuration
def load_transreid_config(config_path, pretrained_path):
    from TransReID.config.defaults import _C as cfg
    cfg = cfg.clone()
    cfg.merge_from_file(config_path)
    cfg.MODEL.PRETRAIN_PATH = pretrained_path
    cfg.freeze()
    return cfg

# Build TransReID model from config
def build_transreid_model(cfg):
    from TransReID.model.make_model import make_model
    model = make_model(cfg, num_class=1000, camera_num=1, view_num=1)
    model.load_param(cfg.MODEL.PRETRAIN_PATH)
    model.eval()
    return model

# Feature extractor using TransReID model
def extract_transreid_features(model, cfg, frame):
    img = cv2.resize(frame, (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)  # Shape: (1, 3, H, W)

    with torch.no_grad():
        feat = model(img)

    feat_np = feat.cpu().numpy().flatten().astype(np.float32)

    #Normalize feature vector
    norm = np.linalg.norm(feat_np)
    if norm > 0:
        feat_np = feat_np / norm

    return feat_np
