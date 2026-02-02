visual_scattering.py 
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import json
from pytorch_wavelets import ScatLayer

# -----------------------------
# Utilities
# -----------------------------
def load_hdf5_images(path):
    with h5py.File(path, 'r') as f:
        return {k: f[k][()] for k in f.keys() if k.startswith("colors")}

def to_grayscale(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    if img.ndim == 3:
        img = 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]
    return img

def build_scattering(device="cuda"):
    scat = torch.nn.Sequential(ScatLayer(), ScatLayer()).to(device).eval()
    return scat

def compute_scattering(img, scat):
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    x = x.to(next(scat.parameters()).device)
    with torch.no_grad():
        return scat(x)

def scattering_energy(S):
    return S.pow(2).sum(dim=1, keepdim=False)

def max_orientation_response(S):
    return S.abs().max(dim=1)[0]

def nms_2d(score, radius=3):
    pooled = F.max_pool2d(score.unsqueeze(0).unsqueeze(0),
                          kernel_size=2*radius+1, stride=1, padding=radius)
    return score == pooled.squeeze()

def select_sparse_keypoints(S, K=300):
    energy = scattering_energy(S)[0]
    orient = max_orientation_response(S)[0]
    score = energy * orient
    keep = nms_2d(score, radius=3)
    ys, xs = torch.where(keep)
    vals = score[ys, xs]
    if len(vals) == 0:
        return np.empty((0,2), dtype=int)
    topk = torch.topk(vals, min(K, len(vals))).indices
    return torch.stack([xs[topk], ys[topk]], dim=1).cpu().numpy()

def extract_sparse_descriptors(S, keypoints, patch_size=8):
    C, H, W = S.shape[1], S.shape[2], S.shape[3]
    half = patch_size // 2
    descs = []
    for x, y in keypoints:
        x0, x1 = x - half, x + half
        y0, y1 = y - half, y + half
        patch = torch.zeros((C, patch_size, patch_size),
                            device=S.device, dtype=S.dtype)
        px0, py0 = max(0, x0), max(0, y0)
        px1, py1 = min(W, x1), min(H, y1)
        patch_x0, patch_y0 = px0 - x0, py0 - y0
        patch[:, patch_y0:patch_y0+(py1-py0), patch_x0:patch_x0+(px1-px0)] = S[0,:,py0:py1,px0:px1]
        descs.append(F.normalize(patch.flatten(), dim=0))
    return torch.stack(descs) if descs else torch.empty((0,), device=S.device)

def scattering_to_image_coords(kps, order=2):
    return kps * (2 ** order)

def match_descriptors(descL, descR):
    sim = torch.mm(descL, descR.t())
    idx_L2R = sim.argmax(dim=1)
    idx_R2L = sim.argmax(dim=0)
    return [(i, j) for i, j in enumerate(idx_L2R) if idx_R2L[j] == i]

def triangulate_points(matches, kpsL, kpsR, K, R0, t0, R1, t1):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    C0, C1 = -R0.T @ t0, -R1.T @ t1
    points = []
    for i, j in matches:
        uL, vL, uR, vR = kpsL[i,0], kpsL[i,1], kpsR[j,0], kpsR[j,1]
        xL = np.array([(uL-cx)/fx, (vL-cy)/fy, 1.0])
        xR = np.array([(uR-cx)/fx, (vR-cy)/fy, 1.0])
        v0, v1 = R0.T @ xL, R1.T @ xR
        v0 /= np.linalg.norm(v0); v1 /= np.linalg.norm(v1)
        w = C0 - C1
        a,b,c = np.dot(v0,v0), np.dot(v0,v1), np.dot(v1,v1)
        d,e = np.dot(v0,w), np.dot(v1,w)
        denom = a*c - b*b
        if abs(denom) < 1e-8: continue
        s0, s1 = (b*e - c*d)/denom, (a*e - b*d)/denom
        points.append(0.5*((C0 + s0*v0) + (C1 + s1*v1)))
    return np.array(points)

# -----------------------------
# Main pipeline
# -----------------------------
images = load_hdf5_images("/home/lang/RenderImages/test/low_light/100/Coast_Guard/frame/0.hdf5")
grayL, grayR = to_grayscale(images["colors_0"]), to_grayscale(images["colors_1"])

scat = build_scattering()
SL, SR = compute_scattering(grayL, scat), compute_scattering(grayR, scat)

kpsL_scat, kpsR_scat = select_sparse_keypoints(SL, K=300), select_sparse_keypoints(SR, K=300)
descL, descR = extract_sparse_descriptors(SL, kpsL_scat), extract_sparse_descriptors(SR, kpsR_scat)
matches = match_descriptors(descL, descR)

kpsL, kpsR = scattering_to_image_coords(kpsL_scat), scattering_to_image_coords(kpsR_scat)

with open("/home/lang/RenderImages/test/low_light/100/Coast_Guard/camera_intrinsics.json") as f:
    K = np.array(json.load(f)["K"])
with open("/home/lang/RenderImages/test/low_light/100/Coast_Guard/camera_poses.json") as f:
    poses = json.load(f)

# Convert Blender camera-to-world → world-to-camera
T0, T1 = np.array(poses["left"]["T"]), np.array(poses["right"]["T"])
R0_bw, t0_bw = T0[:3,:3], T0[:3,3]
R1_bw, t1_bw = T1[:3,:3], T1[:3,3]
R0, t0 = R0_bw.T, -R0_bw.T @ t0_bw
R1, t1 = R1_bw.T, -R1_bw.T @ t1_bw

points3D = triangulate_points(matches, kpsL, kpsR, K, R0, t0, R1, t1)
valid_points3D = points3D[points3D[:,2] > 0]

print("Sparse 3D points:", valid_points3D.shape[0])
depths = valid_points3D[:,2]
print(f"Depth range: {depths.min():.2f} to {depths.max():.2f}, median {np.median(depths):.2f}")


# overlay original image with upsampled feature map from a random scattering channel to illustrate highlighted regions
import matplotlib.pyplot as plt
import cv2
def overlay_scattering_features(img, S, channel=10):
    C, H, W = S.shape[1], S.shape[2], S.shape[3]
    feat_map = S[0, channel, :, :].cpu().numpy()
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
    feat_map_upsampled = cv2.resize(feat_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    plt.imshow(img, cmap='gray')
    plt.imshow(feat_map_upsampled, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()  
    
overlay_scattering_features(grayL, SL, channel=10)
