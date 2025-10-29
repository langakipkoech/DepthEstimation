# run_stereo_morlet.py
import cv2
import numpy as np
import json
import h5py
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt


class ScatteringTransform:
    """Compute scattering coefficients using Morlet wavelets"""
    
    def __init__(self, patch_size: int = 8, scales: List[int] = [1, 2, 4]):
        self.patch_size = patch_size
        self.scales = scales
        
    def compute_scattering_coefficients(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return np.array([])
        patch = patch.astype(np.float32)
        coefficients = [np.mean(patch), np.std(patch)]

        # THRESHOLD (adjust based on image intensity)
        THRESHOLD = 1.0  # for uint8 images

        # First-order
        for scale in self.scales:
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel_size = min(patch.shape[0], patch.shape[1], 2*scale+1)
                if kernel_size < 3:
                    continue
                kernel = self._create_morlet_kernel(kernel_size, scale, theta)
                filtered = cv2.filter2D(patch, -1, kernel)
                modulus = np.abs(filtered)
                modulus = np.maximum(modulus - THRESHOLD, 0)  # Threshold
                coefficients.append(np.mean(modulus))

        # Second-order
        for scale1 in self.scales[:2]:
            for scale2 in self.scales[:2]:
                if scale2 > scale1:
                    k1 = self._create_morlet_kernel(
                        min(patch.shape[0], patch.shape[1], 2*scale1+1), scale1, 0
                    )
                    f1 = np.abs(cv2.filter2D(patch, -1, k1))
                    f1 = np.maximum(f1 - THRESHOLD, 0)

                    k2 = self._create_morlet_kernel(
                        min(f1.shape[0], f1.shape[1], 2*scale2+1), scale2, np.pi/2
                    )
                    f2 = cv2.filter2D(f1, -1, k2)
                    modulus2 = np.abs(f2)
                    modulus2 = np.maximum(modulus2 - THRESHOLD, 0)
                    coefficients.append(np.mean(modulus2))

        return np.array(coefficients)
    
    def _create_morlet_kernel(self, size: int, scale: float, theta: float) -> np.ndarray:
        """
        Create 2D Morlet wavelet kernel (real part).
        Morlet = Gaussian envelope * cos(plane wave)
        """
        if size < 3:
            size = 3
        size = size if size % 2 == 1 else size + 1  # odd size
        center = size // 2
        
        # Morlet parameters
        sigma = scale * 1.5        # envelope spread
        lambd = scale * 4.0        # wavelength (controls frequency)
        gamma = 0.5                # aspect ratio
        psi = 0.0                  # phase offset
        
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        
        # Rotate coordinates
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Gaussian envelope
        envelope = np.exp(-0.5 * (x_theta**2 / sigma**2 + (y_theta * gamma)**2 / sigma**2))
        
        # Plane wave
        wave = np.cos(2 * np.pi * x_theta / lambd + psi)
        
        # Morlet = envelope * wave
        kernel = envelope * wave
        
        # Normalize by L1 (energy-preserving)
        kernel = kernel / (np.sum(np.abs(kernel)) + 1e-10)
        
        return kernel.astype(np.float32)
    
    def extract_patch(self, img: np.ndarray, point: Tuple[float, float]) -> Optional[np.ndarray]:
        x, y = int(round(point[0])), int(round(point[1]))
        half_size = self.patch_size // 2
        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)
        if (y_max - y_min) < self.patch_size // 2 or (x_max - x_min) < self.patch_size // 2:
            return None
        return img[y_min:y_max, x_min:x_max]
    
    def get_morlet_feature_maps(self, patch: np.ndarray) -> list:
        """Return Morlet-filtered maps for visualization"""
        if patch.size == 0:
            return []
        patch = patch.astype(np.float32)
        feature_maps = []
        for scale in self.scales:
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel_size = min(patch.shape[0], patch.shape[1], 2*scale+1)
                if kernel_size < 3:
                    continue
                kernel = self._create_morlet_kernel(kernel_size, scale, theta)
                filtered = cv2.filter2D(patch, -1, kernel)
                feature_maps.append(filtered)
        return feature_maps


class StereoReconstruction:
    def __init__(self, hdf_file_path: str, cam_config_path: str):
        self.hdf_file_path = hdf_file_path
        self.cam_file_path = cam_config_path
        self.P_left = None
        self.P_right = None
        self.P_left_estimated = None
        self.P_right_estimated = None
        self.left_kpts = None
        self.right_kpts = None
        self.baseline = None
        self.focal_length = None
        self.rendered_data: Dict[str, Any] = {}
        self.matches = None
        self.K_mtx = None
        self.img_shape = None
        self.scattering_transform = ScatteringTransform(patch_size=8, scales=[1, 2, 4])
        
    def load_hdf_file(self):
        with h5py.File(self.hdf_file_path, "r") as hdf_data:
            for key in ["colors_0", "colors_1", "depth_0", "depth_1"]:
                img = hdf_data[key][()]
                if img.ndim >= 3 and img.shape[0] == 1:
                    img = img[0]
                self.rendered_data[key] = img
        self.img_shape = self.rendered_data["colors_0"].shape[:2]
        return self.rendered_data

    def load_projection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        with open(self.cam_file_path, "r") as f:
            cam = json.load(f)
        K = np.array(cam["intrinsics"]["K"]).reshape(3, 3)
        self.K_mtx = K
        self.focal_length = float(K[0, 0])
        T_left = np.array(cam["cameras"]["left"]["T"]).reshape(3)
        T_right = np.array(cam["cameras"]["right"]["T"]).reshape(3)
        self.baseline = float(np.linalg.norm(T_left - T_right))

        def build_projection(R, T):
            R = np.array(R).reshape(3, 3)
            C = np.array(T).reshape(3, 1)
            Rt = np.hstack([R, -R @ C])
            return K @ Rt

        self.P_left = build_projection(cam["cameras"]["left"]["R"], cam["cameras"]["left"]["T"])
        self.P_right = build_projection(cam["cameras"]["right"]["R"], cam["cameras"]["right"]["T"])

        print(f"Baseline: {self.baseline:.6f}")
        print(f"Focal length: {self.focal_length:.2f}")
        return self.P_left, self.P_right
    
    @staticmethod
    def prepare_image_cv(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)
    
    def compute_ncc(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        if patch1.shape != patch2.shape:
            return 0.0
        p1 = patch1.astype(np.float32) - np.mean(patch1)
        p2 = patch2.astype(np.float32) - np.mean(patch2)
        num = np.sum(p1 * p2)
        den = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
        return num / den if den > 1e-10 else 0.0
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if len(vec1) == 0 or len(vec2) == 0 or len(vec1) != len(vec2):
            return 0.0
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot / norm if norm > 1e-10 else 0.0
    
    def detect_and_match_with_scattering(
        self,
        ratio_test: float = 0.7,
        use_ransac: bool = True,
        ransac_thresh: float = 1.0,
        epipolar_thresh: float = 2.0,
        ncc_threshold: float = 0.7,
        scattering_similarity_threshold: float = 0.85
    ):
        print(f"\n{'='*60}")
        print("MORLET SCATTERING MATCHING")
        print(f"{'='*60}\n")
        
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        left_img = self.rendered_data["colors_0"]
        right_img = self.rendered_data["colors_1"]
        left_gray = self.prepare_image_cv(left_img)
        right_gray = self.prepare_image_cv(right_img)
        
        kpL, desL = sift.detectAndCompute(left_gray, None)
        kpR, desR = sift.detectAndCompute(right_gray, None)
        print(f"Keypoints: {len(kpL)} (L), {len(kpR)} (R)")

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(desL, desR, k=2)
        candidate_matches = [m for m, n in matches if len([m, n]) == 2 and m.distance < ratio_test * n.distance]
        print(f"After ratio test: {len(candidate_matches)}")

        refined_matches = []
        scattering_scores = []
        ncc_scores = []

        for idx, m in enumerate(candidate_matches):
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(candidate_matches)}...")
            ptL = kpL[m.queryIdx].pt
            ptR = kpR[m.trainIdx].pt
            if abs(ptL[1] - ptR[1]) >= epipolar_thresh or ptL[0] - ptR[0] <= 0.2:
                continue
            patchL = self.scattering_transform.extract_patch(left_gray, ptL)
            patchR = self.scattering_transform.extract_patch(right_gray, ptR)
            if patchL is None or patchR is None:
                continue
            scatterL = self.scattering_transform.compute_scattering_coefficients(patchL)
            scatterR = self.scattering_transform.compute_scattering_coefficients(patchR)
            if len(scatterL) == 0 or len(scatterR) == 0:
                continue
            scatter_sim = self.cosine_similarity(scatterL, scatterR)
            ncc = self.compute_ncc(patchL, patchR)
            if scatter_sim >= scattering_similarity_threshold and ncc >= ncc_threshold:
                refined_matches.append(m)
                scattering_scores.append(scatter_sim)
                ncc_scores.append(ncc)

        print(f"After Morlet+NCC: {len(refined_matches)}")
        print(f"Mean scatter sim: {np.mean(scattering_scores):.3f}")
        print(f"Mean NCC: {np.mean(ncc_scores):.3f}")

        if use_ransac and len(refined_matches) > 8:
            ptsL = np.float32([kpL[m.queryIdx].pt for m in refined_matches])
            ptsR = np.float32([kpR[m.trainIdx].pt for m in refined_matches])
            F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, ransac_thresh, 0.99)
            if mask is not None:
                refined_matches = [refined_matches[i] for i in range(len(refined_matches)) if mask[i]]
                print(f"After RANSAC: {len(refined_matches)}")

        self.left_kpts = kpL
        self.right_kpts = kpR
        self.matches = refined_matches
        print(f"Final matches: {len(refined_matches)}\n")
        return refined_matches

    def compute_depth(self):
        depths = []
        for m in self.matches:
            xL = self.left_kpts[m.queryIdx].pt[0]
            xR = self.right_kpts[m.trainIdx].pt[0]
            disparity = xL - xR
            if disparity <= 0:
                continue
            depth = (self.focal_length * self.baseline) / disparity
            depths.append(depth)
        depths = np.array(depths)
        if len(depths) > 0:
            print(f"Depths: min={depths.min():.3f}, max={depths.max():.3f}, median={np.median(depths):.3f}")
        return depths

    def depth_estimation_proj_matrix(self):
        ptsL = np.float32([self.left_kpts[m.queryIdx].pt for m in self.matches])
        ptsR = np.float32([self.right_kpts[m.trainIdx].pt for m in self.matches])
        K = self.K_mtx
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        ptsL_norm = ((ptsL - [cx, cy]) / [fx, fy]).reshape(-1, 1, 2)
        ptsR_norm = ((ptsR - [cx, cy]) / [fx, fy]).reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(ptsL_norm, ptsR_norm, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print("WARNING: Essential matrix estimation failed")
            return
        
        print(f"\n{'='*60}")
        print("ESSENTIAL MATRIX CONDITIONING")
        print(f"{'='*60}")
        print(f"Original E:\n{E}")
        print(f"E determinant: {np.linalg.det(E):.6e}")
        
        U, S, Vt = np.linalg.svd(E)
        print(f"Original singular values: {S}")
        
        sigma = (S[0] + S[1]) / 2.0
        S_cond = np.array([sigma, sigma, 0.0])
        print(f"Conditioned singular values: {S_cond}")
        
        E_cond = U @ np.diag(S_cond) @ Vt
        print(f"Conditioned E:\n{E_cond}")
        print(f"Conditioned det: {np.linalg.det(E_cond):.6e}")
        print(f"{'='*60}\n")
        
        _, R, t, _ = cv2.recoverPose(E_cond, ptsL_norm, ptsR_norm, focal=fx, pp=(cx, cy))
        t_scaled = t.flatten() / (np.linalg.norm(t.flatten()) + 1e-12) * self.baseline
        
        self.P_left_estimated = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        self.P_right_estimated = K @ np.hstack((R, t_scaled.reshape(3,1)))
        
        print(f"Estimated R:\n{R}")
        print(f"Estimated t (scaled): {t_scaled}")
        print(f"Translation norm: {np.linalg.norm(t_scaled):.6f} (expected: {self.baseline:.6f})\n")
        
        return self.P_left_estimated, self.P_right_estimated

    def compare_with_gt(self, gt_file: str, use_estimated_matrices: bool = False):
        with h5py.File(gt_file, "r") as f:
            gt_map = f["depth_0"][()]
            if gt_map.ndim >= 3 and gt_map.shape[0] == 1:
                gt_map = gt_map[0]

        P_left_use = self.P_left_estimated if use_estimated_matrices and self.P_left_estimated is not None else self.P_left
        P_right_use = self.P_right_estimated if use_estimated_matrices and self.P_right_estimated is not None else self.P_right
        print(f"Using {'ESTIMATED' if use_estimated_matrices else 'CALIBRATION'} matrices")

        gt_depths, depth_disp, depth_triang = [], [], []

        for m in self.matches:
            ptL = self.left_kpts[m.queryIdx].pt
            ptR = self.right_kpts[m.trainIdx].pt
            xL, yL = ptL
            xR, yR = ptR

            disp = xL - xR
            if disp <= 0:
                continue
            depth_d = (self.focal_length * self.baseline) / (disp + 1e-12)
            depth_disp.append(depth_d)

            # Triangulate with (2,1) points
            pts1 = np.array([[xL], [yL]], dtype=np.float32)
            pts2 = np.array([[xR], [yR]], dtype=np.float32)
            X4D = cv2.triangulatePoints(P_left_use, P_right_use, pts1, pts2)
            X = X4D[:3, 0] / (X4D[3, 0] + 1e-12)
            if X[2] < 0:
                X = -X
            depth_triang.append(X[2])

            xi, yi = int(round(xL)), int(round(yL))
            if not (0 <= xi < gt_map.shape[1] and 0 <= yi < gt_map.shape[0]):
                continue
            gt_val = float(gt_map[yi, xi])
            if not (0.1 < gt_val < 5000):
                continue
            gt_depths.append(gt_val)

        min_len = min(len(gt_depths), len(depth_disp), len(depth_triang))
        gt_depths = np.array(gt_depths[:min_len])
        depth_disp = np.array(depth_disp[:min_len])
        depth_triang = np.array(depth_triang[:min_len])

        if len(gt_depths) == 0:
            print("No valid GT points")
            return

        def metrics(est):
            err = est - gt_depths
            mask = np.abs(err - np.mean(err)) / (np.std(err) + 1e-12) < 3
            err = err[mask]
            return np.mean(np.abs(err)), np.sqrt(np.mean(err**2)), len(err)

        mae_d, rmse_d, n_d = metrics(depth_disp)
        mae_t, rmse_t, n_t = metrics(depth_triang)

        print(f"\n{'='*60}")
        print("DEPTH METRICS")
        print(f"{'='*60}")
        print(f"Disparity:  points={n_d}, RMSE={rmse_d:.4f}, MAE={mae_d:.4f}")
        print(f"Triang:     points={n_t}, RMSE={rmse_t:.4f}, MAE={mae_t:.4f}")
        print(f"{'='*60}\n")

        return depth_disp, depth_triang, gt_depths

    def run(self, gt_file: str):
        print(f"{'='*60}")
        print("MORLET STEREO PIPELINE")
        print(f"{'='*60}\n")
        self.load_hdf_file()
        self.load_projection_matrices()
        self.detect_and_match_with_scattering()
        self.depth_estimation_proj_matrix()
        self.compute_depth()
        self.compare_with_gt(gt_file, use_estimated_matrices=False)
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}\n")


def plot_feature_maps(feature_maps: list, title_prefix: str = "Morlet"):
    n = len(feature_maps)
    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 3*rows))
    for i, fmap in enumerate(feature_maps):
        plt.subplot(rows, cols, i + 1)
        norm_fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-10)
        plt.imshow(norm_fmap, cmap='inferno')
        plt.title(f"{title_prefix} {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stereo_file = "output/Binocular/500/Coast_Guard/0.hdf5"
    camera_json = "output/Binocular/500/Coast_Guard/camera_params.json"
    
    stereo = StereoReconstruction(stereo_file, camera_json)
    stereo.run(gt_file=stereo_file)
    
    # Visualize Morlet features on a sample patch
    #patch = stereo.scattering_transform.extract_patch(stereo.rendered_data["colors_0"], (100, 100))
    #if patch is not None:
        #feature_maps = stereo.scattering_transform.get_morlet_feature_maps(patch)
        #plot_feature_maps(feature_maps, title_prefix="Morlet")