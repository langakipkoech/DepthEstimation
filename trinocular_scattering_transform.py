import cv2
import numpy as np
import json
import h5py
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from skimage.feature.texture import graycomatrix, graycoprops
from dataclasses import dataclass



@dataclass
class RectificationPair:
    """Store rectification data for a stereo pair"""
    name: str
    H1: np.ndarray
    H2: np.ndarray
    img1_rect: np.ndarray
    img2_rect: np.ndarray
    baseline: float
    

class ScatteringTransform:
    """Compute scattering coefficients for image patches"""
    
    def __init__(self, patch_size: int = 16, scales: List[int] = [1, 2, 4]):
        self.patch_size = patch_size
        self.scales = scales
        
    def compute_scattering_coefficients(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute scattering coefficients for a patch.
        Uses multi-scale wavelet-like features with different orientations.
        """
        if patch.size == 0:
            return np.array([])
        
        # Ensure patch is float
        patch = patch.astype(np.float32)
        
        coefficients = []
        
        # Order 0: Mean intensity (zeroth order scattering)
        coefficients.append(np.mean(patch))
        coefficients.append(np.std(patch))
        
        # Order 1: First-order scattering at multiple scales
        for scale in self.scales:
            # Gabor-like filters at different orientations
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                # Create oriented filter
                kernel_size = min(patch.shape[0], patch.shape[1], 2*scale+1)
                if kernel_size < 3:
                    continue
                    
                kernel = self._create_gabor_kernel(kernel_size, scale, theta)
                
                # Convolve and take modulus
                filtered = cv2.filter2D(patch, -1, kernel)
                
                # Scattering coefficients: mean of modulus
                coefficients.append(np.mean(np.abs(filtered)))
                
        # Order 2: Second-order scattering (modulus of modulus)
        for scale1 in self.scales[:2]:  # Use fewer scales for second order
            for scale2 in self.scales[:2]:
                if scale2 > scale1:
                    # First layer
                    kernel1 = self._create_gabor_kernel(
                        min(patch.shape[0], patch.shape[1], 2*scale1+1), 
                        scale1, 0
                    )
                    filtered1 = np.abs(cv2.filter2D(patch, -1, kernel1))
                    
                    # Second layer
                    kernel2 = self._create_gabor_kernel(
                        min(filtered1.shape[0], filtered1.shape[1], 2*scale2+1),
                        scale2, np.pi/2
                    )
                    filtered2 = cv2.filter2D(filtered1, -1, kernel2)
                    
                    coefficients.append(np.mean(np.abs(filtered2)))
        
        return np.array(coefficients)
    
    def _create_gabor_kernel(self, size: int, scale: float, theta: float) -> np.ndarray:
        """Create a Gabor-like kernel for scattering"""
        if size < 3:
            size = 3
        sigma = scale
        lambd = scale * 2
        gamma = 0.5
        psi = 0
        
        kernel = cv2.getGaborKernel(
            (size, size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
        )
        return kernel / (np.sum(np.abs(kernel)) + 1e-10)
    
    def extract_patch(self, img: np.ndarray, point: Tuple[float, float]) -> Optional[np.ndarray]:
        """Extract a patch around a keypoint"""
        x, y = int(round(point[0])), int(round(point[1]))
        half_size = self.patch_size // 2
        
        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)
        
        if (y_max - y_min) < self.patch_size // 2 or (x_max - x_min) < self.patch_size // 2:
            return None
            
        patch = img[y_min:y_max, x_min:x_max]
        return patch
    
    
class TriStereoReconstruction:
    def __init__(self, hdf_file_path: str, cam_config_path: str):
        self.hdf_file_path = hdf_file_path
        self.cam_file_path = cam_config_path
        self.P_left = None
        self.P_right = None
        self.P_top = None
        self.left_kpts = None
        self.right_kpts = None
        self.top_kpts = None
        self.baseline = None
        self.focal_length = None
        self.rendered_data: Dict[str, Any] = {}
        self.matches = None
        self.K_mtx = None
        self.img_shape = None
        self.calibration_verified = False
        self.scattering_transform = ScatteringTransform(patch_size=16, scales=[1, 2, 4])
        
    
    def load_hdf_file(self):
        if self.hdf_file_path is None:
            raise RuntimeError("Missing hdf file path")
        view_keys = ["colors_left", "colors_right", "colors_front", "depth_left", "depth_right", "depth_front"]
        with h5py.File(self.hdf_file_path, "r") as hdf_data:
            for key in view_keys:
                if key not in hdf_data:
                    raise KeyError(f"Missing key {key} in HDF file")
                img = hdf_data[key][()]
                if img.ndim >= 3 and img.shape[0] == 1:
                    img = img[0]
                self.rendered_data[key] = img
        
        self.img_shape = self.rendered_data["colors_left"].shape[:2]
        return self.rendered_data
    
    def load_projection_matrices(self, verify: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(self.cam_file_path, "r") as f:
            cam = json.load(f)

        K = np.array(cam["intrinsics"]["K"]).reshape(3, 3)
        self.K_mtx = K
        self.focal_length = float(K[0, 0])

        T_left = np.array(cam["cameras"]["left"]["T"]).reshape(3)
        T_right = np.array(cam["cameras"]["right"]["T"]).reshape(3)
        T_front = np.array(cam["cameras"]["front"]["T"]).reshape(3)

        self.baselines["lr"] = float(np.linalg.norm(T_left - T_right))
        self.baselines["lf"] = float(np.linalg.norm(T_left - T_front))
        self.baselines["rf"] = float(np.linalg.norm(T_right - T_front))
        self.baseline_lr = self.baselines["lr"]

        def build_projection(R, T):
            R = np.array(R).reshape(3, 3)
            C = np.array(T).reshape(3, 1)
            Rt = np.hstack([R, -R @ C])
            return K @ Rt

        self.P_left = build_projection(cam["cameras"]["left"]["R"], cam["cameras"]["left"]["T"])
        self.P_right = build_projection(cam["cameras"]["right"]["R"], cam["cameras"]["right"]["T"])
        self.P_front = build_projection(cam["cameras"]["front"]["R"], cam["cameras"]["front"]["T"])

        
        print(f"Baseline (L-R): {self.baselines['lr']:.6f}")
        print(f"Baseline (L-F): {self.baselines['lf']:.6f}")
        print(f"Baseline (R-F): {self.baselines['rf']:.6f}")
        print(f"Focal length (fx): {self.focal_length:.2f} pixels")
        print(f"Principal point: ({K[0,2]:.2f}, {K[1,2]:.2f})")
        print(f"Image size: {self.img_shape}")
        
        if verify:
            self.verify_calibration()
        
        return self.P_left, self.P_right, self.P_front
    
    
    @staticmethod
    def prepare_image_cv(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            return img.astype(np.uint8)
        else:
            raise ValueError(f"Unexpected image shape {img.shape}")

    def rectify_stereo_pair(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        pair_name : str,
        baseline: float
            ) -> RectificationPair:
        
        #compute fundaqmental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, 
                                         cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is None:
            raise RuntimeError(f"Invalid fundamental matrix")
        
        #rectification homographies
        h, w = img1.shape[:2]
        _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))
        
        #apply rectification
        img1_rect = cv2.warpPerspective(img1, H1, (w, h))
        img2_rect = cv2.warpPerspective(img2, H2, (w, h))
        
        return RectificationPair(
            name=pair_name,
            H1=H1,
            H2=H2,
            img1_rect=img1_rect,
            img2_rect=img2_rect,
            baseline=baseline
        )
        
    def compute_ncc(self, patch1: np.ndarray, patch2: np.ndarray) -> float:
        if patch1.shape != patch2.shape:
            return 0.0 
        
        #convert to float
        patch2 = patch2.astype(np.float32)
        patch1 = patch1.astype(np.float32)
        
        patch1_norm = patch1 - np.mean(patch1)
        patch2_norm = patch2 - np.mean(patch2)
        
        numerator = np.sum(patch1_norm * patch2_norm)
        
        denominator = np.sqrt(np.sum(patch1_norm**2)) * np.sum(patch2_norm**2)
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator/denominator 
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if len(vec1) == 0 or len(vec2) == 0 or len(vec1) != len(vec2):
            return 0.0 
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0 
        
        return dot_product / (norm1 * norm2)
    
    def detect_and_match_with_scattering(
        self,
        ratio_test: float = 0.7,
        use_ransac: bool = True,
        ransac_thresh: float = 1.0,
        epipolar_thresh: float = 2.0,
        ncc_threshold: float = 0.7,
        scattering_similarity_threshold: float = 0.85
    ):
        """
        Detect keypoints and match using scattering coefficients + NCC + cosine similarity
        across all three images (equilateral camera setup).
        Handles pairs: (Left–Right), (Left–Top), (Right–Top)
        """

        def _match_pair(imgA, imgB):
            
            # Initialize SIFT
            sift = cv2.SIFT_create(
                nfeatures=2000,
                nOctaveLayers=6,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )

            # Convert to grayscale
            grayA = self.prepare_image_cv(imgA)
            grayB = self.prepare_image_cv(imgB)

            # Detect keypoints and descriptors
            kpA, desA = sift.detectAndCompute(grayA, None)
            kpB, desB = sift.detectAndCompute(grayB, None)
            if len(kpA) == 0 or len(kpB) == 0:
                raise RuntimeError(f"No keypoints detected ")

          
            # FLANN matching + ratio test
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desA, desB, k=2)

            candidate_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]
            print(f"After ratio test: {len(candidate_matches)} candidate matches")

            if len(candidate_matches) < 8:
                print("Too few candidate matches — skipping pair.")
                return kpA, kpB, []

            # Estimate fundamental matrix
            ptsA = np.float32([kpA[m.queryIdx].pt for m in candidate_matches])
            ptsB = np.float32([kpB[m.trainIdx].pt for m in candidate_matches])
            F, _ = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 2.0, 0.99)
            if F is not None:
                print("Estimated fundamental matrix.")
            else:
                print(" Could not estimate F — skipping epipolar filtering.")
                return kpA, kpB, []

            # Scattering + NCC refinement with geometric epipolar distance
            refined_matches = []
            scattering_scores, ncc_scores = [], []

            for idx, m in enumerate(candidate_matches):
                if idx % 100 == 0:
                    print(f"  Processing match {idx}/{len(candidate_matches)}...")

                ptA = kpA[m.queryIdx].pt
                ptB = kpB[m.trainIdx].pt

                # Compute geometric epipolar distance
                p1 = np.array([ptA[0], ptA[1], 1.0])
                p2 = np.array([ptB[0], ptB[1], 1.0])
                Fp1 = F @ p1
                denom = np.sqrt(Fp1[0]**2 + Fp1[1]**2) + 1e-12
                epipolar_dist = abs(p2.T @ Fp1) / denom

                if epipolar_dist > epipolar_thresh:
                    continue 

                # Extract patches
                patchA = self.scattering_transform.extract_patch(grayA, ptA)
                patchB = self.scattering_transform.extract_patch(grayB, ptB)
                if patchA is None or patchB is None:
                    continue

                scA = self.scattering_transform.compute_scattering_coefficients(patchA)
                scB = self.scattering_transform.compute_scattering_coefficients(patchB)
                if len(scA) == 0 or len(scB) == 0:
                    continue

                scatter_sim = self.cosine_similarity(scA, scB)
                ncc_sim = self.compute_ncc(patchA, patchB)

                if scatter_sim >= scattering_similarity_threshold and ncc_sim >= ncc_threshold:
                    refined_matches.append(m)
                    scattering_scores.append(scatter_sim)
                    ncc_scores.append(ncc_sim)

            print(f"\nAfter scattering + NCC filtering: {len(refined_matches)} matches")
            print(f"Mean scattering similarity: {np.mean(scattering_scores):.3f}" if scattering_scores else "N/A")
            print(f"Mean NCC score: {np.mean(ncc_scores):.3f}" if ncc_scores else "N/A")

            # Optional RANSAC cleanup
            if use_ransac and len(refined_matches) > 8:
                ptsA_ref = np.float32([kpA[m.queryIdx].pt for m in refined_matches])
                ptsB_ref = np.float32([kpB[m.trainIdx].pt for m in refined_matches])
                F_refined, mask = cv2.findFundamentalMat(ptsA_ref, ptsB_ref, cv2.FM_RANSAC, ransac_thresh, 0.99)
                if mask is not None:
                    refined_matches = [refined_matches[i] for i in range(len(refined_matches)) if mask[i]]
                    print(f"After RANSAC: {len(refined_matches)} matches")

            return kpA, kpB, refined_matches

        # === Load three images ===
        left_img = self.rendered_data["colors_left"]
        right_img = self.rendered_data["colors_right"]
        top_img = self.rendered_data["colors_fromt"]

        # === Match each pair ===
        kpL, kpR, matches_LR = _match_pair(left_img, right_img)
        kpL2, kpT, matches_LT = _match_pair(left_img, top_img)
        kpR2, kpT2, matches_RT = _match_pair(right_img, top_img)

        # === Store results ===
        self.matches = {
            "Left-Right": matches_LR,
            "Left-Top": matches_LT,
            "Right-Top": matches_RT,
        }
        self.keypoints = {
            "Left": kpL,
            "Right": kpR,
            "Top": kpT,
        }

        print("\nSummary:")
        print(f"  Left–Right: {len(matches_LR)} matches")
        print(f"  Left–Top:   {len(matches_LT)} matches")
        print(f"  Right–Top:  {len(matches_RT)} matches")

        return self.matches


        
   