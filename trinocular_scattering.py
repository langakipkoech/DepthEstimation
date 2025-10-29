# trinocular_morlet_fixed_full.py
# --------------------------------------------------------------
# Robust trinocular reconstruction using Morlet-like scattering + stable triangulation
# --------------------------------------------------------------
import cv2
import numpy as np
import json
import h5py
from typing import Tuple, Dict, Any, List, Optional


# -------------------------
# Scattering / Gabor utils
# -------------------------
class ScatteringTransform:
    """Compute scattering-like coefficients for image patches (Gabor responses)."""

    def __init__(self, patch_size: int = 16, scales: List[int] = [1, 2, 4]):
        self.patch_size = patch_size
        self.scales = scales

    def compute_scattering_coefficients(self, patch: np.ndarray) -> np.ndarray:
        if patch is None or patch.size == 0:
            return np.array([])
        patch = patch.astype(np.float32)
        coeffs = [np.mean(patch), np.std(patch)]
        for scale in self.scales:
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                ksize = min(patch.shape[0], patch.shape[1], 2 * scale + 1)
                if ksize < 3:
                    continue
                kernel = self._create_gabor_kernel(ksize, scale, theta)
                filtered = cv2.filter2D(patch, -1, kernel)
                coeffs.append(np.mean(np.abs(filtered)))
        # two-stage interactions (small set)
        for s1 in self.scales[:2]:
            for s2 in self.scales[:2]:
                if s2 > s1:
                    k1 = self._create_gabor_kernel(min(patch.shape[0], patch.shape[1], 2 * s1 + 1), s1, 0)
                    f1 = np.abs(cv2.filter2D(patch, -1, k1))
                    k2 = self._create_gabor_kernel(min(f1.shape[0], f1.shape[1], 2 * s2 + 1), s2, np.pi/2)
                    f2 = cv2.filter2D(f1, -1, k2)
                    coeffs.append(np.mean(np.abs(f2)))
        return np.array(coeffs, dtype=np.float32)

    def _create_gabor_kernel(self, size: int, scale: float, theta: float) -> np.ndarray:
        if size < 3:
            size = 3
        sigma = float(scale)
        lambd = float(scale * 2)
        gamma = 0.5
        psi = 0.0
        kernel = cv2.getGaborKernel((size, size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        return kernel / (np.sum(np.abs(kernel)) + 1e-10)

    def extract_patch(self, img: np.ndarray, point: Tuple[float, float]) -> Optional[np.ndarray]:
        if img is None:
            return None
        x, y = int(round(point[0])), int(round(point[1]))
        h = self.patch_size // 2
        y0, y1 = max(0, y - h), min(img.shape[0], y + h)
        x0, x1 = max(0, x - h), min(img.shape[1], x + h)
        if (y1 - y0) < self.patch_size // 2 or (x1 - x0) < self.patch_size // 2:
            return None
        return img[y0:y1, x0:x1]


# -------------------------
# Trinocular reconstruction
# -------------------------
class TrinocularReconstruction:
    def __init__(self, hdf_file_path: str, cam_config_path: str):
        self.hdf_file_path = hdf_file_path
        self.cam_file_path = cam_config_path

        # camera data
        self.K: Optional[np.ndarray] = None
        self.focal: Optional[float] = None
        self.baselines: Dict[str, float] = {}

        # projection matrices (calibrated)
        self.P_left = self.P_right = self.P_front = None
        # estimated (from matches)
        self.P_left_est = self.P_right_est = self.P_front_est = None

        # features and matches
        self.left_kpts = self.right_kpts = self.front_kpts = []
        self.matches_lr = self.matches_lf = self.matches_rf = []
        self.triplets = []

        # images and scattering
        self.rendered_data: Dict[str, Any] = {}
        self.img_shape = None
        self.scattering = ScatteringTransform(patch_size=8, scales=[1, 2, 4])

    # -------------------------
    # Data loading
    # -------------------------
    def load_hdf_file(self):
        with h5py.File(self.hdf_file_path, "r") as f:
            for k in ("colors_left", "colors_right", "colors_front"):
                if k not in f:
                    raise KeyError(f"Missing {k} in HDF file")
                img = f[k][()]
                if img.ndim >= 3 and img.shape[0] == 1:
                    img = img[0]
                self.rendered_data[k] = img
            # optional depth keys
            for k in ("depth_left", "depth_right", "depth_front"):
                if k in f:
                    img = f[k][()]
                    if img.ndim >= 3 and img.shape[0] == 1:
                        img = img[0]
                    self.rendered_data[k] = img
        self.img_shape = self.rendered_data["colors_left"].shape[:2]

    def load_projection_matrices(self):
        with open(self.cam_file_path, "r") as f:
            cam = json.load(f)

        self.K = np.array(cam["intrinsics"]["K"]).reshape(3, 3)
        self.focal = float(self.K[0, 0])

        T = {name: np.array(cam["cameras"][name]["T"]).reshape(3) for name in ("left", "right", "front")}
        self.baselines["lr"] = float(np.linalg.norm(T["left"] - T["right"]))
        self.baselines["lf"] = float(np.linalg.norm(T["left"] - T["front"]))
        self.baselines["rf"] = float(np.linalg.norm(T["right"] - T["front"]))

        def build_P(R, Tvec):
            Rm = np.array(R).reshape(3, 3)
            C = np.array(Tvec).reshape(3, 1)
            Rt = np.hstack([Rm, -Rm @ C])
            return self.K @ Rt

        self.P_left = build_P(cam["cameras"]["left"]["R"], cam["cameras"]["left"]["T"])
        self.P_right = build_P(cam["cameras"]["right"]["R"], cam["cameras"]["right"]["T"])
        self.P_front = build_P(cam["cameras"]["front"]["R"], cam["cameras"]["front"]["T"])

        print(f"Baseline L-R : {self.baselines['lr']:.6f}")
        print(f"Baseline L-F : {self.baselines['lf']:.6f}")

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    @staticmethod
    def _ncc(p1: np.ndarray, p2: np.ndarray) -> float:
        if p1.shape != p2.shape or p1.size == 0:
            return 0.0
        a = p1.astype(np.float32) - p1.mean()
        b = p2.astype(np.float32) - p2.mean()
        den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        return float(np.sum(a * b) / den) if den > 1e-10 else 0.0

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.size == 0 or v2.size == 0 or v1.shape != v2.shape:
            return 0.0
        dot = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return dot / den if den > 1e-10 else 0.0

    @staticmethod
    def refine_keypoints(gray: np.ndarray, kpts: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """Subpixel refine SIFT keypoints using cornerSubPix (modifies keypoint.pt in-place)."""
        if len(kpts) == 0:
            return kpts
        pts = np.float32([kp.pt for kp in kpts]).reshape(-1, 1, 2)
        win = (5, 5)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        try:
            cv2.cornerSubPix(gray, pts, win, (-1, -1), criteria)
            for kp, p in zip(kpts, pts.reshape(-1, 2)):
                kp.pt = (float(p[0]), float(p[1]))
        except Exception:
            # fallback: keep original keypoints
            pass
        return kpts

    # -------------------------
    # Matching (pair)
    # -------------------------
    def _match_pair(self,
                    imgA, imgB,
                    nameA: str, nameB: str,
                    ratio: float,
                    ncc_thr: float,
                    scat_thr: float,
                    use_ransac: bool,
                    ransac_thr: float):
        grayA = self.to_gray(imgA)
        grayB = self.to_gray(imgB)

        sift = cv2.SIFT_create()
        kpA, desA = sift.detectAndCompute(grayA, None)
        kpB, desB = sift.detectAndCompute(grayB, None)

        # Subpixel refinement for better triangulation at long ranges
        kpA = self.refine_keypoints(grayA, kpA)
        kpB = self.refine_keypoints(grayB, kpB)

        # Safety: descriptors may be None
        if desA is None or desB is None or len(desA) == 0 or len(desB) == 0:
            print(f"  WARNING: No descriptors for {nameA}-{nameB}")
            return kpA, kpB, []

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        knn = flann.knnMatch(desA, desB, k=2)
        good = [m for m, n in knn if len([m, n]) == 2 and m.distance < ratio * n.distance]

        refined = []
        for m in good:
            ptA = kpA[m.queryIdx].pt
            ptB = kpB[m.trainIdx].pt

            patchA = self.scattering.extract_patch(grayA, ptA)
            patchB = self.scattering.extract_patch(grayB, ptB)
            if patchA is None or patchB is None:
                continue

            if self._ncc(patchA, patchB) < ncc_thr:
                continue

            sA = self.scattering.compute_scattering_coefficients(patchA)
            sB = self.scattering.compute_scattering_coefficients(patchB)
            if sA.size == 0 or sB.size == 0:
                continue
            if self._cosine(sA, sB) < scat_thr:
                continue

            refined.append(m)

        # RANSAC filtering with fundamental matrix
        if use_ransac and len(refined) >= 8:
            ptsA = np.float32([kpA[m.queryIdx].pt for m in refined])
            ptsB = np.float32([kpB[m.trainIdx].pt for m in refined])
            F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC,
                                             ransacReprojThreshold=ransac_thr,
                                             confidence=0.99)
            if mask is not None:
                refined = [m for i, m in enumerate(refined) if mask[i][0] == 1]

        print(f"  {nameA}-{nameB} → {len(refined)} matches (after Morlet+NCC+RANSAC)")
        return kpA, kpB, refined

    def detect_and_match(self,
                         ratio_test: float = 0.7,
                         use_ransac: bool = True,
                         ransac_thresh: float = 1.0,
                         ncc_thr: float = 0.7,
                         scat_thr: float = 0.85):
        imgL = self.rendered_data["colors_left"]
        imgR = self.rendered_data["colors_right"]
        imgF = self.rendered_data["colors_front"]

        self.left_kpts, self.right_kpts, self.matches_lr = \
            self._match_pair(imgL, imgR, "left", "right", ratio_test, ncc_thr, scat_thr, use_ransac, ransac_thresh)

        self.left_kpts, self.front_kpts, self.matches_lf = \
            self._match_pair(imgL, imgF, "left", "front", ratio_test, ncc_thr, scat_thr, use_ransac, ransac_thresh)

        self.right_kpts, self.front_kpts, self.matches_rf = \
            self._match_pair(imgR, imgF, "right", "front", ratio_test, ncc_thr, scat_thr, use_ransac, ransac_thresh)

    # -------------------------
    # Triplet building
    # -------------------------
    def build_triplets(self):
        lr = {m.queryIdx: m.trainIdx for m in self.matches_lr}
        lf = {m.queryIdx: m.trainIdx for m in self.matches_lf}
        rf = {m.queryIdx: m.trainIdx for m in self.matches_rf}

        triplets = []
        for l_idx, r_idx in lr.items():
            f_l = lf.get(l_idx)
            if f_l is None:
                continue
            f_r = rf.get(r_idx)
            if f_r == f_l:
                triplets.append((l_idx, r_idx, f_l))

        self.triplets = triplets
        print(f"Consistent triplets : {len(triplets)}")
        return triplets

    # -------------------------
    # Pose estimation (anchor to left)
    # -------------------------
    def _estimate_pose(self, ptsA: np.ndarray, ptsB: np.ndarray, baseline: float):
        """
        ptsA, ptsB: raw pixel Nx2 arrays (not normalized)
        returns R (3x3) and t (3,) scaled to baseline with cheirality check
        """
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        E, mask = cv2.findEssentialMat(ptsA, ptsB, cameraMatrix=self.K,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            raise RuntimeError("Essential estimation failed")

        # condition E
        U, S, Vt = np.linalg.svd(E)
        sigma = (S[0] + S[1]) / 2.0
        E_cond = U @ np.diag([sigma, sigma, 0.0]) @ Vt

        # recoverPose with cameraMatrix
        _, R, t, mask_rec = cv2.recoverPose(E_cond, ptsA, ptsB, cameraMatrix=self.K)
        t = t.flatten()
        # scale to baseline
        if np.linalg.norm(t) < 1e-12:
            raise RuntimeError("Recovered t near-zero")
        t = t / (np.linalg.norm(t) + 1e-12) * baseline

        # cheirality (triangulate small subset)
        inliers = np.where(mask_rec.ravel() == 1)[0] if mask_rec is not None else np.arange(len(ptsA))
        if inliers.size > 5:
            sample = inliers[:min(50, inliers.size)]
            ptsA_s = ptsA[sample].T.astype(np.float32)  # 2xM
            ptsB_s = ptsB[sample].T.astype(np.float32)
            P0 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P1 = self.K @ np.hstack((R, t.reshape(3, 1)))
            X4 = cv2.triangulatePoints(P0, P1, ptsA_s, ptsB_s)
            Zs = X4[2, :] / (X4[3, :] + 1e-12)
            pos = np.sum(Zs > 1e-6)
            neg = np.sum(Zs < -1e-6)
            if neg > pos:
                t = -t
        return R, t

    def estimate_projection_matrices(self):
        # L-R
        ptsL_lr = np.float32([self.left_kpts[m.queryIdx].pt for m in self.matches_lr])
        ptsR_lr = np.float32([self.right_kpts[m.trainIdx].pt for m in self.matches_lr])
        R_lr, t_lr = self._estimate_pose(ptsL_lr, ptsR_lr, self.baselines["lr"])

        # L-F
        ptsL_lf = np.float32([self.left_kpts[m.queryIdx].pt for m in self.matches_lf])
        ptsF_lf = np.float32([self.front_kpts[m.trainIdx].pt for m in self.matches_lf])
        R_lf, t_lf = self._estimate_pose(ptsL_lf, ptsF_lf, self.baselines["lf"])

        # store anchored to left
        self.P_left_est = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P_right_est = self.K @ np.hstack((R_lr, t_lr.reshape(3, 1)))
        self.P_front_est = self.K @ np.hstack((R_lf, t_lf.reshape(3, 1)))

        # debug baseline angle
        a = t_lr / (np.linalg.norm(t_lr) + 1e-12)
        b = t_lf / (np.linalg.norm(t_lf) + 1e-12)
        angle = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
        print(f"[DEBUG] angle between L-R and L-F baselines: {angle:.3f}°")
        if angle < 5.0:
            print("WARNING: baselines nearly parallel -> triangulation will be ill-conditioned for far points")

    # -------------------------
    # Damped triangulation (pixel-form A with full P)
    # -------------------------
    def _triangulate_damped(self, pts: List[Tuple[float, float]], Ps: List[np.ndarray],
                            lambda_damp: float = 1e-3, verbose: bool = False) -> Optional[np.ndarray]:
        """
        pts: list of (u,v) pixel coordinates
        Ps: list of 3x4 projection matrices (containing K)
        returns X (3,) or None on failure. Uses pixel-form A so we DO NOT apply K^{-1}.
        """
        if len(pts) != len(Ps):
            return None
        # build A using pixel coords
        A_rows = []
        for (u, v), P in zip(pts, Ps):
            P = np.asarray(P, dtype=np.float64)
            A_rows.append(u * P[2, :] - P[0, :])
            A_rows.append(v * P[2, :] - P[1, :])
        A = np.vstack(A_rows)  # shape (2n, 4)

        ATA = A.T @ A
        # adaptive damping proportional to ATA norm
        scale = np.linalg.norm(ATA, ord=2)
        lam = lambda_damp * (1.0 + scale)
        ATA_damped = ATA + lam * np.eye(4)

        try:
            _, _, Vt = np.linalg.svd(ATA_damped)
            Xh = Vt[-1]
            if not np.isfinite(Xh).all():
                if verbose: print("triang: Xh contains non-finite")
                return None
            if abs(Xh[3]) < 1e-12:
                if verbose: print("triang: tiny homogeneous w (degenerate)")
                return None
            X = Xh[:3] / Xh[3]
            if not np.isfinite(X).all():
                if verbose: print("triang: X not finite after dehomog")
                return None
            # cheirality
            if X[2] < 0:
                X = -X
            return X.astype(np.float64)
        except np.linalg.LinAlgError:
            if verbose: print("triang: SVD failed")
            return None

    # -------------------------
    # Depth computation with robust filtering
    # -------------------------
    def compute_depth(self, use_estimated: bool = False, verbose: bool = True) -> np.ndarray:
        if not self.triplets:
            if verbose: print("No triplets – depth computation skipped")
            return np.array([])

        P_L = self.P_left_est if use_estimated and self.P_left_est is not None else self.P_left
        P_R = self.P_right_est if use_estimated and self.P_right_est is not None else self.P_right
        P_F = self.P_front_est if use_estimated and self.P_front_est is not None else self.P_front

        if P_L is None or P_R is None or P_F is None:
            raise RuntimeError("Projection matrices missing")

        depths = []
        failed = 0
        total = len(self.triplets)

        for i, (l_idx, r_idx, f_idx) in enumerate(self.triplets):
            ptL = tuple(self.left_kpts[l_idx].pt)
            ptR = tuple(self.right_kpts[r_idx].pt)
            ptF = tuple(self.front_kpts[f_idx].pt)

            X = self._triangulate_damped([ptL, ptR, ptF], [P_L, P_R, P_F], lambda_damp=1e-3, verbose=False)
            if X is None:
                failed += 1
                continue
            z = float(X[2])
            if not (np.isfinite(z) and 0.05 < z < 1e6):
                failed += 1
                continue
            depths.append(z)

        depths = np.array(depths, dtype=np.float64)
        if depths.size == 0:
            if verbose:
                print(f"compute_depth: all {total} triplets failed (failed={failed})")
            return np.array([])

        # Robust trimming (3-sigma)
        mu, sigma = float(depths.mean()), float(depths.std())
        if np.isfinite(sigma) and sigma > 0:
            mask = np.abs(depths - mu) <= 3 * sigma
            depths_filtered = depths[mask]
        else:
            depths_filtered = depths

        if depths_filtered.size == 0:
            if verbose:
                print("compute_depth: all depths removed by 3-sigma filter")
            return np.array([])

        if verbose:
            print("\nDEPTH STATISTICS (robust):")
            print(f"  total triplets: {total}, kept: {depths_filtered.size}, failed: {failed}")
            print(f"  mean={depths_filtered.mean():.3f}, std={depths_filtered.std():.3f}, min={depths_filtered.min():.3f}, max={depths_filtered.max():.3f}")
        return depths_filtered

    # -------------------------
    # Bilinear GT sample + safe comparison
    # -------------------------
    def _bilinear_sample(self, img: np.ndarray, x: float, y: float) -> float:
        h, w = img.shape
        if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
            return float(img[int(round(y)), int(round(x))])
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x0, y - y0
        v00 = float(img[y0, x0]); v10 = float(img[y0, x0 + 1])
        v01 = float(img[y0 + 1, x0]); v11 = float(img[y0 + 1, x0 + 1])
        return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11

    def compare_with_gt(self, gt_file: str, use_estimated: bool = False):
        with h5py.File(gt_file, "r") as f:
            gt_map = f["depth_left"][()]
            if gt_map.ndim >= 3 and gt_map.shape[0] == 1:
                gt_map = gt_map[0]

        depths = self.compute_depth(use_estimated=use_estimated, verbose=True)
        if depths.size == 0:
            print("compare_with_gt: no valid depths to compare")
            return None

        # Align depths with triplets as best-effort: iterate triplets and take GT for same L keypoint
        gt_vals = []
        kept_depths = []
        d_idx = 0
        for l_idx, _, _ in self.triplets:
            if d_idx >= len(depths):
                break
            pt = self.left_kpts[l_idx].pt
            x, y = float(pt[0]), float(pt[1])
            if not (0 <= x < gt_map.shape[1] and 0 <= y < gt_map.shape[0]):
                continue
            dgt = float(self._bilinear_sample(gt_map, x, y))
            if not (0.05 < dgt < 5000):
                continue
            gt_vals.append(dgt)
            kept_depths.append(depths[d_idx])
            d_idx += 1

        if len(kept_depths) == 0:
            print("No GT samples aligned with computed depths")
            return None

        depths_arr = np.array(kept_depths)
        gt_arr = np.array(gt_vals[:len(depths_arr)])
        err = depths_arr - gt_arr
        if err.size == 0:
            print("No errors to evaluate")
            return None
        mask = np.abs(err - err.mean()) / (err.std() + 1e-12) < 3
        err = err[mask]
        if err.size == 0:
            print("All errors removed by 3-sigma masking")
            return None

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        print("\n=== TRINOCULAR DEPTH vs GT ===")
        print(f"Points   : {len(err)}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"MAE      : {mae:.4f}")
        print("================================")
        return depths_arr, gt_arr

    # -------------------------
    # Runner
    # -------------------------
    def run(self, gt_file: str,
            ratio_test: float = 0.7,
            use_ransac: bool = True,
            ransac_thr: float = 1.0,
            ncc_thr: float = 0.7,
            scat_thr: float = 0.85,
            estimate_pose: bool = True,
            use_estimated_for_depth: bool = False):
        print("=" * 60)
        print("TRINOCULAR MORLET PIPELINE (ROBUST)")
        print("=" * 60)
        self.load_hdf_file()
        self.load_projection_matrices()
        self.detect_and_match(ratio_test, use_ransac, ransac_thr, ncc_thr, scat_thr)
        self.build_triplets()

        if estimate_pose:
            try:
                self.estimate_projection_matrices()
            except Exception as e:
                print("estimate_projection_matrices failed:", e)
                print("Falling back to calibrated projection matrices for triangulation.")
                estimate_pose = False

        # If user asked to use estimated for depth, pass that flag
        self.compare_with_gt(gt_file, use_estimated=use_estimated_for_depth)
        print("PIPELINE FINISHED")
        print("=" * 60)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Update paths to your dataset
    hdf_path = "output/quad/500/Coast_Guard/views.hdf5"
    cam_json = "output/quad/500/Coast_Guard/camera_params.json"

    tri = TrinocularReconstruction(hdf_path, cam_json)
    tri.run(gt_file=hdf_path,
            ratio_test=0.7,
            use_ransac=True,
            ransac_thr=1.0,
            ncc_thr=0.7,
            scat_thr=0.85,
            estimate_pose=True,
            use_estimated_for_depth=False)
