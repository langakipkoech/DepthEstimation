# --------------------------------------------------------------
#  trinocular_morlet.py
# --------------------------------------------------------------
import cv2
import numpy as np
import json
import h5py
from typing import Tuple, Dict, Any, List, Optional


class ScatteringTransform:
    """Compute scattering coefficients for image patches"""

    def __init__(self, patch_size: int = 16, scales: List[int] = [1, 2, 4]):
        self.patch_size = patch_size
        self.scales = scales

    def compute_scattering_coefficients(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
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

        for s1 in self.scales[:2]:
            for s2 in self.scales[:2]:
                if s2 > s1:
                    k1 = self._create_gabor_kernel(
                        min(patch.shape[0], patch.shape[1], 2 * s1 + 1), s1, 0
                    )
                    f1 = np.abs(cv2.filter2D(patch, -1, k1))
                    k2 = self._create_gabor_kernel(
                        min(f1.shape[0], f1.shape[1], 2 * s2 + 1), s2, np.pi / 2
                    )
                    f2 = cv2.filter2D(f1, -1, k2)
                    coeffs.append(np.mean(np.abs(f2)))

        return np.array(coeffs)

    def _create_gabor_kernel(self, size: int, scale: float, theta: float) -> np.ndarray:
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
        x, y = int(round(point[0])), int(round(point[1]))
        h = self.patch_size // 2
        y0, y1 = max(0, y - h), min(img.shape[0], y + h)
        x0, x1 = max(0, x - h), min(img.shape[1], x + h)
        if (y1 - y0) < self.patch_size // 2 or (x1 - x0) < self.patch_size // 2:
            return None
        return img[y0:y1, x0:x1]


# ----------------------------------------------------------------------
#  MAIN TRINOCULAR CLASS
# ----------------------------------------------------------------------
class TrinocularReconstruction:
    def __init__(self, hdf_file_path: str, cam_config_path: str):
        self.hdf_file_path = hdf_file_path
        self.cam_file_path = cam_config_path

        # projection matrices (ground-truth)
        self.P_left = self.P_right = self.P_front = None
        # estimated matrices (optional)
        self.P_left_est = self.P_right_est = self.P_front_est = None

        self.left_kpts = self.right_kpts = self.front_kpts = None
        self.matches_lr = self.matches_lf = self.matches_rf = None
        self.triplets = None          # list of (l_idx, r_idx, f_idx)

        self.K = None
        self.focal = None
        self.baselines = {}           # lr, lf, rf

        self.rendered_data: Dict[str, Any] = {}
        self.img_shape = None
        self.scattering = ScatteringTransform(patch_size=8, scales=[1, 2, 4])

    # ------------------------------------------------------------------
    # 1. Load images
    # ------------------------------------------------------------------
    def load_hdf_file(self):
        required = ["colors_left", "colors_right", "colors_front"]
        optional = ["depth_left", "depth_right", "depth_front"]

        with h5py.File(self.hdf_file_path, "r") as f:
            for k in required:
                if k not in f:
                    raise KeyError(f"Missing required key {k}")
                img = f[k][()]
                if img.ndim >= 3 and img.shape[0] == 1:
                    img = img[0]
                self.rendered_data[k] = img

            for k in optional:
                if k in f:
                    img = f[k][()]
                    if img.ndim >= 3 and img.shape[0] == 1:
                        img = img[0]
                    self.rendered_data[k] = img

        self.img_shape = self.rendered_data["colors_left"].shape[:2]

    # ------------------------------------------------------------------
    # 2. Load camera parameters
    # ------------------------------------------------------------------
    def load_projection_matrices(self):
        with open(self.cam_file_path, "r") as f:
            cam = json.load(f)

        self.K = np.array(cam["intrinsics"]["K"]).reshape(3, 3)
        self.focal = float(self.K[0, 0])

        T = {name: np.array(cam["cameras"][name]["T"]).reshape(3) for name in ("left", "right", "front")}

        self.baselines["lr"] = float(np.linalg.norm(T["left"] - T["right"]))
        self.baselines["lf"] = float(np.linalg.norm(T["left"] - T["front"]))
        self.baselines["rf"] = float(np.linalg.norm(T["right"] - T["front"]))

        def build_P(R, T):
            R = np.array(R).reshape(3, 3)
            C = np.array(T).reshape(3, 1)
            Rt = np.hstack([R, -R @ C])
            return self.K @ Rt

        self.P_left  = build_P(cam["cameras"]["left"]["R"],  cam["cameras"]["left"]["T"])
        self.P_right = build_P(cam["cameras"]["right"]["R"], cam["cameras"]["right"]["T"])
        self.P_front = build_P(cam["cameras"]["front"]["R"], cam["cameras"]["front"]["T"])

        print(f"Baseline L-R : {self.baselines['lr']:.6f}")
        print(f"Baseline L-F : {self.baselines['lf']:.6f}")

    # ------------------------------------------------------------------
    # 3. Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    def ncc(self, p1: np.ndarray, p2: np.ndarray) -> float:
        if p1.shape != p2.shape:
            return 0.0
        a = p1.astype(np.float32) - p1.mean()
        b = p2.astype(np.float32) - p2.mean()
        num = np.sum(a * b)
        den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        return num / den if den > 1e-10 else 0.0

    def cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.size == 0 or v2.size == 0 or v1.shape != v2.shape:
            return 0.0
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 1e-10 else 0.0

    # ------------------------------------------------------------------
    # 4. Pair-wise matching with Morlet + NCC
    # ------------------------------------------------------------------
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

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        knn = flann.knnMatch(desA, desB, k=2)

        good = [m for m, n in knn if len([m, n]) == 2 and m.distance < ratio * n.distance]

        refined = []
        for m in good:
            ptA = kpA[m.queryIdx].pt
            ptB = kpB[m.trainIdx].pt

            # === REMOVE ALL HARD EPIPOLAR COORDINATE CHECKS ===
            # For diagonal baselines (L-F, R-F), they are meaningless.
            # Let RANSAC + Morlet + NCC do the filtering.

            patchA = self.scattering.extract_patch(grayA, ptA)
            patchB = self.scattering.extract_patch(grayB, ptB)
            if patchA is None or patchB is None:
                continue

            if self.ncc(patchA, patchB) < ncc_thr:
                continue

            sA = self.scattering.compute_scattering_coefficients(patchA)
            sB = self.scattering.compute_scattering_coefficients(patchB)
            if self.cosine(sA, sB) < scat_thr:
                continue

            refined.append(m)

        # === RANSAC on Fundamental Matrix (this is the REAL epipolar filter) ===
        if use_ransac and len(refined) >= 8:
            ptsA = np.float32([kpA[m.queryIdx].pt for m in refined])
            ptsB = np.float32([kpB[m.trainIdx].pt for m in refined])
            F, mask = cv2.findFundamentalMat(
                ptsA, ptsB, cv2.FM_RANSAC,
                ransacReprojThreshold=ransac_thr,
                confidence=0.99
            )
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

        self.left_kpts,  self.right_kpts,  self.matches_lr = \
            self._match_pair(imgL, imgR, "left", "right", ratio_test, ncc_thr, scat_thr,
                             use_ransac, ransac_thresh)

        self.left_kpts, self.front_kpts, self.matches_lf = \
            self._match_pair(imgL, imgF, "left", "front", ratio_test, ncc_thr, scat_thr,
                             use_ransac, ransac_thresh)

        self.right_kpts, self.front_kpts, self.matches_rf = \
            self._match_pair(imgR, imgF, "right", "front", ratio_test, ncc_thr, scat_thr,
                             use_ransac, ransac_thresh)

    # ------------------------------------------------------------------
    # 5. Build consistent triplets
    # ------------------------------------------------------------------
    def build_triplets(self):
        lr = {m.queryIdx: m.trainIdx for m in self.matches_lr}
        lf = {m.queryIdx: m.trainIdx for m in self.matches_lf}
        rf = {m.queryIdx: m.trainIdx for m in self.matches_rf}

        triplets = []
        for l_idx in lr:
            r_idx = lr[l_idx]
            f_idx_l = lf.get(l_idx)
            if f_idx_l is None:
                continue
            f_idx_r = rf.get(r_idx)
            if f_idx_r == f_idx_l:
                triplets.append((l_idx, r_idx, f_idx_l))

        self.triplets = triplets
        print(f"Consistent triplets : {len(triplets)}")
        return triplets

    # ------------------------------------------------------------------
    # 6. (Optional) Estimate projection matrices from pairs
    # ------------------------------------------------------------------
    def estimate_projection_matrices(self):
        # ---- L-R ----
        ptsL = np.float32([self.left_kpts[m.queryIdx].pt for m in self.matches_lr])
        ptsR = np.float32([self.right_kpts[m.trainIdx].pt for m in self.matches_lr])
        R_lr, t_lr = self._estimate_pose(ptsL, ptsR, self.baselines["lr"])

        # ---- L-F ----
        ptsL = np.float32([self.left_kpts[m.queryIdx].pt for m in self.matches_lf])
        ptsF = np.float32([self.front_kpts[m.trainIdx].pt for m in self.matches_lf])
        R_lf, t_lf = self._estimate_pose(ptsL, ptsF, self.baselines["lf"])

        self.P_left_est  = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P_right_est = self.K @ np.hstack((R_lr, t_lr.reshape(3, 1)))
        self.P_front_est = self.K @ np.hstack((R_lf, t_lf.reshape(3, 1)))

    def _estimate_pose(self, ptsA, ptsB, baseline):
        """
        Estimate relative pose between two views using pixel coordinates.
        ptsA, ptsB : Nx2 float32 arrays of pixel coordinates (not normalized).
        baseline : known baseline length (same units as camera translations)
        Returns R (3x3) and t (3,) scaled to baseline.
        """
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Use raw pixel coords with focal/pp parameters
        E, mask = cv2.findEssentialMat(
            ptsA, ptsB, focal=float(fx), pp=(float(cx), float(cy)),
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            raise RuntimeError("Essential matrix estimation failed")

        # Enforce internal conditioning (optional but helps)
        U, S, Vt = np.linalg.svd(E)
        sigma = (S[0] + S[1]) / 2.0
        E_cond = U @ np.diag([sigma, sigma, 0.0]) @ Vt

        # recoverPose expects pixel coordinates if focal/pp passed
        _, R, t, mask_rec = cv2.recoverPose(E_cond, ptsA, ptsB, cameraMatrix=self.K)

        t = t.flatten()
        # scale t to the known baseline (direction preserved)
        t = t / (np.linalg.norm(t) + 1e-12) * baseline

        # --- CHEIRALITY CHECK: ensure majority of triangulated points are in front ---
        # triangulate a small subset of inlier points to check Z sign
        inlier_idx = np.where(mask_rec.ravel() == 1)[0]
        if inlier_idx.size > 5:
            sample_idx = inlier_idx[:min(50, inlier_idx.size)]
            ptsA_s = ptsA[sample_idx].T  # 2xM
            ptsB_s = ptsB[sample_idx].T  # 2xM
            P0 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P1 = self.K @ np.hstack((R, t.reshape(3, 1)))
            X4 = cv2.triangulatePoints(P0, P1, ptsA_s.astype(np.float32), ptsB_s.astype(np.float32))
            Zs = (X4[2, :] / (X4[3, :] + 1e-12))
            pos = np.sum(Zs > 1e-6)
            neg = np.sum(Zs < -1e-6)
            # If majority negative, flip sign of t (and optionally R)
            if neg > pos:
                t = -t
                # Flipping t alone is usually sufficient; flipping R would be incorrect.
        return R, t


    # ------------------------------------------------------------------
    # 7. Depth from two triangulations + averaging
    # ------------------------------------------------------------------
    def compute_depth(self, use_estimated: bool = False):
        if not self.triplets:
            print("No triplets – depth computation skipped")
            return np.array([])

        P_L = self.P_left_est if use_estimated and self.P_left_est is not None else self.P_left
        P_R = self.P_right_est if use_estimated and self.P_right_est is not None else self.P_right
        P_F = self.P_front_est if use_estimated and self.P_front_est is not None else self.P_front

        depths_lr = []
        depths_lf = []
        depths_avg = []

        for l_idx, r_idx, f_idx in self.triplets:
            ptL = self.left_kpts[l_idx].pt
            ptR = self.right_kpts[r_idx].pt
            ptF = self.front_kpts[f_idx].pt

            # ---- L-R ----
            X4 = cv2.triangulatePoints(
                P_L, P_R,
                np.array([[ptL[0]], [ptL[1]]], dtype=np.float32),
                np.array([[ptR[0]], [ptR[1]]], dtype=np.float32)
            )
            X = X4[:3, 0] / (X4[3, 0] + 1e-12)
            if X[2] < 0:
                X = -X
            depths_lr.append(X[2])

            # ---- L-F ----
            X4 = cv2.triangulatePoints(
                P_L, P_F,
                np.array([[ptL[0]], [ptL[1]]], dtype=np.float32),
                np.array([[ptF[0]], [ptF[1]]], dtype=np.float32)
            )
            X = X4[:3, 0] / (X4[3, 0] + 1e-12)
            if X[2] < 0:
                X = -X
            depths_lf.append(X[2])

            # ---- geometric mean ----
            depths_avg.append(np.sqrt(depths_lr[-1] * depths_lf[-1]))

        depths_lr = np.array(depths_lr)
        depths_lf = np.array(depths_lf)
        depths_avg = np.array(depths_avg)

        print("\nDEPTH STATISTICS")
        print(f"  L-R   : {depths_lr.mean():.3f} ± {depths_lr.std():.3f}")
        print(f"  L-F   : {depths_lf.mean():.3f} ± {depths_lf.std():.3f}")
        print(f"  AVG   : {depths_avg.mean():.3f} ± {depths_avg.std():.3f}")

        return depths_avg

    # ------------------------------------------------------------------
    # 8. GT comparison (averaged depth)
    # ------------------------------------------------------------------
    def compare_with_gt(self, gt_file: str, use_estimated: bool = False):
        with h5py.File(gt_file, "r") as f:
            gt_map = f["depth_left"][()]
            if gt_map.ndim >= 3 and gt_map.shape[0] == 1:
                gt_map = gt_map[0]

        depths = self.compute_depth(use_estimated=use_estimated)
        if depths.size == 0:
            return

        gt_vals = []
        for l_idx, _, _ in self.triplets:
            pt = self.left_kpts[l_idx].pt
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < gt_map.shape[1] and 0 <= y < gt_map.shape[0]):
                continue
            d = float(gt_map[y, x])
            if 0.1 < d < 5000:
                gt_vals.append(d)

        n = min(len(depths), len(gt_vals))
        depths = depths[:n]
        gt_vals = np.array(gt_vals[:n])

        err = depths - gt_vals
        mask = np.abs(err - err.mean()) / (err.std() + 1e-12) < 3
        err = err[mask]

        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))

        print("\n=== TRINOCULAR DEPTH vs GT ===")
        print(f"Points   : {len(err)}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"MAE      : {mae:.4f}")
        print("================================\n")

        return depths, gt_vals

    # ------------------------------------------------------------------
    # 9. One-call runner
    # ------------------------------------------------------------------
    def run(self, gt_file: str,
            ratio_test: float = 0.7,
            use_ransac: bool = True,
            ransac_thr: float = 1.0,
            ncc_thr: float = 0.7,
            scat_thr: float = 0.85,
            estimate_pose: bool = True,
            use_estimated_for_depth: bool = False):
        print("=" * 60)
        print("TRINOCULAR MORLET PIPELINE")
        print("=" * 60)

        self.load_hdf_file()
        self.load_projection_matrices()
        self.detect_and_match(ratio_test, use_ransac, ransac_thr, ncc_thr, scat_thr)
        self.build_triplets()

        if estimate_pose:
            self.estimate_projection_matrices()

        self.compare_with_gt(gt_file, use_estimated=use_estimated_for_depth)

        print("PIPELINE FINISHED")
        print("=" * 60)


# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
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
            use_estimated_for_depth=True)   # set True to use estimated matrices