"""
features.py
===========
Convert (17, 3) keypoints into a feature vector for the GRU.

Feature layout (54 dims):
    - normalized coords  : 17 * 2 = 34
    - inter-frame speed  : 17
    - trunk_angle        :  1   (spine tilt from vertical, normalised to [0,1])
    - body_aspect_ratio  :  1   (log(1 + bbox_w/bbox_h); large when lying flat)
    - trunk_angle_delta  :  1   (frame-to-frame change in trunk_angle)
    -----------------------------
    total                : 54

Why add geometric features vs original 51:
    - trunk_angle is the single most discriminative fall signal.
      A standing person has angle~0; a fallen person has angle~1.
      The GRU can *derive* this from raw coords, but giving it explicitly
      speeds up learning and reduces the amount of training data needed.
    - body_aspect_ratio captures the same "horizontal body" signal from a
      different axis (width >> height when lying flat).
    - trunk_angle_delta captures the *rate of falling* (rapid tilt change).

Design notes:
    - Normalize by torso scale, not absolute pixels, so the same pose at
      different distances/sizes yields the same features.
    - Speed depends on the previous frame, so the extractor is *stateful*.
      Call reset() before processing a new video.
    - We do NOT zero out low-confidence keypoints. Empirically that
      introduces zero-spikes that hurt the GRU.
"""

import numpy as np


FEATURE_DIM      = 54   # full feature set (new training)
FEATURE_DIM_COMPAT = 51   # original feature set (compatible with fall_gru_v2.pth)

# COCO-17 indices used by geometric features
_L_SHOULDER = 5
_R_SHOULDER = 6
_L_HIP      = 11
_R_HIP      = 12


class KeypointFeatureExtractor(object):
    """
    Convert (17, 3) keypoints frame-by-frame into a (54,) feature vector.

    Usage:
        ext = KeypointFeatureExtractor()
        ext.reset()                    # call before each new video
        for kpts in keypoint_seq:      # kpts: (17, 3)
            feat = ext.extract(kpts)   # feat: (54,)

    Or process a whole sequence at once:
        feats = ext.extract_sequence(keypoint_seq)   # (T,17,3) -> (T,54)
    """

    def __init__(self, smooth_alpha=0.0, extra_features=True):
        """
        smooth_alpha   : EMA smoothing factor for keypoint coords in [0, 1).
                         0 disables smoothing. Typical values: 0.1~0.3.
        extra_features : if True, produce 54-dim output (trunk_angle + aspect_ratio
                         + trunk_angle_delta appended). Set False to keep the
                         original 51-dim output for backward compat with old models.
        """
        if smooth_alpha < 0.0 or smooth_alpha >= 1.0:
            raise ValueError("smooth_alpha must be in [0, 1).")
        self.smooth_alpha    = float(smooth_alpha)
        self.extra_features  = bool(extra_features)
        self.prev_coords     = None
        self.ema_coords      = None
        self.ref_center      = None
        self.ref_scale       = None
        self.prev_trunk_angle = None   # for trunk_angle_delta

    @property
    def feature_dim(self):
        return FEATURE_DIM if self.extra_features else FEATURE_DIM_COMPAT

    def reset(self):
        """Call before processing a new video."""
        self.prev_coords      = None
        self.ema_coords       = None
        self.ref_center       = None
        self.ref_scale        = None
        self.prev_trunk_angle = None

    def _smooth_coords(self, coords):
        if self.smooth_alpha <= 0.0:
            return coords
        if self.ema_coords is None:
            self.ema_coords = coords.copy()
        else:
            a = self.smooth_alpha
            self.ema_coords = (1.0 - a) * self.ema_coords + a * coords
        return self.ema_coords

    def _compute_reference(self, coords, scores):
        """
        Initialize or update the normalization reference (center, scale).

        Strategy:
            - First call: estimate from current frame's valid keypoints.
            - Subsequent: exponential moving average to smooth jitter.
            - All-invalid frame: keep previous reference.
        """
        valid = scores > 0.2
        if not np.any(valid):
            if self.ref_center is None:
                self.ref_center = np.array([0.0, 0.0], dtype=np.float32)
                self.ref_scale  = 1.0
            return

        cx    = coords[valid, 0].mean()
        cy    = coords[valid, 1].mean()
        scale = max(np.ptp(coords[valid, 0]),
                    np.ptp(coords[valid, 1]),
                    1.0)

        if self.ref_center is None:
            self.ref_center = np.array([cx, cy], dtype=np.float32)
            self.ref_scale  = float(scale)
        else:
            alpha = 0.1
            self.ref_center[0] = (1 - alpha) * self.ref_center[0] + alpha * cx
            self.ref_center[1] = (1 - alpha) * self.ref_center[1] + alpha * cy
            self.ref_scale     = (1 - alpha) * self.ref_scale + alpha * float(scale)

    def _trunk_angle_norm(self, coords, scores):
        """
        Spine tilt from vertical, normalised to [0, 1].
          0  = perfectly upright
          1  = fully horizontal (lying flat)

        Uses shoulder-centre -> hip-centre vector. Falls back to 0 if
        either landmark is low-confidence.
        """
        l_sh_ok = scores[_L_SHOULDER] > 0.15
        r_sh_ok = scores[_R_SHOULDER] > 0.15
        l_hp_ok = scores[_L_HIP]      > 0.15
        r_hp_ok = scores[_R_HIP]      > 0.15

        if (l_sh_ok or r_sh_ok) and (l_hp_ok or r_hp_ok):
            if l_sh_ok and r_sh_ok:
                sh = (coords[_L_SHOULDER] + coords[_R_SHOULDER]) * 0.5
            elif l_sh_ok:
                sh = coords[_L_SHOULDER]
            else:
                sh = coords[_R_SHOULDER]

            if l_hp_ok and r_hp_ok:
                hp = (coords[_L_HIP] + coords[_R_HIP]) * 0.5
            elif l_hp_ok:
                hp = coords[_L_HIP]
            else:
                hp = coords[_R_HIP]

            vec   = sh - hp                            # points upward when standing
            deg   = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))
            return np.float32(np.clip(deg / 90.0, 0.0, 1.0))
        return np.float32(0.0)

    def _body_aspect_ratio(self, coords, scores):
        """
        log(1 + bbox_width / bbox_height) over valid keypoints.
        Standing: height >> width  -> value near 0.
        Lying:    width  >> height -> value large (> 1).
        """
        valid = scores > 0.15
        if valid.sum() < 2:
            return np.float32(0.0)
        v      = coords[valid]
        bbox_h = max(float(np.ptp(v[:, 1])), 1.0)
        bbox_w = max(float(np.ptp(v[:, 0])), 1.0)
        return np.float32(np.log1p(bbox_w / bbox_h))

    def extract(self, keypoints):
        """
        Parameters
        ----------
        keypoints : (17, 3) float32

        Returns
        -------
        features : (54,) float32
                   [0:34]  normalized coords (x0,y0,...,x16,y16)
                   [34:51] inter-frame speed per keypoint
                   [51]    trunk_angle (spine tilt, normalised 0–1)
                   [52]    body_aspect_ratio (log w/h)
                   [53]    trunk_angle_delta (change from previous frame)
        """
        kpts   = np.asarray(keypoints, dtype=np.float32)
        coords = kpts[:, :2]
        scores = kpts[:, 2]

        # Optional temporal smoothing to reduce keypoint jitter.
        coords = self._smooth_coords(coords)

        # Update normalization reference
        self._compute_reference(coords, scores)
        center = self.ref_center
        scale  = max(self.ref_scale, 1.0)

        # 1. Normalized coords (34)
        norm_coords = (coords - center) / scale     # (17, 2)
        coord_feat  = norm_coords.flatten()          # (34,)

        # 2. Inter-frame speed (17)
        if self.prev_coords is not None:
            prev_norm  = (self.prev_coords - center) / scale
            speed      = np.linalg.norm(norm_coords - prev_norm, axis=1)
        else:
            speed = np.zeros(17, dtype=np.float32)
        speed_feat = speed.astype(np.float32)

        self.prev_coords = coords.copy()

        if self.extra_features:
            # 3. Trunk angle (1)
            trunk_angle = self._trunk_angle_norm(coords, scores)

            # 4. Body aspect ratio (1)
            aspect_ratio = self._body_aspect_ratio(coords, scores)

            # 5. Trunk angle delta (1)
            if self.prev_trunk_angle is not None:
                trunk_delta = np.float32(trunk_angle - self.prev_trunk_angle)
            else:
                trunk_delta = np.float32(0.0)
            self.prev_trunk_angle = float(trunk_angle)

            feat = np.concatenate([coord_feat, speed_feat,
                                    [trunk_angle], [aspect_ratio], [trunk_delta]
                                    ]).astype(np.float32)
        else:
            feat = np.concatenate([coord_feat, speed_feat]).astype(np.float32)

        expected = self.feature_dim
        assert feat.shape == (expected,), \
            "feature dim mismatch: got {}, expected {}".format(feat.shape, expected)
        return feat

    def extract_sequence(self, keypoints_seq):
        """
        Process a whole keypoint sequence. Calls reset() automatically.

        Parameters
        ----------
        keypoints_seq : (T, 17, 3) ndarray

        Returns
        -------
        features : (T, feature_dim) ndarray  — 54 or 51 depending on extra_features
        """
        self.reset()
        T   = keypoints_seq.shape[0]
        out = np.zeros((T, self.feature_dim), dtype=np.float32)
        for t in range(T):
            out[t] = self.extract(keypoints_seq[t])
        return out
