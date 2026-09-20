"""
Self-Blended Images (SBI) for CrossFuse v5.

STATUS: CUT -- kept for the record, imported by no current notebook.
Four attempts scored 0.99 AUC on their own held-out self-blend task but
0.39 -> 0.38 -> 0.30 zero-shot on Celeb-DF (consistently BELOW chance).  One
shortcut was found and fixed (global photometric statistics leaking the label)
but zero-shot got worse, so the diagnosis was incomplete.  Superseded by the
CLIP ViT-L/14 + FF++ pretraining in ffpp_v5.py / B1-ffpp-pretrain.ipynb, which
passed its gate (Celeb-DF 0.9186, DFDC 0.8494).  See the negative-results
ledger in CLAUDE.md.

WHY THIS EXISTS
---------------
The v4 visual head was trained on FakeAVCeleb, whose visual fakes are ~97%
Wav2Lip -- a MOUTH-REGION lip-sync generator.  DFDC and Celeb-DF fakes are
FULL-FACE swaps.  A detector trained on the former learns mouth artifacts and
transfers badly (v3.2 measured 0.788 val -> 0.627 DFDC, a 0.161 gap).

SBI (Shiohara & Yamasaki, CVPR 2022, arXiv:2204.08376) sidesteps the problem
by not training on anyone's fakes at all.  It synthesises fakes from REAL
video only, by blending a mildly perturbed copy of a face back onto itself
through a landmark-derived mask.  The resulting artifacts -- blending
boundary, resolution mismatch, colour/statistical inconsistency -- are the
artifacts that EVERY face-swap pipeline leaves, rather than the fingerprint
of one generator.  Trained on FF++ real videos with EfficientNet-B4, the
paper reports cross-dataset AUC of 93.18 (Celeb-DF), 86.15 (DFDCP), 84.83
(FFIW) and 72.42 (DFDC).

The encoder trained here is loaded into CrossFuseModelV5.backbone by
`load_visual_encoder`, so the multimodal stage starts from forensic features
instead of ImageNet ones.

RECIPE (following the paper's Fig. 3 and the reference implementation at
github.com/mapooon/SelfBlendedImages)

    I  --STG-->  (I_s, I_t)          source/target transforms
    landmarks --MG--> M              convex hull, deformed, blurred
    I_sb = I_s * M + I_t * (1 - M)

The source branch additionally gets a small affine jitter, which is what
creates the landmark/resolution mismatch that a real swap produces.

LANDMARKS
---------
The reference implementation uses dlib's 81-point predictor. Three tiers are
tried in order, each falling back to the next only if the previous is
unavailable or fails on a given frame:

  1. MediaPipe FaceMesh (468 points) -- best quality, but its pip install has
     its own dependency chain (protobuf, attrs, flatbuffers, ...) that can
     fail to resolve on a given image/environment.
  2. facenet_pytorch's MTCNN 5-point landmarks (eyes, nose, mouth corners),
     extrapolated into a rough face-shaped polygon. Coarser than MediaPipe,
     but real, image-specific geometry rather than a fixed shape -- and MTCNN
     is already a hard dependency of this pipeline (used for face detection
     in Stage A/A2), so this tier has no extra install risk at all.
  3. A parametric ellipse, jittered per-vertex. This is a LAST resort: an
     early run that fell through to this tier scored 0.99 AUC on its own
     held-out self-blend task but 0.39 (worse than chance) zero-shot on
     Celeb-DF -- the model had learned the ellipse's specific geometry as a
     shortcut rather than anything resembling a real forgery artifact. Do
     not trust an encoder trained on this tier; `landmark_backend()` and the
     printed per-tier usage counts exist specifically to catch it.
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from crossfuse_v5 import build_backbone, build_detector, safe_auc

__all__ = [
    "landmark_backend", "get_face_landmarks", "self_blend_image",
    "SBIFrameDataset", "SBIClassifier", "train_sbi_encoder",
    "score_frames_video_level", "landmark_usage_report", "TIER_NAMES",
    "precompute_landmarks", "matched_real_image",
]


# =====================================================================
# Landmarks
# =====================================================================

_MP_MESH = None
_MP_TRIED = False


def _mediapipe_mesh():
    global _MP_MESH, _MP_TRIED
    if _MP_TRIED:
        return _MP_MESH
    _MP_TRIED = True
    try:
        import mediapipe as mp
        _MP_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.3)
    except Exception as e:
        print(f"[sbi] MediaPipe unavailable ({type(e).__name__}); "
              f"falling back to a parametric face hull. "
              f"`pip install mediapipe` for the full SBI recipe.")
        _MP_MESH = None
    return _MP_MESH


def landmark_backend():
    return "mediapipe" if _mediapipe_mesh() is not None else "degraded (see landmark_usage_report)"


_MTCNN_DET = None
_MTCNN_TRIED = False
_LANDMARK_STATS = {"mediapipe": 0, "mtcnn5": 0, "parametric": 0}


def landmark_usage_report():
    """Per-tier landmark counts for THIS process. Print this at the end of
    training -- a run that's mostly 'parametric' produced an encoder that
    should not be trusted; see the LANDMARKS section in the module docstring
    for why (0.99 self-blend AUC / 0.39 zero-shot Celeb-DF AUC on an earlier
    all-parametric run)."""
    total = sum(_LANDMARK_STATS.values())
    return dict(_LANDMARK_STATS, total=total)


def _mtcnn_detector():
    """Lazy, per-process MTCNN instance reused across every landmark call.
    Reuses crossfuse_v5.build_detector rather than constructing MTCNN
    directly, so this is exactly the same detector already proven to work in
    Stage A/A2 -- no new dependency, no new failure mode.

    ALWAYS built on CPU. This function is only ever called from
    SBIFrameDataset.__getitem__, which runs inside DataLoader WORKER
    processes when num_workers > 0. Those are forked from a parent that has
    already initialized a CUDA context (the model itself lives on the GPU),
    and re-initializing CUDA in a forked child is unsupported by PyTorch --
    it raises 'Cannot re-initialize CUDA in forked subprocess' rather than
    silently working. A single small face crop is cheap enough to detect on
    CPU that this is not a training bottleneck, and it's only invoked on the
    ~50% of samples that draw the self-blend branch."""
    global _MTCNN_DET, _MTCNN_TRIED
    if _MTCNN_TRIED:
        return _MTCNN_DET
    _MTCNN_TRIED = True
    try:
        _MTCNN_DET = build_detector("cpu")
    except Exception as e:
        print(f"[sbi] MTCNN landmark fallback unavailable ({type(e).__name__}: {e})")
        _MTCNN_DET = None
    return _MTCNN_DET


def _mtcnn_5pt_hull(img_rgb):
    """5-point landmarks (left eye, right eye, nose, mouth-left, mouth-right)
    from MTCNN, extrapolated into a rough face-shaped octagon. Coarser than a
    real jawline, but the extrapolation is driven by THIS face's actual eye
    spacing and eye-to-mouth distance, not a fixed shape -- it moves, scales,
    and rotates with the real face. Returns None if no face is detected."""
    det = _mtcnn_detector()
    if det is None:
        return None
    try:
        boxes, probs, landmarks = det.detect(Image.fromarray(img_rgb), landmarks=True)
    except Exception:
        return None
    if boxes is None or landmarks is None or len(landmarks) == 0:
        return None

    idx = int(np.argmax(probs)) if probs is not None and len(probs) > 1 else 0
    return _hull_from_5pt(*landmarks[idx].astype(np.float32))


def _hull_from_5pt(left_eye, right_eye, nose, mouth_l, mouth_r):
    """Shared 5-point -> face-octagon geometry, used by both the in-worker
    path (_mtcnn_5pt_hull) and the up-front GPU path (precompute_landmarks),
    so the two can never drift apart."""
    left_eye = np.asarray(left_eye, dtype=np.float32)
    right_eye = np.asarray(right_eye, dtype=np.float32)
    mouth_l = np.asarray(mouth_l, dtype=np.float32)
    mouth_r = np.asarray(mouth_r, dtype=np.float32)

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_l + mouth_r) / 2.0
    axis = eye_mid - mouth_mid                       # points "up" the face
    axis_len = float(np.linalg.norm(axis)) + 1e-6
    up = axis / axis_len
    side = np.array([-up[1], up[0]], dtype=np.float32)
    face_h = axis_len * 2.6                          # eye-to-mouth ~ 38% of eye-to-chin
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    face_w = eye_dist * 2.3

    return np.asarray([
        eye_mid + up * face_h * 0.55,                        # forehead
        eye_mid + up * face_h * 0.30 + side * face_w * 0.55, # temple (R)
        left_eye - side * face_w * 0.10,                     # cheekbone (R)
        mouth_l - side * face_w * 0.05,                       # jaw (R)
        mouth_mid - up * face_h * 0.55,                       # chin
        mouth_r + side * face_w * 0.05,                       # jaw (L)
        right_eye + side * face_w * 0.10,                     # cheekbone (L)
        eye_mid + up * face_h * 0.30 - side * face_w * 0.55, # temple (L)
    ], dtype=np.float32)


def _parametric_hull(h, w, rng):
    """Elliptical polygon standing in for a jawline hull, with per-vertex
    radial jitter so the mask boundary is not a perfect ellipse (a perfectly
    smooth boundary is itself a learnable shortcut)."""
    cy, cx = h * rng.uniform(0.50, 0.56), w * 0.5
    ry, rx = h * rng.uniform(0.33, 0.40), w * rng.uniform(0.28, 0.35)
    pts = []
    for k in range(24):
        a = 2 * np.pi * k / 24
        j = rng.uniform(0.90, 1.10)
        pts.append([cx + rx * np.cos(a) * j, cy + ry * np.sin(a) * j])
    return np.asarray(pts, dtype=np.float32)


TIER_NAMES = ("mediapipe", "mtcnn5", "parametric")  # index matches the tier_idx below


def _get_face_landmarks_with_tier(img_rgb, rng=None):
    """(landmarks, tier_name). `_LANDMARK_STATS` is updated too, but that
    dict is PROCESS-LOCAL -- with DataLoader num_workers > 0, each worker has
    its own private copy and increments there never reach the main process
    that would print landmark_usage_report(). Do not trust that report under
    multi-worker training; SBIFrameDataset instead returns the tier per
    sample through the normal (cross-process-safe) DataLoader return path,
    which is what train_sbi_encoder aggregates and prints."""
    rng = rng or random.Random(0)
    mesh = _mediapipe_mesh()
    if mesh is not None:
        try:
            res = mesh.process(np.ascontiguousarray(img_rgb))
            if res.multi_face_landmarks:
                h, w = img_rgb.shape[:2]
                lm = res.multi_face_landmarks[0].landmark
                _LANDMARK_STATS["mediapipe"] += 1
                return np.asarray([[p.x * w, p.y * h] for p in lm], dtype=np.float32), "mediapipe"
        except Exception:
            pass

    hull = _mtcnn_5pt_hull(img_rgb)
    if hull is not None:
        _LANDMARK_STATS["mtcnn5"] += 1
        return hull, "mtcnn5"

    _LANDMARK_STATS["parametric"] += 1
    return _parametric_hull(img_rgb.shape[0], img_rgb.shape[1], rng), "parametric"


def get_face_landmarks(img_rgb, rng=None):
    """(N, 2) float32 landmark array in pixel coordinates, tried in order:
    MediaPipe (468 pts) -> MTCNN 5-point extrapolated hull -> parametric
    ellipse.  `img_rgb` is HxWx3 uint8 RGB."""
    lm, _tier = _get_face_landmarks_with_tier(img_rgb, rng)
    return lm


# =====================================================================
# Source / target transforms  (the "STG" of the paper)
# =====================================================================

def _rgb_shift(img, rng, limit=20):
    out = img.astype(np.int16)
    for c in range(3):
        out[..., c] += rng.randint(-limit, limit)
    return np.clip(out, 0, 255).astype(np.uint8)


def _hue_sat_value(img, rng, h_lim=10, s_lim=30, v_lim=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-h_lim, h_lim)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-s_lim, s_lim), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-v_lim, v_lim), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _brightness_contrast(img, rng, b=0.15, c=0.15):
    a = 1.0 + rng.uniform(-c, c)
    beta = 255.0 * rng.uniform(-b, b)
    return np.clip(img.astype(np.float32) * a + beta, 0, 255).astype(np.uint8)


def _random_downscale(img, rng, lo=1.5, hi=4):
    """Resolution mismatch: the single most reliable swap artifact, because a
    generator's output is almost never native-resolution with its target.

    Range moderated from the original (2, 5): at 5x on a 256px crop the face
    drops to ~51px, which made blur so overwhelming a cue that the model
    keyed on it globally rather than on the source/target resolution STEP at
    the mask boundary, which is the artifact that actually transfers."""
    h, w = img.shape[:2]
    f = rng.uniform(lo, hi)
    small = cv2.resize(img, (max(4, int(w / f)), max(4, int(h / f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=rng.choice(
        [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_CUBIC]))


def _sharpen(img, rng):
    blur = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    a = rng.uniform(0.3, 1.0)
    return np.clip(img.astype(np.float32) * (1 + a) - blur.astype(np.float32) * a,
                   0, 255).astype(np.uint8)


def _apply_stg(img, rng, p=0.5):
    """One independent draw over the transform bank."""
    out = img
    if rng.random() < p:
        out = _rgb_shift(out, rng)
    if rng.random() < p:
        out = _hue_sat_value(out, rng)
    if rng.random() < p:
        out = _brightness_contrast(out, rng)
    if rng.random() < p:
        out = _random_downscale(out, rng)
    if rng.random() < 0.3:
        out = _sharpen(out, rng)
    return out


# =====================================================================
# Mask generation  (the "MG" of the paper)
# =====================================================================

def _elastic_deform_mask(mask, rng, grid=8, sigma=None):
    """Low-frequency random displacement field applied to the mask, so its
    boundary does not trace the landmark polygon exactly.  A mask that
    follows landmarks perfectly teaches the network to find landmarks, not
    blending."""
    h, w = mask.shape[:2]
    sigma = sigma or max(h, w) * rng.uniform(0.02, 0.05)
    dx = cv2.resize(np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
                    .randn(grid, grid).astype(np.float32), (w, h)) * sigma
    dy = cv2.resize(np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
                    .randn(grid, grid).astype(np.float32), (w, h)) * sigma
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, xx + dx, yy + dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def _make_mask(landmarks, h, w, rng):
    """Convex hull of the landmarks -> deform -> erode/dilate -> feather."""
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)

    k = rng.randint(1, max(2, int(min(h, w) * 0.06)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.erode(mask, kern) if rng.random() < 0.5 else cv2.dilate(mask, kern)

    mask = _elastic_deform_mask(mask, rng)

    blur = int(min(h, w) * rng.uniform(0.02, 0.10)) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask, 0.0, 1.0)


def matched_real_image(img_rgb, rng=None):
    """The REAL-class counterpart to self_blend_image.

    Built from the same two independently STG-transformed copies and the same
    ratio ladder, but mixed with a SPATIALLY UNIFORM weight instead of a
    face-shaped mask. The two classes therefore share their construction and
    their global photometric distribution exactly, and differ only in whether
    the mixing weight has a spatial DISCONTINUITY -- which is the artifact
    the detector is supposed to learn.

    Measured global-statistic separability (worst |AUC - 0.5| over
    brightness / contrast / colour / blur / high-frequency energy, synthetic
    faces, n=144 per class):

        old construction, pristine real ... 0.109   <- shipped, scored 0.38 zero-shot
        fixed blend, single-STG real ..... 0.035
        this (uniform-alpha real) ........ 0.022

    The old row leaked worst through high-frequency energy (AUC 0.391): its
    fakes were systematically blurrier than its reals, so the model learned
    "blurry => fake". That rule inverts on Celeb-DF, whose reals are soft
    YouTube uploads while its v2 fakes are deliberately high-resolution --
    which is why the failure was a stable sub-chance 0.38 rather than a
    noisy 0.5.
    """
    rng = rng or random.Random()
    source = _apply_stg(img_rgb, rng)
    target = _apply_stg(img_rgb, rng)
    a = rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    blended = target.astype(np.float32) * (1 - a) + source.astype(np.float32) * a
    return np.clip(blended, 0, 255).astype(np.uint8)


def self_blend_image(img_rgb, landmarks=None, rng=None):
    """One self-blended fake from one real face crop.

    Returns (blended uint8 HxWx3, mask float32 HxW).  The mask is returned so
    a caller can visualise or localise; training only needs the image.
    """
    rng = rng or random.Random()
    h, w = img_rgb.shape[:2]
    if landmarks is None:
        landmarks = get_face_landmarks(img_rgb, rng)

    source = _apply_stg(img_rgb, rng)
    target = _apply_stg(img_rgb, rng)

    # Small affine on the source only: shifts the face a few pixels relative
    # to the mask, reproducing the landmark misalignment of a real swap.
    if rng.random() < 0.8:
        M = np.float32([[1, 0, rng.uniform(-0.02, 0.02) * w],
                        [0, 1, rng.uniform(-0.02, 0.02) * h]])
        s = rng.uniform(0.96, 1.04)
        M[:, :2] *= s
        M[0, 2] += (1 - s) * w / 2
        M[1, 2] += (1 - s) * h / 2
        source = cv2.warpAffine(source, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    mask = _make_mask(landmarks, h, w, rng)
    # Discrete blend ratios, weighted toward full opacity as in the paper --
    # partial ratios produce the subtle, hard cases that drive generalization.
    ratio = rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    m = (mask * ratio)[..., None]

    # I_sb = I_s * M + I_t * (1 - M), the paper's Eq. 1. BOTH branches are
    # independently STG-transformed, so the fake's only distinguishing
    # property is the DISCONTINUITY where the two meet.
    #
    # An earlier version blended `source` onto the raw, untransformed
    # img_rgb and only substituted `target` where the mask was exactly 0.
    # That made every fake globally transformed while every real stayed
    # pristine, so the classifier could separate the classes on global
    # photometric statistics without ever looking at the boundary. Because
    # _random_downscale dominates that transform bank, the rule it actually
    # learned was "blurry => fake, crisp => real" -- which INVERTS on
    # Celeb-DF, whose reals are soft YouTube uploads and whose v2 fakes are
    # deliberately high-resolution and colour-corrected. Measured effect:
    # 0.99 self-blend AUC but 0.38 zero-shot Celeb-DF AUC, i.e. a strong
    # signal pointing the wrong way (1 - 0.38 = 0.62), stable across runs
    # and unaffected by landmark quality.
    #
    # The symmetric counterpart to this fix lives in
    # SBIFrameDataset.__getitem__, which must apply _apply_stg to the REAL
    # branch too. Both halves are required; either alone leaves the leak.
    blended = target.astype(np.float32) * (1 - m) + source.astype(np.float32) * m
    return np.clip(blended, 0, 255).astype(np.uint8), mask


# =====================================================================
# Dataset
# =====================================================================

def precompute_landmarks(index, batch_size=64, device=None, verbose=True):
    """Detect landmarks for every frame ONCE, in the main process, on GPU.

    Returns {(path, frame_idx): (landmarks, tier_name)} to hand to
    SBIFrameDataset(landmarks=...).

    Two problems this solves, both visible in a real run:
      * SPEED. Doing MTCNN per-sample inside DataLoader workers pegged the
        CPU (~100%) while GPU0 sat at 30% and GPU1 at 0% -- training was
        CPU-bound on face detection, so adding GPUs did nothing. Detection
        is a fixed one-time cost over the frame set; paying it once up front
        and caching removes it from the training loop entirely.
      * CUDA-IN-FORK. Worker processes cannot initialise CUDA after being
        forked from a CUDA-using parent, which is why the in-worker detector
        is pinned to CPU. Running here, in the main process before any
        workers spawn, lets detection use the GPU at full speed.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        det = build_detector(str(device))
    except Exception as e:
        print(f"[sbi] precompute_landmarks: detector unavailable "
              f"({type(e).__name__}: {e}); falling back to per-sample detection.")
        return None

    from collections import Counter
    out, tiers = {}, Counter()
    arr_cache = {}
    mesh = _mediapipe_mesh()

    for n, (path, fi) in enumerate(index):
        arr = arr_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            if len(arr_cache) > 64:
                arr_cache.clear()
            arr_cache[path] = arr
        img = np.array(arr[fi])

        lm = tier = None
        if mesh is not None:
            try:
                res = mesh.process(np.ascontiguousarray(img))
                if res.multi_face_landmarks:
                    h, w = img.shape[:2]
                    pts = res.multi_face_landmarks[0].landmark
                    lm = np.asarray([[p.x * w, p.y * h] for p in pts], dtype=np.float32)
                    tier = "mediapipe"
            except Exception:
                pass
        if lm is None:
            try:
                boxes, probs, landmarks5 = det.detect(Image.fromarray(img), landmarks=True)
            except Exception:
                boxes = landmarks5 = None
            if boxes is not None and landmarks5 is not None and len(landmarks5):
                idx = int(np.argmax(probs)) if probs is not None and len(probs) > 1 else 0
                lm = _hull_from_5pt(*landmarks5[idx].astype(np.float32))
                tier = "mtcnn5"
        if lm is None:
            lm = _parametric_hull(img.shape[0], img.shape[1], random.Random(n))
            tier = "parametric"

        out[(path, fi)] = (lm, tier)
        tiers[tier] += 1
        if verbose and n and n % 1000 == 0:
            print(f"[sbi] precomputed landmarks {n}/{len(index)} ...")

    if verbose:
        print(f"[sbi] landmark precompute complete on {len(index)} frames "
              f"(device={device}): {dict(tiers)}")
        if tiers["parametric"] / max(len(index), 1) > 0.05:
            print(f"[sbi] WARNING: {tiers['parametric']} frames fell back to the "
                  f"parametric ellipse -- those samples carry a weaker, "
                  f"shape-generic blend mask.")
    return out


class SBIFrameDataset(Dataset):
    """Frame-level dataset over cached REAL face crops.

    Each item independently becomes either the untouched real frame (label 0)
    or a self-blended fake derived from that SAME frame (label 1).  Because
    both classes come from one source image, identity, pose, lighting and
    background are held constant and the ONLY thing separating the labels is
    the blend -- which is exactly the paired-source condition GenD
    (arXiv:2508.06248) identifies as decisive for suppressing shortcuts.

    `frames` is an (N, S, S, 3) uint8 array or a list of .npy paths whose
    arrays are (T, S, S, 3); the latter is memory-mapped.
    """

    def __init__(self, index, out_size=380, training=True, seed=42,
                 fake_p=0.5, cache_landmarks=True, landmarks=None):
        self.index = index                # list of (npy_path, frame_i)
        self.out_size = out_size
        self.training = training
        self.seed = seed
        self.fake_p = fake_p
        self.cache_landmarks = cache_landmarks
        # Optional {(path, frame_i): (landmarks, tier)} from
        # precompute_landmarks(). When present, no detector runs inside the
        # worker at all -- which is both far faster and sidesteps the
        # CUDA-in-forked-worker restriction entirely.
        self._lm_cache = dict(landmarks) if landmarks else {}
        self._arr_cache = {}

    def __len__(self):
        return len(self.index)

    def _frame(self, path, i):
        arr = self._arr_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            if len(self._arr_cache) > 64:     # bound the open-handle count
                self._arr_cache.clear()
            self._arr_cache[path] = arr
        return np.array(arr[i])

    def __getitem__(self, i):
        path, fi = self.index[i]
        rng = random.Random(self.seed * 7919 + i * 31 +
                            (random.getrandbits(24) if self.training else 0))
        img = self._frame(path, fi)

        # -1 = real/untouched frame, no landmark tier applies. This travels
        # back to the main process through the normal DataLoader collation
        # path -- unlike a global counter dict, that path is guaranteed to
        # survive worker-process boundaries, which is why train_sbi_encoder
        # aggregates tier_idx instead of reading _LANDMARK_STATS directly.
        tier_idx = -1
        make_fake = (rng.random() < self.fake_p) if self.training else (i % 2 == 1)
        if make_fake:
            key = (path, fi)
            cached = self._lm_cache.get(key)
            if cached is None:
                lm, tier = _get_face_landmarks_with_tier(img, rng)
                if self.cache_landmarks and len(self._lm_cache) < 4096:
                    self._lm_cache[key] = (lm, tier)
            else:
                lm, tier = cached
            tier_idx = TIER_NAMES.index(tier)
            img, _ = self_blend_image(img, lm, rng)
            label = 1.0
        else:
            # The REAL branch is built the SAME way as the fake, minus the
            # spatial boundary -- see matched_real_image for the measured
            # justification. Leaving the real class pristine here is what
            # produced the 0.38 (inverted) zero-shot Celeb-DF result.
            img = matched_real_image(img, rng)
            label = 0.0

        if self.training:
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
            if rng.random() < 0.3:
                # Post-blend compression, applied identically to both classes,
                # so JPEG statistics cannot act as a label proxy.
                q = rng.randint(40, 90)
                ok, enc = cv2.imencode(".jpg", img[:, :, ::-1],
                                       [int(cv2.IMWRITE_JPEG_QUALITY), q])
                if ok:
                    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]

        if img.shape[0] != self.out_size:
            img = cv2.resize(img, (self.out_size, self.out_size),
                             interpolation=cv2.INTER_AREA)
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(tier_idx, dtype=torch.int64))


# =====================================================================
# Classifier
# =====================================================================

class SBIClassifier(nn.Module):
    """Plain frame-level binary classifier.

    The backbone is built by the SAME `build_backbone` the multimodal model
    uses, so `backbone.state_dict()` drops straight into
    CrossFuseModelV5.backbone with no key surgery."""

    def __init__(self, backbone="efficientnet_b4", pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone, self.dim = build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.dim, 1))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_uint8):
        x = x_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.img_mean) / self.img_std
        f = self.pool(self.backbone(x)).flatten(1)
        return self.head(f).squeeze(-1)


def train_sbi_encoder(train_index, val_index, out_path, backbone="efficientnet_b4",
                      out_size=380, epochs=8, batch_size=16, lr=1e-4,
                      num_workers=4, seed=42, device=None, log_every=50,
                      precompute_lm=True, multi_gpu=False):
    """Train the SBI encoder and save {'backbone_state', 'meta'} to out_path.

    Only the backbone is saved: the multimodal stage rebuilds its own heads,
    and carrying a stale classifier head across would be misleading.

    `multi_gpu` wraps the model in DataParallel across all visible GPUs and
    scales the batch accordingly. DEFAULT FALSE: DataParallel's master GPU
    gathers every replica's output for loss computation and gradient
    reduction, so it carries more than an even share of memory -- this
    crashed a live 2xT4 run (kernel died) at efficientnet_b4 + res 380 +
    effective batch 32. The sharding still runs correctly when enabled; it
    just needs a smaller effective batch before it's safe to turn back on.
    `precompute_lm` runs face detection once up front on the GPU rather than
    per-sample inside CPU workers -- on a 2xT4 session the un-precomputed
    path was CPU-bound (GPU0 ~30%, GPU1 0%), so this is what actually makes
    the second GPU worth having."""
    from torch.utils.data import DataLoader

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_dp = bool(multi_gpu and n_gpu > 1)
    eff_batch = batch_size * n_gpu if use_dp else batch_size

    lm_map = None
    if precompute_lm:
        print(f"[sbi] precomputing landmarks for "
              f"{len(train_index) + len(val_index)} frames on {device} ...")
        lm_map = precompute_landmarks(list(train_index) + list(val_index), device=device)

    tr_ds = SBIFrameDataset(train_index, out_size, training=True, seed=seed,
                            landmarks=lm_map)
    va_ds = SBIFrameDataset(val_index, out_size, training=False, seed=seed,
                            landmarks=lm_map)
    tr = DataLoader(tr_ds, batch_size=eff_batch, shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=(num_workers > 0))
    va = DataLoader(va_ds, batch_size=eff_batch, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    model = SBIClassifier(backbone, pretrained=True).to(device)
    if use_dp:
        print(f"[sbi] DataParallel across {n_gpu} GPUs "
              f"(per-GPU batch {batch_size}, effective {eff_batch})")
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(tr)))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    crit = nn.BCEWithLogitsLoss()

    print(f"[sbi] backbone={backbone} res={out_size} landmarks={landmark_backend()} "
          f"(this line only reflects the MAIN process; see the tier-usage line "
          f"after epoch 1, which is aggregated from actual DataLoader worker output "
          f"and is the number to trust)")
    print(f"[sbi] train frames={len(tr_ds)} val frames={len(va_ds)} "
          f"steps/epoch={len(tr)}")

    from collections import Counter
    best_auc, best_ep = -1.0, -1
    epoch1_tier_counts = {}
    for ep in range(epochs):
        model.train()
        run, nb = 0.0, 0
        tier_counts = Counter()
        for step, (x, y, tier) in enumerate(tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach())
            nb += 1
            if ep == 0:
                for t in tier.tolist():
                    tier_counts[TIER_NAMES[t] if t >= 0 else "real"] += 1
            if log_every and step % log_every == 0:
                print(f"[sbi] ep{ep + 1} step {step}/{len(tr)} loss {run / max(nb, 1):.4f}")

        if ep == 0:
            epoch1_tier_counts = dict(tier_counts)
            fake_total = sum(v for k, v in epoch1_tier_counts.items() if k != "real")
            print(f"[sbi] landmark tier usage, epoch 1 train pass "
                  f"({fake_total} self-blended samples): {epoch1_tier_counts}")
            parametric_frac = epoch1_tier_counts.get("parametric", 0) / max(fake_total, 1)
            if fake_total > 0 and parametric_frac > 0.05:
                print(f"[sbi] WARNING: {100 * parametric_frac:.1f}% of self-blended "
                      f"samples fell all the way back to the parametric ellipse. An "
                      f"earlier all-parametric run scored 0.99 self-blend AUC but 0.39 "
                      f"(worse than chance) zero-shot Celeb-DF -- do not trust this "
                      f"checkpoint until the Celeb-DF gate cell actually clears.")

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for x, y, _tier in va:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    p = torch.sigmoid(model(x).float())
                probs.extend(p.cpu().numpy().tolist())
                trues.extend(y.numpy().tolist())
        auc = safe_auc(trues, probs)
        star = ""
        if auc > best_auc:
            best_auc, best_ep, star = auc, ep, "  * (New Best)"
            # Unwrap DataParallel before saving: otherwise every key gains a
            # "module." prefix and load_visual_encoder's strict check in
            # crossfuse_v5 rejects the whole checkpoint.
            _bb = (model.module if isinstance(model, nn.DataParallel) else model).backbone
            torch.save({"backbone_state": _bb.state_dict(),
                        "meta": {"backbone": backbone, "out_size": out_size,
                                 "epoch": ep + 1, "val_auc": float(auc),
                                 "landmark_usage": epoch1_tier_counts}}, out_path)
        print(f"[sbi] epoch {ep + 1}/{epochs} loss {run / max(nb, 1):.4f} "
              f"| held-out SBI AUC {auc:.4f}{star}")

    print(f"[sbi] done. best epoch {best_ep + 1}, AUC {best_auc:.4f} -> {out_path}")
    print(f"[sbi] landmark tier usage (measured on the epoch 1 train pass, "
          f"aggregated from actual worker output): {epoch1_tier_counts}")
    print("[sbi] NOTE: this AUC is on SELF-BLENDED validation frames, i.e. the "
          "training distribution. It says the model learned the blending task; "
          "it says nothing about generalization. The gate for that is the "
          "zero-shot Celeb-DF score in NB-B0's last cell.")
    return {"path": out_path, "best_val_auc": float(best_auc), "best_epoch": best_ep + 1,
            "landmark_usage": epoch1_tier_counts}


# =====================================================================
# Video-level scoring
# =====================================================================

@torch.no_grad()
def score_frames_video_level(model, crops_uint8, device=None, batch_size=32,
                             out_size=380):
    """Mean frame probability for one clip -- the video-level aggregation the
    SBI paper uses, and the one every cross-dataset number here reports."""
    device = device or next(model.parameters()).device
    model.eval()
    probs = []
    for i in range(0, len(crops_uint8), batch_size):
        chunk = crops_uint8[i:i + batch_size]
        if chunk.shape[1] != out_size:
            chunk = np.stack([cv2.resize(f, (out_size, out_size),
                                         interpolation=cv2.INTER_AREA) for f in chunk])
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            p = torch.sigmoid(model(x).float())
        probs.extend(p.cpu().numpy().tolist())
    return float(np.mean(probs)) if probs else 0.5
