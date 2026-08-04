"""
CrossFuse v5 -- shared library.

Upload this directory as a Kaggle Dataset (suggested slug: `crossfuse-v5-lib`),
then start every notebook with:

    import sys; sys.path.insert(0, "/kaggle/input/crossfuse-v5-lib")
    from crossfuse_v5 import *

WHAT CHANGED FROM v4, AND WHY
-----------------------------
1. Cached mel features.  v4 called librosa three times inside every
   __getitem__ (mfcc_from_waveform + two logmel_window).  With NUM_WORKERS=2
   against Kaggle's 4 vCPUs that was the dominant per-epoch cost and made
   training loader-bound, not GPU-bound.  Stage A now computes the full-clip
   log-mel ONCE and caches it; the dataset slices windows out of the cache.

2. Leading-silence control.  FakeAVCeleb fakes open with ~25-30 ms of digital
   silence while reals open with ambient noise; a silence-duration classifier
   alone scores 98.4% AUC on this corpus (arXiv:2412.00175).  The v4 audio
   head's 0.997 AUC is therefore not trustworthy.  `AUDIO_LEAD_SKIP` drops a
   fixed leading window before the audio-forgery MFCC is computed.  A FIXED
   skip is used rather than energy-based trimming because a fixed skip is
   provably label-blind -- an energy trim removes a different amount of audio
   from reals than from fakes, which is itself a leak.
   Both MFCC variants are cached, so the ablation is a config flag with zero
   re-extraction cost.
   The skip applies ONLY to the audio-forgery head.  The sync head keeps the
   untrimmed waveform so its log-mel stays time-aligned with the visual
   window.

3. Frame-level auxiliary supervision.  v4's step pushed BATCH_CLIPS(4) x
   WINDOW_FRAMES(32) = 128 images through the backbone to produce 4 labels.
   `frame_video_head` attaches a shared linear head to every valid frame
   token, so the same forward pass yields B*T gradient signals for the visual
   forgery task.  The clip-level head remains the reported verdict.

4. Configurable backbone.  The visual encoder is built by name so the
   FakeAVCeleb multimodal model and the FF++/SBI pretrained encoder are the
   same architecture and weights transfer directly.

5. Paired batching.  GenD (arXiv:2508.06248) finds that batching real and
   fake from the SAME source identity is what suppresses shortcut learning.
   `PairedBatchSampler` emits matched real/fake pairs from one identity
   group.

Carried over from v4 unchanged because it was correct: identity union-find
grouping, contiguous crop extraction with per-frame validity, two-phase
freeze->fine-tune with layer-wise LR decay, MC-Dropout + temperature scaling
+ bootstrap thresholds fit under the matched estimator, per-tag RNG streams.
"""

import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
from collections import defaultdict

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

# roc_auc_score/roc_curve/balanced_accuracy_score are reimplemented in pure
# NumPy below instead of imported from scikit-learn. This Kaggle image ships
# a scikit-learn build that imports `numpy.strings` (a NumPy>=2.0-only
# submodule) while the active numpy is 1.26.4 -- a pre-existing mismatch in
# the base image itself, present in a totally fresh kernel before any of our
# code runs, so no combination of pip flags on our end fixes it. These three
# functions are all we ever needed from sklearn, so removing the dependency
# removes the failure mode entirely, on any Kaggle image state.

try:
    from facenet_pytorch import MTCNN
except ImportError:  # notebooks that only need the model/eval side
    MTCNN = None

from torchvision.models import (
    efficientnet_b0, efficientnet_b4,
    EfficientNet_B0_Weights, EfficientNet_B4_Weights,
)

__all__ = [
    "CONFIG", "seed_everything", "make_rng", "device",
    "ffmpeg_extract_wav", "cmvn_normalize", "maybe_cmvn", "mfcc_from_waveform",
    "logmel_full", "slice_logmel_window", "LOGMEL_HOP", "LOGMEL_FPS",
    "UnionFind", "get_identity_tokens", "build_identity_groups",
    "build_detector", "extract_face_crops_contiguous",
    "get_fakeavceleb_label", "get_fakeavceleb_category", "CATEGORY_SHORT",
    "category_to_modality_labels", "make_sample_key",
    "TemporalConvNet", "masked_mean_std", "CrossFuseModelV5", "build_model",
    "build_backbone", "load_visual_encoder",
    "portable_state_dict", "load_portable_state_dict",
    "unwrap", "enable_mc_dropout", "mc_logits",
    "CrossFuseCropDataset", "PairedBatchSampler", "worker_init_fn",
    "self_blend_clip", "compression_augment_clip", "photometric_augment_clip",
    "build_3way_split_packed", "safe_auc", "smooth_labels",
    "roc_auc_score", "roc_curve", "balanced_accuracy_score", "classification_report_text",
    "run_training_pipeline", "evaluate_deterministic",
    "mc_collect", "fit_temperature", "bootstrap_stable_threshold",
    "expected_calibration_error", "bootstrap_ci", "youden_threshold",
]


# =====================================================================
# Configuration
# =====================================================================

CONFIG = {
    # ---- Stage A: what gets cached -----------------------------------
    # 256 (not v4's 160): blending-boundary and compression-inconsistency
    # artifacts live in high spatial frequencies that a 160 px resize
    # destroys.  CACHED_FRAMES drops 64 -> 32 to keep the cache under
    # Kaggle's 20 GB notebook-output limit at the larger crop:
    #   1500 clips * 32 * 256^2 * 3 = 9.4 GB.
    "CROP_SIZE":               256,
    "CACHED_FRAMES":           32,
    "FACE_MARGIN":             0.30,
    "AUDIO_SR":                16000,
    "CAP_FVRA":                250,
    "CAP_FVFA":                250,

    # ---- Stage B: what the model sees --------------------------------
    "WINDOW_FRAMES":           12,     # v4 used 32; see BATCH_CLIPS note
    "AUDIO_FRAMES":            128,
    "SYNC_MEL_FRAMES":         128,
    "N_MELS":                  80,
    "BACKBONE":                "efficientnet_b4",
    "FREEZE_BLOCKS":           4,
    "TRAIN_RES":               224,    # crops are resized to this on-GPU

    # ---- The leading-silence shortcut --------------------------------
    # Seconds dropped from the head of the waveform before the
    # audio-FORGERY MFCC is computed.  0.0 reproduces v4 exactly.
    # The sync head is unaffected either way.
    "AUDIO_LEAD_SKIP":         0.15,

    # ---- Ablation toggles --------------------------------------------
    "GENERATE_HARD_NEGATIVES": True,
    "USE_COMPRESSION_AUG":     True,
    "COMPRESSION_AUG_P":       0.30,   # v4 applied this to every sample
    "USE_MFCC_CMVN":           True,
    "USE_CROSS_ATTENTION":     True,
    "USE_SYNC_HEAD":           True,
    "USE_FUSION_HEAD":         True,
    "LATE_FUSION_BASELINE":    False,
    "SYNC_DENSE_ALIGNED":      True,
    "IDENTITY_DISJOINT_SPLIT": True,
    "USE_PAIRED_BATCHES":      True,
    "SBI_PRETRAINED_ENCODER":  None,   # path to the FF++/SBI checkpoint
    "MODALITY":                "both",
    "HELD_OUT_CATEGORY":       None,

    # ---- Splits, training, evaluation --------------------------------
    "SPLIT_FRACS":             (0.60, 0.20, 0.20),
    "EPOCHS_FROZEN":           3,
    "EPOCHS_FINETUNE":         20,
    "PATIENCE":                6,
    # 12x12 = 144 images/step, close to v4's 128, but 3x the labels per
    # step -- and with the frame head, 144 frame-level gradients as well.
    "BATCH_CLIPS":             12,
    "GRAD_ACCUM":              1,
    "LR_HEAD":                 1e-4,
    # 1e-5, not v4's 3e-5: when the backbone starts from SBI-pretrained
    # forensic features the job is to preserve them, not relearn them.
    "LR_BACKBONE":             1e-5,
    "LLRD":                    0.75,
    "LAMBDA_FRAME":            0.5,
    "LAMBDA_SYNC":             0.5,
    "LAMBDA_FUSION":           0.5,
    "N_MC_SAMPLES":            20,
    "N_SEEDS":                 1,      # 3 only for the final reported run
    "SEED":                    42,
    "SYNC_MIN_AUC":            0.70,   # PREREGISTERED
    "NUM_WORKERS":             4,
    "WORK_DIR":                "/kaggle/working",
    # DataParallel's master GPU gathers every replica's output for loss
    # computation and gradient reduction, so it carries more than its even
    # share of memory -- this crashed a live 2xT4 SBI run (kernel died) at
    # efficientnet_b4 + res 380 + effective batch 32. Off by default until
    # re-validated with a smaller effective batch; the sharding code in
    # CrossFuseModelV5.enable_multi_gpu is still here to flip back on.
    "MULTI_GPU":               False,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGMEL_HOP = 160          # 10 ms at 16 kHz
LOGMEL_NFFT = 400         # 25 ms
LOGMEL_FPS = 100.0        # frames per second of the cached mel


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_rng(tag, seed=None):
    """Per-tag deterministic RNG stream.  Each independent random decision
    gets its OWN seeded generator keyed by a string, instead of sharing one
    global state whose position depends on how many earlier calls ran."""
    seed = CONFIG["SEED"] if seed is None else seed
    h = int(hashlib.md5(f"{tag}:{seed}".encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(h)


# =====================================================================
# Audio front-end
# =====================================================================

def ffmpeg_extract_wav(video_path, sr=None):
    """(y, sr) like librosa.load, but demuxed through ffmpeg first so every
    caller shares one decode path and librosa always reads a clean WAV via
    the fast soundfile backend rather than silently dropping to audioread."""
    sr = CONFIG["AUDIO_SR"] if sr is None else sr
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
                 "-ar", str(sr), "-f", "wav", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            if result.returncode != 0 or not os.path.exists(wav_path):
                raise RuntimeError(result.stderr.decode(errors="ignore")[-300:])
            return librosa.load(wav_path, sr=sr)
        except Exception as e:
            print(f"[!] ffmpeg_extract_wav failed for "
                  f"'{os.path.basename(video_path)}' ({e}); falling back to "
                  f"librosa.load (backend may differ!)")
            return librosa.load(video_path, sr=sr)


def cmvn_normalize(mfcc, eps=1e-8):
    mfcc = np.asarray(mfcc, dtype=np.float32)
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)
    return (mfcc - mean) / (std + eps)


def maybe_cmvn(mfcc, use_cmvn=None):
    use_cmvn = CONFIG["USE_MFCC_CMVN"] if use_cmvn is None else use_cmvn
    return cmvn_normalize(mfcc) if use_cmvn else np.asarray(mfcc, dtype=np.float32)


def mfcc_from_waveform(y, sr=None, n_frames=None, lead_skip=0.0):
    """Full-clip RAW (not CMVN'd) MFCC-40, time-interpolated to a fixed
    length, for the audio-FORGERY head.

    `lead_skip` drops that many seconds from the head of the waveform BEFORE
    analysis.  This is the control for the FakeAVCeleb leading-silence
    shortcut.  It is a fixed offset, not an energy-based trim, so exactly the
    same amount of signal is removed from real and fake clips -- an energy
    trim would remove more from fakes (which start silent) than from reals
    (which start with ambient noise), reintroducing the very leak it is
    meant to close."""
    sr = CONFIG["AUDIO_SR"] if sr is None else sr
    n_frames = CONFIG["AUDIO_FRAMES"] if n_frames is None else n_frames
    try:
        y = np.asarray(y, dtype=np.float32)
        if lead_skip > 0:
            i0 = int(lead_skip * sr)
            # Only skip if enough audio survives to still be analysable.
            if len(y) - i0 >= sr // 2:
                y = y[i0:]
        if y is None or len(y) < sr // 10:
            raise ValueError("audio too short or empty")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        t = F.interpolate(t, size=n_frames, mode="linear", align_corners=False)
        return t.squeeze(0).numpy().astype(np.float32)
    except Exception:
        return np.zeros((40, n_frames), dtype=np.float32)


def logmel_full(y, sr=None, n_mels=None):
    """Whole-clip log-mel at LOGMEL_FPS frames/sec, cached once by Stage A.

    v4 recomputed a windowed log-mel inside every __getitem__, twice (matched
    and time-shifted negative).  Caching the full clip and slicing it is the
    single biggest loader win: librosa disappears from the training path.

    Returned UNNORMALISED -- slice_logmel_window normalises per window, which
    is what v4 did and what keeps a quiet window from being scaled against a
    loud clip's statistics."""
    sr = CONFIG["AUDIO_SR"] if sr is None else sr
    n_mels = CONFIG["N_MELS"] if n_mels is None else n_mels
    try:
        y = np.asarray(y, dtype=np.float32)
        if len(y) < int(0.05 * sr):
            raise ValueError("clip too short")
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=LOGMEL_NFFT, hop_length=LOGMEL_HOP, n_mels=n_mels)
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    except Exception:
        return np.zeros((n_mels, 1), dtype=np.float32)


def slice_logmel_window(mel_full, t0, t1, out_frames=None):
    """Cut [t0, t1) seconds out of a cached full-clip log-mel and resample to
    a fixed length.  Replaces v4's logmel_window() without calling librosa.

    Mel frame k and visual frame j refer to the same instant, up to the
    fixed-length resample -- which preserves RELATIVE timing, the only thing
    lip-sync depends on."""
    out_frames = CONFIG["SYNC_MEL_FRAMES"] if out_frames is None else out_frames
    n_mels = mel_full.shape[0]
    try:
        i0 = max(0, int(round(t0 * LOGMEL_FPS)))
        i1 = min(mel_full.shape[1], int(round(t1 * LOGMEL_FPS)))
        seg = mel_full[:, i0:i1]
        if seg.shape[1] < 5:
            raise ValueError("window too short")
        t = torch.from_numpy(np.ascontiguousarray(seg)).float().unsqueeze(0)
        t = F.interpolate(t, size=out_frames, mode="linear", align_corners=False)
        arr = t.squeeze(0).numpy()
        return ((arr - arr.mean()) / (arr.std() + 1e-8)).astype(np.float32)
    except Exception:
        return np.zeros((n_mels, out_frames), dtype=np.float32)


# =====================================================================
# Identity grouping  (carried over from v4 verbatim -- it was correct)
# =====================================================================

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def get_identity_tokens(video_path):
    """FakeAVCeleb layout is .../{category}/{ethnicity}/{gender}/{id}/{video}.mp4,
    but manipulated filenames also encode a SOURCE identity
    (id00076_id00812_wavtolip.mp4).  Pull every id-token from both the parent
    folder AND the filename, so source-identity leakage is caught too."""
    path_norm = video_path.replace("\\", "/")
    parent = os.path.basename(os.path.dirname(path_norm))
    fname = os.path.basename(path_norm)
    tokens = set(re.findall(r"id\d+", parent.lower())) | set(re.findall(r"id\d+", fname.lower()))
    if not tokens:
        tokens = {f"folder::{parent.lower()}"}
    return tokens


def build_identity_groups(video_paths):
    """Union every id-token that co-occurs on any single clip into one
    connected component; returns {video_path: canonical_group_id}."""
    uf = UnionFind()
    path_tokens = {}
    for v in video_paths:
        toks = get_identity_tokens(v)
        path_tokens[v] = toks
        toks = list(toks)
        for t in toks[1:]:
            uf.union(toks[0], t)
        if len(toks) == 1:
            uf.find(toks[0])
    return {v: uf.find(next(iter(path_tokens[v]))) for v in video_paths}


# =====================================================================
# Face-crop extraction
# =====================================================================

def build_detector(device_str):
    if MTCNN is None:
        raise ImportError("facenet_pytorch is required for extraction "
                          "(`pip install facenet-pytorch`)")
    return MTCNN(image_size=CONFIG["CROP_SIZE"], margin=0, keep_all=False,
                 select_largest=True, post_process=False, device=device_str)


def _crop_from_box(pil_img, box, out_size, margin_frac):
    """Square, margin-expanded crop, clamped to the frame.  Square-FIRST
    rather than resizing a rectangle: an aspect-distorted face would corrupt
    the very blending-boundary geometry we are trying to detect."""
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + margin_frac) / 2.0
    l, t = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
    r, b = min(W, int(round(cx + half))), min(H, int(round(cy + half)))
    if r - l < 8 or b - t < 8:
        return None
    return pil_img.crop((l, t, r, b)).resize((out_size, out_size), Image.BILINEAR)


def extract_face_crops_contiguous(video_path, mtcnn_local, rng=None,
                                  n_frames=None, out_size=None, batch_size=16):
    """Cache a CONTIGUOUS run of `n_frames` face crops as uint8.

    Returns (crops (n,S,S,3) uint8, valid (n,) bool, fps, start_frame,
    timestamps (n,) float32) or (None, None, 0.0, 0, None) on failure.

    `valid` is PER FRAME, not a count, so the attention mask can exclude
    frames where detection failed.  Contiguity (rather than striding across
    the whole clip) is what gives the sync head a usable time base.
    """
    n_frames = CONFIG["CACHED_FRAMES"] if n_frames is None else n_frames
    out_size = CONFIG["CROP_SIZE"] if out_size is None else out_size
    rng = rng or make_rng(f"crops:{os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0.0, 0, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    if not np.isfinite(fps) or fps <= 1.0:
        fps = 25.0

    start = rng.randint(0, max(0, total - n_frames)) if total > n_frames else 0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    pil_frames, idxs, fi = [], [], start
    while cap.isOpened() and len(pil_frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idxs.append(fi)
        fi += 1
    cap.release()
    if not pil_frames:
        return None, None, fps, start, None

    boxes_all = []
    for i in range(0, len(pil_frames), batch_size):
        chunk = pil_frames[i:i + batch_size]
        try:
            boxes, _ = mtcnn_local.detect(chunk)
        except Exception:
            boxes = [None] * len(chunk)
        if boxes is None:
            boxes = [None] * len(chunk)
        for b in boxes:
            boxes_all.append(None if b is None or len(b) == 0 else b[0])

    crops = np.zeros((n_frames, out_size, out_size, 3), dtype=np.uint8)
    valid = np.zeros(n_frames, dtype=bool)
    last_box = None
    for j, pil_img in enumerate(pil_frames):
        box = boxes_all[j] if j < len(boxes_all) else None
        detected = box is not None
        if not detected:
            box = last_box                    # faces do not teleport frame-to-frame
        crop = (_crop_from_box(pil_img, box, out_size, CONFIG["FACE_MARGIN"])
                if box is not None else None)
        if crop is None:
            W, H = pil_img.size               # last resort: centre square
            s = min(W, H)
            crop = pil_img.crop(((W - s) // 2, (H - s) // 2,
                                 (W + s) // 2, (H + s) // 2)) \
                          .resize((out_size, out_size), Image.BILINEAR)
        else:
            if detected:
                last_box = box
            valid[j] = True
        crops[j] = np.asarray(crop, dtype=np.uint8)

    if len(pil_frames) < n_frames:
        valid[len(pil_frames):] = False       # padding tail is never valid

    timestamps = np.zeros(n_frames, dtype=np.float32)
    for j in range(len(idxs)):
        timestamps[j] = idxs[j] / fps
    for j in range(len(idxs), n_frames):
        timestamps[j] = timestamps[max(0, len(idxs) - 1)] + (j - len(idxs) + 1) / fps

    return crops, valid, fps, start, timestamps


# =====================================================================
# FakeAVCeleb bookkeeping
# =====================================================================

CATEGORY_SHORT = {
    "RealVideo-RealAudio": "RVRA", "RealVideo-FakeAudio": "RVFA",
    "FakeVideo-RealAudio": "FVRA", "FakeVideo-FakeAudio": "FVFA", "Unknown": "UNK",
}


def get_fakeavceleb_category(video_path):
    path_norm = video_path.replace("\\", "/").lower()
    for cat in CATEGORY_SHORT:
        if cat != "Unknown" and cat.lower() in path_norm:
            return cat
    return "Unknown"


def get_fakeavceleb_label(video_path):
    cat = get_fakeavceleb_category(video_path)
    if cat == "RealVideo-RealAudio":
        return 0
    if cat != "Unknown":
        return 1
    base = os.path.basename(video_path).lower()
    return 0 if ("real" in base and "fake" not in base) else 1


def category_to_modality_labels(category):
    """(video_label, audio_label).  The 2x2 structure of FakeAVCeleb is what
    makes per-modality attribution possible at all."""
    c = category.lower()
    return (1 if "fakevideo" in c else 0), (1 if "fakeaudio" in c else 0)


def make_sample_key(category, identity_group, video_path):
    stem = os.path.basename(video_path).split(".")[0]
    rel_hash = hashlib.md5(video_path.replace("\\", "/").encode()).hexdigest()[:8]
    return f"{CATEGORY_SHORT.get(category, 'UNK')}_{identity_group}_{stem}_{rel_hash}"


# =====================================================================
# Model
# =====================================================================

_BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights, 1280),
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights, 1792),
}


def build_backbone(name, pretrained=True):
    """(features_module, feature_dim).  Named construction so the SBI
    pretraining stage and the multimodal stage instantiate the SAME
    architecture and weights transfer without surgery."""
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'; choose from {list(_BACKBONES)}")
    ctor, weights_enum, dim = _BACKBONES[name]
    weights = weights_enum.DEFAULT if pretrained else None
    return ctor(weights=weights).features, dim


class TemporalConvNet(nn.Module):
    """Dilated temporal CNN over (B, C_in, T) -> (B, T_out, out_dim).
    Parameterised so one block serves both the 40-band MFCC (audio-forgery
    head) and the 80-band log-mel (sync head)."""

    def __init__(self, num_inputs=40, num_channels=(64, 128, 256), kernel_size=3,
                 dropout=0.3, target_seq_len=60, out_dim=256):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=(kernel_size - 1) * dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
            ]
        self.network = nn.Sequential(*layers)
        self.temporal_pool = nn.AdaptiveAvgPool1d(target_seq_len)
        self.proj = nn.Linear(num_channels[-1], out_dim)

    def forward(self, x):
        out = self.temporal_pool(self.network(x)).transpose(1, 2)
        return self.proj(out)


def masked_mean_std(x, mask=None, eps=1e-8):
    """x: (B,T,D), mask: (B,T) bool where True = valid.  Returns
    concat[mean, std] -> (B, 2D).  The std half carries frame-to-frame
    INCONSISTENCY, which a plain mean discards and which is the strongest
    single cue for manipulated video."""
    if mask is None:
        mean, var = x.mean(dim=1), x.var(dim=1, unbiased=False)
    else:
        mask_f = mask.unsqueeze(-1).float()
        count = mask_f.sum(dim=1).clamp(min=1.0)
        mean = (x * mask_f).sum(dim=1) / count
        var = ((x - mean.unsqueeze(1)) ** 2 * mask_f).sum(dim=1) / count
    return torch.cat([mean, torch.sqrt(var.clamp(min=eps))], dim=-1)


def _mlp_head(in_dim, hidden, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1))


class CrossFuseModelV5(nn.Module):
    """
    Heads
    -----
    video_head        clip-level, visual branch ONLY -> P(visual track fake)
    frame_video_head  per-frame, visual branch ONLY  -> auxiliary supervision
    audio_head        audio branch ONLY              -> P(audio track synthetic)
    sync_head         mouth <-> log-mel cross-attention -> P(AV mismatch)
    fusion_head       fused representation           -> P(clip fake at all)

    The two attribution heads stay strictly unimodal: cross-attention cannot
    leak audio evidence into the video verdict or vice versa.  That is what
    keeps the 4-way modality-attribution matrix interpretable, and it is why
    the fusion head exists separately rather than replacing them.
    """

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.4,
                 window_frames=12, n_mels=80, backbone="efficientnet_b4",
                 train_res=224, use_cross_attention=True, use_sync_head=True,
                 use_fusion_head=True, late_fusion_baseline=False,
                 sync_dense_aligned=True, freeze_blocks=4, pretrained=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.train_res = train_res
        self.use_cross_attention = use_cross_attention
        self.use_sync_head = use_sync_head and use_cross_attention
        self.use_fusion_head = use_fusion_head
        self.late_fusion_baseline = late_fusion_baseline
        self.sync_dense_aligned = sync_dense_aligned
        self.freeze_blocks = freeze_blocks

        self.backbone, self.backbone_dim = build_backbone(backbone, pretrained)
        self.spatial_pool = nn.AdaptiveAvgPool2d(3)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.set_backbone_trainable(freeze_blocks)

        # ---- Visual branch (feeds video_head ONLY) ---------------------
        self.visual_proj = nn.Linear(self.backbone_dim, hidden_dim)
        self.visual_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=512,
            dropout=dropout, batch_first=True)
        self.video_head = _mlp_head(2 * hidden_dim, 128, dropout)
        # Auxiliary per-frame head.  Deliberately a bare Linear: it exists to
        # push gradient into every frame's tokens, not to be a second
        # classifier competing with video_head for capacity.
        self.frame_video_head = nn.Linear(hidden_dim, 1)

        # ---- Audio branch (feeds audio_head ONLY) ---------------------
        self.audio_encoder = TemporalConvNet(num_inputs=40, target_seq_len=window_frames,
                                             dropout=0.3, out_dim=hidden_dim)
        self.audio_head = _mlp_head(2 * hidden_dim, 128, dropout)

        # ---- Sync branch ----------------------------------------------
        if self.use_sync_head:
            self.mouth_proj = nn.Linear(self.backbone_dim, hidden_dim)
            sync_in = n_mels if sync_dense_aligned else 40
            self.mel_encoder = TemporalConvNet(num_inputs=sync_in,
                                               target_seq_len=2 * window_frames,
                                               dropout=0.3, out_dim=hidden_dim)
            self.sync_ca_m2a = nn.MultiheadAttention(hidden_dim, num_heads,
                                                     dropout=dropout, batch_first=True)
            self.sync_ca_a2m = nn.MultiheadAttention(hidden_dim, num_heads,
                                                     dropout=dropout, batch_first=True)
            self.sync_ln_m = nn.LayerNorm(hidden_dim)
            self.sync_ln_a = nn.LayerNorm(hidden_dim)
            self.sync_head = _mlp_head(4 * hidden_dim, 128, dropout)

        # ---- Fusion branch --------------------------------------------
        # Both arms produce 4*hidden and pass through an identically shaped
        # MLP.  The ONLY difference is cross-attention, so the late-fusion
        # ablation is a fair comparison rather than a capacity comparison.
        if self.use_fusion_head:
            if use_cross_attention and not late_fusion_baseline:
                self.fuse_ca_v2a = nn.MultiheadAttention(hidden_dim, num_heads,
                                                         dropout=dropout, batch_first=True)
                self.fuse_ca_a2v = nn.MultiheadAttention(hidden_dim, num_heads,
                                                         dropout=dropout, batch_first=True)
                self.fuse_ln_v = nn.LayerNorm(hidden_dim)
                self.fuse_ln_a = nn.LayerNorm(hidden_dim)
            self.fusion_head = _mlp_head(4 * hidden_dim, 128, dropout)

    # ---- Backbone freezing -------------------------------------------
    def raw_backbone(self):
        """The underlying Sequential, whether or not the backbone is wrapped
        in DataParallel.  Every helper that INDEXES or ITERATES the backbone
        must go through this -- a DataParallel wrapper is not indexable and
        exposes no .parameters() per block."""
        return (self.backbone.module if isinstance(self.backbone, nn.DataParallel)
                else self.backbone)

    def enable_multi_gpu(self):
        """Wrap ONLY the visual backbone in DataParallel.

        The training loop calls encode_visual()/forward_from_tokens()
        directly rather than forward(), and DataParallel only intercepts
        forward() -- so wrapping the whole model would silently run
        everything on one GPU (or raise on the custom method names). The
        backbone is where the B*T image compute actually is, and
        encode_visual invokes it as self.backbone(x), a real forward call,
        so wrapping just that shards the images across GPUs correctly while
        the cheap heads stay on cuda:0."""
        if torch.cuda.device_count() > 1 and not isinstance(self.backbone, nn.DataParallel):
            self.backbone = nn.DataParallel(self.backbone)
        return self

    def set_backbone_trainable(self, freeze_blocks):
        """features[:freeze_blocks] frozen, with their BatchNorms pinned to
        eval so a ~1.5k-clip dataset cannot drift pretrained low-level
        filters."""
        self.freeze_blocks = freeze_blocks
        for i, block in enumerate(self.raw_backbone()):
            for p in block.parameters():
                p.requires_grad = i >= freeze_blocks

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for i, block in enumerate(self.raw_backbone()):
                if i < self.freeze_blocks:
                    block.eval()
        return self

    def backbone_param_groups(self, lr_backbone, llrd=0.75):
        """Layer-wise LR decay: later, more task-specific blocks get the full
        LR; earlier blocks progressively less.  The main guard against the
        backbone collapsing onto identity cues on a small corpus."""
        raw = self.raw_backbone()
        groups, n = [], len(raw)
        for i in range(self.freeze_blocks, n):
            params = [p for p in raw[i].parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr_backbone * (llrd ** (n - 1 - i))})
        return groups

    # ---- Visual encoding, split out so MC-Dropout can reuse it --------
    def encode_visual(self, crops):
        """crops: (B,T,S,S,3) uint8 -> (v_tok, m_tok), each (B,T,backbone_dim).

        v_tok is the 3x3 grid mean (global token); m_tok is the BOTTOM ROW of
        the grid, which is where the mouth sits in a margin-expanded face
        crop.  Deterministic -- the backbone holds no Dropout, so MC-Dropout
        never re-runs it."""
        B, T = crops.shape[0], crops.shape[1]
        x = crops.reshape(B * T, *crops.shape[2:])
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-1] != self.train_res:
            x = F.interpolate(x, size=(self.train_res, self.train_res),
                              mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        grid = self.spatial_pool(self.backbone(x))
        v_tok = grid.mean(dim=(2, 3)).view(B, T, self.backbone_dim)
        m_tok = grid[:, :, 2, :].mean(dim=2).view(B, T, self.backbone_dim)
        return v_tok, m_tok

    def forward_from_tokens(self, v_tok, m_tok, mfcc, sync_audio=None, visual_mask=None):
        B, T = v_tok.shape[0], v_tok.shape[1]
        if visual_mask is None:
            visual_mask = torch.ones(B, T, dtype=torch.bool, device=v_tok.device)
        key_padding_mask = ~visual_mask

        V = self.visual_proj(v_tok)
        V_enc = self.visual_encoder(V, src_key_padding_mask=key_padding_mask)
        v_rep = masked_mean_std(V_enc, visual_mask)
        video_logit = self.video_head(v_rep).squeeze(-1)
        frame_logits = self.frame_video_head(V_enc).squeeze(-1)        # (B, T)

        A_enc = self.audio_encoder(mfcc)
        a_rep = masked_mean_std(A_enc, None)
        audio_logit = self.audio_head(a_rep).squeeze(-1)

        sync_logit = fusion_logit = attn_v2a = attn_a2v = None

        if self.use_sync_head and sync_audio is not None:
            M = self.mouth_proj(m_tok)
            S_enc = self.mel_encoder(sync_audio)
            ca_m, _ = self.sync_ca_m2a(query=M, key=S_enc, value=S_enc)
            fused_m = self.sync_ln_m(M + ca_m)
            ca_a, _ = self.sync_ca_a2m(query=S_enc, key=M, value=M,
                                       key_padding_mask=key_padding_mask)
            fused_s = self.sync_ln_a(S_enc + ca_a)
            sync_logit = self.sync_head(torch.cat(
                [masked_mean_std(fused_m, visual_mask),
                 masked_mean_std(fused_s, None)], dim=-1)).squeeze(-1)

        if self.use_fusion_head:
            if self.use_cross_attention and not self.late_fusion_baseline:
                ca_v2a, attn_v2a = self.fuse_ca_v2a(query=V_enc, key=A_enc, value=A_enc)
                fused_v = self.fuse_ln_v(V_enc + ca_v2a)
                ca_a2v, attn_a2v = self.fuse_ca_a2v(query=A_enc, key=V_enc, value=V_enc,
                                                    key_padding_mask=key_padding_mask)
                fused_a = self.fuse_ln_a(A_enc + ca_a2v)
                fused = torch.cat([masked_mean_std(fused_v, visual_mask),
                                   masked_mean_std(fused_a, None)], dim=-1)
            else:
                fused = torch.cat([v_rep, a_rep], dim=-1)
            fusion_logit = self.fusion_head(fused).squeeze(-1)

        return {"video_logit": video_logit, "frame_logits": frame_logits,
                "audio_logit": audio_logit, "sync_logit": sync_logit,
                "fusion_logit": fusion_logit, "attn_v2a": attn_v2a, "attn_a2v": attn_a2v}

    def forward(self, crops, mfcc, sync_audio=None, visual_mask=None):
        v_tok, m_tok = self.encode_visual(crops)
        return self.forward_from_tokens(v_tok, m_tok, mfcc, sync_audio, visual_mask)


def build_model(cfg, pretrained=True):
    """Single construction path shared by every notebook, so an ablation
    checkpoint always loads into the architecture it was trained as."""
    return CrossFuseModelV5(
        window_frames=cfg["WINDOW_FRAMES"], n_mels=cfg["N_MELS"],
        backbone=cfg["BACKBONE"], train_res=cfg["TRAIN_RES"],
        use_cross_attention=cfg["USE_CROSS_ATTENTION"],
        use_sync_head=cfg["USE_SYNC_HEAD"], use_fusion_head=cfg["USE_FUSION_HEAD"],
        late_fusion_baseline=cfg["LATE_FUSION_BASELINE"],
        sync_dense_aligned=cfg["SYNC_DENSE_ALIGNED"],
        freeze_blocks=cfg["FREEZE_BLOCKS"], pretrained=pretrained)


def portable_state_dict(model):
    """state_dict with any DataParallel 'backbone.module.' prefix normalised
    back to 'backbone.', so a checkpoint written by a 2-GPU run loads into
    the single-GPU model NB-C/NB-D build, and vice versa."""
    return {k.replace("backbone.module.", "backbone."): v
            for k, v in unwrap(model).state_dict().items()}


def load_portable_state_dict(model, state):
    """Inverse of portable_state_dict: re-adds the 'module.' segment when the
    target model's backbone happens to be DataParallel-wrapped."""
    m = unwrap(model)
    if isinstance(m.backbone, nn.DataParallel):
        state = {k.replace("backbone.", "backbone.module.", 1)
                 if k.startswith("backbone.") and not k.startswith("backbone.module.")
                 else k: v for k, v in state.items()}
    return m.load_state_dict(state)


def load_visual_encoder(model, ckpt_path, verbose=True):
    """Load an SBI-pretrained `backbone.*` state dict into a CrossFuseModelV5.

    The checkpoint from the FF++/SBI stage stores the backbone under
    'backbone_state'.  Nothing else is transferred -- the multimodal heads
    have no counterpart there and must train from scratch.

    Raises rather than warns on a shape mismatch: silently falling back to
    ImageNet weights would make the SBI ablation row meaningless while still
    appearing to run."""
    if ckpt_path is None:
        return model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["backbone_state"] if "backbone_state" in ckpt else ckpt
    missing, unexpected = model.raw_backbone().load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"SBI encoder does not fit this backbone ({len(missing)} missing keys, "
            f"e.g. {missing[:3]}). Check that CONFIG['BACKBONE'] matches the "
            f"backbone the SBI stage was trained with.")
    if verbose:
        n = sum(p.numel() for p in model.backbone.parameters())
        print(f"  Loaded SBI-pretrained visual encoder from {ckpt_path} "
              f"({n:,} params, {len(unexpected)} unexpected keys ignored)")
    return model


# ---- MC-Dropout ------------------------------------------------------

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def enable_mc_dropout(model):
    """Switch ONLY Dropout back on; BatchNorm stays in eval with frozen
    running stats."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_logits(model, crops, mfcc, sync_audio=None, visual_mask=None,
              n_samples=20, tokens=None):
    """N stochastic HEAD passes over ONE deterministic backbone pass.

    Returns raw logits {'video','audio','sync','fusion'} each (n, B) --
    callers do their own temperature scaling, which needs logits not
    probabilities.  Only the heads contain dropout, so re-running the
    backbone per sample would be N x the compute for identical tokens.

    `tokens`: pass a precomputed (v_tok, m_tok) when scoring the SAME crops
    against MULTIPLE audio inputs (the sync head's matched/mismatched pair),
    so the backbone runs once across both calls."""
    m = unwrap(model)
    m.eval()
    enable_mc_dropout(m)
    out = {}
    with torch.no_grad():
        v_tok, m_tok = m.encode_visual(crops) if tokens is None else tokens
        for _ in range(n_samples):
            o = m.forward_from_tokens(v_tok, m_tok, mfcc, sync_audio, visual_mask)
            for k_out, k_in in (("video", "video_logit"), ("audio", "audio_logit"),
                                ("sync", "sync_logit"), ("fusion", "fusion_logit")):
                if o[k_in] is not None:
                    out.setdefault(k_out, []).append(o[k_in].float().cpu().numpy())
    m.eval()
    return {k: np.stack(v, axis=0) for k, v in out.items()}


# =====================================================================
# Train-time augmentation
# =====================================================================

def _soft_ellipse_mask(h, w, rng):
    """Feathered elliptical mask roughly covering the face region of a
    centred, margin-expanded crop."""
    cy, cx = h * rng.uniform(0.45, 0.55), w * rng.uniform(0.45, 0.55)
    ry, rx = h * rng.uniform(0.30, 0.42), w * rng.uniform(0.26, 0.38)
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask = np.clip(1.4 - d, 0.0, 1.0).astype(np.float32)
    k = int(max(3, (h // 16) * 2 + 1))
    return cv2.GaussianBlur(mask, (k, k), 0)[..., None]


def self_blend_clip(window, rng):
    """Blend a mildly perturbed copy of each frame back onto itself through a
    soft mask, producing a blending boundary from a single real identity with
    no external generator fingerprint.

    NOTE: this is the lightweight in-training version, kept for the
    FakeAVCeleb hard-negative role it played in v4.  The FF++ pretraining
    stage uses the full landmark-driven SBI recipe in `sbi_v5.py`, which is
    what the cross-dataset generalization claim rests on."""
    out = window.copy()
    T, h, w, _ = window.shape
    mask = _soft_ellipse_mask(h, w, rng)
    dx, dy = rng.uniform(-3, 3), rng.uniform(-3, 3)
    scale = rng.uniform(0.97, 1.03)
    alpha = rng.uniform(0.30, 0.85)
    gain = rng.uniform(0.93, 1.07)
    bias = rng.uniform(-9, 9)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2.5, 2.5), scale)
    M[0, 2] += dx
    M[1, 2] += dy
    for t in range(T):
        src = cv2.warpAffine(window[t], M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        src = np.clip(src.astype(np.float32) * gain + bias, 0, 255)
        if rng.random() < 0.5:                        # resolution mismatch
            f = rng.uniform(0.5, 0.9)
            small = cv2.resize(src, (max(8, int(w * f)), max(8, int(h * f))),
                               interpolation=cv2.INTER_AREA)
            src = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        m = mask * alpha
        out[t] = np.clip(window[t].astype(np.float32) * (1 - m) + src * m,
                         0, 255).astype(np.uint8)
    return out


def compression_augment_clip(window, rng, p_frame=0.5):
    """JPEG / downscale / noise, to stop the model keying on FakeAVCeleb's
    encoder settings rather than on manipulation.

    v4 re-encoded all 32 frames of every training sample; that was a large
    share of the loader budget.  Here the whole call is gated by
    COMPRESSION_AUG_P upstream, and only a random subset of frames within a
    selected clip is touched."""
    out = window.copy()
    T = window.shape[0]
    q = rng.randint(35, 85)
    do_scale = rng.random() < 0.5
    f = rng.uniform(0.5, 0.9)
    noise_sd = rng.uniform(0.0, 3.0)
    h, w = window.shape[1], window.shape[2]
    for t in range(T):
        if rng.random() > p_frame:
            continue
        img = out[t]
        if do_scale:
            small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if noise_sd > 0:
            img = np.clip(img.astype(np.float32) +
                          np.random.RandomState(rng.randint(0, 2 ** 31 - 1))
                          .randn(h, w, 3) * noise_sd, 0, 255).astype(np.uint8)
        out[t] = img
    return out


def photometric_augment_clip(window, rng):
    """Clip-consistent brightness/contrast/saturation plus optional
    horizontal flip.  Applied per CLIP, not per frame, so temporal
    consistency (which the std half of masked_mean_std reads) is preserved."""
    gain = rng.uniform(0.85, 1.15)
    bias = rng.uniform(-18, 18)
    out = np.clip(window.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = out[:, :, ::-1, :].copy()
    if rng.random() < 0.2:
        gray = out.mean(axis=3, keepdims=True)
        sat = rng.uniform(0.5, 1.0)
        out = np.clip(gray + (out - gray) * sat, 0, 255).astype(np.uint8)
    return out


# =====================================================================
# Dataset
# =====================================================================

class CrossFuseCropDataset(Dataset):
    """One contiguous WINDOW_FRAMES window of cached crops plus the audio
    covering exactly the same seconds.

    Train mode draws a random window start and applies fresh augmentation;
    eval mode always takes the CENTRE window with no augmentation, so val and
    test are deterministic across epochs, seeds and ablation rows.

    All audio features come from Stage A's cache -- no librosa call happens
    on the training path.
    """

    SELFBLEND_P = 0.25

    def __init__(self, rows, indices, crop_dir, cfg, training=False):
        self.rows = rows
        # int(x) not list(indices): indices arrives as np.array(dtype=int) and
        # list() yields numpy.int64, which random.Random() rejects with a
        # TypeError raised inside the DataLoader worker on the first batch.
        self.indices = [int(x) for x in indices]
        self.crop_dir = crop_dir
        self.cfg = cfg
        self.training = training
        self.W = cfg["WINDOW_FRAMES"]
        self.sr = cfg["AUDIO_SR"]
        self.group_to_id = {g: i for i, g in
                            enumerate(sorted({r["identity_group"] for r in rows}))}

    def __len__(self):
        return len(self.indices)

    def _load(self, ridx):
        r = self.rows[ridx]
        base = os.path.join(self.crop_dir, r["key"])
        crops = np.load(base + ".npy", mmap_mode="r")
        meta = np.load(base + "_meta.npz")
        # Stage A caches BOTH MFCC variants so AUDIO_LEAD_SKIP is a free
        # config flip rather than a re-extraction.
        mfcc_key = "mfcc_trim" if self.cfg["AUDIO_LEAD_SKIP"] > 0 else "mfcc_full"
        return r, crops, meta[mfcc_key], meta["mel"], meta["valid"], meta["timestamps"]

    def __getitem__(self, i):
        ridx = self.indices[i]
        r, crops, mfcc_raw, mel_full, valid, ts = self._load(ridx)
        rng = random.Random(
            (self.cfg["SEED"] * 1_000_003 + ridx * 9176 + (i if self.training else 0))
            ^ (random.getrandbits(30) if self.training else 0))

        T_cached = crops.shape[0]
        W = min(self.W, T_cached)
        max_start = max(0, T_cached - W)
        start = rng.randint(0, max_start) if self.training else max_start // 2
        window = np.array(crops[start:start + W])
        win_valid = np.array(valid[start:start + W], dtype=bool)
        fps = r["fps"] if r["fps"] > 1 else 25.0
        t0 = float(ts[start])
        t1 = t0 + W / fps

        v_label, a_label = r["video_label"], r["audio_label"]

        if self.training:
            if (self.cfg["GENERATE_HARD_NEGATIVES"] and v_label == 0
                    and rng.random() < self.SELFBLEND_P):
                window = self_blend_clip(window, rng)
                v_label = 1
            if self.cfg["USE_COMPRESSION_AUG"] and rng.random() < self.cfg["COMPRESSION_AUG_P"]:
                window = compression_augment_clip(window, rng)
            window = photometric_augment_clip(window, rng)
            drop = np.random.RandomState(rng.randint(0, 2 ** 31 - 1)).rand(W) < 0.10
            drop &= win_valid
            if drop.any() and (win_valid & ~drop).any():
                window = window.copy()
                window[drop] = 0
                win_valid = win_valid & ~drop

        if not win_valid.any():
            win_valid[0] = True

        mfcc = maybe_cmvn(mfcc_raw, use_cmvn=self.cfg["USE_MFCC_CMVN"])

        if self.cfg["SYNC_DENSE_ALIGNED"]:
            sync_a = slice_logmel_window(mel_full, t0, t1, self.cfg["SYNC_MEL_FRAMES"])
            # SyncNet-style hard negative: same clip, same speaker, same
            # recording conditions -- ONLY the time offset differs, so the
            # only way to tell it apart is actual lip-sync.
            dur = mel_full.shape[1] / LOGMEL_FPS
            shift = rng.choice([-1, 1]) * rng.uniform(0.4, max(0.5, dur / 3))
            o0 = float(np.clip(t0 + shift, 0.0, max(0.0, dur - (t1 - t0))))
            sync_off = slice_logmel_window(mel_full, o0, o0 + (t1 - t0),
                                           self.cfg["SYNC_MEL_FRAMES"])
        else:
            # Ablation: the v3.3 arrangement -- whole-clip MFCC against a
            # visual window it has no time correspondence with.
            sync_a = mfcc.copy()
            sync_off = mfcc.copy()

        pad = self.W - W
        if pad > 0:
            window = np.concatenate(
                [window, np.zeros((pad, *window.shape[1:]), np.uint8)], 0)
            win_valid = np.concatenate([win_valid, np.zeros(pad, bool)])

        return (torch.from_numpy(np.ascontiguousarray(window)),
                torch.from_numpy(mfcc),
                torch.from_numpy(sync_a),
                torch.from_numpy(sync_off),
                torch.tensor(float(v_label)), torch.tensor(float(a_label)),
                torch.from_numpy(win_valid),
                torch.tensor(self.group_to_id[r["identity_group"]], dtype=torch.long))


class PairedBatchSampler(Sampler):
    """Emit batches in which real and fake samples come from the SAME source
    identity group wherever possible.

    GenD (arXiv:2508.06248) reports that paired real/fake batching from a
    common source is the decisive factor in suppressing shortcut learning:
    within a pair, identity, lighting, pose and recording conditions are
    held constant, so the only thing separating the two labels is the
    manipulation itself.  FakeAVCeleb fakes derive from identified reals, so
    the pairing is available for free from the identity groups the split
    already computes.

    Groups with only one label contribute their samples as filler, so no
    training data is discarded.
    """

    def __init__(self, rows, indices, batch_size, seed=42, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        by_group = defaultdict(lambda: {0: [], 1: []})
        for idx in indices:
            r = rows[int(idx)]
            by_group[r["identity_group"]][int(r["video_label"])].append(int(idx))
        self.pairs, self.singles = [], []
        for g, d in by_group.items():
            n_pair = min(len(d[0]), len(d[1]))
            for k in range(n_pair):
                self.pairs.append((d[0][k], d[1][k]))
            self.singles.extend(d[0][n_pair:] + d[1][n_pair:])
        self.n_indices = len(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 7919 + self.epoch)
        pairs = list(self.pairs)
        singles = list(self.singles)
        rng.shuffle(pairs)
        rng.shuffle(singles)
        flat = []
        for a, b in pairs:
            flat.extend([a, b])
        # Interleave leftovers rather than appending them, so the tail
        # batches are not systematically unpaired.
        if singles:
            step = max(1, len(flat) // (len(singles) + 1))
            for j, s in enumerate(singles):
                flat.insert(min(len(flat), (j + 1) * step + j), s)
        batch = []
        for idx in flat:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.pairs) * 2 + len(self.singles)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)


def worker_init_fn(worker_id):
    base = CONFIG["SEED"] + worker_id
    random.seed(base)
    np.random.seed(base)
    torch.manual_seed(base)


# =====================================================================
# Splitting
# =====================================================================

def build_3way_split_packed(rows, identity_disjoint=True, fracs=(0.60, 0.20, 0.20),
                            seed=42, verbose=True):
    """Train/val/test indices targeting `fracs` by SAMPLE count, not group
    count -- identity groups differ wildly in size, so splitting on group
    count alone produces badly skewed sample fractions.

    With identity_disjoint=True no identity group appears in more than one
    partition, and an assertion enforces it.  identity_disjoint=False is the
    leakage ablation and reproduces the naive per-clip random split.
    """
    rng = random.Random(seed)
    idx_all = list(range(len(rows)))
    if not identity_disjoint:
        rng.shuffle(idx_all)
        n = len(idx_all)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        tr, va, te = idx_all[:a], idx_all[a:b], idx_all[b:]
    else:
        by_group = defaultdict(list)
        for i in idx_all:
            by_group[rows[i]["identity_group"]].append(i)
        groups = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
        rng.shuffle(groups)
        # Largest-remainder packing: walk groups and drop each into whichever
        # partition is furthest below its target share.
        n_total = len(idx_all)
        targets = [fracs[0] * n_total, fracs[1] * n_total, fracs[2] * n_total]
        buckets = [[], [], []]
        counts = [0, 0, 0]
        for g in groups:
            deficits = [targets[k] - counts[k] for k in range(3)]
            k = int(np.argmax(deficits))
            buckets[k].extend(by_group[g])
            counts[k] += len(by_group[g])
        tr, va, te = buckets

        g_tr = {rows[i]["identity_group"] for i in tr}
        g_va = {rows[i]["identity_group"] for i in va}
        g_te = {rows[i]["identity_group"] for i in te}
        assert not (g_tr & g_va), f"Identity leak train/val: {sorted(g_tr & g_va)[:5]}"
        assert not (g_tr & g_te), f"Identity leak train/test: {sorted(g_tr & g_te)[:5]}"
        assert not (g_va & g_te), f"Identity leak val/test: {sorted(g_va & g_te)[:5]}"

    if verbose:
        def _summary(name, ids):
            vf = sum(int(rows[i]["video_label"]) for i in ids)
            af = sum(int(rows[i]["audio_label"]) for i in ids)
            ng = len({rows[i]["identity_group"] for i in ids})
            print(f"    {name}: {len(ids):5d} samples | {ng:4d} groups | "
                  f"video fake {vf:4d} | audio fake {af:4d}")
        print(f"  Split (identity_disjoint={identity_disjoint}):")
        _summary("train", tr)
        _summary("val  ", va)
        _summary("test ", te)
    return np.array(tr), np.array(va), np.array(te)


# =====================================================================
# Metrics helpers (pure NumPy -- see the import block for why sklearn is
# deliberately not used here)
# =====================================================================

def roc_auc_score(y_true, y_score):
    """Binary AUC-ROC via the Mann-Whitney U / rank-sum identity:
    AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    Ties get averaged ranks, exactly matching sklearn's convention, so this
    is not an approximation -- it is the same statistic computed directly."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Only one class present in y_true.")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # 1-indexed average rank over the tied block [i, j]
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_by_orig = np.empty(len(y_score), dtype=np.float64)
    rank_by_orig[order] = ranks

    sum_ranks_pos = rank_by_orig[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(y_true, y_score):
    """(fpr, tpr, thresholds) at every distinct score value, thresholds
    descending -- the same standard algorithm sklearn's roc_curve uses
    internally (cumulative true/false positive counts at each cut point)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, len(y_true_sorted) - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_idxs]]

    n_pos, n_neg = tps[-1], fps[-1]
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=np.float64)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps, dtype=np.float64)
    return fpr, tpr, thresholds


def balanced_accuracy_score(y_true, y_pred):
    """Mean per-class recall -- identical definition to sklearn's binary
    balanced accuracy."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recalls = []
    for c in np.unique(y_true):
        mask = y_true == c
        recalls.append(float((y_pred[mask] == c).mean()) if mask.sum() > 0 else 0.0)
    return float(np.mean(recalls))


def classification_report_text(y_true, y_pred, target_names=("class 0", "class 1")):
    """Minimal drop-in for sklearn's classification_report (precision,
    recall, f1-score, support per class, plus accuracy and macro/weighted
    averages), formatted the same way. Binary-only, which is all this
    codebase ever needs it for."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rows = []
    supports, precisions, recalls, f1s = [], [], [], []
    for c, name in enumerate(target_names):
        support = int((y_true == c).sum())
        tp = int(((y_pred == c) & (y_true == c)).sum())
        pred_pos = int((y_pred == c).sum())
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((name, prec, rec, f1, support))
        supports.append(support); precisions.append(prec); recalls.append(rec); f1s.append(f1)

    total = sum(supports)
    acc = float((y_true == y_pred).mean()) if total > 0 else 0.0
    macro = (np.mean(precisions), np.mean(recalls), np.mean(f1s))
    w = np.asarray(supports, dtype=np.float64) / max(total, 1)
    weighted = (float(np.sum(w * precisions)), float(np.sum(w * recalls)), float(np.sum(w * f1s)))

    lines = [f"{'':>16}{'precision':>12}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for name, p, r, f, s in rows:
        lines.append(f"{name:>16}{p:>12.2f}{r:>10.2f}{f:>10.2f}{s:>10d}")
    lines.append("")
    lines.append(f"{'accuracy':>16}{'':>12}{'':>10}{acc:>10.2f}{total:>10d}")
    lines.append(f"{'macro avg':>16}{macro[0]:>12.2f}{macro[1]:>10.2f}{macro[2]:>10.2f}{total:>10d}")
    lines.append(f"{'weighted avg':>16}{weighted[0]:>12.2f}{weighted[1]:>10.2f}"
                 f"{weighted[2]:>10.2f}{total:>10d}")
    return "\n".join(lines)


def safe_auc(y_true, y_score):
    try:
        if len(set(int(v) for v in y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


def smooth_labels(y, eps=0.05):
    return y * (1.0 - eps) + 0.5 * eps


def youden_threshold(y_true, y_prob, default=0.5):
    y_true = np.asarray(y_true)
    if len(set(y_true.astype(int).tolist())) < 2:
        return default
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def expected_calibration_error(probs, labels, n_bins=15):
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf = np.maximum(probs, 1 - probs)
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(probs)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(mean_logits, labels, max_iter=100, lr=0.05):
    """Single-parameter temperature scaling on held-out logits."""
    logits = torch.tensor(np.asarray(mean_logits, dtype=np.float32))
    y = torch.tensor(np.asarray(labels, dtype=np.float32))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    lossf = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(logits / torch.exp(log_t), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return 1.0
    t = float(torch.exp(log_t).detach())
    return t if np.isfinite(t) and 0.05 < t < 20 else 1.0


def bootstrap_stable_threshold(y_true, y_prob, n_boot=200, seed=42, default=0.5):
    """Median Youden threshold over bootstrap resamples.  A single-shot
    Youden point on a few-hundred-sample validation set is high-variance;
    the median over resamples is what actually transfers."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(set(y_true.astype(int).tolist())) < 2:
        return default
    rs = np.random.RandomState(seed)
    thrs = []
    for _ in range(n_boot):
        idx = rs.randint(0, len(y_true), len(y_true))
        if len(set(y_true[idx].astype(int).tolist())) < 2:
            continue
        thrs.append(youden_threshold(y_true[idx], y_prob[idx], default))
    return float(np.median(thrs)) if thrs else default


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, seed=42, alpha=0.05):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rs.randint(0, len(y_true), len(y_true))
        if len(set(y_true[idx].astype(int).tolist())) < 2:
            continue
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# =====================================================================
# Training
# =====================================================================

def evaluate_deterministic(model_local, loader, cfg):
    """Single deterministic pass (dropout off) for per-epoch tracking and
    ablation ranking.  Reported numbers use the MC-Dropout estimator instead,
    and calibrate under that SAME estimator -- mixing the two is what made
    v3.3's calibration meaningless."""
    m = unwrap(model_local)
    m.eval()
    acc = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            crops, mfcc, sync_a, sync_off, vlab, alab, mask, _gid = [
                b.to(device, non_blocking=True) for b in batch]
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # One backbone pass per batch.  The sync head's negative needs
                # the SAME crops against shifted audio, so it reuses the
                # tokens; calling m(...) again would re-run the backbone on
                # identical pixels and roughly double eval cost, which is paid
                # once per epoch.
                v_tok, m_tok = m.encode_visual(crops)
                out = m.forward_from_tokens(v_tok, m_tok, mfcc, sync_a, visual_mask=mask)
                out_neg = (m.forward_from_tokens(v_tok, m_tok, mfcc, sync_off, visual_mask=mask)
                           if out["sync_logit"] is not None else None)
            acc["v_prob"].extend(torch.sigmoid(out["video_logit"].float()).cpu().numpy())
            acc["v_true"].extend(vlab.cpu().numpy())
            acc["a_prob"].extend(torch.sigmoid(out["audio_logit"].float()).cpu().numpy())
            acc["a_true"].extend(alab.cpu().numpy())
            if out["fusion_logit"] is not None:
                acc["f_prob"].extend(torch.sigmoid(out["fusion_logit"].float()).cpu().numpy())
                acc["f_true"].extend(torch.clamp(vlab + alab, 0, 1).cpu().numpy())
            if out_neg is not None:
                # Matched window scores 0; the time-shifted negative scores 1.
                acc["s_prob"].extend(torch.sigmoid(out["sync_logit"].float()).cpu().numpy())
                acc["s_true"].extend([0] * crops.size(0))
                acc["s_prob"].extend(torch.sigmoid(out_neg["sync_logit"].float()).cpu().numpy())
                acc["s_true"].extend([1] * crops.size(0))
    return {"video_auc": safe_auc(acc["v_true"], acc["v_prob"]),
            "audio_auc": safe_auc(acc["a_true"], acc["a_prob"]),
            "sync_auc": safe_auc(acc["s_true"], acc["s_prob"]) if acc["s_true"] else float("nan"),
            "fusion_auc": safe_auc(acc["f_true"], acc["f_prob"]) if acc["f_true"] else float("nan"),
            "raw": dict(acc)}


def run_training_pipeline(manifest, crop_dir, cfg, tag="main", seed=None, verbose=True):
    """Two-phase fine-tune (frozen head warmup -> layer-wise-decayed backbone
    fine-tune).  Returns a result dict whose "model" holds the BEST
    checkpoint's weights, reloaded before return."""
    from torch.utils.data import DataLoader

    seed = cfg["SEED"] if seed is None else seed
    seed_everything(seed)
    cfg = dict(cfg, SEED=seed)
    torch.backends.cudnn.benchmark = True

    train_idx, val_idx, test_idx = build_3way_split_packed(
        manifest, identity_disjoint=cfg["IDENTITY_DISJOINT_SPLIT"],
        fracs=cfg["SPLIT_FRACS"], seed=seed, verbose=verbose)

    train_ds = CrossFuseCropDataset(manifest, train_idx, crop_dir, cfg, training=True)
    val_ds = CrossFuseCropDataset(manifest, val_idx, crop_dir, cfg, training=False)
    test_ds = CrossFuseCropDataset(manifest, test_idx, crop_dir, cfg, training=False)

    nw = cfg["NUM_WORKERS"]
    common = dict(num_workers=nw, pin_memory=True, worker_init_fn=worker_init_fn)
    if cfg["USE_PAIRED_BATCHES"]:
        sampler = PairedBatchSampler(manifest, train_idx, cfg["BATCH_CLIPS"], seed=seed)
        train_loader = DataLoader(train_ds, batch_sampler=_RemapBatchSampler(sampler, train_idx),
                                  persistent_workers=(nw > 0), **common)
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_CLIPS"], shuffle=True,
                                  drop_last=True, persistent_workers=(nw > 0), **common)
    val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_CLIPS"], shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg["BATCH_CLIPS"], shuffle=False, **common)

    model_local = build_model(cfg).to(device)
    load_visual_encoder(model_local, cfg.get("SBI_PRETRAINED_ENCODER"), verbose=verbose)
    if cfg.get("MULTI_GPU", False):
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpu > 1:
            model_local.enable_multi_gpu()
            if verbose:
                print(f"  [{tag}] visual backbone on DataParallel across {n_gpu} GPUs "
                      f"({cfg['BATCH_CLIPS']} clips x {cfg['WINDOW_FRAMES']} frames = "
                      f"{cfg['BATCH_CLIPS'] * cfg['WINDOW_FRAMES']} images/step, sharded)")
    modality = cfg.get("MODALITY", "both")

    tv = [int(manifest[i]["video_label"]) for i in train_idx]
    n_pos, n_neg = sum(tv), len(tv) - sum(tv)
    if cfg["GENERATE_HARD_NEGATIVES"]:
        conv = CrossFuseCropDataset.SELFBLEND_P * n_neg
        n_pos, n_neg = n_pos + conv, n_neg - conv
    pos_weight = torch.tensor(max(n_neg, 1.0) / max(n_pos, 1.0),
                              dtype=torch.float32).to(device)
    crit_video = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_frame = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    crit_plain = nn.BCEWithLogitsLoss()

    total_epochs = cfg["EPOCHS_FROZEN"] + cfg["EPOCHS_FINETUNE"]
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    os.makedirs(cfg["WORK_DIR"], exist_ok=True)
    ckpt_path = os.path.join(cfg["WORK_DIR"], f"crossfuse_v5_{tag}_s{seed}.pth")

    def make_optim(phase):
        if phase == 0:
            model_local.set_backbone_trainable(len(model_local.raw_backbone()))
            groups = [{"params": [p for n_, p in model_local.named_parameters()
                                  if p.requires_grad and not n_.startswith("backbone")],
                       "lr": cfg["LR_HEAD"] * 10}]
        else:
            model_local.set_backbone_trainable(cfg["FREEZE_BLOCKS"])
            groups = model_local.backbone_param_groups(cfg["LR_BACKBONE"], cfg["LLRD"])
            groups.append({"params": [p for n_, p in model_local.named_parameters()
                                      if p.requires_grad and not n_.startswith("backbone")],
                           "lr": cfg["LR_HEAD"]})
        return torch.optim.AdamW(groups, weight_decay=1e-4)

    optimizer = make_optim(0)
    best = {"mean_auc": -1.0, "epoch": -1, "video": float("nan"), "audio": float("nan"),
            "sync": float("nan"), "fusion": float("nan"), "train_video": float("nan"),
            "train_sync": float("nan")}
    since_best = 0

    if verbose:
        print(f"  [{tag}] modality={modality} CA={cfg['USE_CROSS_ATTENTION']} "
              f"sync={cfg['USE_SYNC_HEAD']} fusion={cfg['USE_FUSION_HEAD']} "
              f"paired={cfg['USE_PAIRED_BATCHES']} seed={seed} -- "
              f"phase 1 frozen x{cfg['EPOCHS_FROZEN']}, "
              f"phase 2 fine-tune x{cfg['EPOCHS_FINETUNE']}")

    for epoch in range(total_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        phase = 0 if epoch < cfg["EPOCHS_FROZEN"] else 1
        if epoch == cfg["EPOCHS_FROZEN"]:
            optimizer = make_optim(1)
            if verbose:
                n_tr = sum(p.numel() for p in model_local.parameters() if p.requires_grad)
                print(f"  [{tag}] --- unfreezing backbone from block "
                      f"{cfg['FREEZE_BLOCKS']} ({n_tr:,} trainable) ---")
        if phase == 1:
            # Cosine decay with 2-epoch warmup, scaled per-group off each
            # group's OWN base LR, which keeps layer-wise decay intact
            # instead of collapsing every group onto one curve.
            prog = (epoch - cfg["EPOCHS_FROZEN"]) / max(1, cfg["EPOCHS_FINETUNE"])
            warm = min(1.0, (epoch - cfg["EPOCHS_FROZEN"] + 1) / 2.0)
            factor = max(warm * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))), 0.02)
            for g in optimizer.param_groups:
                g.setdefault("initial_lr", g["lr"])
                g["lr"] = g["initial_lr"] * factor

        model_local.train()
        run_loss, nb = 0.0, 0
        tr = defaultdict(list)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            crops, mfcc, sync_a, sync_off, vlab, alab, mask, gid = [
                b.to(device, non_blocking=True) for b in batch]
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # encode_visual ONCE; the sync loss needs a second HEAD pass
                # on the same crops with mismatched audio, and re-running the
                # backbone would double the dominant cost for zero benefit.
                # Gradients still reach the backbone from both passes via the
                # shared tokens.
                v_tok, m_tok = model_local.encode_visual(crops)
                out = model_local.forward_from_tokens(v_tok, m_tok, mfcc, sync_a,
                                                      visual_mask=mask)
                loss = torch.zeros((), device=device)
                if modality in ("both", "video_only"):
                    loss = loss + crit_video(out["video_logit"], smooth_labels(vlab, 0.05))
                    # Auxiliary per-frame supervision, masked to valid frames.
                    fl = crit_frame(out["frame_logits"],
                                    smooth_labels(vlab, 0.05).unsqueeze(1)
                                    .expand_as(out["frame_logits"]))
                    mf = mask.float()
                    loss = loss + cfg["LAMBDA_FRAME"] * (
                        (fl * mf).sum() / mf.sum().clamp(min=1.0))
                if modality in ("both", "audio_only"):
                    loss = loss + crit_plain(out["audio_logit"], alab)

                if out["fusion_logit"] is not None and modality == "both":
                    any_lab = torch.clamp(vlab + alab, 0, 1)
                    loss = loss + cfg["LAMBDA_FUSION"] * crit_plain(out["fusion_logit"], any_lab)

                if out["sync_logit"] is not None and modality == "both":
                    B = crops.size(0)
                    perm = torch.randperm(B, device=device)
                    ok = gid[perm] != gid            # negative 1: another identity
                    neg_a = torch.where(ok.view(-1, 1, 1), sync_a[perm], sync_off)
                    out_neg = model_local.forward_from_tokens(v_tok, m_tok, mfcc, neg_a,
                                                             visual_mask=mask)
                    s_logit = torch.cat([out["sync_logit"], out_neg["sync_logit"]], 0)
                    s_lab = torch.cat([torch.zeros(B, device=device),
                                       torch.ones(B, device=device)], 0)
                    loss = loss + cfg["LAMBDA_SYNC"] * crit_plain(s_logit, s_lab)
                    tr["s_prob"].extend(torch.sigmoid(s_logit.detach().float()).cpu().numpy())
                    tr["s_true"].extend([0] * B + [1] * B)

            scaler.scale(loss / cfg["GRAD_ACCUM"]).backward()
            if (step + 1) % cfg["GRAD_ACCUM"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_local.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss += float(loss.detach())
            nb += 1
            tr["v_prob"].extend(torch.sigmoid(out["video_logit"].detach().float()).cpu().numpy())
            tr["v_true"].extend(vlab.detach().cpu().numpy())

        val = evaluate_deterministic(model_local, val_loader, cfg)
        train_video_auc = safe_auc(tr["v_true"], tr["v_prob"])
        train_sync_auc = safe_auc(tr["s_true"], tr["s_prob"]) if tr["s_true"] else float("nan")

        if modality == "video_only":
            mean_auc = val["video_auc"]
        elif modality == "audio_only":
            mean_auc = val["audio_auc"]
        else:
            # Selection uses ONLY the two attribution heads, so fusion
            # ablation arms are checkpoint-selected identically and their
            # comparison stays fair.
            mean_auc = 0.5 * (val["video_auc"] + val["audio_auc"])

        improved = mean_auc > best["mean_auc"]
        if improved:
            best.update(mean_auc=mean_auc, epoch=epoch, video=val["video_auc"],
                        audio=val["audio_auc"], sync=val["sync_auc"],
                        fusion=val["fusion_auc"], train_video=train_video_auc,
                        train_sync=train_sync_auc)
            since_best = 0
            torch.save(portable_state_dict(model_local), ckpt_path)
        else:
            since_best += 1

        if verbose:
            print(f"  [{tag}] Ep [{epoch + 1:02d}/{total_epochs}] p{phase} "
                  f"lr {optimizer.param_groups[-1]['lr']:.2e} "
                  f"loss {run_loss / max(nb, 1):.4f} | "
                  f"train v/s {train_video_auc:.4f}/{train_sync_auc:.4f} | "
                  f"val v/a/s/f {val['video_auc']:.4f}/{val['audio_auc']:.4f}/"
                  f"{val['sync_auc']:.4f}/{val['fusion_auc']:.4f} | "
                  f"gap {train_video_auc - val['video_auc']:+.4f}"
                  f"{' * (New Best)' if improved else ''}")

        if since_best >= cfg["PATIENCE"] and phase == 1:
            if verbose:
                print(f"  [{tag}] Early stopping at epoch {epoch + 1}.")
            break

    load_portable_state_dict(
        model_local, torch.load(ckpt_path, map_location=device, weights_only=False))
    model_local.eval()

    if verbose:
        print(f"  [{tag}] Done. Best epoch {best['epoch'] + 1} -> "
              f"val video {best['video']:.4f} | audio {best['audio']:.4f} | "
              f"sync {best['sync']:.4f} | fusion {best['fusion']:.4f}")
        gap = best["train_video"] - best["video"]
        print(f"  [{tag}] Video train/val gap: {gap:+.4f} "
              f"({'OK' if gap < 0.10 else 'HIGH -- check augmentation is firing'})")
        if not np.isnan(best["sync"]) and best["sync"] < cfg["SYNC_MIN_AUC"]:
            print(f"  [{tag}] NOTE: val sync AUC {best['sync']:.4f} < preregistered "
                  f"{cfg['SYNC_MIN_AUC']}; train sync AUC {best['train_sync']:.4f}. "
                  f"Train near chance => feature bottleneck; train well above "
                  f"chance => generalisation gap.")

    return {"tag": tag, "seed": seed, "checkpoint_path": ckpt_path, "cfg": cfg,
            "best_epoch": best["epoch"] + 1, "best_val": best,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "train_loader": train_loader, "val_loader": val_loader,
            "test_loader": test_loader, "model": model_local}


class _RemapBatchSampler(Sampler):
    """PairedBatchSampler yields MANIFEST row indices; CrossFuseCropDataset is
    positional over its own `indices` list.  This maps one to the other so
    the sampler can stay expressed in manifest terms."""

    def __init__(self, inner, indices):
        self.inner = inner
        self.pos = {int(v): i for i, v in enumerate(indices)}

    def set_epoch(self, epoch):
        self.inner.set_epoch(epoch)

    def __iter__(self):
        for batch in self.inner:
            yield [self.pos[i] for i in batch if i in self.pos]

    def __len__(self):
        return len(self.inner)


def mc_collect(model, loader, cfg, n_samples=20, need_sync=True, need_fusion=True):
    """Run the MC-Dropout estimator over a whole loader.

    Returns {head: {'mean_logit', 'std_prob', 'true'}}.  Calibration is fit on
    THESE outputs and applied to THESE outputs -- fitting a temperature on a
    single dropout-off pass and then applying it to a 20-sample dropout-on
    average (v3.3) makes the calibration meaningless."""
    m = unwrap(model)
    m.eval()
    acc = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for batch in loader:
            crops, mfcc, sync_a, sync_off, vlab, alab, mask, _gid = [
                b.to(device, non_blocking=True) for b in batch]
            tokens = m.encode_visual(crops)
            got = mc_logits(m, crops, mfcc, sync_a, mask, n_samples, tokens=tokens)
            pairs = [("video", vlab), ("audio", alab)]
            if need_fusion and "fusion" in got:
                pairs.append(("fusion", torch.clamp(vlab + alab, 0, 1)))
            for head, lab in pairs:
                if head not in got:
                    continue
                lg = got[head]
                acc[head]["mean_logit"].extend(lg.mean(axis=0).tolist())
                acc[head]["std_prob"].extend(
                    (1 / (1 + np.exp(-lg))).std(axis=0).tolist())
                acc[head]["true"].extend(lab.cpu().numpy().tolist())
            if need_sync and "sync" in got:
                neg = mc_logits(m, crops, mfcc, sync_off, mask, n_samples, tokens=tokens)
                for src, lab in ((got["sync"], 0), (neg["sync"], 1)):
                    acc["sync"]["mean_logit"].extend(src.mean(axis=0).tolist())
                    acc["sync"]["std_prob"].extend(
                        (1 / (1 + np.exp(-src))).std(axis=0).tolist())
                    acc["sync"]["true"].extend([lab] * src.shape[1])
    return {h: {k: np.asarray(v) for k, v in d.items()} for h, d in acc.items()}
