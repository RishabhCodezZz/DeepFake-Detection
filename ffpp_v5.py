"""
FF++ visual pretraining for CrossFuse -- shared library.

STATUS (2026-09): RAN, GATE PASSED.  NB-B1 trained the CLIP ViT-L/14
LayerNorm-tuned encoder on 5,000 FF++ c23 videos (real + 4 families) and it
cleared both preregistered gates zero-shot: Celeb-DF 0.9186 (gate 0.85),
DFDC 0.8494 (gate 0.75).  The encoder (`ffpp-encoder-v5`) then drove the
cross-dataset gain in B-train/C-eval (mean AUC 0.61 -> 0.855); the D2 ablation
suggests this pretraining stage, not mixing FF++ rows into training, is what
matters.  That reading is a hypothesis: the no-pretraining arm (A15) also ran
at LN LR 1e-5 against 1e-4 here, so the two are confounded.  Numbers and
caveats: README.md and CLAUDE.md.

WHY THIS STAGE EXISTS
---------------------
CrossFuse v5 scores 0.95 AUC in-domain on FakeAVCeleb but 0.58 (DFDC) and
0.64 (Celeb-DF v2) zero-shot.  Two independent causes:

1. WRONG MANIPULATION FAMILY.  FakeAVCeleb is ~97% Wav2Lip, a mouth-region
   reenactment.  Celeb-DF and DFDC are full-face identity swaps.  The v5
   video head never saw a face swap during training, so no amount of
   architecture work could have fixed the gap.  FF++ supplies four swap /
   reenactment families (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

2. NON-TRANSFERABLE BACKBONE.  LNCLIP-DF (arXiv:2508.06248) and Effort
   (ICML 2025, arXiv:2411.15633) independently land on CLIP ViT-L/14 frozen
   except its LayerNorms, trained on FF++ c23.  LNCLIP-DF's Celeb-DF v2
   ablation: linear probe 78.1 -> +LN-tuning 94.9 -> +L2-norm 96.2, with
   full fine-tuning and LoRA both overfitting outright.  This module
   implements LN-tuning + L2-norm and skips their alignment/uniformity/slerp
   extras, which are worth +1.7 combined against considerably more moving
   parts.

This stage occupies the slot the cut SBI stage was built for and writes the
same {'backbone_state', 'meta'} checkpoint format, so `load_visual_encoder`
in crossfuse_v5.py consumes it with no changes.

CRITICAL INVARIANT
------------------
`FFPPClassifier` computes its visual token through crossfuse_v5's
`encode_frames`/`image_norm_stats` -- the SAME functions
`CrossFuseModelV5.encode_visual` uses.  A divergence there (different resize,
normalisation constant, or pooling) would still let the checkpoint LOAD
cleanly and simply stop transferring, which is close to undiagnosable from
training curves alone.  Do not inline either function here.
"""

import csv
import math
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

from crossfuse_v5 import (
    CONFIG, UnionFind, build_backbone, encode_frames, image_norm_stats,
    compression_augment_clip, photometric_augment_clip,
    safe_auc, bootstrap_ci, seed_everything, make_rng,
)

__all__ = [
    "FFPP_METHODS", "FFPP_CONFIG",
    "discover_ffpp_videos", "ffpp_identity_tokens", "parse_ffpp_name",
    "build_ffpp_manifest", "split_ffpp_identity_disjoint", "load_ffpp_official_splits",
    "FFPPFrameDataset", "PairedFrameSampler", "FFPPClassifier",
    "train_ffpp_encoder", "score_video_frames", "evaluate_ffpp_videos",
]

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

FFPP_CONFIG = {
    # Crop convention MUST match CONFIG's, or the encoder does not transfer.
    "CROP_SIZE":        CONFIG["CROP_SIZE"],     # 256
    "FACE_MARGIN":      CONFIG["FACE_MARGIN"],   # 0.30  (NOT A2's 0.45)
    "TRAIN_RES":        CONFIG["TRAIN_RES"],     # 224
    # 16, not the paper's 32: 720 real + 2880 fake = 3600 videos, and
    # 3600 * 32 * 256^2 * 3 = 22.6 GB overflows Kaggle's 20 GB notebook
    # output cap.  At 16 it is 11.3 GB.  57,600 frames is ample for tuning
    # ~40k LayerNorm parameters.
    "FRAMES_PER_VIDEO": 16,
    # Oversampled so `extract_face_crops_contiguous` has slack to drop
    # frames where detection failed and still return a full window.
    "DETECT_FRAMES":    32,

    "EPOCHS":           10,
    "BATCH_FRAMES":     32,
    "LR":               1e-4,    # LN-tuning LR; see CONFIG["LR_LN"] note
    "LR_MIN":           1e-5,
    "WARMUP_EPOCHS":    1,
    "DROPOUT":          0.3,
    "NUM_WORKERS":      4,
    "MULTI_GPU":        False,
    "USE_PAIRED":       True,
    "USE_COMPRESSION_AUG": True,
    "COMPRESSION_AUG_P":   0.30,
    # PREREGISTERED, in the same spirit as CONFIG["SYNC_MIN_AUC"]: written
    # down before the run so a disappointing result cannot be quietly
    # reinterpreted afterwards.  Below this, do NOT proceed to the
    # multimodal stage -- debug or report the reproduction failure.
    "GATE_CELEBDF_AUC": 0.85,
    "GATE_DFDC_AUC":    0.75,
}


# =====================================================================
# Discovery and identity bookkeeping
# =====================================================================

def parse_ffpp_name(video_path):
    """(target_id, source_id) from an FF++ filename.

    Reals are `033.mp4` -> ("033", None).  Manipulated clips are
    `033_097.mp4`, meaning 097's face rendered onto 033's video, so 033 is
    the base clip and its paired real is `033.mp4` -> ("033", "097").
    """
    stem = os.path.splitext(os.path.basename(video_path.replace("\\", "/")))[0]
    parts = re.findall(r"\d+", stem)
    if not parts:
        return stem, None
    return parts[0], (parts[1] if len(parts) > 1 else None)


def ffpp_identity_tokens(video_path):
    """Identity tokens for union-find grouping.

    crossfuse_v5.get_identity_tokens matches FakeAVCeleb's `id\\d+` pattern
    and finds NOTHING in FF++'s bare numerics -- it would fall back to one
    group per parent folder, silently collapsing the whole dataset into ~5
    groups and destroying the identity-disjoint split.  Hence this variant.
    """
    target, source = parse_ffpp_name(video_path)
    toks = {f"ff::{target}"}
    if source is not None:
        toks.add(f"ff::{source}")
    return toks


def discover_ffpp_videos(roots=("/kaggle/input",), compression="c23", verbose=True):
    """{'original': [...], 'Deepfakes': [...], ...} of FF++ video paths.

    Globs the canonical layout first, then falls back to a path-token scan
    for mirrors that flatten directories.  Returns whatever it finds; the
    CALLER is responsible for asserting completeness (NB-A3 does, loudly,
    before extracting anything -- a missing manipulation family is the
    difference between this stage working and quietly training on 3 of 4
    forgery types).
    """
    import glob as _glob

    found = {k: [] for k in ("original",) + FFPP_METHODS}
    for root in roots:
        for pat in (f"{root}/**/original_sequences/**/{compression}/**/*.mp4",
                    f"{root}/**/original_sequences/**/*.mp4",
                    # Flattened mirrors (e.g. xdxd003/ff-c23's "FaceForensics++_C23")
                    # drop the original_sequences/ wrapper entirely and just have a
                    # bare "original" folder as a sibling of Deepfakes/Face2Face/...
                    # -- same style of fallback as the per-method {meth}/**/*.mp4
                    # pattern below, just for the one family whose folder name isn't
                    # also its FFPP_METHODS key.
                    f"{root}/**/original/**/*.mp4"):
            found["original"] += _glob.glob(pat, recursive=True)
            if found["original"]:
                break
        for meth in FFPP_METHODS:
            for pat in (f"{root}/**/manipulated_sequences/{meth}/{compression}/**/*.mp4",
                        f"{root}/**/manipulated_sequences/{meth}/**/*.mp4",
                        f"{root}/**/{meth}/**/*.mp4"):
                found[meth] += _glob.glob(pat, recursive=True)
                if found[meth]:
                    break

    for k in found:
        allp = sorted(set(found[k]))
        # A broad glob can also match a c40 mirror nested underneath, which
        # would train on a compression level we never meant to include.  Keep
        # the filtered set when it is non-empty; fall back to everything for
        # mirrors that carry no compression level in the path at all.
        filtered = [p for p in allp
                    if compression in p.replace("\\", "/").split("/")]
        found[k] = filtered or allp

    if verbose:
        print(f"FF++ ({compression}) discovery:")
        for k in ("original",) + FFPP_METHODS:
            print(f"  {k:16s} {len(found[k]):5d} videos")
    return found


def build_ffpp_manifest(found, cap_per_method=None, seed=42, verbose=True):
    """Manifest rows: key, path, method, label, target_id, source_id.

    `label` is 1 for every manipulated family and 0 for originals.  Row keys
    are unique across methods because the method name is part of the key --
    `Deepfakes/033_097.mp4` and `FaceSwap/033_097.mp4` are different clips
    with the same filename.
    """
    rng = random.Random(seed)
    rows = []
    for meth in ("original",) + FFPP_METHODS:
        paths = list(found.get(meth, []))
        rng.shuffle(paths)
        if cap_per_method and meth != "original":
            paths = paths[:cap_per_method]
        for p in sorted(paths):
            target, source = parse_ffpp_name(p)
            stem = os.path.splitext(os.path.basename(p.replace("\\", "/")))[0]
            rows.append({
                "key": f"{meth}_{stem}",
                "path": p,
                "method": meth,
                "label": 0 if meth == "original" else 1,
                "target_id": target,
                "source_id": source or "",
            })

    uf = UnionFind()
    for r in rows:
        toks = list(ffpp_identity_tokens(r["path"]))
        for t in toks[1:]:
            uf.union(toks[0], t)
        uf.find(toks[0])
    for r in rows:
        r["identity_group"] = uf.find(next(iter(ffpp_identity_tokens(r["path"]))))

    if verbose:
        n_fake = sum(r["label"] for r in rows)
        print(f"Manifest: {len(rows)} rows ({len(rows) - n_fake} real / {n_fake} fake), "
              f"{len({r['identity_group'] for r in rows})} identity groups")
    return rows


def load_ffpp_official_splits(roots=("/kaggle/input",)):
    """{'train'|'val'|'test': set(video_id)} from FF++'s splits JSON, or None.

    The canonical protocol (720/140/140 videos) that LNCLIP-DF and Effort
    both train under.  The files are `train.json` / `val.json` / `test.json`,
    each a list of [target_id, source_id] pairs, and are constructed so a
    pair never straddles two splits.  Not every Kaggle mirror ships them.
    """
    import glob as _glob
    import json as _json

    out = {}
    for name in ("train", "val", "test"):
        hits = [p for root in roots
                for p in _glob.glob(f"{root}/**/{name}.json", recursive=True)]
        for p in hits:
            try:
                with open(p) as f:
                    data = _json.load(f)
                ids = {str(x) for pair in data for x in pair}
                if ids:
                    out[name] = ids
                    break
            except Exception:
                continue
    return out if len(out) == 3 else None


def split_ffpp_identity_disjoint(rows, fracs=(0.85, 0.15), seed=42,
                                 official_splits=None, verbose=True):
    """(train_idx, val_idx) for checkpoint selection.

    No test split: FF++ is the training corpus, and every number this stage
    reports is zero-shot on Celeb-DF v2 / DFDC.  A held-in FF++ test score
    would only measure how well the model fits its own training
    distribution -- exactly the quantity that misled v5.

    Splitting FF++ by identity is subtler than FakeAVCeleb.  FF++ pairs its
    1000 videos by a permutation (each id is a target once and a source
    once), so union-find over {target, source} follows that permutation's
    CYCLES -- and a random permutation's largest cycle typically covers
    ~60% of its elements.  Strict transitive closure therefore collapses
    most of FF++ into one giant component and cannot yield a split at all.

    Resolution order:
      1. `official_splits` (FF++'s own train/val/test json) -- preferred,
         and the protocol the reference papers use.
      2. Split on target_id, then MEASURE and PRINT how many val rows share
         a source identity with train.  Reported rather than asserted away,
         because with a giant component zero leakage is unachievable.
    """
    def frac_fake(ix):
        return float(np.mean([rows[i]["label"] for i in ix])) if ix else float("nan")

    if official_splits:
        train_ids = official_splits["train"]
        val_ids = official_splits["val"] | official_splits["test"]
        train_idx = [i for i, r in enumerate(rows) if r["target_id"] in train_ids]
        val_idx = [i for i, r in enumerate(rows) if r["target_id"] in val_ids]
        if verbose:
            print(f"Split (FF++ OFFICIAL): train {len(train_idx)} "
                  f"(fake {frac_fake(train_idx):.2f}) | val {len(val_idx)} "
                  f"(fake {frac_fake(val_idx):.2f})")
        return train_idx, val_idx

    comps = defaultdict(int)
    for r in rows:
        comps[r["identity_group"]] += 1
    biggest = max(comps.values()) / max(len(rows), 1)

    targets = sorted({r["target_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(targets)
    n_train = max(1, int(round(len(targets) * fracs[0])))
    train_t = set(targets[:n_train])
    train_idx = [i for i, r in enumerate(rows) if r["target_id"] in train_t]
    val_idx = [i for i, r in enumerate(rows) if r["target_id"] not in train_t]

    train_ids = {rows[i]["target_id"] for i in train_idx}
    for i in train_idx:
        if rows[i]["source_id"]:
            train_ids.add(rows[i]["source_id"])
    leaked = sum(1 for i in val_idx
                 if rows[i]["source_id"] and rows[i]["source_id"] in train_ids)

    if verbose:
        print(f"Split (target_id fallback -- FF++ official json not found): "
              f"train {len(train_idx)} (fake {frac_fake(train_idx):.2f}) | "
              f"val {len(val_idx)} (fake {frac_fake(val_idx):.2f})")
        print(f"  largest identity component covers {biggest:.0%} of rows, so a "
              f"strictly identity-disjoint split is not available; "
              f"{leaked}/{len(val_idx)} val rows ({leaked / max(len(val_idx), 1):.0%}) "
              f"share a SOURCE identity with train.")
        print(f"  This affects checkpoint selection only -- the reported numbers "
              f"for this stage are zero-shot on Celeb-DF v2 / DFDC, where no "
              f"FF++ identity appears at all. Attach FF++'s splits json to "
              f"remove the caveat entirely.")
    return train_idx, val_idx


# =====================================================================
# Dataset
# =====================================================================

class FFPPFrameDataset(Dataset):
    """One randomly-chosen cached frame per __getitem__.

    Frame-level, not clip-level: FF++ manipulations are per-frame spatial
    artifacts, and LN-tuning needs image diversity far more than it needs
    temporal context.  Eval mode pins the middle frame so validation is
    deterministic across epochs and seeds.
    """

    def __init__(self, rows, indices, crop_dir, cfg=None, training=False, seed=42):
        self.rows = rows
        self.indices = [int(x) for x in indices]
        self.crop_dir = crop_dir
        self.cfg = cfg or FFPP_CONFIG
        self.training = training
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ridx = self.indices[i]
        r = self.rows[ridx]
        arr = np.load(os.path.join(self.crop_dir, r["key"] + ".npy"), mmap_mode="r")
        rng = random.Random(
            (self.seed * 1_000_003 + ridx * 9176)
            ^ (random.getrandbits(30) if self.training else 0))

        f = rng.randrange(arr.shape[0]) if self.training else arr.shape[0] // 2
        # (1,H,W,3) so the clip augmentations apply unchanged -- they are
        # written against a leading time axis.
        frame = np.array(arr[f])[None, ...]

        if self.training:
            if (self.cfg["USE_COMPRESSION_AUG"]
                    and rng.random() < self.cfg["COMPRESSION_AUG_P"]):
                frame = compression_augment_clip(frame, rng, p_frame=1.0)
            frame = photometric_augment_clip(frame, rng)

        return (torch.from_numpy(np.ascontiguousarray(frame[0])),
                torch.tensor(float(r["label"])))


class PairedFrameSampler(Sampler):
    """Emit each manipulated clip adjacent to the REAL clip it was built on.

    LNCLIP-DF finds paired real/fake batching from a common source to be the
    decisive factor against shortcut learning: within a pair, identity, pose,
    lighting and capture pipeline are held constant, so the only thing
    separating the two labels is the manipulation.  In FF++ that pairing is
    exact -- `Deepfakes/033_097.mp4` renders onto the frames of `033.mp4`.

    A real clip is the base for up to 4 fakes and is therefore drawn ~4x per
    epoch.  That is deliberate: it also balances the 1:4 real/fake ratio, so
    no pos_weight is needed on the paired path.
    """

    def __init__(self, rows, indices, batch_size, seed=42, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        idx_set = set(int(i) for i in indices)
        real_by_target = {}
        for i in idx_set:
            r = rows[i]
            if r["label"] == 0:
                real_by_target[r["target_id"]] = i

        self.pairs, self.singles = [], []
        for i in sorted(idx_set):
            r = rows[i]
            if r["label"] == 0:
                continue
            base = real_by_target.get(r["target_id"])
            if base is None:
                self.singles.append(i)      # base real not in this split
            else:
                self.pairs.append((base, i))
        # Reals whose fakes all landed elsewhere would otherwise never be seen.
        paired_reals = {b for b, _ in self.pairs}
        self.singles += [i for i in sorted(idx_set)
                         if rows[i]["label"] == 0 and i not in paired_reals]

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


class _RemapBatchSampler(Sampler):
    """PairedFrameSampler speaks manifest row indices; FFPPFrameDataset is
    positional over its own `indices` list.  Same shim as crossfuse_v5's."""

    def __init__(self, base, indices):
        self.base = base
        self.pos = {int(v): k for k, v in enumerate(indices)}

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        for batch in self.base:
            remapped = [self.pos[i] for i in batch if i in self.pos]
            if remapped:
                yield remapped

    def __len__(self):
        return len(self.base)


# =====================================================================
# Model
# =====================================================================

class FFPPClassifier(nn.Module):
    """Frame-level binary classifier over the shared visual encoder.

    Only `backbone_state` is ever saved: the multimodal stage rebuilds its
    own heads, and carrying a stale frame-level head across would be
    misleading.
    """

    def __init__(self, backbone="clip_vit_l14", pretrained=True, dropout=0.3,
                 train_res=None, grad_checkpointing=True):
        super().__init__()
        self.backbone, self.dim = build_backbone(backbone, pretrained,
                                                 grad_checkpointing)
        self.is_vit = getattr(self.backbone, "is_vit", False)
        self.train_res = train_res or FFPP_CONFIG["TRAIN_RES"]
        if self.is_vit:
            self.backbone.check_input_res(self.train_res)
        self.spatial_pool = nn.AdaptiveAvgPool2d(3)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.dim, 1))
        mean, std = image_norm_stats(self.is_vit)
        self.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std).view(1, 3, 1, 1))

    def raw_backbone(self):
        return (self.backbone.module if isinstance(self.backbone, nn.DataParallel)
                else self.backbone)

    def set_ln_trainable(self):
        """LN-tuning: freeze everything, then re-enable LayerNorm affines.

        Returns (n_trainable, n_total) so the caller can PRINT the ratio --
        a silent all-frozen backbone looks exactly like a bad result."""
        raw = self.raw_backbone()
        for p in raw.parameters():
            p.requires_grad = False
        if self.is_vit:
            for ln in raw.layernorms():
                for p in ln.parameters():
                    p.requires_grad = True
        else:
            # CNN comparison arm (ablation A14): fine-tune from the last
            # blocks, matching CONFIG["FREEZE_BLOCKS"].
            for i, block in enumerate(raw):
                if i >= CONFIG["FREEZE_BLOCKS"]:
                    for p in block.parameters():
                        p.requires_grad = True
        n_tr = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        return n_tr, sum(p.numel() for p in raw.parameters())

    def forward(self, x_uint8):
        f = encode_frames(self.backbone, x_uint8, self.img_mean, self.img_std,
                          self.train_res, self.is_vit, self.spatial_pool)
        return self.head(f).squeeze(-1)


# =====================================================================
# Training
# =====================================================================

def train_ffpp_encoder(rows, train_idx, val_idx, crop_dir, out_path,
                       backbone="clip_vit_l14", cfg=None, seed=42,
                       device=None, pretrained=True, verbose=True):
    """Train the FF++ visual encoder; save {'backbone_state', 'meta'}.

    Adam with NO weight decay and a 1-epoch warmup into cosine decay, per
    LNCLIP-DF.  fp16 AMP rather than their bf16: Kaggle's T4 is Turing and
    has no bf16 path, so bf16 there silently falls back to fp32 and halves
    throughput.

    `pretrained=False` starts from random CLIP weights -- meaningless as a
    detector, but it is the honest control for "how much of the result is
    CLIP's pretraining rather than FF++ supervision", and it lets the whole
    pipeline be exercised offline without a 1.7 GB download.
    """
    from torch.utils.data import DataLoader

    cfg = dict(FFPP_CONFIG, **(cfg or {}))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    model = FFPPClassifier(backbone, pretrained=pretrained, dropout=cfg["DROPOUT"],
                           train_res=cfg["TRAIN_RES"]).to(device)
    n_tr, n_all = model.set_ln_trainable()
    if verbose:
        print(f"[ffpp] backbone={backbone} trainable {n_tr:,}/{n_all:,} "
              f"({100 * n_tr / max(n_all, 1):.3f}%)")
    assert n_tr > 0, ("no trainable backbone parameters -- LN-tuning would be a "
                      "frozen linear probe and score ~0.78, not ~0.95")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cfg["MULTI_GPU"] and n_gpu > 1:
        model.backbone = nn.DataParallel(model.backbone)
        if verbose:
            print(f"[ffpp] visual backbone sharded across {n_gpu} GPUs")

    tr_ds = FFPPFrameDataset(rows, train_idx, crop_dir, cfg, training=True, seed=seed)
    va_ds = FFPPFrameDataset(rows, val_idx, crop_dir, cfg, training=False, seed=seed)
    common = dict(num_workers=cfg["NUM_WORKERS"], pin_memory=True)
    if cfg["USE_PAIRED"]:
        sampler = PairedFrameSampler(rows, train_idx, cfg["BATCH_FRAMES"], seed=seed)
        tr = DataLoader(tr_ds, batch_sampler=_RemapBatchSampler(sampler, train_idx),
                        persistent_workers=(cfg["NUM_WORKERS"] > 0), **common)
    else:
        sampler = None
        tr = DataLoader(tr_ds, batch_size=cfg["BATCH_FRAMES"], shuffle=True,
                        drop_last=True,
                        persistent_workers=(cfg["NUM_WORKERS"] > 0), **common)
    va = DataLoader(va_ds, batch_size=cfg["BATCH_FRAMES"], shuffle=False, **common)

    # Paired batching already balances 1:4; pos_weight is the guard for the
    # USE_PAIRED=False ablation arm.
    lab = [rows[i]["label"] for i in train_idx]
    n_pos = max(sum(lab), 1)
    pos_w = torch.tensor(max(len(lab) - n_pos, 1) / n_pos, dtype=torch.float32,
                         device=device) if not cfg["USE_PAIRED"] else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=cfg["LR"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    steps_per_epoch = max(1, len(tr))
    warm_steps = cfg["WARMUP_EPOCHS"] * steps_per_epoch
    total_steps = cfg["EPOCHS"] * steps_per_epoch

    def lr_at(step):
        if step < warm_steps:
            return cfg["LR_MIN"] + (cfg["LR"] - cfg["LR_MIN"]) * step / max(warm_steps, 1)
        prog = (step - warm_steps) / max(total_steps - warm_steps, 1)
        return cfg["LR_MIN"] + 0.5 * (cfg["LR"] - cfg["LR_MIN"]) * \
            (1 + math.cos(math.pi * min(prog, 1.0)))

    best = {"auc": -1.0, "epoch": -1}
    gstep = 0
    for epoch in range(cfg["EPOCHS"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        run, nb, tp, tt = 0.0, 0, [], []
        for x, y in tr:
            for g in optim.param_groups:
                g["lr"] = lr_at(gstep)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logit = model(x)
                loss = crit(logit, y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim)
            scaler.update()
            run += float(loss.detach()); nb += 1; gstep += 1
            tp.extend(torch.sigmoid(logit.detach().float()).cpu().numpy())
            tt.extend(y.detach().cpu().numpy())

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in va:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logit = model(x)
                vp.extend(torch.sigmoid(logit.float()).cpu().numpy())
                vt.extend(y.numpy())
        tr_auc, va_auc = safe_auc(tt, tp), safe_auc(vt, vp)

        improved = va_auc > best["auc"]
        if improved:
            best.update(auc=va_auc, epoch=epoch)
            torch.save({
                "backbone_state": {k: v.cpu() for k, v in
                                   model.raw_backbone().state_dict().items()},
                "meta": {"backbone": backbone, "epoch": epoch, "val_auc": va_auc,
                         "train_res": cfg["TRAIN_RES"], "crop_size": cfg["CROP_SIZE"],
                         "face_margin": cfg["FACE_MARGIN"], "seed": seed,
                         "n_train": len(train_idx), "paired": cfg["USE_PAIRED"],
                         "clip_pretrained": pretrained},
            }, out_path)
        if verbose:
            print(f"  [ffpp] Ep [{epoch + 1:02d}/{cfg['EPOCHS']}] "
                  f"lr {optim.param_groups[0]['lr']:.2e} loss {run / max(nb, 1):.4f} | "
                  f"train {tr_auc:.4f} val {va_auc:.4f} | gap {tr_auc - va_auc:+.4f}"
                  f"{' * (New Best)' if improved else ''}")

    ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
    model.raw_backbone().load_state_dict(ckpt["backbone_state"])
    model.to(device).eval()
    if verbose:
        print(f"  [ffpp] Done. Best epoch {best['epoch'] + 1}, "
              f"FF++ val AUC {best['auc']:.4f} -> {out_path}")
        print(f"  [ffpp] NOTE: this is an IN-DOMAIN number. The gate that "
              f"matters is zero-shot Celeb-DF v2 >= {cfg['GATE_CELEBDF_AUC']}.")
    return {"model": model, "checkpoint_path": out_path, "best": best, "cfg": cfg}


# =====================================================================
# Zero-shot evaluation
# =====================================================================

def score_video_frames(model, crops_uint8, device=None, batch_size=32):
    """Mean frame probability for one clip.

    Averaging over ALL extracted frames, not one window: v5's cross-dataset
    scorer judged each video on the first 12 frames of a randomly-placed
    32-frame run, which added avoidable variance to every reported CI.
    """
    device = device or next(model.parameters()).device
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(crops_uint8), batch_size):
            x = torch.from_numpy(
                np.ascontiguousarray(crops_uint8[i:i + batch_size])).to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                p = torch.sigmoid(model(x).float())
            probs.extend(p.cpu().numpy().tolist())
    return float(np.mean(probs)) if probs else 0.5


def evaluate_ffpp_videos(model, entries, name, detector=None, n_frames=32,
                         device=None, verbose=True):
    """Zero-shot video-level AUC over [(video_path, label), ...].

    Reports n and a bootstrap 95% CI, matching how every other number in this
    project is reported.
    """
    from crossfuse_v5 import build_detector, extract_face_crops_contiguous

    device = device or next(model.parameters()).device
    detector = detector or build_detector(str(device))
    y_true, y_score = [], []
    for vpath, label in entries:
        crops, valid, *_ = extract_face_crops_contiguous(
            vpath, detector, rng=make_rng(vpath), n_frames=n_frames,
            out_size=FFPP_CONFIG["CROP_SIZE"])
        if crops is None or int(valid.sum()) == 0:
            continue
        y_score.append(score_video_frames(model, crops[valid], device=device))
        y_true.append(label)

    if len(set(y_true)) < 2:
        if verbose:
            print(f"[{name}] skipped -- fewer than 2 classes (n={len(y_true)})")
        return None
    auc = safe_auc(y_true, y_score)
    lo, hi = bootstrap_ci(y_true, y_score, safe_auc)
    if verbose:
        print(f"[{name}] n={len(y_true)} | AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")
    return {"name": name, "n": len(y_true), "auc": auc, "auc_ci": [lo, hi],
            "y_true": y_true, "y_score": y_score}
