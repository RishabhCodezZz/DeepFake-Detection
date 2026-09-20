# CLAUDE.md — CrossFuse v5 project notes

Living doc. Update the **Status / next steps** section at the bottom after every Kaggle run —
everything above it should stay accurate for longer.

## What this project is

Audio-visual deepfake detection. A shared visual backbone + a temporal audio encoder feed four
heads (`video`, `audio`, `sync`, `fusion`) — see [README.md](README.md) for the architecture
table and the full committed results. The project's stated emphasis is **evaluation honesty**:
negative results are reported, not tuned away. Two are already on the record (SBI pretraining,
cut; the sync head, chance-level) and this doc tracks a third investigation in progress.

A paper draft covering the results in README.md has been submitted and is frozen — not touched
unless it's rejected. Everything below is a **separate, ungraded "major project" track**: real
further work on the same codebase, not busywork, but not judged against the paper.

## Two unrelated "A" naming schemes — do not conflate them

1. **Notebook stage names** (`notebooks/*.ipynb`) — pipeline stages, run in order on Kaggle:
   `A-extract` → `A2-ffpp-extract` (superseded) / `A3-ffpp-full-extract` → `B0-sbi-pretrain`
   (cut) / `B1-ffpp-pretrain` → `B-train` → `C-eval` → `D-ablations`. See the table below.
2. **Ablation arm names**, inside `D-ablations.ipynb`, results in `ablation_results_v5.csv` —
   16 config variants of the *same* B-train model (`A1_video_only` … `A16_clip_full_finetune`).

`A3-ffpp-full-extract.ipynb` (a notebook) and `A3_two_head_no_fusion` (an ablation row) share a
letter by pure coincidence and mean nothing to each other.

## Repo layout

| Path | Role |
|---|---|
| `crossfuse_v5.py` | The library. Config, data pipeline, model, training loop, eval/calibration, NumPy reimplementations of the sklearn metrics used. Imported by every notebook. |
| `ffpp_v5.py` | FF++-specific pipeline for the CLIP-pretraining track ("Track C" below): discovery, identity-disjoint splitting, a lightweight `FFPPClassifier`, `train_ffpp_encoder`, zero-shot video-level eval. |
| `sbi_v5.py` | Self-Blended Images pretraining. **Cut** — see Negative results. Kept for the record; not imported by any current notebook. |
| `notebooks/A-extract.ipynb` | FakeAVCeleb crops + cached audio features → Kaggle Dataset `crops-v5`. |
| `notebooks/A2-ffpp-extract.ipynb` | Old FF++ real-only extraction, built for the cut SBI stage. Superseded by A3, not deleted. |
| `notebooks/A3-ffpp-full-extract.ipynb` | **Unrun.** FF++ real + all 4 fake families (Deepfakes/Face2Face/FaceSwap/NeuralTextures) → `ffpp-crops-v5`. Feeds Track C. |
| `notebooks/B0-sbi-pretrain.ipynb` | SBI pretraining. **Cut.** |
| `notebooks/B1-ffpp-pretrain.ipynb` | **Unrun.** CLIP ViT-L/14, LayerNorm-only tuning, pretrained on FF++ → `ffpp-encoder-v5`. Preregistered gate: Celeb-DF ≥ 0.85, DFDC ≥ 0.75. |
| `notebooks/B2-sync-probe.ipynb` | **Unrun.** Cheap decisive probe for the sync-head rewrite — FakeAVCeleb only, EfficientNet-B4 backbone fully frozen throughout, both `SYNC_ARCH` values trained side by side (~20–30 min on 2×T4). Gate: `SYNC_MIN_AUC=0.70`. Run this before `B-train`, independent of Track C. |
| `notebooks/B-train.ipynb` | Main multimodal training. Already rewritten to consume the merged FakeAVCeleb+FF++ manifest and the FF++-pretrained encoder, but **not yet run against them**. |
| `notebooks/C-eval.ipynb` | Calibration, in-domain test report, cross-dataset suite, figures. |
| `notebooks/D-ablations.ipynb` | 16-row ablation grid → `ablation_results_v5.csv`. Scores IN-DOMAIN only, so it cannot show what fixed cross-dataset transfer. Not run against the CLIP config. |
| `notebooks/D2-ablation-attribution.ipynb` | **Unrun.** Trimmed attribution ablation (A4/A13/A14/A15, optional X1) scored on zero-shot Celeb-DF + DFDC AUC, one session per group of arms (`ARMS_TO_RUN`). Writes `ablation_attrib_<arms>.csv`. |
| `README.md` | Committed results writeup. **Currently describes the EfficientNet-B4 run — see caveat below.** |

## ⚠ Current state vs. README — read this before trusting any number

`README.md` and every `*.png` / `*.json` / `*.csv` result file at the repo root were produced
by the **EfficientNet-B4, FakeAVCeleb-only** configuration (the run the paper reports).

The working tree's `CONFIG` in `crossfuse_v5.py` has since been switched to
`BACKBONE = "clip_vit_l14"`, `FREEZE_BLOCKS = 0` (LayerNorm-tuning), and the training pipeline
now expects a merged FakeAVCeleb+FF++ manifest and an FF++-pretrained encoder
(`PRETRAINED_ENCODER`). **This configuration has now been executed end to end (2026-09)**:
`A3`, `B1` (gate passed), `B-train` and `C-eval` all ran on Kaggle — see the Status log for the
numbers. The ablation grid (`D-ablations`) has NOT been run against it.

So: code in the repo ≠ results in the repo, right now. Don't read README's 0.95/0.99/0.61
numbers as describing what `CONFIG` would currently produce — they describe the *previous*
CNN configuration. This will be resolved by running Track C (below) and updating README
afterward, not before.

## Kaggle dataset handoff chain

```
crossfuse-v5-lib (crossfuse_v5.py, ffpp_v5.py — uploaded manually, re-upload after any library edit)
        │
        ├─► crops-v5            (NB-A)         FakeAVCeleb crops + audio cache
        ├─► ffpp-crops-v5       (NB-A3)        FF++ crops, all 4 families      ─┐
        │                                                                       ├─► ffpp-encoder-v5 (NB-B1)
        └─► [Celeb-DF v2, DFDC sample — attached, not produced by this repo]  ─┘
                                                                                       │
        crops-v5 + ffpp-crops-v5 + ffpp-encoder-v5 ──────────► NB-B ──► crossfuse-v5-ckpt
                                                                                       │
                                                                        NB-C, NB-D  ◄──┘
```

Every notebook `glob`s for `crossfuse_v5.py` / `ffpp_v5.py` by exact filename under
`/kaggle/input`, so **re-upload the `crossfuse-v5-lib` dataset after any edit to either file,
before running a notebook that imports it.**

## Invariants — do not break these silently

- **Identity-disjoint splitting** (union-find over id-tokens in folder + filename) is enforced
  by assertion in both `crossfuse_v5.py` and `ffpp_v5.py`. `IDENTITY_DISJOINT_SPLIT=False` is
  an explicit ablation (`A8_random_split`), not a default.
- **The two attribution heads (`video`, `audio`) stay strictly unimodal.** Cross-attention must
  never leak audio evidence into the video verdict or vice versa — that's what keeps the 4-way
  modality-attribution matrix interpretable. Only `sync` and `fusion` cross-attend.
- **Calibration under a matched estimator**: temperature and bootstrap-median Youden thresholds
  are fit using the *same* 20-sample MC-Dropout estimator applied at test time. Don't calibrate
  against `evaluate_deterministic`'s single-pass output and apply it to MC-Dropout scores.
- **Preregistered gates, fixed before the run they gate, not adjusted after seeing results:**
  - `CONFIG["SYNC_MIN_AUC"] = 0.70` — sync head must clear this to be reported as working.
  - `FFPP_CONFIG["GATE_CELEBDF_AUC"] = 0.85`, `GATE_DFDC_AUC = 0.75` — NB-B1's encoder must
    clear both before NB-B is allowed to consume it.
- **No scikit-learn dependency.** `roc_auc_score`, `roc_curve`, `balanced_accuracy_score`,
  a `classification_report` equivalent are reimplemented in NumPy in `crossfuse_v5.py` (Kaggle's
  image ships an sklearn build that imports a NumPy-2.0-only submodule while pinning NumPy
  1.26). Verified numerically identical to sklearn across randomised trials. Don't reach for
  `from sklearn... import` in new code here.
- **Crop convention must match across corpora.** `FFPP_CONFIG["CROP_SIZE"]` / `FACE_MARGIN`
  read from `CONFIG`, not redeclared — a local override would silently break transfer between
  the FakeAVCeleb cache and the FF++ cache (A2's `FACE_MARGIN=0.45` bug, fixed in A3).
- **`MULTI_GPU` is off by default** (`CONFIG["MULTI_GPU"] = False`), **and stays off for
  `clip_vit_l14` specifically — confirmed unusable twice, for two different reasons.**
  DataParallel wrapping the visual backbone crashed a live 2×T4 SBI run (kernel died) at
  `efficientnet_b4` + res 380 + effective batch 32. Re-tested smaller (`B2-sync-probe.ipynb`,
  frozen `efficientnet_b4`, 12×12 images/step): instead of crashing, it stalled indefinitely
  (GPU utilization ~0%). **2026-09-03, third attempt, live `B-train` run on CLIP ViT-L/14:**
  `enable_multi_gpu()` was patched with two targeted fixes (`NCCL_P2P_DISABLE=1` before any CUDA
  call; gradient checkpointing turned off on the wrapped backbone — checkpointing + DataParallel
  is a documented bad interaction) and validated clean on `notebooks/multigpu-probe.ipynb`
  (200-row slice, two runs, no hang). **The hang was genuinely gone at real scale too** — but the
  run was ~3× *slower* than single-GPU instead: 2904s/epoch vs. 934.7s, consistent across 5
  epochs, not noise. Root cause this time is different from the earlier hang: `nn.DataParallel`
  re-replicates the *entire* wrapped module to the second GPU on *every forward call* (not once);
  ViT-L/14 is 303M params, and `BATCH_CLIPS=4` means ~1,400+ forward calls/epoch, so replication
  overhead dominates and swamps any parallelism benefit. A small probe (200 rows, ~31 steps)
  cannot catch this — it only tests for a hang, not throughput, and this failure mode is specific
  to large models with many small steps. `B-train.ipynb` now forces `CONFIG["MULTI_GPU"] = False`
  unconditionally with this reasoning inline, regardless of GPU count. **Don't re-enable
  `nn.DataParallel` for `clip_vit_l14` again** — the fix needed is a different mechanism
  (DistributedDataParallel, which keeps persistent per-process replicas instead of
  re-replicating), not another tweak to `enable_multi_gpu()`, and a DDP rewrite hasn't been
  attempted. Still usable as-is for `efficientnet_b4` if the original crash's cause (not the
  replication-overhead one) is ever isolated — smaller model, proportionally tiny replication cost.

## Negative-results ledger

**1. SBI pretraining — cut.** Four attempts scored 0.99 AUC on FF++'s own held-out self-blend
task but 0.39 → 0.38 → 0.30 zero-shot on Celeb-DF — consistently *below* chance. One shortcut
was found and fixed (global photometric statistics leaking the label; worst single-feature
separability 0.109 → 0.022) but zero-shot got worse, so the diagnosis was incomplete. Code
remains in `sbi_v5.py`, unused. Superseded by the CLIP+FF++ approach in `B1-ffpp-pretrain.ipynb`.

**2. Sync head — chance-level, root cause now understood (see Sync head investigation below).**
Reported: AUC 0.5074 in-domain, exactly 0.500 with unaligned input (`A10_sync_unaligned`), train
AUC also at chance. Two prior fix attempts:
  - *Attempt 1*: replaced the mouth token (previously the bottom row of the shared backbone's
    coarse 3×3 pooled grid — nearly identical to the global video token at very low resolution)
    with a dedicated `MouthEncoder` CNN cropping the actual mouth region from the raw 256px face
    crop. Verified offline (187 checks, full checkpoint round-trip). **Kaggle result: no
    change** — 0.5075 vs 0.5074. No regression to the other three heads.
  - *Attempt 2*: instrumented the negative-construction logic to log
    `cross_identity` vs `same_clip_shift` negative counts. Disproved the leading theory ("most
    negatives are hard same-clip time-shifts") — **72% of training negatives were the easy
    cross-identity case** — yet train sync AUC was still ~0.51 across multiple epochs, with
    gradient confirmed reaching every sync-branch parameter. So the negatives aren't the
    problem either.
  - *Root cause (this round)*: the sync branch's output is provably invariant to the ORDER of
    its encoded audio sequence — no positional encoding anywhere downstream of `mel_encoder`,
    and both cross-attention outputs get pooled by `masked_mean_std` (mean+std over time) before
    the classifier head. A head that cannot represent *when* things happen cannot solve a task
    that is entirely about *when* things happen. See "Sync head investigation" below for the
    implemented fix and the important caveat about what a fixed proxy task does and doesn't
    prove about FakeAVCeleb's Wav2Lip-dominated fakes.

## Sync head investigation — architecture rewrite implemented, awaiting the Kaggle probe

**The permutation-invariance proof (precise version — corrected once already, see below).**
`mel_encoder` is a `TemporalConvNet` (a CNN): its local receptive fields mean permuting its
*raw input* mel bins produces a genuinely different encoding, not a relabelling of the same
one — that is NOT where the invariance lives. One step later, it is exact: `sync_ca_m2a` pools
over the `{S_enc}` key/value set (order-blind), `sync_ca_a2m`'s per-position queries against
the fixed mouth set are pointwise (permuting `S_enc` just permutes its output in lockstep), and
`masked_mean_std` pools both arms over time before the classifier. No positional signal enters
any of that. So once `mel_encoder` has produced its set of per-frame content vectors,
`sync_logit` cannot depend on *which* position each one came from relative to the mouth
sequence — only on the order-blind pooled set of content. `verify_sync_fix.py` proves this
exactly (`torch.allclose` to 1e-5) by permuting `S_enc` directly (not the raw mel — an earlier
draft of this test permuted the raw mel and got a false failure, `max abs diff 1.76e-02`, before
this was caught and corrected).

Two supporting defects, also fixed: `MouthEncoder` used to encode each frame independently (no
lip-motion representation across time), and `TemporalConvNet`'s symmetric padding grew the
sequence length at each layer, distorting the fine-grained alignment the new head depends on.

**The fix, implemented in `crossfuse_v5.py` (gated behind `CONFIG["SYNC_ARCH"]`):**
1. `MouthEncoder` gained a small 1D-conv temporal stack (`self.temporal`) over its per-frame
   features, so lip *motion* is representable, not just per-frame appearance.
2. `pooled_ca` (the old path) is **byte-for-byte unchanged** — no positional encoding was added
   to it. An earlier draft added sinusoidal positional encoding here as "the fix"; that broke
   the old path's own reproducibility (it stopped matching 0.5074/A10's 0.500) and was reverted
   once the verification script caught it. Positional encoding turned out to be unnecessary
   anyway — `offset_profile` (below) encodes position through explicit array indexing instead
   of through attention, which is a cleaner fix for the same root cause.
3. New default, `SYNC_ARCH = "offset_profile"`: mouth and audio project to a shared
   L2-normalised space; cosine similarity between `mouth[t]` and `audio[t+k]` for
   `k ∈ [-SYNC_MAX_OFFSET, SYNC_MAX_OFFSET]`, averaged over `t`, gives a `(B, 2K+1)` profile.
   A matched window peaks at `k=0` by construction; a small MLP (`sync_head`) maps the profile
   → `sync_logit`. Core math factored into a standalone `offset_similarity_profile()` so it's
   testable with synthetic embeddings, independent of trained weights.
4. Auxiliary InfoNCE loss (`LAMBDA_SYNC_NCE`, `SYNC_NCE_TEMP`) — cross-entropy over the profile
   with target `k=0` for matched windows, positive pass only — dense per-clip supervision
   instead of one bit per clip. Wired into `run_training_pipeline`'s sync loss block.
5. `CrossFuseCropDataset` slices the mel over a *wider* window
   (`[t0 - K/fps, t1 + K/fps]`, length `WINDOW_FRAMES + 2*SYNC_MAX_OFFSET`) when
   `SYNC_ARCH="offset_profile"`, so offsets are representable. Reads from the already-cached
   full-clip mel — no Stage-A re-extraction needed.
6. `offset_audio_encoder` uses `TemporalConvNet(..., causal=True)` — a new `Chomp1d`-based
   causal mode, applied ONLY here (never to `audio_encoder` or the old `mel_encoder`, both of
   which already work well and were left untouched) — so its output length doesn't grow via
   padding and the offset indexing stays precisely aligned to real time.
7. `SYNC_DENSE_ALIGNED=False` (the `A10_sync_unaligned` ablation) is asserted incompatible with
   `SYNC_ARCH="offset_profile"` at both `CrossFuseModelV5.__init__` and `build_model` — that
   ablation has no time base to profile over, and must stay on `pooled_ca`.

**Verified offline, 69/69 checks, CPU-only, `verify_sync_fix.py` (repo root):** the corrected
invariance proof; `offset_similarity_profile` peaking exactly at `k=0` for synthetic aligned
embeddings and NOT reproducing that peak under a time-shuffle; the full `offset_profile` model
forward changing under a raw-audio permutation (contrast with `pooled_ca`); checkpoint
round-trip for both archs including a regression guard that video/audio/fusion logits are
byte-identical after the sync change; the invalid `offset_profile`+unaligned combo raising at
both the constructor and `build_model`; gradient reaching every sync-branch parameter for both
archs; the real `CrossFuseCropDataset` code path producing correctly-shaped widened slices
(fixed on-disk fixture, no real video data); and `evaluate_deterministic` plus the exact
training-loop sync loss block (BCE + InfoNCE) running end to end without crashing.

**Decision rule (fixed in advance, not yet run):** the planned `notebooks/B2-sync-probe.ipynb`
(sync-only, backbone frozen, ~20–30 min on 2×T4) will report proxy-task AUC for both
architectures side by side. **≥ 0.70 → fold `offset_profile` into `B-train.ipynb`.**
**< 0.70 → cut sync for good**, reported as a third fully-diagnosed negative result with the
(corrected) invariance proof as the mechanism.

**Important caveat to keep in any writeup:** fixing the architecture's ability to represent
timing is not the same claim as "sync will become a useful deepfake signal on FakeAVCeleb."
~97% of FakeAVCeleb's visual fakes are Wav2Lip-generated, and Wav2Lip is trained with a SyncNet
expert loss — i.e. its objective *is* correct lip-sync, so natural desynchronization may
genuinely be rare in this corpus regardless of whether the head can represent it. The
architecture fix and the data-scarcity concern are two different, independently true things;
the probe is designed to tell them apart (a fixed architecture that still can't beat 0.70 despite
being provably no longer permutation-invariant would point back at data, not model).

## Track C — FF++-pretrained CLIP backbone (targets the 0.58/0.64 cross-dataset gap)

**Why:** the 0.95 in-domain vs. ~0.61 cross-dataset gap is the project's headline finding.
FakeAVCeleb is ~97% Wav2Lip (mouth-region reenactment); DFDC and Celeb-DF are full-face identity
swaps. The video head never saw a face swap during training — a training-data problem, not
purely a modelling one. Two papers (LNCLIP-DF, arXiv:2508.06248; Effort, arXiv:2411.15633) both
report CLIP ViT-L/14 with LayerNorm-only tuning on FF++ reaching 96.5/87.0 and 95.6/85.4 AUC on
Celeb-DF/DFDC respectively, against this project's 64.0/58.0 with EfficientNet-B4/ImageNet.

**Status: steps 1-4a below have run (2026-09); D-ablations has not.** Sequencing (each publishes a Kaggle Dataset for the next):
1. `A3-ffpp-full-extract.ipynb` → `ffpp-crops-v5`
2. `B1-ffpp-pretrain.ipynb` → gate (Celeb-DF≥0.85, DFDC≥0.75) → `ffpp-encoder-v5`
3. `B-train.ipynb` (merged manifest + pretrained encoder — run its timing-gate cell first)
4. `C-eval.ipynb`, `D-ablations.ipynb`

This is independent of the sync investigation and is the higher-confidence lever on the
project's actual weak point. Confirmed sequencing with the user: the sync probe (cheap) runs
first since it's already staged, Track C (expensive, GPU-bound) after.

---

## Status / next steps

_Update this section after every Kaggle run. Most recent first._

- **2026-09-20 (attribution ablation: 4 of 5 arms — A4 added; X1 not run)** — A4 (full recipe at
  the reduced budget) landed **DFDC 0.8008 [0.752, 0.845] / Celeb-DF 0.9004 [0.850, 0.941] / mean
  0.8506**, time-cut at 425.8 min with best epoch 7 (same situation as A15, so A4 vs A15 is a
  like-for-like comparison). Combined table now in `results/ablation_attribution_v5.csv`, means:
  **A4 0.851 ≈ A13 0.846 > A15 0.750 > A14 0.550**. So: mixing FF++ into fine-tuning adds nothing
  (A4≈A13); the dedicated FF++ pretraining stage is the ingredient (A4−A15 ≈ +0.10, and it shows up
  on Celeb-DF, 0.900 vs 0.716 with non-overlapping CIs; on DFDC 0.801 vs 0.785 overlaps); FF++ data
  without CLIP does nothing (A14). Note A4 Celeb-DF 0.900 at 7-10 epochs vs 0.845 for the fully
  trained B-train model — CIs overlap so not proven, but consistent with (b) above: longer
  fine-tuning on FakeAVCeleb may cost some transfer. **Open question: X1 (plain CLIP, no FF++ at
  all) was never run**, so "is it just CLIP?" is only partly answered (A15 shows CLIP + FF++ mix
  without pretraining reaches 0.75, above the 0.61 baseline, but cannot separate CLIP from the
  mix). Per-session CSVs are in `results/ablation_attrib_*.csv`. **Project status: main goal met
  and attribution mostly answered; remaining work is optional (X1, seeds) plus writing it up
  (README section for post-paper results).**

- **2026-09-20 (attribution ablation: 3 of 5 arms done — the FF++ PRETRAINED ENCODER is what fixed
  cross-dataset)** — Results in `results/ablation_attribution_v5.csv` (single seed, 2 frozen + ≤8
  fine-tune epochs, patience 4, same Celeb-DF/DFDC pools as C-eval). Cross-dataset video AUC
  (DFDC / Celeb-DF / mean): **A13 no-FF++-in-mix (CLIP + FF++-pretrained encoder)
  0.831 / 0.862 / 0.846**; **A15 CLIP + FF++ mix, no pretraining 0.785 / 0.716 / 0.750**
  (CUT SHORT by the time budget at 421 min, best epoch 7 — not a fair full-budget result);
  **A14 EfficientNet-B4 + FF++ mix 0.514 / 0.587 / 0.550** (in-domain video only 0.727, best epoch
  = 10 = the last, so likely undertrained). References: paper baseline mean 0.610; full B-train
  0.855. **Reading:** (1) A13 ≈ full B-train (0.846 vs 0.855, within noise) — mixing FF++ into the
  fine-tuning set adds nothing once the encoder is FF++-pretrained; (2) A15 is ~0.10 below A13
  (Celeb-DF CIs do not overlap; DFDC CIs do) — the dedicated FF++ pretraining stage matters beyond
  merely including FF++ rows; (3) A14 ≈ baseline — FF++ data alone, without CLIP, does not fix
  it. Caveats: A15 time-cut and A14 epoch-capped, so both are lower bounds; 887-row A13 was also
  still improving (best epoch 9/10); single seed. **Not yet run: A4 (reference at this budget;
  ~7h, would be time-cut like A15, so B-train's 0.855 is a reasonable stand-in) and X1 (plain CLIP,
  no FF++ at all, ~1-1.5h) — X1 is the one that separates "it is just CLIP" from "FF++ matters".**

- **2026-09-19 (attribution ablation notebook prepared, not run)** — Wrote
  `notebooks/D2-ablation-attribution.ipynb` (old `D-ablations.ipynb` left untouched). Why a new
  file: the old grid only scores in-domain FakeAVCeleb, where every arm is ~0.99, so it could not
  say which change fixed cross-dataset transfer. New notebook scores each arm on the same Celeb-DF
  (400, `make_rng("celebdf_eval")`) and DFDC pools as `C-eval`, with face crops extracted once and
  cached for all arms. Arms: A4 (full recipe, reduced budget), A13 (no FF++ in mix), A14
  (EfficientNet-B4), A15 (no FF++ pretraining), optional X1 (plain CLIP, no FF++ at all — added
  because A13 keeps the FF++-pretrained encoder, so it cannot separate "CLIP" from "FF++").
  Budget 2 frozen + ≤8 fine-tune epochs, patience 4, one seed; per-arm `TIME_BUDGET_S` from
  `plan_arm_budget_s`. Library change: `run_training_pipeline` now also returns `stop_reason`
  (`None`/`"patience"`/`"time_budget"`) so a time-cut arm is flagged; **re-upload
  `crossfuse-v5-lib`**. Verified locally: syntax, the budget helper, and the scoring loop against
  stand-ins; NOT run against the real model/datasets. Suggested sessions: (A14, A13) → A15 → A4 →
  optional X1.

- **2026-09-19 (C-eval done — TRACK C WORKED: cross-dataset AUC 0.61 → 0.855)** — Files landed in
  `results/` (`crossdataset_results_v5.json`, `calibration_v5.json`, `figures/`); the repo-root
  copies are still the paper's old EfficientNet-B4 results and were deliberately not overwritten.
  **Zero-shot cross-dataset, video head, same eval pools as the paper (n≈400 each):**
  DFDC **0.8649** [0.8239, 0.9012] vs 0.5799 [0.5127, 0.6447]; Celeb-DF v2 **0.8447**
  [0.7813, 0.9025] vs 0.6399 [0.5509, 0.7246]; mean **0.8548 vs 0.6099** (+0.245). The new CIs
  do not overlap the old ones on either dataset, so the gain is not bootstrap noise.
  **In-domain FakeAVCeleb test (n=309, identity-disjoint):** video AUC 0.998 (paper 0.9509),
  fusion 0.983 (0.9686), audio 0.978 (0.9866); sync still chance, `sync_reportable=false`.
  **Caveats to state alongside these numbers:** (a) *Accuracy does not transfer with AUC*: DFDC
  accuracy is only 0.557 (paper 0.477) because the decision threshold (0.786) was fit on
  FakeAVCeleb val — ranking transferred, calibration did not; Celeb-DF acc is 0.8675. (b) Celeb-DF
  0.845 is *below* the FF++-only B1 encoder's zero-shot 0.9186 (DFDC 0.865 is above its 0.849) —
  different scoring code and sample, so not proven, but fine-tuning on FakeAVCeleb may cost some
  Celeb-DF transfer. Both are still below the literature's 96.5/87.0 and 95.6/85.4. (c) One seed,
  no std. (d) The test split has only 2 identity groups, so the near-perfect in-domain numbers
  should not be over-read. (e) **No ablation attributes the gain** to FF++ data vs CLIP backbone vs
  FF++ pretraining — `D-ablations` arms A13 (no FF++ data), A14 (EfficientNet backbone), A15 (no
  FF++ pretrain) are the ones that would; not run yet. (f) `results/__results___files/` and
  `results/__huggingface_repos__.json` are Kaggle auto-generated leftovers, safe to delete.
  **Next:** run A4/A13/A14/A15 only (A16 is a duplicate of A4), then a README section for the
  post-paper results (leave the paper's section and tag `paper-v1` untouched).

- **2026-09-19 (B-train finished on Kaggle, single GPU; C-eval fixed, not yet run)** —
  `B-train.ipynb` completed with the new timing gate / `TIME_BUDGET_S`. `best_metrics_v5.json`
  (1 seed, seed 42): **val video AUC 0.9991, val audio 0.9971, val sync 0.5042** (sync at chance
  as already established; cut). These are *in-domain validation* numbers only — the result that
  matters is cross-dataset Celeb-DF/DFDC from `C-eval`, still to come (baseline 0.6399/0.5799,
  B1 encoder alone reached 0.9186/0.8494 zero-shot). Checkpoint published as Kaggle Dataset
  `crossfuse-v5-ckpt-1` (C-eval globs `crossfuse_v5_main_*.pth`, so the dataset name is
  irrelevant). Before running `C-eval` two bugs were fixed in the notebook: it never installed
  `open_clip` (would have failed with `open_clip is None`, same as B-train earlier), and it took
  the first `manifest.csv` by glob order (would pick the FF++ one if `ffpp-crops-v5` is attached);
  it now selects the manifest with an `audio_label` column. The C-eval split is safe:
  `build_3way_split_packed` excludes `train_only` (FF++) rows, so FakeAVCeleb-only val/test
  match what B-train saw (304/309). **Next: run `C-eval.ipynb`, then update README with the
  measured cross-dataset numbers (not the paper).**

- **2026-09-19 (housekeeping so the Track C run can finish; no Kaggle run yet)** — Executed
  `docs/superpowers/plans/2026-09-19-finishable-run-and-cleanup.md`. (1) `crossfuse_v5.py`:
  new pure `should_stop(...)` (patience in fine-tune phase + a wall-clock `TIME_BUDGET_S`,
  default `None`), unit-tested in `tests/test_should_stop.py` (4 pass). (2) `B-train.ipynb`
  timing gate now measures one *fine-tune* epoch (the old frozen-epoch probe hid the slow part;
  its ">90s" warning was an EfficientNet-era threshold), prints a full-run ETA, and sets
  `TIME_BUDGET_S = 9h - 1h safety - probe` so the run stops cleanly after its best checkpoint
  instead of being killed by Kaggle's 9h cap. Single-GPU only. (3) `D-ablations.ipynb`: `A10`
  now passes `SYNC_ARCH="pooled_ca"` (it used to crash `build_model`); all 16 arms verified to
  build. **Known issue, not fixed: `A16_clip_full_finetune` is not full fine-tuning** — ViTs
  are LayerNorm-only by design in `set_backbone_trainable`, so A16 has the same 102,400
  trainable params as A4 and is effectively a duplicate. Drop, relabel, or add a real path.
  (4) Cleanup: removed root `ablation_results_v5.csv` (identical to `results/`), the two
  multigpu-probe notebooks (outputs saved to `results/executed/multigpu-probe.output.txt`),
  `__pycache__`; moved the executed sync probe to `results/executed/`. `AGENTS.md` kept: it is
  a stale copy of this file, safe to delete. Local env note: `torch 2.9.1` vs `torchvision
  0.22.0` mismatch on the dev laptop breaks `import torchvision` locally only (not Kaggle);
  `verify_sync_fix.py` could not be re-run after these edits.
  **Next: run `B-train.ipynb` single-GPU (steps in the chat handoff), then `C-eval`.**

- **2026-09-02 (B1-ffpp-pretrain GATE PASSED — Track C's encoder is real)** — Corrected re-run of
  `B2-sync-probe.ipynb` (no `nan`, clean two-phase 3+12 split, `torch.allclose` diff shows
  invariance holds on the trained model too, max diff 1.04e-07) landed **pooled_ca 0.5062 /
  offset_profile 0.5129 val sync AUC, both below the 0.70 gate** (train AUC 0.4848 / 0.4754).
  Top-1 offset accuracy 100% (n=304) confirms the architecture genuinely finds the audio-video
  offset; it just doesn't separate real from fake. Per the preregistered decision rule: **sync
  is cut for good** — third fully-diagnosed negative result, closing that investigation.
  Then hit and fixed three infra bugs running Track C for the first time: (1) `ffpp_v5.py` was
  never uploaded to the `crossfuse-v5-lib` Kaggle Dataset (only `crossfuse_v5.py` was) —
  re-uploaded. (2) `discover_ffpp_videos` required a literal `original_sequences/` folder for
  real videos with no fallback for flattened mirrors; the attached `xdxd003/ff-c23` dataset
  ships real videos in a bare `original/` folder as a sibling of `Deepfakes/`, `Face2Face/`,
  etc. — added a third fallback pattern (`{root}/**/original/**/*.mp4`) mirroring the one the
  manipulated methods already had. (3) `B1-ffpp-pretrain.ipynb`'s `!pip install open_clip_torch`
  (no `--no-deps`) let pip's resolver disturb the Kaggle image's torch/torchvision pairing,
  breaking `torchvision.models.efficientnet_b0` import at `crossfuse_v5.py`'s module level —
  made that import optional (same try/except pattern as `MTCNN`/`open_clip`, loud `ImportError`
  only if an efficientnet backbone is actually requested) and fixed the install to
  `--no-deps open_clip_torch timm safetensors ftfy regex pyyaml` (the verified real dependency
  list, pulled from the wheel metadata — timm and safetensors don't pin torch/torchvision
  either, checked directly). Separately hit a stale-environment Kaggle notebook (`torch 1.3.0`,
  `torchvision 0.4.1a0`, pip returning zero versions for anything = no internet) — fixed via
  notebook Settings: Environment → "always use latest", Internet → on.
  **With those fixed, `B1-ffpp-pretrain.ipynb` ran clean and passed its gate comfortably:**
  `celebdf 0.9186 (gate 0.85, PASS, Δ+0.2787 over the v5 baseline 0.6399)`,
  `dfdc 0.8494 (gate 0.75, PASS, Δ+0.2695 over the v5 baseline 0.5799)` — both in the same range
  as the literature this track was chasing (LNCLIP-DF 96.5/87.0, Effort 95.6/85.4), not just
  barely clearing the bar. `ffpp_encoder_v5.pth` published as Kaggle Dataset `ffpp-encoder-v5`.
  **Next: run `B-train.ipynb`** with `crossfuse-v5-lib` + `crops-v5` + `ffpp-crops-v5` +
  `ffpp-encoder-v5` attached — check the timing-gate cell's probe result before letting the full
  multi-seed run go.
- **2026-08-16 (second Kaggle attempt of B2-sync-probe, config bug found and fixed)** — First
  Kaggle run of `B2-sync-probe.ipynb` completed without crashing (all cells ran, gate table
  printed) but was **not trustworthy**: training loss went `nan` partway through the `pooled_ca`
  run (visible directly in the log from epoch 5 onward), and val metrics froze identically for
  the last 5 epochs — a real numerical divergence, not a sync-architecture symptom. Root cause:
  the probe's config set `EPOCHS_FROZEN=15, EPOCHS_FINETUNE=0` to keep the backbone frozen for
  the whole run, but `run_training_pipeline`'s phase-1 optimizer (`make_optim(0)`) uses
  `LR_HEAD * 10` — a 10x-boosted LR meant for a brief ~3-epoch warmup, not sustained training.
  Staying there for 15 epochs under mixed-precision (`torch.amp` + `GradScaler`) diverged, and
  the resulting NaN/Inf activations got baked into some `_mlp_head`'s `BatchNorm1d` running
  stats — which is why even the reloaded "best" checkpoint later produced non-finite
  `sync_logit` values in the notebook's own invariance-check cell. (That cell's diff was also
  computed via a raw subtraction rather than `torch.allclose`, so matching infinities showed as
  `nan` instead of correctly registering as "still invariant" — fixed alongside the config.)
  **Fix**: use the pipeline's normal two-phase split (`EPOCHS_FROZEN=3, EPOCHS_FINETUNE=12`)
  instead of forcing everything into phase 1, and set `FREEZE_BLOCKS=12` (safely above
  EfficientNet-B4's actual 9 blocks) so phase 2's `backbone_param_groups()` still returns zero
  backbone parameter groups — backbone stays fully frozen through both phases, but the heads
  train at the calmer, intended LR. Verified locally: `named_parameters()` filtered to
  `backbone.*` is exactly 0 with `requires_grad=True` after a full two-phase run on synthetic
  data. **The first run's actual gate numbers (pooled_ca 0.5196, offset_profile 0.5475, both
  FAIL) should be treated as unreliable given the instability and are not the final answer** —
  re-running with the corrected config is next.
- **2026-08-16** — Sync-head architecture rewrite implemented in `crossfuse_v5.py` (see "Sync
  head investigation" above): `MouthEncoder` temporal context, new `offset_profile` head +
  InfoNCE loss as the default, widened dataset mel slice, causal `TemporalConvNet` mode scoped
  to the new head only, `pooled_ca` kept byte-identical to the original for reproducibility.
  Verified offline: 69/69 checks pass in `verify_sync_fix.py` (CPU-only, no Kaggle needed) —
  including a self-correction where an initial version of both the fix and its test wrongly
  claimed invariance to permuting the *raw* mel input; the real, provable claim is invariance to
  permuting the *encoded* sequence one step later (see the investigation section for the exact
  distinction).
