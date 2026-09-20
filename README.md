# CrossFuse v5 — Multi-Modal Deepfake Detection with Shortcut-Controlled Evaluation

Audio-visual deepfake detection with per-modality attribution, trained on
FakeAVCeleb and evaluated zero-shot on DFDC and Celeb-DF v2.

> **Reproducibility note.** The numbers in the first sections below come from the
> EfficientNet-B4, FakeAVCeleb-only code at git tag `paper-v1` (commit `dacd546`).
> Later work on `main` adds a CLIP ViT-L/14 + FaceForensics++ track; its results are
> in [Post-paper results](#post-paper-results-clip--faceforensics) and were not part
> of the paper.

The emphasis of this project is **evaluation honesty**: several of the results
below are negative or smaller-than-expected, and are reported as measured
rather than tuned away.

## Architecture

A shared visual backbone (EfficientNet-B4) and a temporal-convolutional audio
encoder feed four heads:

| Head | Reads | Predicts |
|---|---|---|
| `video` | visual branch only | is the visual track manipulated |
| `audio` | audio branch only | is the audio track synthetic |
| `sync` | mouth tokens × time-aligned log-mel, cross-attention | do lips and speech correspond |
| `fusion` | cross-attended joint representation | is the clip manipulated at all |

The two attribution heads are strictly unimodal by construction, which is what
makes the 4-way modality-attribution matrix (RVRA / RVFA / FVRA / FVFA)
interpretable.

## Results

### In-domain (FakeAVCeleb, identity-disjoint test split, n=309)

| Head | AUC | 95% CI | ECE |
|---|---|---|---|
| Video | 0.9509 | [0.9281, 0.9713] | 0.048 |
| Audio | 0.9866 | [0.9715, 0.9977] | 0.041 |
| Fusion | 0.9686 | [0.9501, 0.9846] | 0.031 |
| Sync | 0.5074 | [0.4602, 0.5534] | 0.014 |

### Zero-shot cross-dataset (video head only)

| Dataset | n | AUC | 95% CI |
|---|---|---|---|
| DFDC (sample) | 398 | 0.5799 | [0.5127, 0.6447] |
| Celeb-DF v2 | 400 | 0.6399 | [0.5509, 0.7246] |
| **Mean** | | **0.6099** | |

The gap between 0.95 in-domain and ~0.61 cross-dataset is the project's
central finding, not an incidental weakness: strong in-domain numbers on this
corpus substantially overstate real-world transfer.

The audio head is deliberately **never** OR-combined into the cross-dataset
verdict. DFDC applies audio swaps only to already-visually-manipulated videos
and releases no per-video audio labels, so the audio head has nothing to
detect there and can only contribute false positives.

## Reported negative results

**The sync head does not work on this corpus** (AUC 0.507, and 0.500 exactly
when its input is time-unaligned). Train AUC is also at chance, so this is a
missing signal rather than a generalisation gap: ~97% of FakeAVCeleb's visual
fakes are Wav2Lip-generated, and Wav2Lip's objective *is* correct lip-sync, so
the corpus contains almost no natural desynchronisation to learn from. A
`SYNC_MIN_AUC = 0.70` threshold was fixed in advance specifically so this
could not be quietly reported as a working signal.

**Self-Blended Images (SBI) pretraining failed and was cut.** Four attempts on
FF++ real video scored 0.99 AUC on their own held-out self-blend task but
0.39 → 0.38 → 0.30 zero-shot on Celeb-DF — consistently *below* chance,
meaning a strong signal pointing the wrong way rather than an absence of
signal. One shortcut was found and fixed (real and fake samples differed in
global photometric statistics rather than only at the blend boundary; measured
worst single-feature separability dropped 0.109 → 0.022), but the zero-shot
score got worse, so the diagnosis was incomplete. The code remains in
`sbi_v5.py`; the shipped pipeline uses ImageNet initialisation.

## Ablations

`results/ablation_results_v5.csv` — 12 configurations, seed 42, reduced epoch
budget. Notable rows, all versus `A4_full_crossfuse` (test AUC):

- `A8_random_split` (identity leakage): video 0.919 vs 0.909. Leakage inflates
  in the expected direction but only slightly, once hard negatives and a
  corrected evaluation pipeline are already in place.
- `A9_late_fusion_baseline`: fusion 0.942 vs 0.927 — plain late fusion
  slightly **outperforms** the cross-attention mechanism.
- `A12_no_silence_trim`: audio 0.982 vs 0.986 — trimming FakeAVCeleb's
  leading-silence artifact barely moves the audio head, suggesting its ~0.98
  AUC rests more on the single-vocoder (SV2TTS) fingerprint than on that
  artifact.
- `A10_sync_unaligned`: sync exactly 0.500.

## Post-paper results: CLIP + FaceForensics++

*Not part of the paper. Code: `main`; paper code: tag `paper-v1`.*

The central finding above is the gap between 0.95 in-domain and ~0.61 cross-dataset.
FakeAVCeleb is ~97% Wav2Lip (mouth-only edits) while Celeb-DF and DFDC are full-face
swaps, so the video head never saw a face swap in training. To test whether the gap is a
training-data problem, the EfficientNet-B4/ImageNet backbone was replaced with
**CLIP ViT-L/14 (LayerNorm-only tuning)**, first pretrained on FaceForensics++ (its four
swap/reenactment families) and then trained on the same identity-disjoint FakeAVCeleb
split. The encoder had to clear a preregistered gate (Celeb-DF >= 0.85, DFDC >= 0.75)
before being used; it scored 0.9186 / 0.8494.

### Zero-shot cross-dataset (video head, same evaluation pools as above)

| Dataset | n | Paper (EffNet-B4) | CLIP + FF++ | 95% CI (new) |
|---|---|---|---|---|
| DFDC (sample) | 397 | 0.5799 | **0.8649** | [0.8239, 0.9012] |
| Celeb-DF v2 | 400 | 0.6399 | **0.8447** | [0.7813, 0.9025] |
| **Mean** | | 0.6099 | **0.8548** | |

The new intervals do not overlap the paper's on either dataset. In-domain (FakeAVCeleb
test, n=309): video AUC 0.998, fusion 0.983, audio 0.978. The sync head remains at chance
(validation AUC 0.504), consistent with the negative result above.

### What caused the improvement (attribution ablation)

Single seed, reduced budget (2 frozen + up to 8 fine-tune epochs), same evaluation pools.
`results/ablation_attribution_v5.csv`.

| Arm | Backbone | FF++ in training mix | FF++ encoder pretraining | DFDC | Celeb-DF | Mean |
|---|---|---|---|---|---|---|
| A4 full recipe (time-limited)  | CLIP | yes | yes | 0.801 | 0.900 | 0.851 |
| A13 no FF++ in mix             | CLIP | no  | yes | 0.831 | 0.862 | 0.846 |
| A15 no FF++ pretraining (time-limited) | CLIP | yes | no | 0.785 | 0.716 | 0.750 |
| A14 EfficientNet-B4            | EffNet | yes | no | 0.514 | 0.587 | 0.550 |

Reading: mixing FF++ into fine-tuning adds nothing once the encoder is FF++-pretrained
(A4 ~ A13); the dedicated FF++ pretraining stage is the ingredient that matters (A4 vs
A15, mainly on Celeb-DF, where the intervals do not overlap); FF++ data without CLIP does
not help (A14).

### Caveats

- Single seed; bootstrap intervals are roughly +/-0.05, so differences below that are noise.
- A4 and A15 were stopped by a wall-clock budget (best epoch 7); A14 hit its epoch cap.
  Treat them as lower bounds.
- Ranking transfers better than calibration: at the FakeAVCeleb-fit threshold, DFDC
  accuracy is only 0.557 (Celeb-DF 0.868) even though AUC is 0.865.
- Results are below the published CLIP-on-FF++ numbers (~96 / ~87 AUC).
- Not run: plain CLIP with no FF++ at all, so "is it just CLIP?" is only partly answered
  (A15 shows CLIP + FF++ mix without pretraining reaches 0.75, above the 0.61 baseline).
- The sync-head rewrite (offset-profile architecture, verified in `verify_sync_fix.py`)
  found the correct audio-video offset 100% of the time but still scored ~0.51 at
  real-vs-fake, i.e. the corpus lacks natural desynchronisation to learn from.
- `nn.DataParallel` on two GPUs was ~3x *slower* than one GPU for this model (per-step
  re-replication of a 303M-parameter backbone); all runs are single-GPU.

Pipeline for this track: `A3-ffpp-full-extract` -> `B1-ffpp-pretrain` (gate) ->
`B-train` -> `C-eval` -> `D2-ablation-attribution`. Figures and JSON are in `results/`.

## Evaluation protocol

- **Identity-disjoint splitting** via union-find over id-tokens in both the
  parent folder and the filename, so a manipulated clip's *source* identity is
  grouped with its target. Enforced by assertion.
- **Calibration under a matched estimator** — temperature and bootstrap-median
  Youden thresholds are fit on validation using the same 20-sample MC-Dropout
  estimator applied at test time.
- **Bootstrap 95% CIs and explicit n** on every reported number.

## Pipeline

Six Kaggle notebooks, each publishing its output as a Dataset for the next.
Split this way so no stage approaches Kaggle's 12-hour session limit.

| Notebook | Purpose |
|---|---|
| `notebooks/A-extract.ipynb` | FakeAVCeleb face crops + cached audio features → `crops-v5` |
| `notebooks/A2-ffpp-extract.ipynb` | FF++ real-video crops (for SBI) |
| `notebooks/B0-sbi-pretrain.ipynb` | SBI visual pretraining (**cut — see above**) |
| `notebooks/B-train.ipynb` | Multimodal training → `crossfuse-v5-ckpt` |
| `notebooks/C-eval.ipynb` | Calibration, test report, cross-dataset suite, figures |
| `notebooks/D-ablations.ipynb` | Ablation grid → `ablation_results_v5.csv` |
| `notebooks/A3-ffpp-full-extract.ipynb` | *(post-paper)* FF++ real + 4 fake families → `ffpp-crops-v5` |
| `notebooks/B1-ffpp-pretrain.ipynb` | *(post-paper)* CLIP ViT-L/14 FF++ pretraining, preregistered gate → `ffpp-encoder-v5` |
| `notebooks/B2-sync-probe.ipynb` | *(post-paper)* sync-head rewrite probe (negative result) |
| `notebooks/D2-ablation-attribution.ipynb` | *(post-paper)* attribution ablation scored on cross-dataset AUC |

`crossfuse_v5.py` and `sbi_v5.py` (repo root) are uploaded as a Kaggle Dataset
(`crossfuse-v5-lib`) and imported by every notebook. The module names are kept
as-is even though the surrounding project no longer uses "v5" in file/folder
names elsewhere — every notebook's import and `glob` search hardcodes these
exact filenames, and the Kaggle dataset you're using is named after them, so
renaming them would require re-uploading to Kaggle and would break re-running
any notebook from this repo against your existing Kaggle setup. Rename them
together (module names + every notebook's imports/globs + the Kaggle dataset)
if you want that consistency later — not done here to avoid breaking the
working pipeline silently.

Audio features (MFCC in both trimmed and untrimmed variants, plus full-clip
log-mel) are cached at extraction time, which keeps librosa off the training
path entirely — the loader, not the GPU, was the original bottleneck.

`scikit-learn` is deliberately not a dependency: `roc_auc_score`, `roc_curve`,
`balanced_accuracy_score` and a `classification_report` equivalent are
reimplemented in NumPy in `crossfuse_v5.py` (verified numerically identical
against scikit-learn across randomised trials including tie-heavy inputs),
because Kaggle's image ships a scikit-learn build that imports a
NumPy-2.0-only submodule while pinning NumPy 1.26.

## Datasets

FakeAVCeleb (training), DFDC sample and Celeb-DF v2 (zero-shot evaluation),
FaceForensics++ c23 (SBI attempt). None are redistributed here.
