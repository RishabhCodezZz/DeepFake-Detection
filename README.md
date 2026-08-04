# CrossFuse v5 — Multi-Modal Deepfake Detection with Shortcut-Controlled Evaluation

Audio-visual deepfake detection with per-modality attribution, trained on
FakeAVCeleb and evaluated zero-shot on DFDC and Celeb-DF v2.

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
