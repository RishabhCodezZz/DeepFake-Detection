# CLAUDE.md — CrossFuse v5 project notes

Living doc. Update **Current status** and the **Status log** at the bottom after every Kaggle run —
everything between them should stay accurate for longer.

## Current status (as of 2026-09-20)

**The main goal is met.** The paper's headline weakness — 0.95 in-domain but only ~0.61 zero-shot on
unseen datasets — is fixed and explained:

| Zero-shot video AUC | DFDC | Celeb-DF v2 | Mean |
|---|---|---|---|
| Paper (EfficientNet-B4, FakeAVCeleb only) | 0.5799 | 0.6399 | **0.6099** |
| CLIP ViT-L/14 + FF++ pretraining (final `B-train` model) | **0.8649** [0.824, 0.901] | **0.8447** [0.781, 0.902] | **0.8548** |

Attribution ablation (single seed, reduced budget; mean cross-dataset AUC): **A4 full recipe 0.851 ≈
A13 no FF++ in the mix 0.846 > A15 no FF++ pretraining 0.750 > A14 EfficientNet + FF++ 0.550.** The
FF++ *pretraining* of the CLIP encoder looks like the ingredient that matters (A15 is confounded by
`LR_LN` 1e-5 vs 1e-4, so this is a hypothesis); mixing FF++ rows into fine-tuning shows no detectable
difference; the EfficientNet arm (A14) was undertrained, so it says nothing about FF++ data without CLIP.

**Done:** paper (submitted, frozen) · sync head investigated and cut · SBI cut · Track C
(A3 → B1 gate → B-train → C-eval) · attribution ablation (D2) · README rewritten to show the current results only ·
pushed to GitHub (`main`, tag `paper-v1` on `dacd546`); the cleanup pass and README rewrite are committed too.

**Open / optional:** `X1` ablation (plain CLIP, no FF++ at all) never run, so "is it just CLIP?" is
only partly answered · more seeds for error bars · the old 16-row `D-ablations` grid was never run
against the CLIP config (and its `A16` row duplicates `A4`) · missing records: `best_metrics_v5.json`
(B-train), `ffpp_zeroshot_v5.json` (B1), the executed `C-eval` notebook · A15 (no FF++ pretraining) was run with `LR_LN=1e-5`
vs 1e-4 in B1, so it confounds "pretraining" with "LayerNorms barely trained"; rerun at 1e-4 to settle it.

**Caveats to state with any number:** one seed, bootstrap CIs ≈ ±0.05 · accuracy does not transfer with
AUC (DFDC acc 0.557 at the FakeAVCeleb-fit threshold; Celeb-DF 0.868) · A4/A15 were stopped by a
wall-clock budget, A14 hit its epoch cap → lower bounds · results are still below the literature
(LNCLIP-DF 96.5/87.0, Effort 95.6/85.4) · the FakeAVCeleb test split has only 2 identity groups.

## What this project is

Audio-visual deepfake detection. A shared visual backbone + a temporal audio encoder feed four
heads (`video`, `audio`, `sync`, `fusion`) — see [README.md](README.md) for the architecture
table and the current results. The project's stated emphasis is **evaluation honesty**: negative results are
reported, not tuned away. Three are on the record (SBI pretraining, cut; the sync head, chance-level
and now fully diagnosed and cut; `nn.DataParallel`, unusable for the CLIP backbone).

A paper draft covering the EfficientNet-B4 baseline results has been submitted and is frozen — its
exact code is git tag `paper-v1` (commit `dacd546`), and its result artifacts are the repo-root PNG/JSON
files. Everything after that is a **separate, ungraded "major project" track** (CLIP + FF++). The README
was rewritten on 2026-09-20 to show only the current results (the paper-era sections and the negative
results were dropped from it by choice); the full history, negative results included, stays here.

## Two unrelated "A" naming schemes — do not conflate them

1. **Notebook stage names** (`notebooks/*.ipynb`) — pipeline stages, run in order on Kaggle:
   `A-extract` → `A2-ffpp-extract` (superseded) / `A3-ffpp-full-extract` → `B0-sbi-pretrain`
   (cut) / `B1-ffpp-pretrain` → `B-train` → `C-eval` → `D-ablations` (old) / `D2-ablation-attribution`.
2. **Ablation arm names** (`A1_video_only` … `A16_clip_full_finetune`, plus `X1`), used inside
   `D-ablations.ipynb` and `D2-ablation-attribution.ipynb`; results in `results/*.csv`.

`A3-ffpp-full-extract.ipynb` (a notebook) and `A3_two_head_no_fusion` (an ablation row) share a
letter by pure coincidence and mean nothing to each other.

## Repo layout

| Path | Role / status |
|---|---|
| `crossfuse_v5.py` | The library: config, data pipeline, model, training loop (incl. `should_stop` + `TIME_BUDGET_S`), eval/calibration, NumPy reimplementations of the sklearn metrics. Imported by every notebook. |
| `ffpp_v5.py` | FF++ pipeline for the CLIP track: discovery (canonical and flat mirrors), identity-disjoint splitting, `FFPPClassifier`, `train_ffpp_encoder`, zero-shot eval. **Ran; gate passed.** |
| `sbi_v5.py` | Self-Blended Images pretraining. **Cut** (negative result). Kept for the record; imported by no current notebook. |
| `verify_sync_fix.py` | Offline (CPU) 69-check verification of the sync rewrite. Passed 2026-08-16; not re-run since (see Current status caveats / env note). |
| `tests/` | CPU unit tests (47 pass + 1 strict xfail): stopping rule, NumPy metrics vs brute force, identity-disjoint splitting (incl. `train_only` pinning and the val/test-invariant-to-FF++ property C-eval relies on), sync-profile / masked pooling helpers, MC-Dropout. The xfail documents that `enable_mc_dropout` does not make transformer/attention layers stochastic. Run: `python -m pytest tests -q`. CI: `.github/workflows/tests.yml`. Deps: `requirements.txt`. |
| `notebooks/A-extract.ipynb` | FakeAVCeleb crops + audio cache → Kaggle Dataset `crops-v5`. Ran (paper). |
| `notebooks/A2-ffpp-extract.ipynb` | Old FF++ real-only extraction for the cut SBI stage. **Superseded by A3.** |
| `notebooks/A3-ffpp-full-extract.ipynb` | FF++ real + 4 fake families → `ffpp-crops-v5` (5,000 videos). **Ran.** |
| `notebooks/B0-sbi-pretrain.ipynb` | SBI pretraining. **Cut.** |
| `notebooks/B1-ffpp-pretrain.ipynb` | CLIP ViT-L/14 LayerNorm-tuned FF++ pretraining → `ffpp-encoder-v5`. **Ran; gate passed** (Celeb-DF 0.9186 / DFDC 0.8494). |
| `notebooks/B2-sync-probe.ipynb` | Sync-head probe. **Ran; both archs failed the 0.70 gate → sync cut.** Executed copy: `results/executed/`. |
| `notebooks/B-train.ipynb` | Multimodal training on the merged FakeAVCeleb+FF++ manifest, from the FF++ encoder. **Ran** (1 seed, single T4). |
| `notebooks/C-eval.ipynb` | Calibration, in-domain test report, cross-dataset suite, figures. **Ran.** |
| `notebooks/D-ablations.ipynb` | Old 16-row grid; scores in-domain only. A1–A12 are the paper's ablations. **Not run against the CLIP config.** |
| `notebooks/D2-ablation-attribution.ipynb` | Attribution ablation scored on Celeb-DF + DFDC. **A4/A13/A14/A15 ran; `X1` not run.** |
| `README.md` | Public write-up of the CURRENT results (CLIP + FF++), how it works, ablation, how to run. Deliberately omits the paper-era baseline sections and the negative results (see this file). |
| `results/` | **Post-paper** results: `crossdataset_results_v5.json`, `calibration_v5.json`, `figures/`, `ablation_attribution_v5.csv` (all 4 D2 arms), `ablation_results_v5.csv` (paper's A1–A12), `executed/`. |
| repo-root `*.png`, `calibration_v5.json`, `crossdataset_results_v5.json` | **The paper's** result artifacts (EfficientNet-B4). Do not overwrite; new runs go in `results/`. |
| `.gitignore` | Excludes checkpoints/features (`*.pth`, `*.npy`, …), caches, data, and Kaggle output leftovers. |

## Results provenance — which numbers belong to which model

- **Paper (EfficientNet-B4, FakeAVCeleb only):** no longer in `README.md` (removed 2026-09-20; older
  versions are in git history and at tag `paper-v1`), the repo-root
  `*.png` / `*.json`, and `results/ablation_results_v5.csv`. Reproducible from tag `paper-v1`.
- **Current (CLIP ViT-L/14 + FF++):** everything in `README.md` and everything in `results/`
  except `ablation_results_v5.csv`. `crossfuse_v5.py`'s `CONFIG` on `main` describes THIS model
  (`BACKBONE="clip_vit_l14"`, `FREEZE_BLOCKS=0`, LayerNorm-only tuning, merged manifest,
  `PRETRAINED_ENCODER` from B1).
- Don't compare the paper's ablation table with the D2 table: different backbone, different metric
  (in-domain vs zero-shot cross-dataset).

## Kaggle dataset handoff chain

```
crossfuse-v5-lib (crossfuse_v5.py, ffpp_v5.py — uploaded manually, re-upload after any library edit)
        │
        ├─► crops-v5            (NB-A)         FakeAVCeleb crops + audio cache
        ├─► ffpp-crops-v5       (NB-A3)        FF++ crops, all 4 families      ─┐
        │                                                                       ├─► ffpp-encoder-v5 (NB-B1)
        └─► [Celeb-DF v2, DFDC sample — attached, not produced by this repo]  ─┘
                                                                                       │
        crops-v5 + ffpp-crops-v5 + ffpp-encoder-v5 ──────────► NB-B ──► crossfuse-v5-ckpt-1
                                                                                       │
        NB-C  ◄── crops-v5 + crossfuse-v5-ckpt-1 + Celeb-DF + DFDC                    │
        NB-D2 ◄── crops-v5 + ffpp-crops-v5 + ffpp-encoder-v5 + Celeb-DF + DFDC (no ckpt needed)
```

`crossfuse-v5-ckpt-1` holds the ONLY copy of the trained CLIP checkpoint (`crossfuse_v5_main_s42.pth`,
~1.2 GB, gitignored). **Do not delete that Kaggle dataset.** `C-eval` finds the checkpoint by the
glob `crossfuse_v5_main_*.pth`, so the dataset name is irrelevant.

Every notebook `glob`s for `crossfuse_v5.py` / `ffpp_v5.py` by exact filename under
`/kaggle/input`, so **re-upload the `crossfuse-v5-lib` dataset after any edit to either file,
before running a notebook that imports it.** Every CLIP notebook installs
`open_clip_torch timm safetensors ftfy regex pyyaml` with `--no-deps` (cell 1) and needs Internet ON
and a "latest" Kaggle environment (an old pinned environment ships torch 1.3 and no pip access).

## Invariants — do not break these silently

- **Identity-disjoint splitting** (union-find over id-tokens in folder + filename) is enforced
  by assertion in both `crossfuse_v5.py` and `ffpp_v5.py`. `IDENTITY_DISJOINT_SPLIT=False` is
  an explicit ablation (`A8_random_split`), not a default. FF++ rows are `train_only`: pinned to
  train, excluded from the val/test split, so FakeAVCeleb val/test stay identical with or without them
  (304/309) — `C-eval` relies on this to rebuild the same split.
- **The two attribution heads (`video`, `audio`) stay strictly unimodal.** Cross-attention must
  never leak audio evidence into the video verdict or vice versa — that's what keeps the 4-way
  modality-attribution matrix interpretable. Only `sync` and `fusion` cross-attend.
- **Calibration under a matched estimator**: temperature and bootstrap-median Youden thresholds
  are fit using the *same* 20-sample MC-Dropout estimator applied at test time. Don't calibrate
  against `evaluate_deterministic`'s single-pass output and apply it to MC-Dropout scores.
- **Preregistered gates, fixed before the run they gate, not adjusted after seeing results:**
  - `CONFIG["SYNC_MIN_AUC"] = 0.70` — sync head must clear this to be reported as working (it did not).
  - `FFPP_CONFIG["GATE_CELEBDF_AUC"] = 0.85`, `GATE_DFDC_AUC = 0.75` — NB-B1's encoder must
    clear both before NB-B may consume it (it did: 0.9186 / 0.8494).
- **No scikit-learn dependency.** `roc_auc_score`, `roc_curve`, `balanced_accuracy_score`,
  a `classification_report` equivalent are reimplemented in NumPy in `crossfuse_v5.py` (Kaggle's
  image ships an sklearn build that imports a NumPy-2.0-only submodule while pinning NumPy
  1.26). Verified numerically identical to sklearn. Don't reach for `from sklearn... import`.
- **Crop convention must match across corpora.** `FFPP_CONFIG["CROP_SIZE"]` / `FACE_MARGIN`
  read from `CONFIG`, not redeclared — a local override would silently break transfer between
  the FakeAVCeleb cache and the FF++ cache (A2's `FACE_MARGIN=0.45` bug, fixed in A3).
- **Single GPU only: `CONFIG["MULTI_GPU"]` stays `False`** (`B-train` and `D2` force it). Failed three
  times: (1) `nn.DataParallel` crashed a live 2×T4 SBI run (`efficientnet_b4`, res 380, batch 32);
  (2) it stalled a small frozen-EfficientNet probe (GPU ~0%); (3) after fixing the hang
  (`NCCL_P2P_DISABLE=1`; gradient checkpointing off on the wrapped backbone) it ran **~3× slower**
  on CLIP ViT-L/14 (2904 s/epoch vs 934.7 s) because `nn.DataParallel` re-replicates the entire
  303M-param backbone on *every forward call* and `BATCH_CLIPS=4` means ~1,400 calls/epoch. A
  200-row probe could not catch this — it tests "does it hang", not throughput. Don't re-enable
  `nn.DataParallel` for `clip_vit_l14`; a real fix would be DistributedDataParallel (not attempted).
- **Runs must fit Kaggle's 9 h session / 30 h-per-week cap.** `should_stop` (patience in fine-tune
  phase + `TIME_BUDGET_S`) ends a run cleanly with its best checkpoint; `run_training_pipeline`
  returns `stop_reason` (`None` / `"patience"` / `"time_budget"`). An arm stopped by the time budget is
  a lower bound, not a fair comparison — flag it.
- **Results live where their model lives:** new runs write to `results/`, never over the repo-root
  paper artifacts.

## Negative-results ledger

**1. SBI pretraining — cut.** Four attempts scored 0.99 AUC on FF++'s own held-out self-blend
task but 0.39 → 0.38 → 0.30 zero-shot on Celeb-DF — consistently *below* chance. One shortcut
was found and fixed (global photometric statistics leaking the label; worst single-feature
separability 0.109 → 0.022) but zero-shot got worse, so the diagnosis was incomplete. Code
remains in `sbi_v5.py`, unused. Superseded by the CLIP+FF++ approach (`B1`), which passed its gate.

**2. Sync head — chance-level, root cause understood, fix built and verified, still cut.**
Reported: AUC 0.5074 in-domain, exactly 0.500 unaligned (`A10_sync_unaligned`), train AUC also at
chance. Attempt 1 (dedicated `MouthEncoder`) changed nothing (0.5075). Attempt 2 (negative-source
logging) disproved "mostly hard same-clip shifts" (72% were easy cross-identity) yet train AUC stayed
~0.51. **Root cause:** the sync branch's output is provably invariant to the order of its encoded
audio sequence (no positional signal; `masked_mean_std` pools over time), so it cannot represent
*when* anything happens. **Fix:** the `offset_profile` architecture (below), verified offline
(69/69 checks). **Outcome (`B2-sync-probe`, 2026-09):** the fix works mechanically — top-1 offset
accuracy 100% (n=304) — but both archs fail the preregistered 0.70 gate (`pooled_ca` val 0.5062,
`offset_profile` 0.5129; train 0.4848 / 0.4754). The corpus (~97% Wav2Lip, whose training objective
*is* correct lip-sync) has no natural desynchronisation to learn. Sync is cut; final `B-train` val
sync AUC 0.5042; `sync_reportable=False`. The head is still built and trained by default so the
4-head checkpoints stay unchanged.

**3. `nn.DataParallel` for the CLIP backbone — unusable.** See the MULTI_GPU invariant above.

## Sync head investigation — CLOSED (fix implemented, verified, gate failed → cut)

*Technical record kept because it backs negative result #2 and the reported proofs.*

**The permutation-invariance proof (precise version — corrected once already).**
`mel_encoder` is a `TemporalConvNet` (a CNN): its local receptive fields mean permuting its
*raw input* mel bins produces a genuinely different encoding — that is NOT where the invariance
lives. One step later, it is exact: `sync_ca_m2a` pools over the `{S_enc}` key/value set
(order-blind), `sync_ca_a2m`'s per-position queries against the fixed mouth set are pointwise
(permuting `S_enc` just permutes its output in lockstep), and `masked_mean_std` pools both arms over
time before the classifier. So once `mel_encoder` has produced its set of per-frame content vectors,
`sync_logit` cannot depend on *which* position each came from — only on the order-blind pooled set.
`verify_sync_fix.py` proves this (`torch.allclose` to 1e-5) by permuting `S_enc` directly (an earlier
draft permuted the raw mel and got a false failure, `max abs diff 1.76e-02`, before this was corrected).
Two supporting defects, also fixed: `MouthEncoder` encoded each frame independently, and
`TemporalConvNet`'s symmetric padding grew the sequence length, distorting alignment.

**The fix, in `crossfuse_v5.py` (gated behind `CONFIG["SYNC_ARCH"]`):**
1. `MouthEncoder` gained a 1D-conv temporal stack so lip *motion* is representable.
2. `pooled_ca` (old path) is **byte-for-byte unchanged** so 0.5074 / A10's 0.500 stay reproducible.
   (An earlier sinusoidal-positional-encoding draft broke that reproducibility and was reverted.)
3. `SYNC_ARCH = "offset_profile"` (default): mouth and audio project to a shared L2-normalised space;
   cosine similarity between `mouth[t]` and `audio[t+k]` for `k ∈ [-SYNC_MAX_OFFSET, SYNC_MAX_OFFSET]`,
   averaged over `t`, gives a `(B, 2K+1)` profile; a small MLP maps it to `sync_logit`. Core math is
   the standalone, testable `offset_similarity_profile()`.
4. Auxiliary InfoNCE loss (`LAMBDA_SYNC_NCE`, `SYNC_NCE_TEMP`), target `k=0` for matched windows.
5. `CrossFuseCropDataset` slices a *wider* mel window (`WINDOW_FRAMES + 2*SYNC_MAX_OFFSET`) from the
   cached full-clip mel — no re-extraction.
6. `offset_audio_encoder` uses `TemporalConvNet(..., causal=True)` (`Chomp1d`), scoped to the new head only.
7. `SYNC_DENSE_ALIGNED=False` (A10) is asserted incompatible with `offset_profile` (constructor and
   `build_model`); A10 must pass `SYNC_ARCH="pooled_ca"`.

**Verified offline, `verify_sync_fix.py`, 69/69:** the corrected invariance proof; the profile peaking
exactly at `k=0` for synthetic aligned embeddings and not under a time-shuffle; the full model changing
under a raw-audio permutation; checkpoint round-trip for both archs with a regression guard that
video/audio/fusion logits are byte-identical; invalid-combo assertions; gradients reaching every
sync-branch parameter; the real dataset path with widened slices; `evaluate_deterministic` and the exact
BCE+InfoNCE loss block running end to end.

**Caveat kept for any writeup:** fixing the architecture's ability to represent timing is not the same
claim as "sync becomes a useful deepfake signal on FakeAVCeleb" — and the probe showed exactly that: the
architecture is fixed, the signal is missing from the data.

## Track C — FF++-pretrained CLIP backbone — DONE

**Why:** the 0.95 in-domain vs ~0.61 cross-dataset gap. FakeAVCeleb is ~97% Wav2Lip (mouth-region);
DFDC and Celeb-DF are full-face swaps, so the video head never saw a face swap — a training-data
problem, not purely a modelling one. LNCLIP-DF (arXiv:2508.06248) and Effort (arXiv:2411.15633) report
CLIP ViT-L/14 with LayerNorm-only tuning on FF++ reaching 96.5/87.0 and 95.6/85.4 on Celeb-DF/DFDC.

**Executed (2026-09):** `A3` → `ffpp-crops-v5` · `B1` gate PASSED (Celeb-DF 0.9186, DFDC 0.8494) →
`ffpp-encoder-v5` · `B-train` (val video 0.9991, audio 0.9971, sync 0.5042) → `crossfuse-v5-ckpt-1` ·
`C-eval` (DFDC 0.8649, Celeb-DF 0.8447; in-domain test n=309: video 0.998, fusion 0.983, audio 0.978) ·
`D2` attribution (see Current status). Two notes: the final model's Celeb-DF (0.845) is below the
FF++-only B1 encoder's own zero-shot (0.9186) — different scoring code/sample so unproven, but FakeAVCeleb
fine-tuning may cost some Celeb-DF transfer (A4, best epoch 7, scored 0.900); and DFDC decision-threshold
accuracy does not transfer with AUC.

**Optional next work:** `X1` (plain CLIP, no FF++), extra seeds, a DDP rewrite if multi-GPU is ever wanted.

---

## Status log

_Most recent first. Add an entry after every Kaggle run or consistency pass._

- **2026-10-02 (review round 2: text fixes, tests, CI — Phase 1)** — Second review (7/10; no leakage, no
  metric bug) listed new issues. Done locally: stale/over-claiming text fixed (CLAUDE.md status, `ffpp_v5.py`
  header, D2 intro, `self_blend_clip` docstring which wrongly said the SBI recipe was in use); README now
  discloses self-blend (25% of real train windows, every arm), 2-identity test split, Celeb-DF class prior,
  audio head FakeAVCeleb-only, sync at chance, baseline-vs-new scoring protocol (baseline `crops[:W]` =
  first window only; new = 2 windows), MC-Dropout scope; B1 notebook `MULTI_GPU` forced False; added
  `requirements.txt`, GitHub Actions pytest, 43 new tests (mutation-checked: 5/5 injected bugs caught).
  Verified MC-Dropout limitation: transformer/attention dropout stays off (strict xfail). **Still open
  (needs Kaggle/GPU):** BatchNorm fed by silent-audio FF++ rows, scorer drops frames 24-31, FF++ frames
  non-contiguous, A15 at `LR_LN=1e-4`, `X1`, audio zero-shot cells, per-clip scores, missing B1/B-train JSONs.
- **2026-10-02 (README claim fixes after code review)** — A review found no leakage or metric bugs but
  several over-claims. README fixes: baseline described as trained on FakeAVCeleb alone (was "same data");
  ablation reading softened (FF++ in the mix = "no detectable difference", A14 undertrained, A15 confounded
  by `LR_LN` 1e-5 vs 1e-4); A14 added to the lower-bound note. Not yet disclosed in the README (optional):
  self-blend augmentation is on in every arm, sync is at chance, audio head validated in-domain only, test
  split has 2 identity groups, baseline and new model were scored with different window protocols.

- **2026-09-20 (README rewrite — committed `db17bc2`)** — On request, `README.md` was rewritten
  to show only the good, current results: dropped the paper-era baseline sections, the negative-results
  sections and the paper ablation table; kept the baseline only as the "before" column. Removed
  material is in git history (and at tag `paper-v1`); the full negative-results story remains in this
  file. Kept honest qualifiers in a short "Notes on the numbers" section (single seed, ±0.05 CIs,
  AUC vs threshold accuracy, time-limited arms) and described `sync` as experimental/not used for
  verdicts, so the README does not overstate. Added a per-clip attribution stat (288/309 correct
  four-way, read from `results/figures/modality_attribution_4way.png`). Cross-references to the old
  README sections were updated in this file, `sbi_v5.py`, `ffpp_v5.py` and the D2 notebook.
- **2026-09-20 (consistency + cleanup pass — committed `d356f85`)** — Audited every tracked file
  against the current state. `CLAUDE.md` rewritten (status-at-a-glance, corrected layout/provenance,
  sync section marked CLOSED, Track C marked DONE, references to removed files dropped).
  `crossfuse_v5.py`: comment/docstring-only fixes (stale "FF++/SBI encoder", `MULTI_GPU` config
  comment, docstring that still described the fix as not yet tried → "tested, then 3× slower", sync status comment at
  `SYNC_ARCH`); `sbi_v5.py` header now says CUT; `ffpp_v5.py` header records the gate result. Notebook
  markdown corrected: A2/B0 superseded/cut banners, A3 status + a stale "no original_sequences" message
  reworded, B1 table row "current" → "paper" + result, B2 result banner, B-train status/time-budget note,
  C-eval no longer claims an FF++ evaluation it never did, D-ablations pointer to D2 + A16 note, D2 status.
  `.gitignore`: caches + Kaggle output leftovers + corrected dataset comment. **Deleted (all recoverable
  from commit `fbf43bb`):** per-session `results/ablation_attrib_A*.csv` (exact duplicates of
  `results/ablation_attribution_v5.csv`, verified row-by-row), the multigpu-probe output log
  (dead-end record; findings are in this file), `docs/` (the executed cleanup plan), local caches.
  Kept deliberately: SBI code/notebooks (`sbi_v5.py`, `B0`, `A2` — evidence for a reported negative
  result) and the paper's root artifacts. Verified after edits: `py_compile` on all `.py`, all 10
  notebooks parse with unique cells, 4/4 tests. Not re-run: `verify_sync_fix.py` (the dev laptop has a
  `torch 2.9.1` / `torchvision 0.22.0` mismatch that breaks `import torchvision` locally only; Kaggle unaffected).
- **2026-09-20 (pushed to GitHub)** — Tagged `paper-v1` = `dacd546` (the paper's exact code) and pushed
  it; pushed commit `fbf43bb` to `main` (34 files). Repo is public. README gained a "Post-paper results"
  section and an updated reproducibility note; junk (a stale copy of this file and Kaggle output leftovers) removed first.
- **2026-09-20 (attribution ablation results)** — `D2` arms, cross-dataset video AUC (DFDC / Celeb-DF /
  mean), CIs in `results/ablation_attribution_v5.csv`: **A4 0.801 / 0.900 / 0.851** (time-cut, best ep 7);
  **A13 0.831 / 0.862 / 0.846**; **A15 0.785 / 0.716 / 0.750** (time-cut, best ep 7); **A14 0.514 / 0.587 /
  0.550** (best ep 10 = cap, in-domain video only 0.727 → likely undertrained). Reading: A4 ≈ A13 (FF++ in
  the mix adds nothing once the encoder is FF++-pretrained); A4 − A15 ≈ +0.10, mostly on Celeb-DF (0.900 vs
  0.716, CIs do not overlap; DFDC 0.801 vs 0.785 overlaps) → the pretraining stage matters; A14 ≈ baseline →
  FF++ data alone, without CLIP, does not help. `X1` not run.
- **2026-09-19 (attribution notebook prepared)** — Wrote `D2-ablation-attribution.ipynb` (old
  `D-ablations` left untouched — it scored in-domain only, where every arm is ~0.99). D2 scores each arm on
  the same Celeb-DF (400, `make_rng("celebdf_eval")`) and DFDC pools as C-eval, with face crops extracted
  once and cached. Arms A4/A13/A14/A15 + optional X1 (added because A13 keeps the FF++-pretrained encoder,
  so it cannot separate "CLIP" from "FF++"). 2 frozen + ≤8 fine-tune epochs, patience 4, one seed, per-arm
  budget from `plan_arm_budget_s`. `run_training_pipeline` now returns `stop_reason`.
- **2026-09-19 (C-eval done — Track C worked)** — Cross-dataset 0.61 → 0.855 (table at top). Before running,
  two `C-eval` bugs were fixed: it never installed `open_clip`, and took the first `manifest.csv` by glob
  order (could pick FF++'s); it now selects the manifest with an `audio_label` column. Kaggle output leftovers
  (auto-generated PNG folder and a HuggingFace-repos JSON) were discarded.
- **2026-09-19 (B-train finished; housekeeping)** — `B-train` ran single-GPU with the new timing gate
  (measures one fine-tune epoch, sets `TIME_BUDGET_S = 9 h − 1 h safety − probe`). Housekeeping done first:
  `should_stop` + `TIME_BUDGET_S` (unit-tested), timing gate rewritten (the old frozen-epoch probe hid the
  slow part and its ">90 s" warning was an EfficientNet-era threshold), `A10` fixed (`SYNC_ARCH="pooled_ca"`;
  it crashed `build_model`), `A16` found to duplicate `A4`. Checkpoint published as `crossfuse-v5-ckpt-1`.
- **2026-09-03 (multi-GPU saga, resolved: single GPU)** — See the MULTI_GPU invariant. Sequence: hang fixed
  → clean 200-row probe → live full run ~3× slower (2904 s vs 934.7 s per epoch; 5 consecutive epochs
  identical) → stopped, `MULTI_GPU` forced off in `B-train`, `enable_multi_gpu` docstring rewritten. The best
  checkpoint from that aborted run (epoch 7, val fusion 0.9782) was superseded by the single-GPU run.
- **2026-09-02 (B1 gate passed; sync cut)** — Corrected `B2-sync-probe` re-run (no NaN, two-phase 3+12 split,
  `torch.allclose` invariance diff 1.04e-07): `pooled_ca` 0.5062 / `offset_profile` 0.5129 val sync AUC, both
  below 0.70 → **sync cut for good**. Track C infra bugs found and fixed: (1) `ffpp_v5.py` never uploaded to
  `crossfuse-v5-lib`; (2) `discover_ffpp_videos` had no fallback for flat mirrors (`xdxd003/ff-c23` ships
  `FaceForensics++_C23/{original,Deepfakes,…}`) → added `{root}/**/original/**/*.mp4`; (3)
  `pip install open_clip_torch` without `--no-deps` disturbed torch/torchvision → efficientnet import made
  optional (loud `ImportError` only if requested) and install fixed to
  `--no-deps open_clip_torch timm safetensors ftfy regex pyyaml` (dependency list read from the wheel
  metadata); plus a stale Kaggle environment (torch 1.3.0, no pip access) fixed via Settings. **B1 then ran
  clean and passed:** Celeb-DF 0.9186 (gate 0.85, Δ+0.2787 over 0.6399), DFDC 0.8494 (gate 0.75, Δ+0.2695
  over 0.5799).
- **2026-08-16 (second Kaggle attempt of B2-sync-probe)** — First run finished but was **untrustworthy**:
  loss went `nan` from epoch 5 of the `pooled_ca` run (its first-run gate numbers 0.5196 / 0.5475 are
  discarded). Root cause: the config used `EPOCHS_FROZEN=15, EPOCHS_FINETUNE=0`, keeping phase 1's 10×
  warm-up LR (`LR_HEAD * 10`) for 15 epochs under mixed precision; NaN activations were baked into some
  `_mlp_head` `BatchNorm1d` running stats (which also produced non-finite `sync_logit` in the notebook's own
  invariance cell; that diff was computed by raw subtraction so matching infinities showed as `nan` — now
  `torch.allclose`). Fix: normal two-phase split (3+12) and `FREEZE_BLOCKS=12` so the backbone stays frozen
  through both phases.
- **2026-08-16** — Sync-head rewrite implemented in `crossfuse_v5.py` (see "Sync head investigation"),
  verified 69/69 offline, including a self-correction where an initial fix and test wrongly claimed
  invariance to permuting the *raw* mel (the real, provable claim is invariance to permuting the *encoded*
  sequence one step later).
