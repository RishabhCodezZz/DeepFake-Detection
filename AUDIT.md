# CrossFuse directory audit

Audited on 2026-10-05. The scope covers the Python libraries, verification script, tests, notebook templates and saved executions, result JSON/CSV/PNG files, README and project notes, Git configuration, CI, and the local Overleaf source packages. The Word report was excluded from content inspection at the user's request. Reports remain local and are ignored by Git.

## Repairs

| Area | Finding and change |
|---|---|
| Training | The final incomplete gradient-accumulation group was discarded. It now receives an optimizer update and is normalized by its actual group size. Regression tests compare the real training loop with direct larger-batch updates. Invalid accumulation settings and empty training loaders fail early. |
| Paired sampling | Inserting leftover examples between pair members could separate them across batches. Both samplers now pack pairs together and shuffle whole batches; paired batch sizes must be even. FakeAVCeleb pairs match connected identity groups, which does not guarantee the same source clip or recording conditions. |
| Sync negatives | A visual-only FF++ row could supply zero audio as a cross-identity negative for an audio-bearing row. The negative selector now requires the donor to have audio, otherwise using the same-clip shift. This does not make the failed sync branch reportable. |
| Split integrity | Pinned training rows were excluded from the final identity-overlap assertion. The assertion now includes them. Token iteration is sorted so newly generated identity groups do not depend on Python set order. Preserve the original manifests when reproducing historical runs. |
| FF++ discovery | Explicit c0/c40 paths could be accepted when c23 was requested. They are now excluded. Fallback patterns are checked separately for each root. Duplicate mirror cache keys fail before extraction can overwrite files. |
| FF++ splits | The loader could accept arbitrary train/val/test JSONs and combine different directories. It now requires co-located numeric pair lists with disjoint IDs. Official split use rejects missing targets and source pairs crossing train/held-out partitions. The historical target-ID fallback remains explicit. |
| Calibration | Tied or inverted predictions could select the ROC origin's infinite Youden threshold. Selection now uses finite thresholds, including in bootstrap calibration. Existing saved thresholds were finite and have not been rewritten. |
| External audio | C-eval omitted the training-time MFCC CMVN, used the video threshold for audio accuracy, ran unnecessary large image batches, and scored decode failures as zero features. It now matches audio preprocessing, uses the audio threshold, evaluates the audio branch directly, and excludes failures. No external audio results were available to revise. |
| Evaluation loading | C-eval restores companion training configuration when available, builds without downloading pretrained weights, and checks saved held-out sample keys from new runs. Legacy checkpoints without companion configuration emit a warning. |
| Reliability figures | The saved plots mixed fake probability with accuracy at a fitted cutoff. The template now uses confidence and correctness at 0.5, matching the ECE definition. Historical PNGs and executed notebooks remain unchanged; their reliability curves need regeneration from the checkpoint. |
| Extraction | Jobs could concurrently share a detector because thread-pool workers are not pinned to devices. Per-detector locks now serialize its jobs while allowing separate devices to run concurrently. Newly written manifests are sorted by cache key. |
| Training evidence | B-train now requires the encoder's two gate scores, rather than silently starting without pretraining. New records include actual FF++ training counts, split sample keys, stop reasons, and typed configuration. B1 now saves its split branch, sample keys, pretrained tag and gate result in encoder metadata. |
| Ablation templates | Removed A16, which claimed full CLIP fine-tuning while duplicating LayerNorm-only A4. The older grid now installs its CLIP dependencies, uses FakeAVCeleb for audio-only training, and describes its in-domain scope. D2 no longer calls the main checkpoint a completed 23-epoch run or treats unequal-budget gaps as causal evidence. |
| Repository | Preserved the downloaded C-eval execution as `results/executed/c-eval.executed.ipynb`. Recovered JSON records now use `.json` extensions. CI includes the 69-check sync verification. Reports, local environments and temporary audit extracts are excluded. |

## Evidence corrections

The main training record is `results/best_metrics_v5.json`: seed 42, selected epoch 13, CLIP ViT-L/14, an FF++ encoder, 887 training clips, 304 validation clips and 309 test clips. The original B-train log is unavailable. The README therefore describes FakeAVCeleb-only multimodal training with FF++ pretraining; it does not attribute a 5,000-row mixture to this checkpoint. The mixture is recorded for A4/A14/A15 in the attribution CSV (5,887 training rows). C-eval's reconstruction of a FakeAVCeleb-only split cannot independently establish the main training mixture.

The recovered B1 score JSON records Celeb-DF AUC 0.9177758 on 400 clips and DFDC AUC 0.8506696 on 398 clips. Earlier notes reported 0.9186/0.8494. Both versions clear the original 0.85/0.75 gates, but the missing execution log prevents resolving which scoring run produced the difference. The executed FF++ split branch is also unknown.

The main external result JSON is internally consistent: DFDC AUC 0.8649351 (397 clips) and Celeb-DF AUC 0.8447268 (400 clips), with mean 0.8548309. Main and ablation intervals and means were checked. Root PNG/JSON files and `results/ablation_results_v5.csv` belong to the older EfficientNet paper and remain intact.

Effort uses orthogonal weight subspace adaptation; it does not use the same LayerNorm-only adaptation as LNCLIP-DF. Source comments now distinguish them. Sources: [LayerNorm adaptation / GenD](https://arxiv.org/abs/2508.06248), [Effort](https://arxiv.org/abs/2411.15633).

## Validation

- A fresh local Python 3.12 environment installs the pinned requirements with torch 2.7.0+cpu and torchvision 0.22.0+cpu. `python -m pip check` reports no broken requirements.
- Direct imports of TorchVision, OpenCLIP, librosa, pandas and SciPy succeed; actual nonzero MFCC/log-mel extraction and slicing pass a CPU smoke check.
- The complete test suite passes; the final count is recorded below. One strict expected failure documents that functional transformer/attention dropout remains deterministic under the current MC estimator.
- `python verify_sync_fix.py`: all 69 checks pass, including complete model forwards, gradients, datasets and checkpoint round trips for both sync architectures.
- All Python sources and notebook Python cells parse. Both saved executed notebooks contain no recorded error outputs. This is structural validation, not a new Kaggle execution.
- All ten PNG files decode, the three local Overleaf ZIPs pass CRC checks, and the current source package's six manifest hashes and sizes match. Its bibliography/citation and label/reference targets are consistent.
- Secret-pattern scanning found no matching private keys, GitHub tokens, AWS access keys or Hugging Face tokens in the files selected for publication. This is not an exhaustive security guarantee.
- `git diff --check` passes. The remote main branch was checked before publication.

Final test result: **78 passed, 1 expected failure**. Two harmless CPU warnings explain that pinned-memory transfers have no accelerator available.

## Remaining limitations and follow-up work

1. **Research evidence:** one seed, only two held-out FakeAVCeleb identity groups, development-informed external pools, and unknown executed FF++ split provenance limit generalization claims. None can be repaired by editing saved scores. New runs should retain manifests, keys, configuration, raw predictions and execution logs.
2. **Controlled comparisons:** rerun A15 at the pretraining LayerNorm learning rate, run the optional X1 arm, and compare repeated seeds under matched budgets. Time-limited scores are not mathematical lower bounds on later performance.
3. **Operating thresholds:** the FakeAVCeleb cutoff gives about 55.7% DFDC accuracy despite useful AUC. A new deployment-domain validation set is needed to choose a cutoff; do not tune against the reported external test labels.
4. **Mixed-modality normalization:** FF++ rows are masked from audio-dependent losses, but their zero features still pass through audio BatchNorm and can influence statistics in mixed batches. Loss masking alone does not fully isolate missing modalities. Redesign and validate this path before drawing stronger mixture conclusions; the main recovered record has no FF++ mixture.
5. **Sync supervision:** the branch still trains but fails its verdict gate. Boundary-clipped mel windows can compress the intended padded time span, and constructed same-clip shifts do not establish natural desynchronization performance. Timestamp-preserving padding and suitable sync data need separate validation.
6. **Historical plots:** rerun C-eval with the actual checkpoint to regenerate reliability curves and save per-clip logits. The old confusion matrices and ROC panels remain useful; the incorrect reliability curves are disclosed beside them.
7. **Reproduction:** the trained checkpoint and datasets are not in this directory, so end-to-end Kaggle training, checkpoint evaluation and GPU throughput were not rerun. Upload the edited libraries and notebook templates to Kaggle before using the fixes. Saved metrics describe the historical code, not the repaired recipe.
8. **Reports:** the current Overleaf ZIP matches its source and manifest; numbered ZIPs are older drafts and should not be mistaken for the latest one. LaTeX compilation, page count and report layout are unverified. The Word report was left alone as requested. Reports are excluded from the GitHub push.
9. **Licensing:** the repository has no project-wide LICENSE. The owner needs to choose the intended reuse terms; this audit does not assign a license. Dataset and bundled third-party class permissions remain their respective owners' terms.

The repository is suitable for publication as a research project with these caveats. The audit does not establish deployment readiness or validate new model performance.
