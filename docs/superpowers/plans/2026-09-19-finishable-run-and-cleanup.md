# Finishable Track C Run + Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CLIP ViT-L/14 + FF++ `B-train` run finish inside Kaggle's 9-hour session cap on a single T4, fix the two known notebook bugs, and remove clutter without losing anything irreplaceable.

**Architecture:** Single-GPU only (nn.DataParallel is confirmed ~3x slower for this model, see CLAUDE.md). Add a wall-clock budget to the existing early-stopping check so a run stops cleanly after its best checkpoint instead of being killed mid-epoch. Cleanup is proposal-first: nothing is deleted until the user confirms the list in Task 4.

**Tech Stack:** Python 3.12, PyTorch, open_clip, Kaggle notebooks (T4), pytest.

**Spec:** `CLAUDE.md` (status log + invariants) and `README.md` (frozen paper results). No separate design doc exists.

## Global Constraints

- `CONFIG["MULTI_GPU"]` stays `False` for `clip_vit_l14`. Do not re-enable `nn.DataParallel`.
- No scikit-learn imports in new code (NumPy reimplementations only).
- Preregistered gates (`SYNC_MIN_AUC=0.70`, `GATE_CELEBDF_AUC=0.85`, `GATE_DFDC_AUC=0.75`) are not adjusted.
- Identity-disjoint splitting stays on by default.
- Do not commit, push, or delete anything the user has not approved. The user said earlier: "do not commit or anything".
- Kaggle limits: 9 hours per session, 30 GPU hours per week.
- Every edit to `crossfuse_v5.py` or `ffpp_v5.py` needs a re-upload of the `crossfuse-v5-lib` Kaggle Dataset before a notebook uses it.

## Why the current run is not working (evidence)

1. **Wall-clock.** Single-GPU frozen epoch = 934.7s on the full 5,887-row manifest. Fine-tune epochs are slower (gradient checkpointing recomputes the forward pass once the LayerNorms train). 23 epochs at that speed is close to the 9h cap.
2. **The multi-GPU attempt made it worse.** 2904s/epoch, 3.1x slower, stable across 5 epochs (log: 7291.4s, 10194.6s, 13099.9s, 16004.0s, 18911.6s).
3. **The run had already peaked.** Best val fusion 0.9782 at epoch 7. Epoch 8 was not a new best. Compute past that point buys little.
4. **The only timing check is blind to the slow part.** The timing gate in `B-train.ipynb` runs one *frozen* epoch. It never measures a fine-tune epoch, and its ">90s" warning is a stale threshold from the EfficientNet-B4 era.
5. **`D-ablations.ipynb` will crash at row 10.** `A10_sync_unaligned` sets `SYNC_DENSE_ALIGNED=False` but inherits `SYNC_ARCH="offset_profile"`, which `build_model` rejects.
6. **Repo state is confusing.** Code, README and result files describe different models; several duplicate files sit at the repo root.

## File Structure

- Modify: `crossfuse_v5.py` (add `should_stop`, `TIME_BUDGET_S` config key, wire into the epoch loop near line 2481)
- Create: `tests/test_should_stop.py`
- Modify: `notebooks/B-train.ipynb` (timing gate measures a fine-tune epoch, real ETA)
- Modify: `notebooks/D-ablations.ipynb` (fix A10 arm)
- Modify: `CLAUDE.md` (record results)
- Delete (only after approval in Task 4): see Task 4 list

---

### Task 1: Wall-clock-aware stopping rule

**Files:**
- Modify: `crossfuse_v5.py` (CONFIG near line 252; loop at line 2481)
- Test: `tests/test_should_stop.py`

**Interfaces:**
- Produces: `should_stop(since_best: int, patience: int, phase: int, elapsed_s: float, budget_s: float | None) -> str | None`. Returns `None` to continue, or a reason string (`"patience"` or `"time_budget"`).
- Consumes: `cfg["PATIENCE"]` (int, exists), `cfg["TIME_BUDGET_S"]` (float or None, new).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_should_stop.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crossfuse_v5 import should_stop


def test_continue_when_nothing_triggers():
    assert should_stop(2, 6, 1, 1000.0, 30000.0) is None


def test_patience_only_applies_in_finetune_phase():
    assert should_stop(9, 6, 0, 10.0, None) is None
    assert should_stop(6, 6, 1, 10.0, None) == "patience"


def test_time_budget_stops_in_any_phase_when_exceeded():
    assert should_stop(0, 6, 1, 30001.0, 30000.0) == "time_budget"
    assert should_stop(0, 6, 0, 30001.0, 30000.0) == "time_budget"


def test_no_budget_means_no_time_stop():
    assert should_stop(0, 6, 1, 10**9, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_should_stop.py -v`
Expected: FAIL with `ImportError: cannot import name 'should_stop'`

- [ ] **Step 3: Implement**

Add above `run_training_pipeline` in `crossfuse_v5.py`:

```python
def should_stop(since_best, patience, phase, elapsed_s, budget_s):
    """Why training should stop now, or None to keep going.

    Patience only counts in the fine-tune phase (phase == 1), matching the
    old inline check. The time budget applies in every phase: it exists so a
    run ends cleanly, with its best checkpoint already on disk, instead of
    being killed by Kaggle's 9h session cap mid-epoch.
    """
    if budget_s is not None and elapsed_s >= budget_s:
        return "time_budget"
    if phase == 1 and since_best >= patience:
        return "patience"
    return None
```

Add to `CONFIG` next to `"PATIENCE": 6,`:

```python
    "TIME_BUDGET_S":           None,   # e.g. 7.5 * 3600 on Kaggle; None = no limit
```

In `run_training_pipeline`, record the start time once, before the epoch loop (`import time` at the top of the file if `time` is not already imported; check with `grep -n "^import time" crossfuse_v5.py`):

```python
    t_run_start = time.time()
```

Replace the block at line 2481:

```python
        if since_best >= cfg["PATIENCE"] and phase == 1:
            if verbose:
                print(f"  [{tag}] Early stopping at epoch {epoch + 1}.")
            break
```

with:

```python
        stop_reason = should_stop(since_best, cfg["PATIENCE"], phase,
                                  time.time() - t_run_start, cfg.get("TIME_BUDGET_S"))
        if stop_reason:
            if verbose:
                print(f"  [{tag}] Stopping at epoch {epoch + 1}: {stop_reason}.")
            break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_should_stop.py -v && python -m py_compile crossfuse_v5.py`
Expected: 4 passed, no compile error

- [ ] **Step 5: Regression check**

Run: `python verify_sync_fix.py`
Expected: `ALL 69 CHECKS PASSED` (it exercises the real training loop)

- [ ] **Step 6: Commit (only if the user approves committing)**

```bash
git add tests/test_should_stop.py crossfuse_v5.py
git commit -m "feat: wall-clock budget in training stop rule"
```

---

### Task 2: Timing gate that measures the slow part and gives a real ETA

**Files:**
- Modify: `notebooks/B-train.ipynb` (the "Timing gate" cell, currently cell index 4; re-read the notebook with the Read tool first, cell ids are positional and shift after edits)

**Interfaces:**
- Consumes: `run_training_pipeline`, `CONFIG`, `MANIFEST`, `CROP_DIR` (defined in earlier cells).
- Produces: sets `CONFIG["TIME_BUDGET_S"]` for the main run (consumed by Task 1's `should_stop`).

- [ ] **Step 1: Replace the timing-gate cell**

```python
# ---- Timing gate: measure ONE FINE-TUNE epoch, then set a time budget ----
# The old gate ran a frozen epoch, which skips the backbone backward pass and
# so under-reports the real per-epoch cost. EPOCHS_FROZEN=0 goes straight to
# fine-tune (LayerNorms trainable, gradient checkpointing active).
_timing_cfg = dict(CONFIG, EPOCHS_FROZEN=0, EPOCHS_FINETUNE=1, N_SEEDS=1,
                   TIME_BUDGET_S=None)
import time
t0 = time.time()
_probe_result = run_training_pipeline(MANIFEST, CROP_DIR, _timing_cfg,
                                      tag="timing_probe", seed=CONFIG["SEED"],
                                      verbose=True)
elapsed = time.time() - t0
del _probe_result
torch.cuda.empty_cache()

SESSION_CAP_S = 9 * 3600
SAFETY_S = 1.0 * 3600            # setup, eval, checkpoint upload
n_epochs = CONFIG["EPOCHS_FROZEN"] + CONFIG["EPOCHS_FINETUNE"]
print(f"\nOne fine-tune epoch: {elapsed:.0f}s ({elapsed / 60:.1f} min).")
print(f"Full {n_epochs}-epoch run would take ~{n_epochs * elapsed / 3600:.1f} h "
      f"(patience={CONFIG['PATIENCE']} usually stops it earlier).")

CONFIG["TIME_BUDGET_S"] = SESSION_CAP_S - SAFETY_S - elapsed   # minus this probe
print(f"TIME_BUDGET_S set to {CONFIG['TIME_BUDGET_S'] / 3600:.2f} h; the run "
      f"stops cleanly after its best checkpoint when this is exceeded.")
```

- [ ] **Step 2: Validate syntax**

Run:
```bash
python -c "import json,ast; nb=json.load(open('notebooks/B-train.ipynb',encoding='utf-8')); [ast.parse('\n'.join(l for l in ''.join(c['source']).split('\n') if not l.strip().startswith('!'))) for c in nb['cells'] if c['cell_type']=='code']; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Re-upload and run on Kaggle**

Re-upload `crossfuse_v5.py` to `crossfuse-v5-lib` (Task 1 changed it), upload the notebook, restart the session, Run All. Read the printed fine-tune epoch time and the estimated hours. If the estimate exceeds ~7 h even with patience, lower `CONFIG["EPOCHS_FINETUNE"]` in the backbone cell (validation peaked at epoch 7 last time, so 12 is a defensible cap).

---

### Task 3: Fix the A10 ablation crash

**Files:**
- Modify: `notebooks/D-ablations.ipynb` (arm `A10_sync_unaligned`)

**Interfaces:**
- Consumes: `cfg_variant(**overrides)` and `BASE` (defined earlier in the notebook).

- [ ] **Step 1: Write the failing check**

```python
# run from repo root
import copy
from crossfuse_v5 import CONFIG, build_model

def cfg_variant(**kw):
    c = copy.deepcopy(CONFIG); c.update(kw); return c

bad = cfg_variant(BACKBONE="efficientnet_b4", FREEZE_BLOCKS=4, SYNC_DENSE_ALIGNED=False)
try:
    build_model(bad)
    raise SystemExit("expected ValueError, got none")
except ValueError as e:
    print("reproduced:", str(e)[:80])

good = cfg_variant(BACKBONE="efficientnet_b4", FREEZE_BLOCKS=4,
                   SYNC_DENSE_ALIGNED=False, SYNC_ARCH="pooled_ca")
build_model(good)
print("fixed config builds")
```

- [ ] **Step 2: Run it**

Run: `python <that script>`
Expected: `reproduced: SYNC_DENSE_ALIGNED=False requires SYNC_ARCH='pooled_ca'...` then `fixed config builds`

- [ ] **Step 3: Patch the notebook arm**

In `notebooks/D-ablations.ipynb`, change
`("A10_sync_unaligned", cfg_variant(**BASE, SYNC_DENSE_ALIGNED=False), M),`
to
`("A10_sync_unaligned", cfg_variant(**BASE, SYNC_DENSE_ALIGNED=False, SYNC_ARCH="pooled_ca"), M),`

Use the Read tool on the notebook first to get fresh cell ids.

- [ ] **Step 4: Validate syntax** (same one-liner as Task 2, pointed at `D-ablations.ipynb`)

Expected: `ok`

---

### Task 4: Repo cleanup (proposal first, delete only after approval)

The request said "delete the files and everything that are" and the sentence ends there. Confirm the intended rule before running anything in this task. Proposed default rule: *duplicates and superseded probes, nothing that holds a result that exists nowhere else.*

**Files:**
- Delete candidates (safe, each is either a byte-identical copy or a finished throwaway):

| Path | Why it is safe to delete |
|---|---|
| `ablation_results_v5.csv` (repo root) | Byte-identical to `results/ablation_results_v5.csv` (verified with `diff -q`) |
| `AGENTS.md` | Identical to `CLAUDE.md` except line 1 (verified) |
| `notebooks/multigpu-probe.ipynb` | Probe finished. Conclusion (DataParallel unusable for CLIP) is recorded in `CLAUDE.md` and in the `enable_multi_gpu` docstring |
| `notebooks/multigpu-probe output.ipynb` | Executed copy of the above |
| `__pycache__/` | Already in `.gitignore`, regenerated automatically |

- Keep, do NOT delete (irreplaceable or history):

| Path | Why keep |
|---|---|
| `audio_eval_panel.png`, `fusion_eval_panel.png`, `sync_eval_panel.png`, `video_eval_panel.png`, `modality_attribution_4way.png`, `calibration_v5.json`, `crossdataset_results_v5.json` | The paper's actual result artifacts. **Untracked in git**, so deleting them is permanent |
| `b2-sync-probe.ipynb` (repo root) | The only executed copy holding the sync-probe outputs (0.5062 / 0.5129). Move it, do not delete (see Step 2) |
| `sbi_v5.py`, `notebooks/B0-sbi-pretrain.ipynb`, `notebooks/A2-ffpp-extract.ipynb` | Part of the reported negative-results story |
| `ffpp_v5.py`, `verify_sync_fix.py` | Needed by Track C / offline verification |

- [ ] **Step 1: Show the user the exact list and get a yes**

Run: `git status --short` and paste the two tables above. Wait for confirmation.

- [ ] **Step 2: Preserve the executed sync probe**

```bash
mkdir -p results/executed
git mv -k b2-sync-probe.ipynb results/executed/ 2>/dev/null || mv b2-sync-probe.ipynb results/executed/b2-sync-probe.executed.ipynb
```

- [ ] **Step 3: Delete the approved duplicates**

```bash
rm ablation_results_v5.csv AGENTS.md
rm "notebooks/multigpu-probe.ipynb" "notebooks/multigpu-probe output.ipynb"
rm -rf __pycache__
```

- [ ] **Step 4: Verify nothing needed broke**

Run: `python -m py_compile crossfuse_v5.py ffpp_v5.py && python -m pytest tests -v && ls results`
Expected: compiles, tests pass, `results/` still contains `ablation_results_v5.csv`

---

### Task 5: Keep the paper version reproducible (needs user approval, touches git)

**Files:**
- Modify: `README.md` (short note near the top)
- Git tag (local): `paper-v1` on `dacd546`

- [ ] **Step 1: Add the note to `README.md` under the title**

```markdown
> **Reproducibility note.** The numbers below come from the EfficientNet-B4,
> FakeAVCeleb-only code at git tag `paper-v1`. `main` continues past that with a
> CLIP ViT-L/14 + FaceForensics++ track that is still being evaluated.
```

- [ ] **Step 2: Create the tag**

```bash
git tag -a paper-v1 dacd546 -m "Code as used for the paper: EfficientNet-B4, FakeAVCeleb-only"
```

- [ ] **Step 3: Update `CLAUDE.md` status log** with the outcome of Tasks 1-3 (fine-tune epoch time, stop reason, final val AUCs). Do not commit unless the user says so.

---

## Self-Review

- **Spec coverage:** Each failure in "Why the current run is not working" maps to a task: 1 and 4 -> Task 1 and 2; 2 -> Global Constraints plus Task 2; 3 -> Task 1 (time budget, patience); 5 -> Task 3; 6 -> Tasks 4 and 5.
- **Placeholders:** none. The deletion rule is flagged as needing user confirmation because their sentence was cut off, not left as a TBD.
- **Type consistency:** `should_stop(since_best, patience, phase, elapsed_s, budget_s)` is the same in the test, the implementation and the call site; `TIME_BUDGET_S` is the same key in CONFIG, `cfg.get(...)` and the notebook.
- **Known gap:** I have not seen the latest single-GPU run's output. If that run already finished or failed differently, "Why it is not working" should be corrected from its log before executing.
