"""Exercise the real training loop without downloading a visual backbone."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import crossfuse_v5 as C


class TinyDataset(Dataset):
    def __init__(self, rows, indices, crop_dir, cfg, training=False):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        x = torch.tensor([float(index + 1) / 10])
        label = torch.tensor(float(index % 2))
        return x, x, x, x, label, label, torch.ones(1, dtype=torch.bool), torch.tensor(index), torch.tensor(1.)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.head = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        self.seen = []

    def n_backbone_blocks(self):
        return 0

    def set_backbone_trainable(self, count):
        pass

    def backbone_param_groups(self, *args, **kwargs):
        return []

    def encode_visual(self, crops):
        self.seen.append(crops.detach().clone())
        return crops, None

    def forward_from_tokens(self, tokens, mouth, mfcc, sync, visual_mask=None):
        logits = self.head(tokens).squeeze(-1)
        return {"video_logit": logits, "frame_logits": logits[:, None],
                "audio_logit": logits, "fusion_logit": None, "sync_logit": None}


@pytest.mark.parametrize("accum", [3, 8])
def test_partial_gradient_group_updates_match_direct_group_averages(monkeypatch, tmp_path, accum):
    monkeypatch.setattr(C, "device", torch.device("cpu"))
    monkeypatch.setattr(C, "CrossFuseCropDataset", TinyDataset)
    monkeypatch.setattr(C, "build_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(C, "build_3way_split_packed", lambda *args, **kwargs:
                        (np.arange(10), np.arange(2), np.arange(2)))
    monkeypatch.setattr(C, "evaluate_deterministic", lambda *args:
                        {"video_auc": 0.75, "audio_auc": 0.75, "sync_auc": float("nan"), "fusion_auc": float("nan")})
    steps = []

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            steps.append(1)
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", lambda groups, **kwargs: RecordingSGD(groups))
    cfg = dict(C.CONFIG, EPOCHS_FROZEN=0, EPOCHS_FINETUNE=1, BATCH_CLIPS=2,
               GRAD_ACCUM=accum, NUM_WORKERS=0, USE_PAIRED_BATCHES=False,
               GENERATE_HARD_NEGATIVES=False, MODALITY="video_only", LAMBDA_FRAME=0.,
               LR_HEAD=0.1, WORK_DIR=str(tmp_path))
    rows = [{"video_label": i % 2} for i in range(10)]
    result = C.run_training_pipeline(rows, str(tmp_path), cfg, verbose=False)
    # Five mini-batches: accum=3 must flush two; accum=8 must flush one.
    assert len(steps) == (2 if accum == 3 else 1)
    # Compare with direct larger batches in the exact shuffled order. This
    # checks the partial group's normalization, not only the update count.
    reference = TinyModel()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)  # epoch-1 warmup factor
    batches = result["model"].seen
    for start in range(0, len(batches), accum):
        x = torch.cat(batches[start:start + accum])
        labels = ((x[:, 0] * 10).round().long() - 1).remainder(2).float()
        loss = nn.functional.binary_cross_entropy_with_logits(
            reference.head(x).squeeze(-1), C.smooth_labels(labels))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        optimizer.step()
    assert torch.allclose(result["model"].head.weight, reference.head.weight, atol=1e-7)
