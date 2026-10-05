"""Run evaluator helpers from the notebook itself to prevent protocol drift."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import crossfuse_v5 as C


def notebook_functions():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "C-eval.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    functions = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "".join(line for line in source.splitlines(keepends=True)
                         if not line.lstrip().startswith(("!", "%")))
        functions.extend(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = dict(np=np, torch=torch, os=os, CONFIG=dict(C.CONFIG),
                     temps={"audio": 1.}, thrs={"video": .9, "audio": .5},
                     safe_auc=C.safe_auc, device=torch.device("cpu"),
                     maybe_cmvn=C.maybe_cmvn,
                     bootstrap_ci=lambda y, p, f: (.5, 1.))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_external_audio_report_uses_audio_threshold():
    ns = notebook_functions()
    result = ns["report_video_generalization"]("audio", [0, 1], [.3, .6], head="audio")
    assert result["acc"] == 1.


def test_external_audio_normalizes_features_and_excludes_decode_failures():
    ns = notebook_functions()
    raw = np.arange(40 * 128, dtype=np.float32).reshape(40, 128)
    ns["mfcc_from_waveform"] = lambda *args, **kwargs: raw

    def decode(path):
        if path == "bad":
            raise ValueError("decode failed")
        return np.ones(16000), 16000

    ns["ffmpeg_extract_wav"] = decode
    captured = []
    ns["predict_audio_only"] = lambda batch: captured.append(batch.numpy()) or np.array([.7])
    labels, scores = ns["score_audio_files"]([("bad", 0), ("good", 1)])
    assert labels == [1] and scores == [.7]
    assert np.allclose(captured[0][0], C.cmvn_normalize(raw))


def test_external_audio_does_not_call_visual_backbone():
    ns = notebook_functions()

    class AudioOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_encoder = nn.Sequential()  # patched to transpose below
            self.audio_head = nn.Linear(80, 1)

        def encode_visual(self, *args):
            raise AssertionError("Audio scoring must not encode images")

    model = AudioOnly()
    model.audio_encoder.forward = lambda mfcc: mfcc.transpose(1, 2)
    ns["model"] = model
    ns["CONFIG"]["N_MC_SAMPLES"] = 2
    scores = ns["predict_audio_only"](torch.ones(2, 40, 128))
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("path", sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")))
def test_notebook_python_cells_parse(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            source = "".join(line for line in source.splitlines(keepends=True)
                             if not line.lstrip().startswith(("!", "%")))
            ast.parse(source, filename=str(path))
