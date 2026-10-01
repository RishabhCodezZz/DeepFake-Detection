"""Small CPU-only checks on the tensor helpers the sync and fusion heads rely on."""
import pytest
import torch
import torch.nn as nn

from crossfuse_v5 import enable_mc_dropout, masked_mean_std, offset_similarity_profile


def test_masked_mean_std_ignores_masked_frames():
    x = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :6] = True
    junk = x.clone()
    junk[:, 6:] = 1e6                       # garbage in the masked region
    assert torch.allclose(masked_mean_std(x, mask), masked_mean_std(junk, mask))
    assert torch.allclose(masked_mean_std(x, mask),
                          masked_mean_std(x[:, :6], None), atol=1e-5)


def test_masked_mean_std_returns_mean_then_std():
    x = torch.tensor([[[1.0], [3.0]]])
    out = masked_mean_std(x)
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(2.0)
    assert out[0, 1] == pytest.approx(1.0, abs=1e-4)


def _unit(t):
    return t / t.norm(dim=-1, keepdim=True)


def test_offset_profile_peaks_at_zero_for_aligned_embeddings():
    torch.manual_seed(0)
    B, W, E, K = 3, 12, 16, 3
    audio = _unit(torch.randn(B, W + 2 * K, E))
    mouth = audio[:, K:K + W, :].clone()    # mouth[t] == audio[t + K], i.e. offset 0
    profile = offset_similarity_profile(mouth, audio, torch.ones(B, W, dtype=torch.bool), K)
    assert profile.shape == (B, 2 * K + 1)
    assert profile.argmax(dim=1).tolist() == [K] * B


def test_offset_profile_moves_when_the_audio_is_shifted():
    torch.manual_seed(1)
    B, W, E, K = 2, 12, 16, 3
    audio = _unit(torch.randn(B, W + 2 * K, E))
    mouth = audio[:, K + 2:K + 2 + W, :].clone()    # true offset is +2
    profile = offset_similarity_profile(mouth, audio, torch.ones(B, W, dtype=torch.bool), K)
    assert profile.argmax(dim=1).tolist() == [K + 2] * B


def test_enable_mc_dropout_turns_dropout_on_and_leaves_batchnorm_in_eval():
    net = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))
    net.eval()
    enable_mc_dropout(net)
    assert net[2].training is True
    assert net[1].training is False


def test_enable_mc_dropout_gives_different_outputs_across_passes_for_plain_dropout():
    net = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5)).eval()
    enable_mc_dropout(net)
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert not torch.equal(net(x), net(x))


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation: enable_mc_dropout only flips nn.Dropout modules, but "
    "nn.TransformerEncoderLayer and nn.MultiheadAttention run their own dropout "
    "off the layer's training flag and the eval fast path, so a transformer "
    "stays deterministic.  The fusion head's cross-attention may be affected."))
def test_enable_mc_dropout_makes_a_transformer_layer_stochastic():
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32,
                                       dropout=0.5, batch_first=True)
    layer.eval()
    enable_mc_dropout(layer)
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert not torch.equal(layer(x), layer(x))
