"""
Offline verification for the sync-head rewrite (CPU only, no Kaggle needed).

Two things this proves, not just checks:
  1. The OLD architecture ("pooled_ca") is, downstream of mel_encoder,
     exactly invariant to any permutation of the ENCODED audio sequence's
     time positions -- the mathematical root cause of the reported 0.5074
     chance-level sync AUC. (NOT invariant to permuting the raw mel bins --
     mel_encoder is a CNN with local receptive fields, so that would be a
     genuinely different encoding, not a relabelling of the same one. See
     the comment above the pooled_ca branch in forward_from_tokens.)
  2. The NEW architecture ("offset_profile") is provably NOT invariant to
     that same kind of permutation, and its core similarity computation
     genuinely peaks at the correct offset when fed a synthetically aligned
     pair -- independent of whether randomly-initialised network weights
     happen to produce a meaningful alignment (they don't, and don't need
     to for this check).

Also exercises: full checkpoint round-trip for both SYNC_ARCH values, the
SYNC_DENSE_ALIGNED=False ablation combo, has_audio=0 (FF++-style) rows
through the real CrossFuseCropDataset code path, the invalid
offset_profile+unaligned combination raising as expected, gradient reaching
every sync-branch parameter, and evaluate_deterministic plus the training
loop's exact sync loss block (BCE + InfoNCE) running end to end.

Run: python verify_sync_fix.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossfuse_v5 as C

torch.manual_seed(0)
np.random.seed(0)

CHECKS = {"n": 0}


def check(cond, msg):
    CHECKS["n"] += 1
    if not cond:
        raise AssertionError(f"CHECK FAILED (#{CHECKS['n']}): {msg}")
    print(f"  [ok] {msg}")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# A small, fast config shared by every model-building test below. Backbone
# is untrained (pretrained=False, random weights) -- every check here is
# about architecture PROPERTIES (invariance, gradient flow, shape/checkpoint
# correctness), not learned behaviour, so random weights are sufficient and
# keep this runnable on a CPU in seconds.
def base_cfg(**overrides):
    cfg = dict(C.CONFIG)
    cfg.update(
        CROP_SIZE=64, WINDOW_FRAMES=4, TRAIN_RES=64, N_MELS=80,
        BACKBONE="efficientnet_b4", FREEZE_BLOCKS=4, GRAD_CHECKPOINTING=False,
        AUDIO_FRAMES=32, SYNC_MEL_FRAMES=32, SYNC_MAX_OFFSET=3, SYNC_EMB_DIM=32,
    )
    cfg.update(overrides)
    return cfg


def build(cfg, pretrained=False):
    return C.build_model(cfg, pretrained=pretrained)


def rand_batch(cfg, B=2):
    W, S = cfg["WINDOW_FRAMES"], cfg["CROP_SIZE"]
    crops = torch.randint(0, 256, (B, W, S, S, 3), dtype=torch.uint8)
    mfcc = torch.randn(B, 40, cfg["AUDIO_FRAMES"])
    mask = torch.ones(B, W, dtype=torch.bool)
    return crops, mfcc, mask


def rand_sync_audio(cfg, B=2):
    frames = (cfg["WINDOW_FRAMES"] + 2 * cfg["SYNC_MAX_OFFSET"]
              if cfg["SYNC_ARCH"] == "offset_profile" else cfg["SYNC_MEL_FRAMES"])
    # mel_encoder's input width is n_mels when SYNC_DENSE_ALIGNED, else 40
    # (the MFCC width) -- see CrossFuseModelV5.__init__'s `sync_in`.
    channels = cfg["N_MELS"] if cfg["SYNC_DENSE_ALIGNED"] else 40
    return torch.randn(B, channels, frames)


# =====================================================================
# 1. Prove the OLD architecture's classifier is invariant to the ORDER of
#    the encoded audio sequence (not the raw mel -- mel_encoder is a CNN
#    with local receptive fields, so permuting ITS input is a genuinely
#    different encoding, not a relabelling of the same one; see the comment
#    above the pooled_ca branch in forward_from_tokens for the precise
#    claim). The provable invariance is one step later, at S_enc.
# =====================================================================
section("1. pooled_ca: proving the downstream permutation-invariance bug")


def pooled_ca_from_encoded(model, M, S_enc, visual_mask, key_padding_mask):
    """Replicates forward_from_tokens's pooled_ca branch from S_enc onward,
    so the invariance claim can be tested at the exact point it applies --
    forward_from_tokens itself always recomputes S_enc fresh from raw audio
    and has no hook to inject an already-encoded, permuted sequence."""
    ca_m, _ = model.sync_ca_m2a(query=M, key=S_enc, value=S_enc)
    fused_m = model.sync_ln_m(M + ca_m)
    ca_a, _ = model.sync_ca_a2m(query=S_enc, key=M, value=M, key_padding_mask=key_padding_mask)
    fused_s = model.sync_ln_a(S_enc + ca_a)
    return model.sync_head(torch.cat(
        [C.masked_mean_std(fused_m, visual_mask), C.masked_mean_std(fused_s, None)],
        dim=-1)).squeeze(-1)


cfg1 = base_cfg(SYNC_ARCH="pooled_ca")
m1 = build(cfg1).eval()
crops, mfcc, mask = rand_batch(cfg1)
sync_audio = rand_sync_audio(cfg1)
key_padding_mask = ~mask

with torch.no_grad():
    v_tok, m_tok = m1.encode_visual(crops)
    out1 = m1.forward_from_tokens(v_tok, m_tok, mfcc, sync_audio, mask)
    check(out1["sync_logit"] is not None, "pooled_ca produces a sync_logit")

    M = m1.mouth_proj(m_tok)
    S_enc = m1.mel_encoder(sync_audio)
    logit_a = pooled_ca_from_encoded(m1, M, S_enc, mask, key_padding_mask)
    check(torch.allclose(out1["sync_logit"], logit_a, atol=1e-5),
          "pooled_ca_from_encoded reproduces forward_from_tokens's own "
          "sync_logit exactly (confirms the replica is faithful before "
          "using it to test the claim)")

    perm = torch.randperm(S_enc.shape[1])
    logit_b = pooled_ca_from_encoded(m1, M, S_enc[:, perm, :], mask, key_padding_mask)

check(torch.allclose(logit_a, logit_b, atol=1e-5),
      "downstream of mel_encoder, sync_logit is EXACTLY invariant to a "
      "random permutation of the ENCODED audio sequence's T positions (max "
      f"abs diff {(logit_a - logit_b).abs().max().item():.2e}) -- this is "
      "the proven root cause of the 0.5074 chance-level result: once local "
      "content is embedded per-frame, nothing downstream can tell WHICH "
      "position it came from relative to the mouth sequence")

# Repeat with several independent permutations so this isn't a fluke of one
# particular shuffle.
for trial in range(5):
    torch.manual_seed(100 + trial)
    perm = torch.randperm(S_enc.shape[1])
    with torch.no_grad():
        logit_t = pooled_ca_from_encoded(m1, M, S_enc[:, perm, :], mask, key_padding_mask)
    check(torch.allclose(logit_a, logit_t, atol=1e-5),
          f"invariance holds under independent random permutation #{trial}")

# Contrast check: permuting the RAW mel (before mel_encoder) is NOT a no-op
# -- mel_encoder's CNN has local receptive fields, so this produces a
# genuinely different encoding, not a relabelling. Confirms the invariance
# really is specific to the post-encoder stage, not a vacuous test.
with torch.no_grad():
    raw_perm = torch.randperm(sync_audio.shape[-1])
    out_raw_shuffled = m1.forward_from_tokens(v_tok, m_tok, mfcc, sync_audio[:, :, raw_perm], mask)
check(not torch.allclose(out1["sync_logit"], out_raw_shuffled["sync_logit"], atol=1e-5),
      "control: shuffling the RAW mel bins (pre-mel_encoder) DOES change "
      "sync_logit -- the invariance is specific to the encoded sequence, "
      "not the raw input, confirming the test isn't vacuous")

# Other heads must NOT be invariant to their own inputs -- a further sanity
# check that this is a property of the sync branch specifically.
with torch.no_grad():
    out_diff_mfcc = m1.forward_from_tokens(v_tok, m_tok, torch.randn_like(mfcc), sync_audio, mask)
check(not torch.allclose(out1["audio_logit"], out_diff_mfcc["audio_logit"], atol=1e-5),
      "control: audio_logit DOES change when its own input changes (rules "
      "out a vacuous test)")


# =====================================================================
# 2. offset_similarity_profile: peaks at k=0 for a synthetically aligned
#    pair, using the EXACT production function with no trained weights
#    involved -- isolates the math from whether training will succeed.
# =====================================================================
section("2. offset_similarity_profile: peak-at-k=0 with synthetic embeddings")

B, W, K, E = 3, 5, 4, 16
mouth_emb = F.normalize(torch.randn(B, W, E), dim=-1)
# Audio embedding: at the true offset (index t+K), reuse the mouth vector
# exactly (cosine similarity 1.0); everywhere else, independent random unit
# vectors (expected cosine similarity ~0 for random high-dim unit vectors).
audio_emb = F.normalize(torch.randn(B, W + 2 * K, E), dim=-1)
for t in range(W):
    audio_emb[:, K + t, :] = mouth_emb[:, t, :]
visual_mask = torch.ones(B, W, dtype=torch.bool)

profile = C.offset_similarity_profile(mouth_emb, audio_emb, visual_mask, K)
check(profile.shape == (B, 2 * K + 1), f"profile shape is (B, 2K+1) = {(B, 2*K+1)}, got {tuple(profile.shape)}")

peak_idx = profile.argmax(dim=-1)
check(bool((peak_idx == K).all()),
      f"profile peaks at index K={K} (offset 0) for every batch row, got argmax={peak_idx.tolist()}")
check(bool((profile[:, K] > 0.99).all()),
      f"peak value is ~1.0 (exact match by construction), got {profile[:, K].tolist()}")
other = torch.cat([profile[:, :K], profile[:, K + 1:]], dim=-1)
check(bool((profile[:, K:K + 1] > other + 0.3).all()),
      "peak clearly separated from every other offset (margin > 0.3)")

# Shuffling the audio embedding's time axis must NOT reproduce the same
# profile -- this is the direct contrast with pooled_ca's exact invariance.
perm = torch.randperm(audio_emb.shape[1])
profile_shuffled = C.offset_similarity_profile(mouth_emb, audio_emb[:, perm, :], visual_mask, K)
check(not torch.allclose(profile, profile_shuffled, atol=1e-4),
      "offset_similarity_profile is NOT invariant to shuffling the audio "
      "time axis (unlike pooled_ca's masked_mean_std(cross_attention(...)))")


# =====================================================================
# 3. Full offset_profile MODEL forward is not permutation-invariant either
#    (end-to-end, through the trained/random encoders, not just the math)
# =====================================================================
section("3. offset_profile: full model forward is not shift-invariant")

cfg2 = base_cfg(SYNC_ARCH="offset_profile")
m2 = build(cfg2).eval()
crops2, mfcc2, mask2 = rand_batch(cfg2)
sync_audio2 = rand_sync_audio(cfg2)

with torch.no_grad():
    v_tok2, m_tok2 = m2.encode_visual(crops2)
    outA = m2.forward_from_tokens(v_tok2, m_tok2, mfcc2, sync_audio2, mask2)
    perm2 = torch.randperm(sync_audio2.shape[-1])
    outB = m2.forward_from_tokens(v_tok2, m_tok2, mfcc2, sync_audio2[:, :, perm2], mask2)

check(outA["sync_logit"] is not None and outA["sync_profile"] is not None,
      "offset_profile produces both sync_logit and sync_profile")
check(outA["sync_profile"].shape == (crops2.shape[0], 2 * cfg2["SYNC_MAX_OFFSET"] + 1),
      "sync_profile has shape (B, 2*SYNC_MAX_OFFSET+1)")
check(not torch.allclose(outA["sync_logit"], outB["sync_logit"], atol=1e-5),
      "full offset_profile model's sync_logit CHANGES under a random "
      "permutation of the audio timesteps (max abs diff "
      f"{(outA['sync_logit'] - outB['sync_logit']).abs().max().item():.2e})")


# =====================================================================
# 4. Checkpoint round-trip: portable_state_dict / load_portable_state_dict,
#    for both SYNC_ARCH values, and the invalid combo raises as designed.
# =====================================================================
section("4. Checkpoint round-trip")

for arch in ("pooled_ca", "offset_profile"):
    cfg = base_cfg(SYNC_ARCH=arch)
    src = build(cfg).eval()
    dst = build(cfg).eval()   # freshly, independently initialised
    state = C.portable_state_dict(src)
    C.load_portable_state_dict(dst, state)

    crops_r, mfcc_r, mask_r = rand_batch(cfg)
    sa_r = rand_sync_audio(cfg)
    with torch.no_grad():
        vt_s, mt_s = src.encode_visual(crops_r)
        vt_d, mt_d = dst.encode_visual(crops_r)
        out_s = src.forward_from_tokens(vt_s, mt_s, mfcc_r, sa_r, mask_r)
        out_d = dst.forward_from_tokens(vt_d, mt_d, mfcc_r, sa_r, mask_r)
    check(torch.allclose(out_s["sync_logit"], out_d["sync_logit"], atol=1e-5),
          f"[{arch}] checkpoint round-trip reproduces identical sync_logit "
          "on a fresh model instance")
    check(torch.allclose(out_s["video_logit"], out_d["video_logit"], atol=1e-5),
          f"[{arch}] checkpoint round-trip reproduces identical video_logit "
          "(regression guard: sync change must not perturb the video head)")
    check(torch.allclose(out_s["audio_logit"], out_d["audio_logit"], atol=1e-5),
          f"[{arch}] checkpoint round-trip reproduces identical audio_logit")
    if out_s["fusion_logit"] is not None:
        check(torch.allclose(out_s["fusion_logit"], out_d["fusion_logit"], atol=1e-5),
              f"[{arch}] checkpoint round-trip reproduces identical fusion_logit")

# SYNC_DENSE_ALIGNED=False is only valid with SYNC_ARCH="pooled_ca" -- build
# it and confirm it still works (the A10_sync_unaligned ablation path).
cfg_unaligned = base_cfg(SYNC_ARCH="pooled_ca", SYNC_DENSE_ALIGNED=False)
m_unaligned = build(cfg_unaligned).eval()
crops_u, mfcc_u, mask_u = rand_batch(cfg_unaligned)
sa_u = rand_sync_audio(cfg_unaligned)
with torch.no_grad():
    vt_u, mt_u = m_unaligned.encode_visual(crops_u)
    out_u = m_unaligned.forward_from_tokens(vt_u, mt_u, mfcc_u, sa_u, mask_u)
check(out_u["sync_logit"] is not None,
      "SYNC_ARCH='pooled_ca' + SYNC_DENSE_ALIGNED=False (A10 ablation combo) still builds and runs")

# The invalid combo (offset_profile + unaligned) must raise, not silently
# misbehave -- checked at BOTH the model constructor and build_model, since
# the model can also be built by hand outside build_model.
raised = False
try:
    C.CrossFuseModelV5(sync_arch="offset_profile", sync_dense_aligned=False,
                       backbone="efficientnet_b4", freeze_blocks=4, pretrained=False)
except AssertionError:
    raised = True
check(raised, "CrossFuseModelV5(sync_arch='offset_profile', sync_dense_aligned=False) raises AssertionError")

raised = False
try:
    build(base_cfg(SYNC_ARCH="offset_profile", SYNC_DENSE_ALIGNED=False))
except ValueError:
    raised = True
check(raised, "build_model raises ValueError for the same invalid combo (caught before construction)")


# =====================================================================
# 5. Gradient reaches every sync-branch parameter (both archs) -- the same
#    check that ruled OUT "gradient isn't arriving" as an explanation last
#    round; re-run here so the new architecture doesn't reintroduce it.
# =====================================================================
section("5. Gradient flow through the sync branch")

for arch in ("pooled_ca", "offset_profile"):
    cfg = base_cfg(SYNC_ARCH=arch)
    m = build(cfg)
    m.train()
    crops_g, mfcc_g, mask_g = rand_batch(cfg, B=3)
    sa_g = rand_sync_audio(cfg, B=3)
    v_tok_g, m_tok_g = m.encode_visual(crops_g)
    out_g = m.forward_from_tokens(v_tok_g, m_tok_g, mfcc_g, sa_g, mask_g)
    loss = out_g["sync_logit"].sum()
    if arch == "offset_profile":
        loss = loss + F.cross_entropy(
            out_g["sync_profile"] / 0.1,
            torch.full((3,), cfg["SYNC_MAX_OFFSET"], dtype=torch.long))
    loss.backward()

    sync_params = {n: p for n, p in m.named_parameters()
                   if n.startswith(("mouth_encoder.", "mouth_proj.", "sync_"))
                   or n.startswith(("offset_audio_encoder.", "mel_encoder."))}
    check(len(sync_params) > 0, f"[{arch}] at least one sync-branch parameter found")
    missing = [n for n, p in sync_params.items() if p.grad is None or bool((p.grad == 0).all())]
    check(not missing, f"[{arch}] every sync-branch parameter received nonzero gradient "
                        f"(missing/zero: {missing})")


# =====================================================================
# 6. End-to-end through the REAL CrossFuseCropDataset code path, with a
#    synthetic on-disk cache fixture (no real video data needed) -- this is
#    the part most likely to have an off-by-one in the widened mel slice.
# =====================================================================
section("6. CrossFuseCropDataset: widened mel slice, real code path")

with tempfile.TemporaryDirectory() as tmp:
    T_cached, S, fps, sr = 16, 64, 25.0, 16000
    key_audio, key_silent = "clip_audio", "clip_silent"
    AF = base_cfg()["AUDIO_FRAMES"]   # must match the has_audio=0 zero-placeholder shape exactly,
                                      # or a DataLoader batching both rows fails to stack them --
                                      # a fixture bug, not a production one, but worth pinning down.

    def write_fixture(key, has_audio):
        crops = np.random.randint(0, 256, (T_cached, S, S, 3), dtype=np.uint8)
        np.save(os.path.join(tmp, key + ".npy"), crops)
        if has_audio:
            mel_len = 400  # generous: LOGMEL_FPS=100 => 4s, covers any slice bound used below
            np.savez(
                os.path.join(tmp, key + "_meta.npz"),
                mfcc_trim=np.random.randn(40, AF).astype(np.float32),
                mfcc_full=np.random.randn(40, AF).astype(np.float32),
                mel=np.random.randn(80, mel_len).astype(np.float32),
                valid=np.ones(T_cached, dtype=bool),
                timestamps=(np.arange(T_cached, dtype=np.float32) / fps),
            )

    write_fixture(key_audio, has_audio=True)
    write_fixture(key_silent, has_audio=False)

    rows = [
        {"key": key_audio, "fps": fps, "video_label": 0, "audio_label": 0,
         "identity_group": "g0", "has_audio": 1, "crop_dir": tmp},
        {"key": key_silent, "fps": fps, "video_label": 1, "audio_label": 0,
         "identity_group": "g1", "has_audio": 0, "crop_dir": tmp},
    ]

    for arch in ("pooled_ca", "offset_profile"):
        cfg = base_cfg(SYNC_ARCH=arch, WINDOW_FRAMES=6, N_MELS=80)
        expected_frames = (cfg["WINDOW_FRAMES"] + 2 * cfg["SYNC_MAX_OFFSET"]
                           if arch == "offset_profile" else cfg["SYNC_MEL_FRAMES"])
        for training in (False, True):
            ds = C.CrossFuseCropDataset(rows, [0, 1], tmp, cfg, training=training)
            for idx, expect_has_audio in ((0, 1), (1, 0)):
                item = ds[idx]
                window, mfcc, sync_a, sync_off = item[0], item[1], item[2], item[3]
                has_a = item[8]
                check(sync_a.shape == (cfg["N_MELS"], expected_frames),
                      f"[{arch} training={training}] row {idx} sync_a shape "
                      f"{tuple(sync_a.shape)} == (N_MELS, {expected_frames})")
                check(sync_off.shape == sync_a.shape,
                      f"[{arch} training={training}] row {idx} sync_off shape matches sync_a")
                check(int(has_a.item()) == expect_has_audio,
                      f"[{arch} training={training}] row {idx} has_audio flag correct")
                if expect_has_audio == 0:
                    check(bool((sync_a == 0).all()) and bool((sync_off == 0).all()),
                          f"[{arch} training={training}] silent row's sync tensors are exactly zero")

    # And the row's sync_a/sync_off must actually be usable by the model
    # (shape compatibility end to end), for the has_audio=1 row.
    for arch in ("pooled_ca", "offset_profile"):
        cfg = base_cfg(SYNC_ARCH=arch, WINDOW_FRAMES=6, N_MELS=80)
        ds = C.CrossFuseCropDataset(rows, [0], tmp, cfg, training=False)
        window, mfcc, sync_a, sync_off, vlab, alab, wv, gid, has_a = ds[0]
        m = build(cfg).eval()
        with torch.no_grad():
            v_tok_e, m_tok_e = m.encode_visual(window.unsqueeze(0))
            out_e = m.forward_from_tokens(v_tok_e, m_tok_e, mfcc.unsqueeze(0),
                                          sync_a.unsqueeze(0), wv.unsqueeze(0))
        check(out_e["sync_logit"] is not None and out_e["sync_logit"].shape == (1,),
              f"[{arch}] dataset output feeds the model end-to-end and produces a scalar sync_logit")

    # =====================================================================
    # 7. evaluate_deterministic (Kaggle's real per-epoch eval call) runs
    #    without crashing, and the training loop's sync loss block --
    #    replicated exactly as in run_training_pipeline, BCE + InfoNCE --
    #    produces a finite loss with gradient for both archs.
    # =====================================================================
    section("7. evaluate_deterministic + training-loop sync loss, both archs")

    from torch.utils.data import DataLoader

    for arch in ("pooled_ca", "offset_profile"):
        cfg = base_cfg(SYNC_ARCH=arch, WINDOW_FRAMES=6, N_MELS=80)
        ds = C.CrossFuseCropDataset(rows, [0, 1], tmp, cfg, training=False)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        m = build(cfg)

        result = C.evaluate_deterministic(m, loader, cfg)
        check("sync_auc" in result and "video_auc" in result,
              f"[{arch}] evaluate_deterministic returns video_auc/sync_auc keys without crashing")
        check(not np.isnan(result["video_auc"]),
              f"[{arch}] video_auc is a real number (got {result['video_auc']:.4f})")

        # run_training_pipeline's sync loss block (BCE matched-vs-shifted,
        # plus InfoNCE for offset_profile), replicated verbatim.
        crops_t, mfcc_t, sync_a_t, sync_off_t, vlab_t, alab_t, mask_t, gid_t, has_a_t = next(iter(loader))
        m.train()
        v_tok_t, m_tok_t = m.encode_visual(crops_t)
        out_t = m.forward_from_tokens(v_tok_t, m_tok_t, mfcc_t, sync_a_t, visual_mask=mask_t)
        B_t = crops_t.size(0)
        perm_t = torch.randperm(B_t)
        ok_t = gid_t[perm_t] != gid_t
        neg_a_t = torch.where(ok_t.view(-1, 1, 1), sync_a_t[perm_t], sync_off_t)
        out_neg_t = m.forward_from_tokens(v_tok_t, m_tok_t, mfcc_t, neg_a_t, visual_mask=mask_t)
        s_logit_t = torch.cat([out_t["sync_logit"], out_neg_t["sync_logit"]], 0)
        s_lab_t = torch.cat([torch.zeros(B_t), torch.ones(B_t)], 0)
        s_w_t = torch.cat([has_a_t, has_a_t], 0)
        ls_t = F.binary_cross_entropy_with_logits(s_logit_t, s_lab_t, reduction="none")
        loss_t = cfg["LAMBDA_SYNC"] * ((ls_t * s_w_t).sum() / s_w_t.sum().clamp(min=1.0))
        if arch == "offset_profile" and out_t["sync_profile"] is not None:
            Kt = cfg["SYNC_MAX_OFFSET"]
            target_t = torch.full((B_t,), Kt, dtype=torch.long)
            n_a_t = has_a_t.sum().clamp(min=1.0)
            nce_t = F.cross_entropy(out_t["sync_profile"] / cfg["SYNC_NCE_TEMP"],
                                    target_t, reduction="none")
            loss_t = loss_t + cfg["LAMBDA_SYNC_NCE"] * ((nce_t * has_a_t).sum() / n_a_t)
        loss_t.backward()
        check(torch.isfinite(loss_t).item(),
              f"[{arch}] training-loop sync loss block (BCE"
              f"{' + InfoNCE' if arch == 'offset_profile' else ''}) runs and "
              f"produces a finite loss ({loss_t.item():.4f})")


print(f"\n{'=' * 70}\nALL {CHECKS['n']} CHECKS PASSED\n{'=' * 70}")
