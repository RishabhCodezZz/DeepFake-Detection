"""Identity-disjoint splitting is the main honesty guarantee of the project."""
import numpy as np
import pytest

from crossfuse_v5 import (
    build_3way_split_packed, build_identity_groups, get_identity_tokens,
)


def make_rows(n_groups=30, per_group=6, ffpp=0):
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            rows.append({"identity_group": f"id{g:05d}", "video_label": k % 2,
                         "audio_label": (k // 2) % 2})
    for k in range(ffpp):
        rows.append({"identity_group": f"ffpp{k}", "video_label": k % 2,
                     "audio_label": 0, "train_only": True})
    return rows


def groups_of(rows, idx):
    return {rows[i]["identity_group"] for i in idx}


def test_identity_tokens_come_from_folder_and_filename():
    toks = get_identity_tokens("/d/FakeVideo/Asian/men/id00076/id00076_id00812_wavtolip.mp4")
    assert toks == {"id00076", "id00812"}


def test_source_and_target_identities_land_in_one_group():
    paths = ["/d/a/id00001/id00001_id00002_x.mp4",
             "/d/b/id00002/id00002_id00003_y.mp4",
             "/d/c/id00009/id00009_z.mp4"]
    g = build_identity_groups(paths)
    assert g[paths[0]] == g[paths[1]]   # chained through id00002
    assert g[paths[2]] != g[paths[0]]


def test_no_identity_group_crosses_partitions():
    rows = make_rows()
    tr, va, te = build_3way_split_packed(rows, seed=0, verbose=False)
    assert not groups_of(rows, tr) & groups_of(rows, va)
    assert not groups_of(rows, tr) & groups_of(rows, te)
    assert not groups_of(rows, va) & groups_of(rows, te)


def test_every_row_lands_in_exactly_one_partition():
    rows = make_rows()
    tr, va, te = build_3way_split_packed(rows, seed=0, verbose=False)
    all_idx = np.concatenate([tr, va, te])
    assert sorted(all_idx.tolist()) == list(range(len(rows)))


def test_split_fractions_are_close_to_the_target_by_sample_count():
    rows = make_rows(n_groups=60)
    tr, va, te = build_3way_split_packed(rows, seed=0, verbose=False)
    n = len(rows)
    assert len(tr) / n == pytest.approx(0.60, abs=0.05)
    assert len(va) / n == pytest.approx(0.20, abs=0.05)
    assert len(te) / n == pytest.approx(0.20, abs=0.05)


def test_train_only_rows_are_pinned_to_train_and_never_reach_val_or_test():
    rows = make_rows(ffpp=50)
    tr, va, te = build_3way_split_packed(rows, seed=0, verbose=False)
    pinned = {i for i, r in enumerate(rows) if r.get("train_only")}
    assert pinned <= set(tr.tolist())
    assert not pinned & set(va.tolist())
    assert not pinned & set(te.tolist())


def test_val_and_test_are_identical_with_or_without_ffpp_rows():
    """C-eval rebuilds the split from the FakeAVCeleb-only manifest and relies on this."""
    base = make_rows()
    _, va0, te0 = build_3way_split_packed(base, seed=42, verbose=False)
    _, va1, te1 = build_3way_split_packed(base + make_rows(n_groups=0, ffpp=40), seed=42,
                                          verbose=False)
    assert va0.tolist() == va1.tolist()
    assert te0.tolist() == te1.tolist()


def test_pinned_row_cannot_reintroduce_a_held_out_identity():
    base = make_rows()
    _, val, _ = build_3way_split_packed(base, seed=42, verbose=False)
    pinned = dict(base[int(val[0])], train_only=True)
    with pytest.raises(AssertionError, match="Pinned identity"):
        build_3way_split_packed(base + [pinned], seed=42, verbose=False)


def test_split_is_deterministic_for_a_seed_and_changes_with_it():
    rows = make_rows()
    a = build_3way_split_packed(rows, seed=1, verbose=False)
    b = build_3way_split_packed(rows, seed=1, verbose=False)
    c = build_3way_split_packed(rows, seed=2, verbose=False)
    assert all(x.tolist() == y.tolist() for x, y in zip(a, b))
    assert any(x.tolist() != y.tolist() for x, y in zip(a, c))


def test_naive_split_leaks_identities_which_is_what_the_ablation_measures():
    rows = make_rows()
    tr, va, te = build_3way_split_packed(rows, identity_disjoint=False, seed=0, verbose=False)
    assert groups_of(rows, tr) & groups_of(rows, te)
