"""Paired examples must stay in the same optimization mini-batch."""
import pytest

from crossfuse_v5 import PairedBatchSampler
from ffpp_v5 import PairedFrameSampler


@pytest.mark.parametrize("sampler_cls", [PairedBatchSampler, PairedFrameSampler])
def test_pairs_stay_together_when_unpaired_examples_are_present(sampler_cls):
    rows = []
    for group in range(6):
        for label in (0, 1):
            rows.append({"identity_group": str(group), "target_id": str(group),
                         "video_label": label, "label": label})
    for group in range(6, 13):
        rows.append({"identity_group": str(group), "target_id": str(group),
                     "video_label": 1, "label": 1})
    sampler = sampler_cls(rows, range(len(rows)), batch_size=4, drop_last=False)
    for epoch in range(4):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert len(batches) == len(sampler)
        assert sorted(i for batch in batches for i in batch) == list(range(len(rows)))
        for real, fake in sampler.pairs:
            assert any(real in batch and fake in batch for batch in batches)


@pytest.mark.parametrize("sampler_cls", [PairedBatchSampler, PairedFrameSampler])
def test_odd_paired_batch_size_is_rejected(sampler_cls):
    with pytest.raises(ValueError, match="even"):
        sampler_cls([], [], batch_size=3)
