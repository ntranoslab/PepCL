from itertools import islice

import pandas as pd
import pytest

from pepcl.pepcl import PepCL, PepCL_Micro


def _classification_sampling_df():
    return pd.DataFrame(
        {
            "domain_id": [0, 0, 0, 0, 0, 0, 1, 1],
            "label":     [1, 1, 0, 0, 1, 0, 1, 0],
            "allele":    ["A0201", "A0201", "A0201", "A0201", "B0702", "B0702", "A0201", "A0201"],
        }
    )


def test_infinite_allele_sampler_samples_positive_and_negative_rows():
    df = _classification_sampling_df()

    sampler = PepCL._InfiniteAlleleSampler(
        df=df,
        domain_id=0,
        n_pos=1,
        neg_ratio=1.0,
        seed=123,
    )

    sampled_idx = list(islice(iter(sampler), 4))
    sampled = df.loc[sampled_idx]

    assert len(sampled) == 4
    assert int((sampled["label"] == 1).sum()) == 2
    assert int((sampled["label"] == 0).sum()) == 2
    assert set(sampled["domain_id"]) == {0}


def test_infinite_allele_sampler_respects_zero_negative_ratio():
    df = _classification_sampling_df()

    sampler = PepCL._InfiniteAlleleSampler(
        df=df,
        domain_id=0,
        n_pos=1,
        neg_ratio=0.0,
        seed=123,
    )

    sampled_idx = list(islice(iter(sampler), 2))
    sampled = df.loc[sampled_idx]

    assert len(sampled) == 2
    assert set(sampled["label"]) == {1}
    assert set(sampled["domain_id"]) == {0}


def test_infinite_allele_sampler_respects_q_pos_by_allele_filtering():
    df = _classification_sampling_df()

    sampler = PepCL._InfiniteAlleleSampler(
        df=df,
        domain_id=0,
        n_pos=10,
        neg_ratio=1.0,
        seed=123,
        q_pos_by_allele={"A0201": 1, "B0702": 0},
    )

    sampled_idx = list(islice(iter(sampler), 2))
    sampled = df.loc[sampled_idx]

    assert len(sampled) == 2
    assert set(sampled["allele"]) == {"A0201"}
    assert int((sampled["label"] == 1).sum()) == 1
    assert int((sampled["label"] == 0).sum()) == 1


def test_infinite_allele_sampler_empty_domain_yields_no_indices():
    df = _classification_sampling_df()

    sampler = PepCL._InfiniteAlleleSampler(
        df=df,
        domain_id=99,
        n_pos=1,
        neg_ratio=1.0,
        seed=123,
    )

    sampled_idx = list(islice(iter(sampler), 5))

    assert sampled_idx == []


def test_infinite_allele_sampler_rejects_negative_neg_ratio():
    df = _classification_sampling_df()

    with pytest.raises(ValueError, match="neg_ratio must be >= 0"):
        PepCL._InfiniteAlleleSampler(
            df=df,
            domain_id=0,
            n_pos=1,
            neg_ratio=-1.0,
            seed=123,
        )


def test_pooled_ft_sampler_samples_global_positive_and_negative_rows():
    df = pd.DataFrame(
        {
            "domain_id": [1, 1, 1, 1, 1, 1],
            "label":     [1, 1, 1, 0, 0, 0],
            "allele":    ["A0201", "A0201", "B0702", "A0201", "B0702", "B0702"],
        }
    )

    sampler = PepCL_Micro._InfinitePooledFTSampler(
        df=df,
        domain_id=1,
        pooled_num_pos=2,
        neg_ratio=1.0,
        batch_size=4,
        seed=123,
    )

    sampled_idx = list(islice(iter(sampler), 4))
    sampled = df.loc[sampled_idx]

    assert len(sampled) == 4
    assert int((sampled["label"] == 1).sum()) == 2
    assert int((sampled["label"] == 0).sum()) == 2
    assert set(sampled["domain_id"]) == {1}


def test_pooled_ft_sampler_empty_domain_yields_no_indices():
    df = pd.DataFrame(
        {
            "domain_id": [0, 0],
            "label": [1, 0],
            "allele": ["A0201", "A0201"],
        }
    )

    sampler = PepCL_Micro._InfinitePooledFTSampler(
        df=df,
        domain_id=1,
        pooled_num_pos=2,
        neg_ratio=1.0,
        batch_size=4,
        seed=123,
    )

    sampled_idx = list(islice(iter(sampler), 5))

    assert sampled_idx == []


def test_pooled_ft_sampler_rejects_negative_neg_ratio():
    df = pd.DataFrame(
        {
            "domain_id": [1, 1],
            "label": [1, 0],
            "allele": ["A0201", "A0201"],
        }
    )

    with pytest.raises(ValueError, match="neg_ratio must be >= 0"):
        PepCL_Micro._InfinitePooledFTSampler(
            df=df,
            domain_id=1,
            pooled_num_pos=1,
            neg_ratio=-1.0,
            batch_size=2,
            seed=123,
        )