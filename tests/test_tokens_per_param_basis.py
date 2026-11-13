"""train.tokens_per_param_basis: validated on every construction path; dense-MTP differs, plain dense doesn't."""

import pytest

from radiance.config import Config, ModelConfig, TrainConfig, load_config_with_overrides
from radiance.pretokenize import run_tokens
from radiance.size import size_config

_DENSE = ["train.tokens_per_param=20", "model.d_model=128", "model.head_dim=32", "model.n_layers=3"]


def test_direct_construction_validates():
    with pytest.raises(ValueError, match="tokens_per_param_basis"):
        TrainConfig(tokens_per_param_basis="unique")
    with pytest.raises(ValueError, match="tokens_per_param_basis"):
        Config(train=TrainConfig(tokens_per_param_basis="bogus"))


def test_case_variant_rejected():
    with pytest.raises(ValueError, match="tokens_per_param_basis"):
        TrainConfig(tokens_per_param_basis="Active")
    with pytest.raises(ValueError, match="tokens_per_param_basis"):
        load_config_with_overrides(None, ["train.tokens_per_param_basis=Active"])


def test_valid_values_accepted():
    assert TrainConfig(tokens_per_param_basis="active").tokens_per_param_basis == "active"
    assert TrainConfig().tokens_per_param_basis == "total"


def test_dense_mtp_differs_between_bases():
    base = _DENSE + ["model.mtp_heads=2"]
    total = load_config_with_overrides(None, base)
    active = load_config_with_overrides(None, base + ["train.tokens_per_param_basis=active"])
    sizes = size_config(total, 1000)
    assert sizes["active"] < sizes["total"]
    assert run_tokens(active, 1000) < run_tokens(total, 1000)


def test_plain_dense_identical():
    total = load_config_with_overrides(None, _DENSE)
    active = load_config_with_overrides(None, _DENSE + ["train.tokens_per_param_basis=active"])
    sizes = size_config(total, 1000)
    assert sizes["active"] == sizes["total"]
    assert run_tokens(active, 1000) == run_tokens(total, 1000)
