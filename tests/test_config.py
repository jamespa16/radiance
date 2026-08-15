"""Config pickle forward-compatibility: a checkpoint is a pickled Config, and pickle restores
__dict__ without running __init__, so fields added after the checkpoint was saved (sft/dpo,
and any future ones) are absent from the restored instance's __dict__ — default_factory only
fires inside __init__, so they would simply not exist. Config.__setstate__ backfills missing
fields with their current defaults instead: a missing field means what an omitted key in a
YAML config means.
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass

import pytest

from radiance.config import _backfill_missing_fields


def _old_shape_config(cfg):
    """Simulate a checkpoint pickled before sft/dpo existed (and before a nested field):
    build the exact state dict pickle would hand to __setstate__, with those attrs absent."""
    data = copy.deepcopy(cfg.data)
    del data.disk_cache_max_gb  # a nested field with a known default (None)
    state = dict(cfg.__dict__)
    state["data"] = data
    del state["sft"], state["dpo"]

    restored = object.__new__(type(cfg))
    restored.__setstate__(state)
    return restored


def test_unpickled_config_backfills_missing_fields_with_defaults(tiny_cfg):
    cfg = tiny_cfg()
    restored = _old_shape_config(cfg)

    assert restored.sft.enabled is False
    assert restored.dpo.enabled is False
    assert restored.data.disk_cache_max_gb is None
    # Fully backfilled == the config that was pickled: nothing else was lost.
    assert restored == cfg


def test_backfill_raises_on_field_without_default():
    # Every current Config field has a default, so the no-default branch only fires for
    # future required fields; pin it with a synthetic schema where it is reachable.
    @dataclass
    class Shaped:
        required: int
        optional: int = 5

    restored = object.__new__(Shaped)
    restored.__dict__.update({"optional": 9})  # `required` predates the schema
    with pytest.raises(ValueError, match="required"):
        _backfill_missing_fields(restored)



def test_config_pickle_roundtrip_is_unchanged(tiny_cfg):
    # The happy path (all fields present) goes through the same __setstate__ and must be a
    # no-op: backfilling nothing leaves the config bit-identical.
    assert pickle.loads(pickle.dumps(tiny_cfg())) == tiny_cfg()
