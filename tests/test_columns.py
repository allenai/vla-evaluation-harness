"""Column batches: nested dicts of arrays (first axis the row) or lists."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval import columns


def _batch() -> dict:
    return {
        "state": np.arange(6.0).reshape(3, 2),
        "images": {"head": np.zeros((3, 4, 4, 3), np.uint8), "wrist": np.ones((3, 2, 2, 3), np.uint8)},
        "task": ["a", "b", "c"],
    }


def test_take_concat_split_stack_round_trip():
    b = _batch()
    assert columns.num_rows(b) == 3
    part = columns.take(b, [2, 0])
    assert part["state"].tolist() == [[4.0, 5.0], [0.0, 1.0]] and part["task"] == ["c", "a"]
    assert part["images"]["head"].shape == (2, 4, 4, 3)
    joined = columns.concat([columns.take(b, [0]), columns.take(b, [1, 2])])
    assert columns.num_rows(joined) == 3 and joined["task"] == ["a", "b", "c"]
    rows = columns.split(b, 3)
    assert rows[1]["images"]["wrist"].shape == (2, 2, 3) and rows[1]["task"] == "b"
    back = columns.stack(rows)
    assert np.array_equal(back["state"], b["state"]) and back["images"]["head"].shape == (3, 4, 4, 3)


def test_values_that_do_not_stack_stay_lists():
    stacked = columns.stack([{"a": np.ones((1, 7))}, {"a": np.ones((2, 7))}, {"a": np.ones((2, 7))}])
    assert isinstance(stacked["a"], list) and [x.shape for x in stacked["a"]] == [(1, 7), (2, 7), (2, 7)]
    assert columns.stack([{"x": 1.0}, {"x": 2.0}]) == {"x": [1.0, 2.0]}
    joined = columns.concat([{"a": np.ones((1, 1, 7))}, {"a": np.ones((1, 2, 7))}])
    assert [x.shape for x in joined["a"]] == [(1, 7), (2, 7)]


def test_batches_with_different_keys_do_not_concat():
    with pytest.raises(ValueError):
        columns.concat([{"a": np.zeros(1)}, {"a": np.zeros(1), "episode_restart": [True]}])


def test_columns_of_different_lengths_are_rejected():
    with pytest.raises(ValueError):
        columns.num_rows({"a": np.zeros(2), "b": [1, 2, 3]})
