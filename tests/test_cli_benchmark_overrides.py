"""`vla-eval run --param` / `--benchmark-field` applied to every benchmark entry."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from vla_eval.cli import main as cli
from vla_eval.config import EvalConfig, merge_benchmark_overrides


def _config(**entry: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"benchmark": "pkg.mod:Bench", "params": {"seed": 1}}
    base.update(entry)
    return {"benchmarks": [base, {"benchmark": "pkg.mod:Other"}]}


def _apply(config: dict[str, Any], *, param=None, field=None) -> list[dict[str, Any]]:
    cli._apply_benchmark_overrides(config, param, field)
    return config["benchmarks"]


def test_benchmark_field_sets_entry_level_keys_on_every_entry() -> None:
    entries = _apply(_config(), field=["max_steps=200", "episodes_per_task=5"])

    for entry in entries:
        assert entry["max_steps"] == 200
        assert entry["episodes_per_task"] == 5


def test_benchmark_field_values_are_yaml_typed_not_strings() -> None:
    entries = _apply(_config(), field=["max_steps=200", "throughput_mode=true", "subname=transport", "hz=20.5"])

    assert entries[0]["max_steps"] == 200
    assert entries[0]["throughput_mode"] is True
    assert entries[0]["subname"] == "transport"
    assert entries[0]["hz"] == 20.5


def test_benchmark_field_replaces_an_existing_entry_value() -> None:
    entries = _apply(_config(max_steps=50, subname="old"), field=["max_steps=200", "subname=new"])

    assert entries[0]["max_steps"] == 200
    assert entries[0]["subname"] == "new"


def test_benchmark_field_dotted_params_key_merges_like_param() -> None:
    entries = _apply(_config(), field=["params.chunk=4"])

    assert entries[0]["params"] == {"seed": 1, "chunk": 4}
    assert entries[1]["params"] == {"chunk": 4}


def test_param_and_benchmark_field_params_are_equivalent() -> None:
    via_param = _apply(_config(), param=["chunk=4"])
    via_field = _apply(_config(), field=["params.chunk=4"])

    assert via_param == via_field


def test_param_still_merges_into_params_and_leaves_entry_keys_alone() -> None:
    entries = _apply(_config(max_steps=50), param=["chunk=4"])

    assert entries[0]["params"] == {"seed": 1, "chunk": 4}
    assert entries[0]["max_steps"] == 50
    assert "chunk" not in entries[0]


def test_benchmark_field_applies_after_param_on_a_shared_key() -> None:
    entries = _apply(_config(), param=["chunk=4"], field=["params.chunk=9"])

    assert entries[0]["params"]["chunk"] == 9


def test_no_overrides_leaves_the_config_untouched() -> None:
    config = _config(max_steps=50)
    before = {"benchmarks": [dict(b) for b in config["benchmarks"]]}

    cli._apply_benchmark_overrides(config, None, None)

    assert config == before


def test_overridden_entry_still_builds_an_eval_config() -> None:
    entries = _apply(_config(), field=["max_steps=200", "subname=transport", "params.chunk=4"])

    cfg = EvalConfig.from_dict(entries[0])

    assert cfg.max_steps == 200
    assert cfg.subname == "transport"
    assert cfg.params == {"seed": 1, "chunk": 4}
    assert cfg.resolved_name() == "Bench_transport"


def test_unknown_entry_key_is_kept_for_other_consumers() -> None:
    """The entry schema is open: `vla-eval test` reads action_dim off benchmark entries."""
    entries = _apply(_config(), field=["action_dim=14"])

    assert entries[0]["action_dim"] == 14
    assert EvalConfig.from_dict(entries[0]).benchmark == "pkg.mod:Bench"


def test_overrides_reach_omegaconf_dictconfig_entries() -> None:
    """vla_eval.run() accepts a Mapping config; dict(DictConfig) keeps DictConfig entries."""
    from omegaconf import OmegaConf

    conf = OmegaConf.create({"benchmarks": [{"benchmark": "pkg.mod:Bench", "params": {"seed": 1}}]})
    cfg: dict[str, Any] = {"benchmarks": conf["benchmarks"]}

    merge_benchmark_overrides(cfg, {"max_steps": 200, "params": {"chunk": 4}})

    entry = cfg["benchmarks"][0]
    assert entry["max_steps"] == 200
    assert entry["params"] == {"seed": 1, "chunk": 4}


def test_override_without_an_equals_sign_is_rejected() -> None:
    """Bare `max_steps` would otherwise parse to None and read as "use the default"."""
    with pytest.raises(ValueError, match="must be KEY=VALUE"):
        cli._dotlist_to_dict(["max_steps"])


def test_override_with_an_empty_value_is_an_explicit_null() -> None:
    assert cli._dotlist_to_dict(["max_steps="]) == {"max_steps": None}


@pytest.mark.parametrize("bad", ["tasks=[", "x=%"])
def test_malformed_yaml_value_raises_value_error_not_a_yaml_error(bad: str) -> None:
    """cmd_run turns ValueError into a clean exit; a raw yaml error would traceback."""
    with pytest.raises(ValueError, match="could not parse override"):
        cli._dotlist_to_dict([bad])


def test_cmd_run_exits_cleanly_on_a_malformed_override(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_load_config", lambda _path: _config())
    args = argparse.Namespace(
        config="unused.yaml",
        server_url=None,
        output_dir=None,
        param=None,
        benchmark_field=["tasks=["],
        shard_id=None,
        num_shards=None,
        eval_id=None,
        no_save=False,
        record_video=None,
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_run(args)

    assert exc.value.code == 1
    assert "could not parse override" in capsys.readouterr().err


def test_non_mapping_benchmark_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"benchmarks\[1\] must be a mapping"):
        merge_benchmark_overrides({"benchmarks": [{"benchmark": "a"}, "oops"]}, {"max_steps": 1})


def test_non_mapping_params_block_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"benchmarks\[0\].params must be a mapping"):
        merge_benchmark_overrides({"benchmarks": [{"params": "oops"}]}, {"params": {"seed": 1}})


def test_missing_benchmarks_key_is_a_no_op() -> None:
    config: dict[str, Any] = {"benchmarks": None}

    merge_benchmark_overrides(config, {"max_steps": 1})

    assert config == {"benchmarks": None}


def test_run_parser_collects_repeated_benchmark_field_flags(monkeypatch) -> None:
    """--benchmark-field reaches cmd_run as a list, in the order given."""
    monkeypatch.setattr(
        "sys.argv",
        ["vla-eval", "run", "-c", "eval.yaml", "--benchmark-field", "max_steps=200", "--benchmark-field", "hz=20"],
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args: captured.update(vars(args)))

    cli.main()

    assert captured["benchmark_field"] == ["max_steps=200", "hz=20"]
    assert captured["param"] is None
