# Reproduction Reports

Model-centric evaluation reports. Each document covers one model's reproduction results across all benchmarks tested with that model.

## Reports

| Model | Benchmarks | Verdict | Report |
|-------|-----------|---------|--------|
| DB-CogACT (CogACT 7B) | LIBERO, CALVIN, SimplerEnv | All reproduced within expected variance | [db-cogact.md](db-cogact.md) |

## Benchmark Determinism Reference

Which benchmarks produce deterministic results when given the same seed? (sync mode only — async/realtime is inherently non-deterministic due to timing.)

| Benchmark | Seed Support | Deterministic | Notes |
|-----------|-------------|---------------|-------|
| LIBERO | `seed` (env.seed) | Yes | `episode_idx` selects initial state |
| CALVIN | `seed` (pl.seed_everything) | Yes | FNV hash for scene layout |
| SimplerEnv | `seed` (optional) | Partial | Non-deterministic when `seed: null` |
| RoboCasa | `seed` (optional) | Partial | Non-deterministic when `seed: null` |
| RoboTwin | `seed` | Yes | `100000*(1+seed)` offset per task |
| Kinetix | `seed` | Yes | JAX PRNG per episode |
| RoboCerebra | `seed` (env.seed) | Yes | gymnasium seed |
| ManiSkill2 | None | No | No simulator seed API |
| RLBench | None | No | CoppeliaSim internal RNG |
| VLABench | None | No | dm_control internal RNG |
| MIKASA | None | No | ManiSkill3 default seeding |

## Adding a New Report

Copy **[_TEMPLATE.md](_TEMPLATE.md)** and fill in the sections. Name the file after the model (e.g. `openvla.md`).

## Blocked Benchmarks

Some benchmarks cannot be integrated due to external blockers. See [BLOCKED.md](BLOCKED.md).
