## Summary

Adds **VSArena** as a remote `StepBenchmark` adapter — the first benchmark in vla-eval that connects to a **hosted WebSocket harness** instead of stepping a local Docker sim.

- **Task:** 3-cube block stacking (4-DOF arm, Rapier 60 Hz on VSArena server)
- **VLA track:** 128×128 RGB + language instruction (no cube GPS to policy)
- **Smoke path:** `dry_run=True` — offline mock, no API key, works with `vla-eval test`
- **Live path:** connects to `wss://vsarena-harness.onrender.com` with `VSARENA_API_KEY`

[VSArena](https://vsarena.vercel.app) is an open browser arena with public ELO (MIT). This PR follows CONTRIBUTING.md and the add-benchmark skill.

## Why remote?

VSArena's thesis is **zero-install public comparison** (LMArena-style for spatial policies). Physics + anti-cheat ingest live on the harness; vla-eval gets a thin client Docker image (~base size, no MuJoCo/SAPIEN).

## Test plan

- [x] `pytest tests/test_vsarena_benchmark.py`
- [ ] `vla-eval test -c configs/benchmarks/vsarena/smoke_test.yaml` (after Docker image build)
- [ ] Live smoke with `VSARENA_API_KEY` + model server

## Links

- Demo: https://vsarena.vercel.app/simulation
- Protocol: https://vsarena.vercel.app/docs
- VSArena repo: https://github.com/NovaCoding-G/VSArena
