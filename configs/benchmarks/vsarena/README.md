# VSArena benchmark

Remote block-stacking eval via the [VSArena](https://vsarena.vercel.app) hosted harness.

Unlike MuJoCo/SAPIEN benchmarks in this repo, physics runs on VSArena's WebSocket server (Rapier 60 Hz). The Docker image is a thin client — no GPU sim required.

## Quick start

### Smoke test (offline, no API key)

```bash
vla-eval test -c configs/benchmarks/vsarena/smoke_test.yaml
```

Uses `dry_run: true` — mock states only, validates the StepBenchmark adapter + EchoModelServer loop.

### Live eval

1. Get an API key from [vsarena.vercel.app/account](https://vsarena.vercel.app/account)
2. Export `VSARENA_API_KEY`
3. Start a model server (e.g. LeRobot / OpenVLA config)
4. Run:

```bash
export VSARENA_API_KEY=...
vla-eval serve --config configs/model_servers/lerobot/pi05_libero.yaml
vla-eval run --config configs/benchmarks/vsarena/eval.yaml
```

Optional: `VSARENA_AGENT_NAME` sets the public leaderboard label.

## Observation format

- **VLA mode** (default): 128×128 RGB `scene` camera + language instruction
- Proprio: 4 joint angles forwarded as `proprio`

## Action format

Default: 7-D LIBERO-style vector `[dx, dy, dz, dax, day, daz, grip]` mapped to VSArena `ee_delta` + gripper.

Advanced: pass `vsarena_action` dict with `joint_targets` / `ee_delta` + `gripper_state`.

## Links

- Live demo: https://vsarena.vercel.app/simulation
- Protocol: https://vsarena.vercel.app/docs
- VSArena repo: https://github.com/NovaCoding-G/VSArena (MIT)

## Status

◇ Integrated — awaiting first reproduction report on live harness.
