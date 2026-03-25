# Configuration Guide

This directory contains two types of YAML configs consumed by different CLI commands.

## Benchmark Configs (`configs/*.yaml`) — `vla-eval run`

Define **what to evaluate**: server URL, Docker image, and benchmark parameters.

```bash
# Basic usage (server at localhost:8000)
vla-eval run -c configs/libero_all.yaml

# Override server URL (no need to edit the YAML)
vla-eval run -c configs/libero_all.yaml --server-url ws://my-host:8000

# Override benchmark params (applied to all benchmark entries)
vla-eval run -c configs/libero_all.yaml \
  --server-url ws://my-host:8000 \
  --param send_wrist_image=true \
  --param send_state=true
```

Schema: `{server, docker, output_dir, benchmarks[]}`

## Model Server Configs (`configs/model_servers/`) — `vla-eval serve`

Define **how to launch a model server**: which script to run and with what arguments.

```bash
vla-eval serve -c configs/model_servers/pi0/libero.yaml
```

Schema: `{script, args, [extends]}`

## Automatic Observation Negotiation

Model servers declare their observation requirements (e.g. wrist camera, proprioceptive
state) via the WebSocket HELLO handshake. The orchestrator reads these and auto-configures
the benchmark — **no manual `--param` flags needed** for most models.

How it works:
1. `vla-eval serve` starts the model server with its config
2. `vla-eval run` connects and receives `observation_params` in the HELLO response
3. The orchestrator merges them into benchmark params before creating the benchmark

Priority (highest wins):
1. `--param` CLI flags — explicit user override
2. Benchmark config YAML `params:` — explicit in the config file
3. Model server `observation_params` — auto-detected from server
4. Benchmark `__init__` defaults — fallback

This means you can run most evaluations with just:

```bash
# No --param needed — model server tells the benchmark what it needs
vla-eval serve -c configs/model_servers/pi0/libero.yaml &
vla-eval run -c configs/libero_all.yaml --server-url ws://localhost:8000
```

The `--param` flag is still available for manual overrides or experimentation.

## Benchmark Params Quick Reference

| Param | Default | Description |
|-------|---------|-------------|
| `send_wrist_image` | `false` | Include wrist camera image |
| `send_state` | `false` | Include proprioceptive state vector |
| `absolute_action` | `false` | Use absolute EE pose (not deltas) |
| `num_steps_wait` | `10` | Idle steps before evaluation starts |
| `seed` | `0` | Random seed for environment reset |

### Auto-configured params by model (LIBERO)

These are automatically set via HELLO negotiation — listed here for reference.

| Model | `send_wrist_image` | `send_state` | `absolute_action` |
|-------|:------------------:|:------------:|:-----------------:|
| OpenVLA | | | |
| CogACT | | | |
| OpenVLA-OFT | `true` | `true` | |
| Pi0 / Pi0.5 | `true` | `true` | |
| GR00T | `true` | `true` | |
| DB-CogACT | `true` | `true` | |
| StarVLA | `true` | | |
| X-VLA | `true` | `true` | `true` |

Empty cells = use default (`false`).
