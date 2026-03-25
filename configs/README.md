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

Model server configs that require non-default benchmark params document them
in a `# Required benchmark params (--param):` comment at the top.

## Benchmark Params Quick Reference

Different models require different observation inputs. Pass these via `--param`
when running evaluation:

| Param | Default | Description |
|-------|---------|-------------|
| `send_wrist_image` | `false` | Include wrist camera image |
| `send_state` | `false` | Include proprioceptive state vector |
| `absolute_action` | `false` | Use absolute EE pose (not deltas) |
| `num_steps_wait` | `10` | Idle steps before evaluation starts |
| `seed` | `0` | Random seed for environment reset |

### Required params by model (LIBERO)

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

Empty cells = use default (`false`). Check each model server config for details.
