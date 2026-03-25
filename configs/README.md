# configs/

Two config schemas, consumed by different CLI commands.

## `configs/*.yaml` — Evaluation configs

Used by `vla-eval run`. Define server URL, Docker image, and benchmark parameters.

Schema: `{server, docker, output_dir, benchmarks[]}`

## `configs/model_servers/` — Model server configs

Used by `vla-eval serve`. Define which script to run and with what arguments.

Schema: `{script, args, [extends]}`

Model servers declare their observation requirements (e.g. wrist camera,
proprioceptive state) via the HELLO handshake, so the benchmark is
auto-configured without manual `--param` flags. See each model server
config's header comments for details.
