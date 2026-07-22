# RC365 S2B hierarchical policy

This server wraps the read-only `rc365_s2b` implementation as one vla-eval
policy. vla-eval keeps ownership of environment construction, reset seeds,
task horizons, stepping, and success accounting. The server queries System 2
on the reset observation and after every complete 16-action System 1 chunk,
then returns one action per harness observation.

Set the reference project and modality metadata paths before starting it:

```bash
export RC365_S2B_ROOT=/path/to/projects/rc365-s2b
export PYTHONPATH="$RC365_S2B_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export ROBOCASA_MODALITY_JSON=/path/to/robocasa/task/lerobot/meta/modality.json
export ROBOCASA_GR00T_N15_CKPT=/path/to/checkpoint-120000
vla-eval serve --config configs/model_servers/rc365_s2b/hierarchical.yaml
```

The reference source is imported in place. No `rc365_s2b` source is copied
into this repository. An editable installation of the same reference project
can be used instead of `PYTHONPATH`.

Select System 2 with `args.system2`:

- `global-only` sends the official global instruction to System 1.
- `random-valid` samples a valid family and stage using the episode seed.
- `mllm-stub` uses the reference stub. Set `mllm_planner_import` to an
  importable `System2` subclass and optionally set `mllm_planner_kwargs` when
  a generator is available.
- `gold-sequence` reads `gold_sequences_path`. This is a predeclared ceiling
  schedule and never receives the environment object or simulator state.

The qualification runner uses the additional internal `gold-oracle` mode.
RoboCasa evaluates the reference privileged completion predicates in the
benchmark container and sends only the resulting typed decision to this
server. Use `python -m vla_eval.rc365_s2b_qualification`; do not select
`gold-oracle` manually for ordinary evaluation.

A gold schedule is a JSON task mapping. `chunk` is zero-based. Omitting it
uses the event list index. A call remains current until a later scheduled
event, so repeated System 2 queries return the same call and do not extend
history. The schedule may end with `finish_task`; otherwise the official task
horizon is the only stopping cap.

```json
{
  "RinseSinkBasin": [
    {
      "chunk": 0,
      "name": "execute_phase",
      "arguments": {
        "skill_family": "Activate",
        "stage": "execute",
        "instruction": "turn on the sink faucet"
      }
    },
    {"chunk": 4, "name": "finish_task", "arguments": {}}
  ]
}
```

`finish_task` is returned as a policy termination request. The synchronous
runner ends the episode without applying another action, then asks the
benchmark for its existing success result.

When qualification output arguments are present, the server writes one
`rc365-s2b-exec-episode-v1` JSONL record. It combines its call and instruction
trace with strict-success and chunk telemetry returned by the benchmark. The
output is directly consumable by the reference qualification scorer.
