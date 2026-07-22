# RC365 S2B hierarchical policy

One stateful policy wrapping the S2B System 2 planner and the GR00T
System 1 executor. System 2 is re-queried after every 16-step action
chunk: an unchanged call continues the current instruction, a new call
switches it, and finish_task terminates the episode (surfaced to the
runner as `terminate_episode`).

`hierarchical.yaml` selects the System 2 implementation (`gold`,
`global_only`, `random_valid`, or `mllm`) and the checkpoint paths.
Set `VLA_EVAL_RENDER=cpu` to render through OSMesa and keep the GPU
exclusively for the policy.
