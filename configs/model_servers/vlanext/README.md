---
smoke_config: libero_spatial.yaml
---

# VLANeXt

VLANeXt-B (Qwen3-VL-2B, flow-matching action chunks), served as upstream's LIBERO eval runs it.
[Paper](https://arxiv.org/abs/2602.18532) | [GitHub](https://github.com/DravenALG/VLANeXt) (pinned `d911e6b`)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `libero_spatial.yaml` | LIBERO Spatial | `DravenALG/VLANeXt` `VLANeXt_libero_spatial.pt` |
| `libero_object.yaml` | LIBERO Object | `DravenALG/VLANeXt` `VLANeXt_libero_object.pt` |
| `libero_goal.yaml` | LIBERO Goal | `DravenALG/VLANeXt` `VLANeXt_libero_goal.pt` |
| `libero_10.yaml` | LIBERO-10 | `DravenALG/VLANeXt` `VLANeXt_libero_10.pt` |

Pair each with `configs/benchmarks/vlanext/<suite>.yaml`. `attn_implementation` defaults to `sdpa`;
upstream uses `flash_attention_2` (needs `flash-attn` in the script env).
