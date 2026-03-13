# Blocked Benchmarks

Benchmarks that were investigated but could not be integrated due to external blockers outside the harness's control.

---

## FurnitureBench

| Field | Value |
|-------|-------|
| **Paper** | FurnitureBench: A Real-World Furniture Assembly Benchmark (RSS 2023) |
| **Repository** | https://github.com/clvrai/furniture-bench |
| **Simulator** | Isaac Gym (NVIDIA) |
| **Blocker** | Isaac Gym binary unavailable |
| **Date investigated** | 2026-02-25 |

### Details

FurnitureBench depends on **Isaac Gym Preview** (`isaacgym`), a proprietary NVIDIA package that was distributed as a pre-built binary via a gated download page. As of 2025, the download page has been retired and the package is no longer available through any public channel. NVIDIA has transitioned to **Isaac Lab** (built on Isaac Sim / Omniverse), but FurnitureBench's codebase has not been ported to the new stack.

- `pip install isaacgym` does not work — the package was never published to PyPI.
- The original preview download page (`https://developer.nvidia.com/isaac-gym`) redirects to Isaac Lab documentation.
- No open-source fork or mirror of the Isaac Gym binary exists.

### Resolution path

Integration becomes possible if any of the following occur:

1. NVIDIA re-releases Isaac Gym binaries or open-sources the runtime.
2. The FurnitureBench authors port their codebase to Isaac Lab / Isaac Sim.
3. A community fork replaces the Isaac Gym dependency with an alternative simulator.

---

## BEHAVIOR-1K

| Field | Value |
|-------|-------|
| **Paper** | BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark (CoRL 2022) |
| **Repository** | https://github.com/StanfordVL/OmniGibson |
| **Simulator** | Isaac Sim / Omniverse (via OmniGibson) |
| **Blocker** | Isaac Sim incompatible with A100 datacenter GPUs |
| **Date investigated** | 2026-02-25 |

### Details

BEHAVIOR-1K uses **OmniGibson**, which requires **NVIDIA Isaac Sim** (Omniverse). Isaac Sim's rendering pipeline does not support A100 datacenter GPUs — it requires RTX-class GPUs (GeForce RTX, Quadro RTX, A10, L4, T4, etc.) for its ray-tracing renderer.

On A100 80GB PCIe (the hardware available for this integration):

- Isaac Sim fails during rendering pipeline initialization.
- This is a known limitation confirmed in NVIDIA's official documentation and bug reports.
- The issue affects all A100 variants (PCIe and SXM).

OmniGibson itself installs cleanly (`pip install omnigibson`) and an official Docker image exists (`stanfordvl/behavior:latest`). The **only** blocker is GPU hardware compatibility.

### Resolution path

Integration becomes possible if any of the following occur:

1. Evaluation is run on a machine with an RTX-class GPU (e.g., RTX 4090, A10, L4).
2. NVIDIA adds A100 support to Isaac Sim's rendering pipeline.
3. OmniGibson adds a software renderer fallback that works without RTX hardware.

