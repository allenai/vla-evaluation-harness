# RFCs

Design documents for `vla-evaluation-harness` — a framework for evaluating Vision-Language-Action (VLA) models across robot simulation benchmarks.

RFCs capture significant architectural decisions and proposed changes. They exist so contributors can understand *why* the system works the way it does, and to provide a lightweight review process for non-trivial changes.

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Proposed** | Under discussion, not yet accepted |
| **Accepted** | Approved for implementation |
| **Partially Implemented** | Accepted and partially landed in the codebase |
| **Implemented** | Fully landed in the codebase |
| **Rejected** | Reviewed and declined |
| **Withdrawn** | Pulled by the author |
| **Superseded** | Replaced by a later RFC |

## Index

| RFC | Title | Status |
|-----|-------|--------|
| [0001](0001-realtime-evaluation.md) | Real-time Evaluation as Architectural Foundation | Implemented |
| [0002](0002-communication-protocol.md) | Communication Protocol | Implemented |
| [0003](0003-model-server-hierarchy.md) | Model Server Hierarchy | Implemented |
| [0004](0004-benchmark-and-episode-execution.md) | Benchmark and Episode Execution | Implemented |
| [0005](0005-model-server-isolation.md) | Model Server Dependency Isolation via uv Scripts | Implemented |
| [0006](0006-episode-sharding.md) | Episode Sharding for Parallel Evaluation | Implemented |
| [0007](0007-batch-predict-model-server.md) | BatchPredictModelServer Implementation | Implemented |

## Proposing a New RFC

1. Create `NNNN-short-title.md` using the next available number.
2. Use the header template below.
3. Open a PR. Discussion happens on the PR.
4. Once accepted, update the status and merge.

### Header Template

```markdown
# RFC-NNNN: Title

- **Author:** @username
- **Status:** Proposed
- **Type:** Standards Track | Informational
- **Created:** YYYY-MM-DD
- **Requires:** RFC-XXXX or —
- **Superseded-By:** RFC-XXXX or —
```

| Field | Description |
|-------|-------------|
| **Author** | GitHub handle(s) of the author(s) |
| **Status** | See [Status Definitions](#status-definitions) above |
| **Type** | *Standards Track* — changes to interfaces, protocols, or behavior. *Informational* — guidelines, conventions, or background |
| **Created** | Date the RFC was first written |
| **Requires** | Other RFCs this one depends on, or `—` if none |
| **Superseded-By** | RFC that replaces this one, or `—` if still active |
