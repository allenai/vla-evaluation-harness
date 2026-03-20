## Summary

<!-- What does this PR do? -->

## Checklist

- [ ] I have read the relevant contributing guide ([CONTRIBUTING.md](https://github.com/allenai/vla-evaluation-harness/blob/main/CONTRIBUTING.md) or [leaderboard/CONTRIBUTING.md](https://github.com/allenai/vla-evaluation-harness/blob/main/leaderboard/CONTRIBUTING.md))

**Code changes:**
- [ ] `make check` passes (ruff + ty)
- [ ] `make test` passes (pytest)
- [ ] Smoke-tested affected configs (if benchmarks, model servers, or Docker were changed)
- [ ] I have updated relevant documentation (if applicable)

Smoke test commands run:
```
(paste the commands you ran here, e.g. vla-eval test -c configs/model_servers/cogact.yaml)
```

**Leaderboard data changes:**
- [ ] `python leaderboard/scripts/validate.py` passes (use `--fix` to auto-sort and format)

<!-- Delete whichever section doesn't apply to your PR. -->
