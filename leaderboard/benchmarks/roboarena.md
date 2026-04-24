---
benchmark: roboarena
display_name: RoboArena
paper_url: https://arxiv.org/abs/2506.18123
metric:
  name: elo_rating
  unit: Elo
  range:
  - 0
  - 2000
  higher_is_better: true
official_leaderboard: https://robo-arena.github.io/
detail_notes: "RoboArena (<a href='https://arxiv.org/abs/2506.18123'>2506.18123</a>). Elo-based ranking via pairwise human evaluation of robot manipulation policies."
---

**Standard**: Elo-based pairwise comparison benchmark with API-synced entries only; `overall_score` = Elo rating from pairwise matches (higher is better).

## Scoring
- `overall_score`: Elo rating as provided by the upstream API; `null` for non-API-synced rows (which are rejected entirely — see Checks).
- `suite_scores`: not used.
- `task_scores`: not used.

## Checks
- Is this entry API-synced? `curated_by` must end with `-api`. Manual paper extractions are forbidden and must be rejected from the candidate set entirely — not kept with `overall_score = null`.
- Has the entry been matched against at least 15 pairwise comparisons? Entries with fewer comparisons have high variance (std > 120 Elo) and must note the comparison count.

## Methodology axes (record in `notes`, do not null)
- Pairwise comparison count: high-variance entries (< 15 comparisons) remain in the leaderboard but the count must be disclosed.
