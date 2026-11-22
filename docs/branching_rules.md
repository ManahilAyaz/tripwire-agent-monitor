# Git workflow (follow strictly)

## Pull requests
- At least one PR per week. If a week has more work, split it into several PRs by theme.
- Always branch from the latest main: `git checkout main && git pull`. Never push to main directly.
- Open PRs with `gh pr create` against main. Title format: `[W1] Trajectory logger + retry handling (T12, T13)`.
- PR description must include: tasks covered (IDs), what changed, how to test (exact commands),
  dependencies/stubs used, and anything left for a follow-up.
- Do not merge PRs yourself. A teammate reviews and merges.
- If the next PR depends on an unmerged one, branch from that PR's branch and say so in the
  description (stacked PR).

## Commits
- One commit per task. Each task ID gets its own commit; don't mix tasks in one commit.
- Message format: `T12: add JSONL trajectory logger following log schema`
- Before committing: run `pytest` (add tests for new code in `tests/`), and make sure no secrets,
  `.env` files or large data files are staged.

## Branch naming
`w<week>/t<ids>-<short-name>`, e.g. `w1/t02-repo-readme-branching`.
