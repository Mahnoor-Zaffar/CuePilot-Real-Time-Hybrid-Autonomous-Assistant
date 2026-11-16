# Contributing

## Setup

```sh
make setup      # server venv, desktop dependencies, docs tooling
make up         # Postgres (pgvector) + Redis via Docker
make migrate
make dev        # gateway on :8000 and the Electron client
```

`make check` is the gate: ruff, desktop typecheck, server tests, the docs
link/mermaid check, and the commit-hook test. CI adds the Postgres migration,
local-embedder, Redis integration, and `promtool` jobs. Those need real
services, so they are opt-in locally:

```sh
make up            # Postgres (pgvector) + Redis
make test-redis    # token bucket and sliding window against real Redis
```

## Commits

- Subject: `type: imperative summary` (`feat`, `fix`, `test`, `docs`, `chore`).
- Body: why the change exists and what it trades away. A change that adds a
  bound or a fallback should say so; the docs do.
- One logical change per commit.

## Attribution

When an agent contributed to a commit, say so in the message:

```
feat: add the outbox age gauge

Assisted-by: OpenCode
```

`make hooks` enables `.githooks/prepare-commit-msg` in your clone. The hook
appends a `Co-authored-by` trailer only when the marker is present, so commits
the agent did not touch stay attributed to their author.

Configure the co-author once per clone. GitHub links a co-author only when the
trailer's email belongs to an account, and the noreply address embeds that
account's numeric id:

```sh
login=cuepilot-agent                                    # the machine account you own
id=$(gh api "users/$login" --jq .id)
git config cuepilot.coauthor "cuePilot Agent <$id+$login@users.noreply.github.com>"
```

The account must already exist; the agent cannot create it, verify its email, or
hold its credentials. Never point the trailer at someone else's identity — that
is false attribution, not a badge. Verify the hook with `make hooks-check`.

For co-author credit to survive the merge, keep the trailer in the merged
commit: use a merge commit or rebase merge, or paste the trailer into the PR
body so a squash merge keeps it.

## Pull requests

Branch from `main`, keep the PR focused, and note the `make check` result in the
description. CI runs on every PR; merges land on `main`.
