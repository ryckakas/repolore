# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in
this repository. `CLAUDE.md` is a symlink to this file.

## What this is

`repolore` is a hand-built RAG pipeline over one person's engineering history on GitHub: PR bodies,
review threads, issues, PR-less commits, repo markdown (`README`, `AGENTS.md`, `docs/**`) and local
`.plans/` documents. It is searched from a Typer CLI (`search`, `eval`) and served read-only
to Claude Code through a stdio MCP server. Claude Code writes the answers; repolore itself calls no LLM.

It is a **learning project first**: every stage (fetch, chunk, embed, retrieve, generate,
evaluate) is written by hand so it stays visible. No LangChain, LlamaIndex, RAGAS or other
RAG/eval frameworks. Prefer the approach that exposes how a stage works over one that hides it.

It runs locally for a single user. Other people use it by running their own copy (`uv tool install` from
the git repo + `claude mcp add`, as in the README quickstart) with their own GitHub token. No model API key
is needed: embeddings run locally. There is no hosted service and no GitHub App.

## Commands

Python 3.12 (pinned in `.python-version`), managed with uv, which always uses its own Python build
(`python-preference = "only-managed"`): some system builds, such as python.org's on macOS, can't load the
sqlite-vec extension. A tool install doesn't read the project's `[tool.uv]` settings, so the README's
install command passes `--managed-python` itself; `[tool.uv.sources]` does apply, so it still gets the CPU torch.
Code that opens files always passes
`encoding="utf-8"`, because the Windows default cp1252 codec breaks on GitHub content. Don't lean on
`PYTHONUTF8=1` instead: installed users won't have it set, and CI's Windows job runs without it.

```sh
uv sync                     # install the project and dev tools from uv.lock
uv run repolore --help      # the CLI
uv run repolore repos       # list the discovered repos and why each is in (owned, member, contributed, local);
                            # --excluded also lists the ones repolore.toml leaves out
uv run repolore ingest      # fetch what changed in them into the raw cache; --full refetches, --offline rereads it
uv run repolore inspect     # chunk the cached documents: counts, token histogram, random samples
uv run repolore build       # embed what changed and write the index (<data-dir>/db/default.db; --db names another,
                            # --max-tokens splits long sections (0: the default, 8192 or a smaller window),
                            # --embedder picks the model, --bot-reviews/--no-bot-reviews keeps or drops review
                            # threads started by bots; a left-out option keeps the index's last value)
uv run repolore refresh     # ingest, then build with the index's recorded settings; one refresh at a time
uv run repolore search "…"  # best chunks, with --repo, --kind, --author, -k, --mode, --rerank, --distinct
uv run repolore server status  # the shared MCP server: pid, address, version, sessions; `server stop` ends it
uv run repolore eval        # score retrieval on the eval questions; --mode, --rerank, --distinct, --show-misses,
                            # --holdout
uv run ruff format          # format
uv run ruff check           # lint
uv run pyright              # type-check (standard mode)
uv run bonsai-lint          # cognitive complexity, threshold in bonsai-lint.toml
uv run pytest               # offline tests; `live` tests are deselected
uv run pytest -m live       # tests that call real GitHub / Voyage APIs, run by hand
uv run pytest --cov         # tests with the coverage floor CI enforces (90%, branch coverage)
uv audit --locked --preview-features audit-command   # known advisories in uv.lock
```

Run format, lint, type-check and tests before considering a change done. CI
(`.github/workflows/ci.yml`) runs the same gates plus a few that only run there. Each job is a
required status check on `main`, except `audit`:

- `lint`, `typecheck`: ruff check, ruff format --check, bonsai-lint (cognitive complexity over 15 fails), pyright.
- `test (ubuntu-latest)`, `test (windows-latest)`, `test (macos-latest)`: pytest with the coverage floor.
- `build`: builds the wheel and runs the installed `repolore` from it.
- `typos`: spelling.
- `zizmor`, `actionlint`: workflow security and correctness, and shellcheck over `.githooks/`.
- `audit`: `uv audit`. It runs on every PR but isn't required.

CodeQL (Python and Actions) runs from `.github/workflows/codeql.yml`, not GitHub's default setup,
which skips pull requests from forks. Its `analyze (python)` and `analyze (actions)` jobs are
required status checks too, and a ruleset rule blocks merging until CodeQL has results, and on an
error-level alert or a high or critical security alert.

Every action is pinned to a commit SHA with its version in a comment, and Dependabot moves both. A
renamed or added job also needs `.github/rulesets/protect-main.json` updated and re-applied (see
`.github/rulesets/README.md`).

## Workflow

`main` only changes through pull requests. A GitHub ruleset blocks direct pushes, force-pushes and
deletion. The versioned hooks in `.githooks/` block direct commits and pushes to `main` locally;
enable them once per clone with `git config core.hooksPath .githooks`. Work happens on
`feat/…`, `fix/…`, `chore/…` or `docs/…` branches.

The version comes from git through hatch-vcs, not from `pyproject.toml`. A release is a `vX.Y.Z` tag on `main`;
any later commit builds as the next patch's dev version with its commit, such as `0.1.1.dev3+g1a2b3c4`, and
uncommitted changes add a date. Every tag must be a version: a tag hatch-vcs can't parse fails the build.
`repolore version` prints the version, and the MCP server reports it to its client. The local editable install only
rebuilds when tags change, so its version can lag behind the checked-out commit.

## Architecture

The plan is in `.plans/initial-build/plan.md`, which is local and gitignored. Its decision log
(D1–D27) is the reasoning behind everything below. The package grows into this layout
(`src/repolore/`), one module per pipeline stage:

- `paths.py` resolves the per-user directories through `platformdirs`. Config is
  `%APPDATA%\repolore\repolore.toml` (roaming). Data is `%LOCALAPPDATA%\repolore` (local),
  holding the raw API cache and the DB files, and `--data-dir` overrides it. **The repo never
  contains corpus data.** It contains private code, so it stays out of git.
- `config.py` loads `repolore.toml`: optional `include` / `exclude` globs over `owner/name` and `local_roots`.
  Without a file, every discovered repo is in.
- `discovery.py` decides what to ingest. Repos the viewer owns or reads as a collaborator or org member (forks
  skipped) are fetched whole. Other people's repos found by an `involves:<login>` search are fetched only for the
  viewer's PRs and issues; `repositoriesContributedTo` is not used, because it misses PRs the viewer only commented
  on. The search leaves out each org whose every visible repo is in the viewer's repo list (forks included), so its
  PRs don't fill the search's 1,000 results; forks in those orgs get a search of their own. Private memberships need
  `read:org`; orgs that can't be read stay in the search. Checkouts directly under `local_roots` that have a `.plans/`
  folder add their plans, matched to a repo by the `origin` remote, and stay in as plans-only when the repo itself is
  gone.
- `sources/` fetches GitHub data with a sync `httpx` client: GraphQL for PRs, issues, commits, discovery and
  contributions (`github.graphql`), REST for the markdown tree and blobs. `models.py` types the responses. The
  client waits for the rate limit to reset when fewer than 100 points are left, and retries a rate-limited request.
  `plans.py` scans the local checkouts. Raw responses are cached as `<data-dir>/raw/<owner>/<repo>/*.json`, so
  chunking and embedding never need the network.
- `ingest.py` syncs the raw cache. A cached repo is fetched incrementally: PRs and issues newest-updated first down
  to the newest cached `updatedAt`, commits `since` the newest cached one, markdown files only when their blob SHA
  changed. Files gone from the tree drop out, and a repo no longer discovered or selected is deleted from the cache,
  both logged, so `build` removes them from the index. Deleted PRs and issues, and default-branch history that was
  rewritten or merged with older commit dates, need `ingest --full`.
- `documents.py` normalizes into `Document`s. Bots are filtered by GraphQL `author.__typename`,
  not by a login list. Deleted accounts become `ghost`. Review threads a bot started are dropped unless the index
  is built with `--bot-reviews`. Plans are private `plan` documents with `file://` URLs. A commit is indexed only
  when no PR has it: its oid is neither a PR's merge commit nor one of its first 100 commits, whatever the PR's state.
  The history query doesn't ask GitHub per commit, which made its pages up to 3x slower; commits cached while it did
  keep GitHub's answer. A rebase merge gives the commits new oids, so all but the last commit of a rebase-merged PR
  are indexed as commits too.
- `chunking.py` splits each document's markdown at headings (markdown-it-py, top-level blocks only, so
  a code fence or table stays one block), keeps each section whole up to a limit, and prefixes each chunk with a
  header like `owner/repo · PR #17 · <title> · <section>`. A nested section's header names its parent headings
  too, as in `owner/repo · CHANGELOG.md · Changelog > [0.3.0] - 2026-09-24 > Changed`, while its `section` stays
  the leaf heading, which the eval matches. The limit is `--max-tokens` or else an 8,192-token
  ceiling, never more than the model reads, counted with the model's tokenizer on the embedded text, header
  included (a header over half the limit keeps half for the text, so those rare chunks pass it). A longer section
  is split between blocks, and a block over the limit by lines: fence pieces reopen and close the fence, table
  pieces repeat the header row, and a single line that long is cut by characters. Each document's sections are
  counted in one call, and only those over the limit are counted block by block.
- `embedding/` has the `Embedder` protocol and `LocalEmbedder` over sentence-transformers in float32. `build` embeds
  on the Apple Silicon GPU (MPS) when there is one, or on the device `REPOLORE_DEVICE` names; everything else, the
  MCP server's queries included, runs on the CPU. Both devices give the same vectors, so the embedding cache is shared.
  The default model is `voyage-4-nano` (1024 dimensions, pinned to a reviewed revision because it runs the
  model's own code); `bge-small` is the smaller alternative. An index records its model, and `embedder_for` picks
  the matching one for search. The embedding and reranking models load from the Hugging Face cache and go to the
  Hub only to download a pinned revision that isn't cached yet. If that download fails, `hub.py` turns the error
  into a `ModelUnavailableError` that names the model and revision, which the CLI prints as one line and the MCP
  tool returns as its error; huggingface_hub's own retries still take a minute or two first. `count_tokens` reads
  only the model's `tokenizer.json` (no weights), and `build_index` uses it to batch by padded tokens, since a batch
  is padded to its longest text. torch comes from PyTorch's CPU index (`[tool.uv.sources]`); PyPI's Linux wheel
  bundles gigabytes of CUDA.
- `store.py` writes one SQLite file per index: `chunks`, a sqlite-vec `vec0` table filtered by repo,
  kind and author inside the KNN query, and an FTS5 table. An index is in WAL mode and rewritten in one transaction,
  so a search during a build reads the previous index until it commits; with a rollback journal, readers are locked
  out once the writer spills its cache. `EmbeddingCache` (`<data-dir>/cache/embeddings.db`)
  keys vectors by model and chunk text, so every index reuses them and a rebuild embeds only changed chunks.
- `indexing.py` builds an index. `retrieval.py` searches one in `vector`, `keyword` (FTS5 BM25) or `hybrid` mode
  (reciprocal rank fusion), optionally reranked by `reranking.py`'s local cross-encoder. The defaults (vector,
  whole sections, no reranking, and in the MCP tool one chunk per source URL) come from the experiments in
  `EXPERIMENTS.md`; change them only with a measured `repolore eval` result.
- `evals/questions.py` loads the eval question set. The real set lives in the config dir
  (`repolore questions` prints where) and is never committed, because it names private repos;
  `evals/questions.example.yaml` shows the format.
- `evals/metrics.py` (recall@k, MRR, nDCG@10, hand-written and checked against ir-measures in the tests),
  `evals/harness.py` (runs the questions, matches hits to expected sources by URL and optional section) and
  `evals/runs.py` (each run saved under `<data-dir>/evals/runs/`, compared with the previous one on the same
  questions). A run also records the index size and the median search time per question, which is how search
  latency is measured at a given corpus size. `EXPERIMENTS.md` logs every retrieval experiment, aggregate numbers only.
- `mcp_server.py` defines the `search`, `read`, `list_repos` and `refresh` tools (official `mcp` SDK v2 with
  `MCPServer`; the v1 `FastMCP` import no longer exists). It opens the index read-only per call, so `repolore build`
  can rewrite it meanwhile. Its server loads the index's model on a background thread at start (about 35 s warm here,
  mostly importing transformers), so the first search finds it ready or waits on `LocalEmbedder`'s load lock.
  `search` cuts each passage at 1,500 characters; `read` returns a source's chunks in document order with their
  headers, about 12,000 characters per call (or one passage, when that is longer), so one long PR can't flood the
  context.
  `list_repos` also returns the index's `built_at` and the last refresh's status. Never print to stdout in code a
  stdio server runs: stdout carries the protocol.
- `relay.py` is `repolore-mcp`, which every Claude Code session starts over stdio. It answers the MCP handshake
  itself and forwards `tools/list` and `tools/call` to one shared server per data dir and index, through an MCP client
  over Streamable HTTP, a new connection per call. It starts that server when none runs, replaces one of another
  version, retries a call once if the server went away, and asks the session to reconnect in `/mcp` when repolore
  was upgraded under it. It imports only the MCP SDK and the small state modules, never torch, numpy or sqlite-vec:
  about 70 MB per session instead of the model's 1.8 GB. `--standalone` serves the tools from the relay process
  itself, model included, as before.
- `shared_server.py` is `repolore server run`: the tools over Streamable HTTP from our own `uvicorn.Server` on
  127.0.0.1, port 0. It holds `server-<db>.lock` for its life, writes its port and a bearer token new at every start to
  `server-<db>.json` (readable by the user only), and needs that token on every request; the SDK checks Host and
  Origin against DNS rebinding. It exits on `repolore server stop` (a `server-<db>.stop` file) or 10 minutes after
  the last relay left. `server_state.py` has those files; each relay holds an OS lock in `clients/<db>/` while it
  runs, so a killed one drops out by itself.
- `refresh.py` runs one refresh at a time. `repolore refresh` holds an OS file lock (`<data-dir>/refresh.lock`, via
  `filelock`, so a killed run leaves no stale lock) and records its start, end and outcome in `refresh.json`. The MCP
  `refresh` tool doesn't refresh in the server: it starts `python -m repolore refresh` as a detached process that
  writes to `refresh.log` and outlives the session, so the server loads no second model and blocks no worker thread.
- `processes.py` starts those detached processes, the refresh and the shared server, in the data dir and with
  `python -P`, so the session's project directory is neither locked nor on their import path. On Windows, Claude Code
  stops a session's server with `taskkill /T /F`, which kills every process still linked to it by parent PID,
  whatever the creation flags, so a short-lived intermediate starts the real process and exits before
  `spawn_detached` returns.
- `cli.py` is Typer wiring only. Logic lives in the stage modules.

Nothing is async except the relay's forwarding and the shared server's watcher, which live on the SDK's event loop:
the MCP SDK runs the plain `def` tools on worker threads (40, shared by every session), and the rest of the pipeline
is synchronous on purpose. `LocalEmbedder` locks its lazy model load because those threads can search in parallel.

While a Claude Code session runs `repolore-mcp`, or the shared server runs, files in `.venv` are open, and
`uv sync` / `uv add` can fail with "Access is denied". Disable the server in `/mcp` or close those sessions, and run
`repolore server stop`, before changing dependencies or adding a tag, since a new tag reinstalls the editable package.
Until then, `uv run --no-sync` runs the checkout's code without reinstalling. A running shared server keeps the code
it started with, since the editable install's version changes only with tags: after changing tool or server code,
run `repolore server stop`.

## Comments

No comments, strictly, on hand-written code. Names and small functions carry the meaning; if a
comment feels needed, that's usually a sign the code should be restructured or renamed instead.

The one exception is a **"why" comment**, only when absolutely necessary: a non-obvious invariant, a
workaround for a specific upstream bug or quirk, or a constraint that isn't visible at the call
site. Never a "what" comment restating what the code already says.

## Testing conventions

Tests live in `tests/`, one file per module (`test_<module>.py`), using plain pytest functions and
fixtures. **Tests never touch the network by default**; servers on 127.0.0.1 (port 0) are fine, such as the shared
server the relay tests run on a thread. HTTP to other hosts is faked with `httpx.MockTransport`
fed from saved JSON samples, and embeddings come from `FakeEmbedder` in `tests/conftest.py`, a deterministic
bag-of-words stand-in. Fixtures in
`tests/fixtures/` come from **public repos only**, because this repo is public; never copy data from a
private repo into it. Anything that
calls a real API is marked `@pytest.mark.live`, which is deselected unless run with `-m live`.
Retrieval quality is measured by the eval harness (`repolore eval`), not by unit tests.
