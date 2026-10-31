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

It runs locally for a single user. Other people use it by running their own copy (`uvx` + `claude
mcp add`) with their own GitHub token. No model API key is needed: embeddings run locally by default, and
Voyage's hosted API is an option. There is no hosted service and no GitHub App.

## Commands

Python 3.12 (pinned in `.python-version`), managed with uv, which always uses its own Python build
(`python-preference = "only-managed"`): some system builds, such as python.org's on macOS, can't load the
sqlite-vec extension. Code that opens files always passes
`encoding="utf-8"`, because the Windows default cp1252 codec breaks on GitHub content. Don't lean on
`PYTHONUTF8=1` instead: `uvx` users won't have it set, and CI's Windows job runs without it.

```sh
uv sync                     # install the project and dev tools from uv.lock
uv run repolore --help      # the CLI
uv run repolore repos       # list the discovered repos and why each is in (owned, member, contributed, local);
                            # --excluded also lists the ones repolore.toml leaves out
uv run repolore ingest      # fetch what changed in them into the raw cache; --full refetches, --offline rereads it
uv run repolore inspect     # chunk the cached documents: counts, token histogram, random samples
uv run repolore build       # embed what changed and write the index (<data-dir>/db/default.db; --db names another,
                            # --max-tokens splits long sections, --embedder picks the model,
                            # --bot-reviews keeps review threads started by bots)
uv run repolore search "…"  # best chunks, with --repo, --kind, --author, -k, --mode, --rerank
uv run repolore eval        # score retrieval on the eval questions; --mode, --rerank, --show-misses, --holdout
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
which skips pull requests from forks. A ruleset rule blocks merging until CodeQL has results, and on
an error-level alert or a high or critical security alert.

Every action is pinned to a commit SHA with its version in a comment, and Dependabot moves both. A
renamed or added job also needs `.github/rulesets/protect-main.json` updated and re-applied (see
`.github/rulesets/README.md`).

## Workflow

`main` only changes through pull requests. A GitHub ruleset blocks direct pushes, force-pushes and
deletion. The versioned hooks in `.githooks/` block direct commits and pushes to `main` locally;
enable them once per clone with `git config core.hooksPath .githooks`. Work happens on
`feat/…`, `fix/…`, `chore/…` or `docs/…` branches.

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
  on. Checkouts directly under `local_roots` that have a `.plans/` folder add their plans, matched to a repo by the
  `origin` remote, and stay in as plans-only when the repo itself is gone.
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
  is built with `--bot-reviews`. Plans are private `plan` documents with `file://` URLs.
- `chunking.py` splits each document's markdown at headings (markdown-it-py, top-level blocks only, so
  a code fence or table is never cut), keeps each section whole (with `max_tokens`, splits longer ones between
  blocks), and prefixes each chunk with a header like `owner/repo · PR #17 · <title> · <section>`.
- `embedding/` has the `Embedder` protocol and `LocalEmbedder` over sentence-transformers on the CPU in float32.
  The default model is `voyage-4-nano` (1024 dimensions, pinned to a reviewed revision because it runs the
  model's own code); `bge-small` is the smaller alternative. An index records its model, and `embedder_for` picks
  the matching one for search. The embedding and reranking models load from the Hugging Face cache and go to the
  Hub only to download a pinned revision that isn't cached yet. torch comes from PyTorch's CPU index
  (`[tool.uv.sources]`); PyPI's Linux wheel bundles gigabytes of CUDA.
- `store.py` writes one SQLite file per index: `chunks`, a sqlite-vec `vec0` table filtered by repo,
  kind and author inside the KNN query, and an FTS5 table. `EmbeddingCache` (`<data-dir>/cache/embeddings.db`)
  keys vectors by model and chunk text, so every index reuses them and a rebuild embeds only changed chunks.
- `indexing.py` builds an index. `retrieval.py` searches one in `vector`, `keyword` (FTS5 BM25) or `hybrid` mode
  (reciprocal rank fusion), optionally reranked by `reranking.py`'s local cross-encoder. The defaults (vector,
  whole sections, no reranking) come from the experiments in `EXPERIMENTS.md`; change them only with a measured
  `repolore eval` result.
- `evals/questions.py` loads the eval question set. The real set lives in the config dir
  (`repolore questions` prints where) and is never committed, because it names private repos;
  `evals/questions.example.yaml` shows the format.
- `evals/metrics.py` (recall@k, MRR, nDCG@10, hand-written and checked against ir-measures in the tests),
  `evals/harness.py` (runs the questions, matches hits to expected sources by URL and optional section) and
  `evals/runs.py` (each run saved under `<data-dir>/evals/runs/`, compared with the previous one on the same
  questions). A run also records the index size and the median search time per question, which is how search
  latency is measured at a given corpus size. `EXPERIMENTS.md` logs every retrieval experiment, aggregate numbers only.
- `mcp_server.py` serves `search` and `list_repos` to Claude Code over stdio (`repolore-mcp`, official `mcp` SDK
  v2 with `MCPServer`; the v1 `FastMCP` import no longer exists). It opens the index read-only per call, so
  `repolore build` can rewrite it meanwhile, and loads the model on the first search (about a minute here).
  Never print to stdout in code the server runs: stdout carries the protocol.
- `cli.py` is Typer wiring only. Logic lives in the stage modules.

Nothing is async: the MCP SDK runs the plain `def` tools on worker threads, and the rest of the pipeline is
synchronous on purpose. `LocalEmbedder` locks its lazy model load because those threads can search in parallel.

While a Claude Code session runs the server, files in `.venv` are open, and `uv sync` / `uv add` can fail with
"Access is denied". Disable the server in `/mcp` or close those sessions before changing dependencies.

## Comments

No comments, strictly, on hand-written code. Names and small functions carry the meaning; if a
comment feels needed, that's usually a sign the code should be restructured or renamed instead.

The one exception is a **"why" comment**, only when absolutely necessary: a non-obvious invariant, a
workaround for a specific upstream bug or quirk, or a constraint that isn't visible at the call
site. Never a "what" comment restating what the code already says.

## Testing conventions

Tests live in `tests/`, one file per module (`test_<module>.py`), using plain pytest functions and
fixtures. **Tests never touch the network by default.** HTTP is faked with `httpx.MockTransport`
fed from saved JSON samples, and embeddings come from `FakeEmbedder` in `tests/conftest.py`, a deterministic
bag-of-words stand-in. Fixtures in
`tests/fixtures/` come from **public repos only**, because this repo is public; never copy data from a
private repo into it. Anything that
calls a real API is marked `@pytest.mark.live`, which is deselected unless run with `-m live`.
Retrieval quality is measured by the eval harness (`repolore eval`), not by unit tests.
