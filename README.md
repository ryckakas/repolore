# repolore

![repolore: RAG over your own engineering history on GitHub, with a CLI and an MCP server](https://raw.githubusercontent.com/ryckakas/repolore/main/docs/images/cover-hero.jpg)

**Ask why your code is the way it is, and get back the PR, review thread or issue that says so.**

*Lore: what a team knows about its own code. Most of it was written down once, in a PR or a review thread, and
then buried.*

RAG over your own engineering history on GitHub: PRs, review threads, issues, commits and docs, searchable from a
CLI and from Claude Code over MCP. It answers questions like "why did we drop the retry middleware?" with the
passages that explain it and links to them.

[![CI](https://github.com/ryckakas/repolore/actions/workflows/ci.yml/badge.svg)](https://github.com/ryckakas/repolore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/repolore-mcp?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/repolore-mcp/)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why this one

- **The history, not just the code.** PR descriptions, review threads, issues, commits that never had a PR, the
  repos' markdown, and the `.plans/` folders in your local checkouts. The reasons behind code live there.
- **Everything stays on your machine.** It uses your own GitHub token and embeds locally, so no model API key is
  needed, and nothing leaves your machine except GitHub API calls and the one-time model download from GitHub.
- **Made for Claude Code.** Sessions get `search`, `read`, `list_repos` and `refresh` tools and answer with links
  to the sources. Codex and other MCP clients work too.
- **One model for every session.** Each session runs a small relay, about 70 MB; one shared server holds the model.
- **Stays current cheaply.** A refresh fetches only what changed on GitHub and embeds only the chunks that changed.
- **Every default was measured.** Each retrieval choice went through `repolore eval` against a fixed question set,
  and [EXPERIMENTS.md](https://github.com/ryckakas/repolore/blob/main/EXPERIMENTS.md) logs every result.
- **Hand-built.** No RAG or eval framework: fetching, chunking, embedding, retrieval and evaluation are each written
  out, so every stage stays visible.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a GitHub login. It runs on Linux, Windows
and macOS 14+ on Apple Silicon.

```sh
uv tool install --managed-python --python 3.12 repolore-mcp   # the repolore and repolore-mcp commands
repolore setup                                                # pick repos, register with Claude Code, first refresh
```

The package is `repolore-mcp`: PyPI's `repolore` is a different project.

For GitHub access, being logged in with the [GitHub CLI](https://cli.github.com/) (`gh auth login`) is enough.
Otherwise set `REPOLORE_GITHUB_TOKEN` to a classic token with the `repo` scope, plus `read:org` for organization
repos. Organizations with SSO also need the token authorized for them.

`repolore setup` checks the GitHub token, lists the repos it found and asks which to index, writes your pick to
`repolore.toml`, registers repolore with Claude Code and Codex when they're on PATH, and starts the first refresh in
the background. It finds the repos you own or can read as a collaborator or organization member, plus your own PRs
and issues in other people's repos. Every question has a flag, which `repolore setup --help` lists, so it can run
without asking: `repolore setup --include "you/*" --register --refresh` (plus `--replace-config` when
`repolore.toml` exists). In Claude Code, the plugin can register it instead and adds a refresh at session start;
see "The Claude Code plugin" below.

The first refresh downloads the embedding model from GitHub, about 1.4 GB, and embeds everything on your machine,
which takes a while for a large account, so start with a few repos. As a rough guide, 3,713 chunks took 22 minutes
to embed on a 12-core desktop CPU, after a 9-minute download. On Apple Silicon it embeds on the GPU through MLX; set
`REPOLORE_DEVICE=cpu` to keep it on the CPU. `repolore doctor` shows how far it has got and which engine a build
uses.

<details>
<summary><b>The Claude Code plugin</b></summary>

This repo is also a Claude Code plugin marketplace. Its plugin registers the MCP server and, when a session starts,
refreshes the index in the background if the last refresh is at least 12 hours old. It needs repolore-mcp 0.3.0 or
later installed as above. Install the plugin before running `repolore setup`, which then sees its server and leaves
the registration to it:

```text
/plugin install repolore --marketplace ryckakas/repolore
```

That works in one step from Claude Code 2.1.275; on older versions, run `/plugin marketplace add ryckakas/repolore`,
then `/plugin install repolore@repolore`. The plugin's tools are named `mcp__plugin_repolore_repolore__search` and so
on.

The plugin's `refresh_hours` option sets the interval, and `0` turns the refresh off: `/plugin`, then Installed,
repolore and Configure options. The refresh skips until `repolore setup` has built the index, and while another
refresh runs.

If Claude Code also has repolore registered for every project with the same command, it uses that registration,
leaves the plugin's server out, and the tools keep their `mcp__repolore__` names. A registration with another command,
such as a checkout run through `uv run`, runs beside the plugin's, and Claude sees every tool twice; keep one, with
`claude mcp remove repolore --scope user` or by disabling the plugin.

Claude Code doesn't update plugins from other people's marketplaces by itself: `/plugin marketplace update repolore`
fetches the latest plugin, or turn on auto-update for the marketplace in `/plugin` under Marketplaces. The plugin
doesn't update repolore itself; `uv tool upgrade repolore-mcp` does.

</details>

<details>
<summary><b>How the install works: Python, PyTorch, the latest main and uvx</b></summary>

`--managed-python` makes uv use its own Python build, which can load the SQLite extension the index needs; some
system builds, such as python.org's on macOS, can't. If the commands aren't found afterwards, run
`uv tool update-shell` and open a new terminal. The install takes about 160 MB.

The first launch of each command compiles its Python code, which took about 25 seconds on Windows, close to Claude
Code's 30-second wait for an MCP server; later launches take a few seconds. Adding `--compile-bytecode` to the
install does that work up front instead. On some Windows machines uv then fails with "os error 1224"
([astral-sh/uv#14017](https://github.com/astral-sh/uv/issues/14017)); install without it there.

Reranking (`--rerank`, off by default) needs PyTorch on top, about 900 MB more, and says how to add it when you ask
for it.

To run the latest `main` instead of the last release, install `git+https://github.com/ryckakas/repolore` the same
way.

Without a tool install, uvx runs it too:
`uvx --managed-python --python 3.12 --from repolore-mcp repolore setup --no-register`, and the other commands the
same way. See [Other MCP clients](https://github.com/ryckakas/repolore#other-mcp-clients) for launching the server
like that.

</details>

<details>
<summary><b>Setting up by hand</b></summary>

`repolore setup` runs these steps, which you can also run yourself:

1. See which repos it found:

   ```sh
   repolore repos
   ```

   Create `repolore.toml` at the config path that `repolore version` prints, then run `repolore repos` again to check
   the selection:

   ```toml
   include = ["you/some-repo", "your-org/*"]   # globs over owner/name; every repo when unset
   exclude = ["your-org/old-*"]                # never these
   local_roots = ["~/projects"]                # checkouts in here add their local .plans/ documents
   ```

2. Fetch and index:

   ```sh
   repolore ingest   # fetch from GitHub; later runs fetch only what changed
   repolore build    # embed and index; later runs embed only what changed
   ```

3. Register the MCP server with Claude Code once:

   ```sh
   claude mcp add --scope user --transport stdio repolore -- repolore-mcp
   ```

   For Codex and other clients, see [Other MCP clients](https://github.com/ryckakas/repolore#other-mcp-clients).

</details>

## Use it

```sh
repolore search "why did we drop the retry middleware?"
repolore search "rate limit handling" --repo you/some-repo --kind pr -k 10
repolore refresh          # fetch what changed and rebuild the index with its recorded settings
repolore doctor           # check every part of the setup, with the fix for each failure
repolore server status    # the shared MCP server; `repolore server stop` stops it
```

Or ask Claude Code in a new session. It gets `search`, `read`, `list_repos` and `refresh` tools and can answer
questions about your history with links to the sources. `search` cuts each passage at 1,500 characters, and `read`
returns a source's whole text from the index, plans and private repos included, a page at a time.

<details>
<summary><b>The shared server</b></summary>

All sessions share one server that holds the model: each session's `repolore-mcp` is a small relay (about 70 MB)
that starts the shared server when none is running and passes the tools through. The server loads the model in the
background when it starts, which takes about half a minute, so only a search in that time waits for it, and it exits
10 minutes after the last session closes. It listens on 127.0.0.1 only and needs a token that changes every time it
starts. `repolore server status` shows it and `repolore server stop` stops it. `repolore-mcp --standalone` serves
from the session's own process, with its own copy of the model, as before.

</details>

<details>
<summary><b>Keeping it fresh</b></summary>

`repolore refresh` ingests what changed and rebuilds the index with the settings it was built with, and only one
refresh runs at a time. Search keeps working meanwhile. An MCP client can do the same: the server's `refresh` tool
starts one in the background, and `list_repos` says when the index was built, how the last refresh went and, while
one runs, how far it has got (repos fetched, chunks embedded). Claude Code asks before the first call; to allow it
for good, add `mcp__repolore__refresh` to `permissions.allow` in your Claude Code settings, or
`mcp__plugin_repolore_repolore__refresh` when the plugin provides the server.

The plugin also refreshes when a session starts and the last refresh is at least 12 hours old. Without the plugin, a
`SessionStart` hook in `~/.claude/settings.json` does the same:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          { "type": "command", "command": "repolore", "args": ["session-start", "--hours", "12"], "async": true }
        ]
      }
    ]
  }
}
```

`repolore session-start` starts the refresh in the background and returns at once, so the session doesn't wait for
it and closing the session doesn't stop it; Claude Code ends a hook's own process with its session.

</details>

<details>
<summary><b>Upgrading, your data, and when something breaks</b></summary>

`uv tool upgrade repolore-mcp` updates repolore itself. On Windows, close the sessions using repolore (in Claude Code
you can disable it in `/mcp` instead) and run `repolore server stop` first, since running programs keep their files
locked. Elsewhere, a session opened before the upgrade says so on its next search; reconnect repolore in your client
(`/mcp` in Claude Code).

The index and the raw GitHub responses live in the data directory that `repolore version` prints; they contain your
private repos' content, so treat that folder like the repos themselves.

When something doesn't work, `repolore doctor` checks each part of the setup and prints the fix for each one that
fails: GitHub access and the token's scopes, `repolore.toml`, the index, the embedding model, the last refresh, the
shared server and the MCP clients that have repolore registered. A client that's installed without repolore is only
a warning. `doctor` makes one GitHub request, and when Claude Code is installed it asks `claude mcp get repolore` (or
`plugin:repolore:repolore`, the plugin's server), which starts repolore to check that it connects.

</details>

## Other MCP clients

`repolore-mcp` is a standard stdio MCP server, so other clients can run it too. Most take a JSON entry in the
portable `mcpServers` format (VS Code's own `mcp.json` calls the object `servers`):

```json
{ "mcpServers": { "repolore": { "command": "repolore-mcp" } } }
```

A client can also launch it through `uvx`, without a tool install:

```json
{ "mcpServers": { "repolore": { "command": "uvx", "args": ["--managed-python", "--python", "3.12", "repolore-mcp"] } } }
```

The first launch downloads about 190 MB, which can take longer than a client waits for a server to start (30
seconds in Claude Code), so run `uvx --managed-python --python 3.12 --from repolore-mcp repolore setup --no-register`
once first: it fills uv's cache and sets repolore up.

<details>
<summary><b>Codex</b></summary>

Add it to `~/.codex/config.toml` with longer timeouts. `repolore setup` adds this entry when `codex` is on PATH, with
an `env_vars` line for the variables below that are set when it runs:

```toml
[mcp_servers.repolore]
command = "repolore-mcp"
startup_timeout_sec = 60
tool_timeout_sec = 300
```

Codex's startup timeout covers the server's `tools/list`, which takes about 8 seconds when no shared server is
running yet and longer after a reboot, and its tool timeout covers a search that waits for the model to load, about
10 seconds warm and longer cold. Its documented defaults are 10 and 60 seconds. When building a session's tool list,
Codex waits only about a second for servers that aren't required (`mcp_optional_startup_grace_ms`); set that to `0`,
or `required = true` on this server, to wait for the startup timeout instead.

Codex gives a server only a short list of environment variables: `PATH`, the home, app data and temp folders, and a
few more. That's enough for `gh auth token` (checked on Windows). If you rely on `REPOLORE_GITHUB_TOKEN`,
`HTTPS_PROXY`, `REPOLORE_DEVICE` or an `XDG_*` directory, list it in `env_vars = ["REPOLORE_GITHUB_TOKEN"]`. The relay
finds its folders through `XDG_*`, and the shared server, which every client's sessions share, and the refreshes it
starts inherit the environment of whichever session started it.

</details>

## What the experiments found

Every retrieval choice was measured with `repolore eval` against 35 questions about the author's public repos, each
with the sources expected to answer it.
[EXPERIMENTS.md](https://github.com/ryckakas/repolore/blob/main/EXPERIMENTS.md) has the full log. In short:

- **Plain vector search won.** Keyword search (BM25) alone was worse on every metric, because questions use plain
  words rather than the exact identifiers. Fusing the two rankings and reranking with a cross-encoder both gave
  gains within the noise and moved some first-place hits down.
- **Bigger chunks were better.** One chunk per markdown section, never split, beat 800, 400 and 200-token caps.
  No section in that corpus came near 8,192 tokens, which is now the most a chunk holds by default: on larger
  corpora, the few longer sections were slow to embed and were cut off at the model's 32K-token window
  ([#17](https://github.com/ryckakas/repolore/issues/17)), so they are now split instead.
- **The model matters most.** `bge-small` (33M parameters) lost a third of the first-place hits compared with
  `voyage-4-nano` (340M), the default.
- **Less can be more.** Indexing every commit instead of only those without a PR lowered MRR: the commits restate
  their PR and push its passages down.
- **One passage per source.** Plain search spent the top 5 on 3.3 sources on average, the rest being more sections
  of the same document. Keeping each source's best section raised recall@5 from 0.857 to 0.904 and lowered no
  question, so Claude Code's search does that by default (`--distinct` on the CLI).
- **"Not found" is the open problem.** Cosine scores of answerable and unanswerable questions overlap. The
  reranker's scores separated them much better on the 3 unanswerable questions so far, which makes them the most
  promising way to say "nothing indexed covers this".

| Default config, 35 questions | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| First baseline (400-token chunks) | 0.818 | 0.919 | 0.809 | 0.804 |
| Now (whole sections) | 0.865 | 0.919 | 0.814 | 0.810 |

A search takes a median of 308 ms over 2,587 chunks on a desktop CPU, query embedding included. With 35 questions,
one question moves recall by about 3%, so these numbers show direction, not precision. The biggest limit is the
question set itself: several "misses" are valid answers that weren't listed as expected.

## Development

```sh
uv sync
uv run repolore --help
uv run pytest
```

To have Claude Code use your checkout instead of the installed tool:

```sh
claude mcp add --scope user --transport stdio repolore -- uv run --directory /path/to/repolore --no-sync repolore-mcp
```

`main` only changes through PRs. After cloning, enable the hooks that block direct commits and pushes to it:

```sh
git config core.hooksPath .githooks
```

## Documentation

- [EXPERIMENTS.md](https://github.com/ryckakas/repolore/blob/main/EXPERIMENTS.md): every retrieval experiment and
  its numbers
- [AGENTS.md](https://github.com/ryckakas/repolore/blob/main/AGENTS.md): the architecture, stage by stage, and the
  conventions for working on it
- [SECURITY.md](https://github.com/ryckakas/repolore/blob/main/SECURITY.md): reporting a vulnerability

## License

MIT
