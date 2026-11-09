# repolore

![repolore: RAG over your own engineering history on GitHub, with a CLI and an MCP server](https://raw.githubusercontent.com/ryckakas/repolore/main/docs/images/cover-hero.jpg)

**Ask why your code is the way it is, and get back the PR, review thread or issue that says so.**

*Lore: what a team knows about its own code. Most of it was written down once, in a PR or a review thread, and
then buried.*

RAG over your own engineering history on GitHub: PRs, review threads, issues, commits and docs, searchable from a
CLI and from any MCP client, such as Claude Code or Codex. It answers questions like "why did we drop the retry
middleware?" with the passages that explain it and links to them.

[![CI](https://github.com/ryckakas/repolore/actions/workflows/ci.yml/badge.svg)](https://github.com/ryckakas/repolore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/repolore-mcp?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/repolore-mcp/)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why this one

- **The history, not just the code.** PR descriptions, review threads, issues, commits that never had a PR, the
  repos' markdown, and the local documents you point it at, such as `.plans/` folders in your checkouts. The
  reasons behind code live there.
- **Everything stays on your machine.** It uses your own GitHub token and embeds locally, so no model API key is
  needed, and nothing leaves your machine except GitHub API calls and the one-time model download from GitHub.
- **Works in your MCP client.** Sessions get `search`, `read`, `list_repos` and `refresh` tools and answer with links
  to the sources. `repolore setup` registers it with the clients it finds, from Claude Code and Codex to Cursor and
  VS Code.
- **One model for every session.** Each session runs a small relay, about 70 MB; one shared server holds the model.
- **Stays current cheaply.** A refresh fetches only what changed on GitHub and embeds only the chunks that changed.
- **Every default was measured.** Each retrieval choice went through `repolore eval` against a fixed question set,
  and [EXPERIMENTS.md](https://github.com/ryckakas/repolore/blob/main/EXPERIMENTS.md) logs every result.
- **Hand-built.** No RAG or eval framework: fetching, chunking, embedding, retrieval and evaluation are each written
  out, so every stage stays visible.

## Install

You need a GitHub login. It runs on Linux, Windows and macOS 14+ on Apple Silicon. On Linux and macOS:

```sh
curl -fsSL https://raw.githubusercontent.com/ryckakas/repolore/main/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/ryckakas/repolore/main/install.ps1 | iex
```

Both scripts ([install.sh](https://github.com/ryckakas/repolore/blob/main/install.sh),
[install.ps1](https://github.com/ryckakas/repolore/blob/main/install.ps1)) install
[uv](https://docs.astral.sh/uv/getting-started/installation/) if it's missing, install or upgrade repolore, and run
`repolore setup`. To do it by hand, with uv already there:

```sh
uv tool install --managed-python --python 3.12 repolore-mcp   # the repolore and repolore-mcp commands
repolore setup                                                # pick repos, register your MCP clients, first refresh
```

The package is `repolore-mcp`: PyPI's `repolore` is a different project.

For GitHub access, being logged in with the [GitHub CLI](https://cli.github.com/) (`gh auth login`) is enough.
Otherwise set `REPOLORE_GITHUB_TOKEN` to a classic token with the `repo` scope, plus `read:org` for organization
repos. Organizations with SSO also need the token authorized for them.

`repolore setup` checks the GitHub token, lists the repos it found and asks which to index, writes your pick to
`repolore.toml`, offers to register repolore with each
[MCP client](https://github.com/ryckakas/repolore#mcp-clients) it finds, and starts the first refresh in the
background. It finds the repos you own or can read as a collaborator or organization member, plus your own PRs and
issues in other people's repos. Replacing `repolore.toml` with a smaller pick removes the other repos from the index
at the next refresh.

With no flags it first offers `repolore ui`, a local web page with the same choices, instead of the questions.

Every question has a flag, which `repolore setup --help` lists, so it can run without asking:
`repolore setup --include "you/*" --register --refresh`, plus `--replace-config` when `repolore.toml` exists, and
`--clients claude,codex` to pick the clients. The install scripts pass their arguments on:

```sh
curl -fsSL https://raw.githubusercontent.com/ryckakas/repolore/main/install.sh | sh -s -- --include "you/*" --register --refresh
```

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/ryckakas/repolore/main/install.ps1))) --include "you/*" --register --refresh
```

The first refresh downloads the embedding model from GitHub, about 1.4 GB, and embeds everything on your machine,
which takes a while for a large account, so start with a few repos. As a rough guide, 3,713 chunks took 22 minutes
to embed on a 12-core desktop CPU, after a 9-minute download. On Apple Silicon it embeds on the GPU through MLX; set
`REPOLORE_DEVICE=cpu` to keep it on the CPU. `repolore doctor` shows how far it has got and which engine a build
uses.

<details>
<summary><b>How the install works: Python, PyTorch, the latest main and uvx</b></summary>

`--managed-python` makes uv use its own Python build, which can load the SQLite extension the index needs; some
system builds, such as python.org's on macOS, can't. If the commands aren't found afterwards, open a new terminal;
after a manual install, run `uv tool update-shell` first. The install takes about 160 MB.

The first launch of each command compiles its Python code, which took about 25 seconds on Windows, close to the
30 seconds Claude Code waits for an MCP server; later launches take a few seconds. Adding `--compile-bytecode` to a
manual install does that work up front instead. On some Windows machines uv then fails with "os error 1224"
([astral-sh/uv#14017](https://github.com/astral-sh/uv/issues/14017)); install without it there.

Reranking (`--rerank`, off by default) needs PyTorch on top, about 900 MB more, and says how to add it when you ask
for it.

To run the latest `main` instead of the last release, set `REPOLORE_INSTALL_FROM_GIT=1` for the install script
(`curl … | REPOLORE_INSTALL_FROM_GIT=1 sh`, or `$env:REPOLORE_INSTALL_FROM_GIT = '1'` first in PowerShell), or install
`git+https://github.com/ryckakas/repolore` by hand. Extras such as `rerank` have to be added again.

Without a tool install, uvx runs it too:
`uvx --managed-python --python 3.12 --from repolore-mcp repolore setup --no-register`, and the other commands the
same way. See [MCP clients](https://github.com/ryckakas/repolore#mcp-clients) for launching the server like that.

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
   local_roots = ["~/projects"]                # where your checkouts are, up to two levels down
   local_patterns = [".plans/**/*.md"]         # markdown to index inside each checkout; nothing local when unset
   checkout_depth = 2                          # optional: how many levels under a root to look

   [[local_folders]]                           # markdown that belongs to no repo; repeat for more
   name = "notes"                              # search it as --repo local/notes
   path = "~/notes"                            # absolute or ~, a folder or one file
   patterns = ["**/*.md"]                      # optional, for a folder

   [[branches]]                                # commits from more than the default branch; repeat for more
   repo = "your-org/app"                       # a glob over owner/name
   patterns = ["staging/*", "live/*"]          # globs over branch names, case-sensitive
   ```

   `repolore ui` can add and remove `[[branches]]` on its Sources page.

   Commits are read from the default branch and from any others `[[branches]]` names, and a commit is indexed only
   when no PR has it. A PR from one of these branches doesn't count, so a fix pushed to `staging/x` and released by a
   `staging/x` to `live/x` PR is still found. A release through a branch you haven't listed counts as that commit's
   PR, so list that branch too. A PR's commit cherry-picked onto another branch is indexed once, repeating the PR.
   Removing a pattern or deleting a branch leaves its commits in the index until `repolore ingest --full`. Only
   branches pushed to GitHub are read, not local ones. Name long-lived branches rather than using `*`, which stops
   every PR from claiming its commits and indexes them all.

2. Fetch and index:

   ```sh
   repolore ingest   # fetch from GitHub; later runs fetch only what changed
   repolore build    # embed and index; later runs embed only what changed
   ```

3. Register the MCP server with your client once; see
   [MCP clients](https://github.com/ryckakas/repolore#mcp-clients). In Claude Code:

   ```sh
   claude mcp add --scope user --transport stdio repolore -- repolore-mcp
   ```

</details>

## Use it

```sh
repolore search "why did we drop the retry middleware?"
repolore search "rate limit handling" --repo you/some-repo --kind pr -k 10
repolore refresh          # fetch what changed and rebuild the index with its recorded settings
repolore doctor           # check every part of the setup, with the fix for each failure
repolore ui               # a local web page for the status, the MCP tools, what's indexed and the clients, until Ctrl-C
repolore server status    # the shared MCP server; `repolore server stop` stops it
```

On a Linux machine shared with other accounts, run `repolore ui --no-browser` and paste the printed address into a
browser yourself: opening a browser puts that address, which works until Ctrl-C, on a command line other accounts can
read.

Or ask your MCP client in a new session. It gets `search`, `read`, `list_repos` and `refresh` tools and can answer
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
one runs, how far it has got (repos fetched, chunks embedded). Nothing refreshes by itself, except in Claude Code,
which can do it when a session starts; see [Claude Code](https://github.com/ryckakas/repolore#mcp-clients).

The shared server, and the refreshes it starts, inherit the environment of whichever client started it. Claude Desktop
never gets `REPOLORE_GITHUB_TOKEN`, for example, so if it started the server, a `refresh` tool call from any client can
fail when you rely on that variable or on a `gh` the app can't find. Searches aren't affected. `repolore refresh` from
a terminal uses your own environment, and `repolore server stop` lets the next client start the server.

</details>

<details>
<summary><b>Upgrading, your data, and when something breaks</b></summary>

Running the install script again, or `uv tool upgrade repolore-mcp`, updates repolore and keeps its extras. On Windows,
close the sessions using repolore first (in Claude Code you can disable it in `/mcp` instead), since running programs
keep their files locked; before a manual upgrade, run `repolore server stop` too. Elsewhere, a session opened before the
upgrade says so on its next search; reconnect repolore in your client (`/mcp` in Claude Code).

The index and the raw GitHub responses live in the data directory that `repolore version` prints; they contain your
private repos' content, so treat that folder like the repos themselves.

When something doesn't work, `repolore doctor` checks each part of the setup and prints the fix for each one that
fails: GitHub access and the token's scopes, `repolore.toml`, what each local source matches, the index, the
embedding model, the last refresh, the shared server and each MCP client it finds. A client that's installed without
repolore is only a warning; one with
two registrations fails, naming how to remove each. `doctor` makes one GitHub request, and for a single Claude Code
registration it asks `claude mcp get`, which starts repolore to check that it connects. Other clients can't be asked,
so they pass as "connection not checked".

</details>

## MCP clients

`repolore-mcp` is a standard stdio MCP server. `repolore setup` offers to register it, for every project, with each of
these it finds; `--clients` picks some by key:

| Client | `--clients` | Registered in |
|---|---|---|
| Claude Code | `claude` | `claude mcp add` at user scope, or the plugin below |
| Codex | `codex` | `~/.codex/config.toml` |
| Gemini CLI | `gemini` | `~/.gemini/settings.json` |
| Claude Desktop | `claude-desktop` | `claude_desktop_config.json` |
| Cursor | `cursor` | `~/.cursor/mcp.json` |
| VS Code | `vscode` | `mcp.json` in its user folder |
| Zed | `zed` | its `settings.json` |
| OpenCode | `opencode` | `opencode.json` or `opencode.jsonc` |
| Devin CLI | `devin` | `devin mcp add` at user scope |

Claude Code and Codex have been run end to end. Registering has also been checked on real installs of Claude Desktop,
Cursor and VS Code. The rest follow each client's documentation, so
[open an issue](https://github.com/ryckakas/repolore/issues) if one doesn't work.

A client that already has repolore, under any name, is left alone. Adding to another app's settings file leaves the
rest of it as it was, comments included, and keeps a `<file>.repolore-backup` from before the first change. Apps started
from a dock or the Start menu get the absolute path to `repolore-mcp`, because their PATH may not have uv's tool folder.
A token only ever goes in as a reference to the variable, for the clients that can expand one.

<details>
<summary><b>Claude Code</b></summary>

This repo is also a Claude Code plugin marketplace. Its plugin registers the MCP server and, when a session starts,
refreshes the index in the background if the last refresh is at least 12 hours old. It needs repolore-mcp 0.3.0 or
later. Install the plugin before running `repolore setup` or the install script, which then sees its server and leaves
the registration to it:

```text
/plugin install repolore --marketplace ryckakas/repolore
```

That works in one step from Claude Code 2.1.275; on older versions, run `/plugin marketplace add ryckakas/repolore`,
then `/plugin install repolore@repolore`. The plugin's tools are named `mcp__plugin_repolore_repolore__search` and so
on.

To switch to the plugin after `setup` registered repolore, remove that registration:
`claude mcp remove repolore --scope user`. `repolore doctor` fails while a client has two. If you leave both and they
run the same command, Claude Code uses the registration and leaves the plugin's server out. With different commands,
such as a checkout run through `uv run`, both run and Claude sees every tool twice.

The plugin's `refresh_hours` option sets the interval, and `0` turns the refresh off: `/plugin`, then Installed,
repolore and Configure options. The refresh skips until `repolore setup` has built the index, and while another
refresh runs.

Claude Code doesn't update plugins from other people's marketplaces by itself: `/plugin marketplace update repolore`
fetches the latest plugin, or turn on auto-update for the marketplace in `/plugin` under Marketplaces. The plugin
doesn't update repolore itself.

Without the plugin, a `SessionStart` hook in `~/.claude/settings.json` does the same refresh:

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

Claude Code asks before the first call to the `refresh` tool. To allow it for good, add `mcp__repolore__refresh` to
`permissions.allow` in your Claude Code settings, or `mcp__plugin_repolore_repolore__refresh` when the plugin provides
the server.

</details>

<details>
<summary><b>Claude Desktop, Cursor, VS Code, Gemini CLI, Zed, OpenCode and Devin CLI</b></summary>

- **Claude Desktop:** quit it fully and start it again right after registering. A copy that's already running hasn't
  loaded the entry, so the next time it saves one of its own settings it removes it. `repolore doctor` shows when that
  has happened, and registering again puts it back. It gets no token, because it has no documented way to reference a
  variable. On Windows, a Store install is reported to read its config from its own folder under `Packages`, and
  `setup` writes there.
- **Claude Desktop and the Cursor CLI** are reported to cut a tool call off at about 60 seconds, with no setting to
  change it. A first search that waits for the model takes about half a minute, so it should fit.
- **Cursor and Devin CLI:** variables that are set when `setup` runs go in as `${env:NAME}`. An app started from a dock
  may not have them; a token it can't expand falls back to `gh`.
- **VS Code:** the entry goes in the default profile's `mcp.json`.
- **Gemini CLI:** variables go in as `$NAME`. Running `gemini mcp add` yourself later rewrites its settings with those
  values expanded, which is why `setup` edits the file instead.
- **Zed:** the entry gets a 300-second timeout, up from its default of 60.
- **OpenCode:** the entry uses the older flat form, which both OpenCode V1 and V2 read.

</details>

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

<details>
<summary><b>Any other client</b></summary>

Most clients take a JSON entry in the portable `mcpServers` format:

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
  question, so the MCP `search` tool does that by default (`--distinct` on the CLI).
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
