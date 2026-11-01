# repolore

RAG over your own engineering history on GitHub: PRs, review threads, issues, commits and docs, searchable from a
CLI and from Claude Code over MCP. It answers questions like "why did we drop the retry middleware?" with the
passages that explain it and links to them.

Everything runs on your machine with your own GitHub token. Embeddings are computed locally, so no model API key is
needed, and nothing leaves your machine except GitHub API calls and the one-time model download from Hugging Face.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a GitHub login. It runs on Linux, Windows
and macOS 14+ on Apple Silicon.

1. Install the `repolore` and `repolore-mcp` commands:

   ```sh
   uv tool install --managed-python --python 3.12 git+https://github.com/ryckakas/repolore
   ```

   `--managed-python` makes uv use its own Python build, which can load the SQLite extension the index needs; some
   system builds, such as python.org's on macOS, can't. If the commands aren't found afterwards, run
   `uv tool update-shell` and open a new terminal. The install takes about 1.2 GB, mostly PyTorch.

2. Give it GitHub access. If you're logged in with the [GitHub CLI](https://cli.github.com/) (`gh auth login`),
   there's nothing to do. Otherwise set `REPOLORE_GITHUB_TOKEN` to a classic token with the `repo` scope, plus
   `read:org` for organization repos. Organizations with SSO also need the token authorized for them.

3. See which repos it found, and narrow them down for a first run:

   ```sh
   repolore repos
   ```

   It finds the repos you own or can read as a collaborator or organization member, plus your own PRs and issues in
   other people's repos. The first build embeds everything locally, which takes a while for a large account, so
   start with a few repos. Create `repolore.toml` at the config path that `repolore version` prints, then run
   `repolore repos` again to check the selection:

   ```toml
   include = ["you/some-repo", "your-org/*"]   # globs over owner/name; every repo when unset
   exclude = ["your-org/old-*"]                # never these
   local_roots = ["~/projects"]                # checkouts in here add their local .plans/ documents
   ```

4. Fetch and index:

   ```sh
   repolore ingest   # fetch from GitHub; later runs fetch only what changed
   repolore build    # embed and index; later runs embed only what changed
   ```

   The first build downloads the embedding model, about 700 MB. As a rough guide, a small repo of 149 chunks took
   4.5 minutes on a 12-core desktop CPU, download included. On Apple Silicon, `build` embeds on the GPU, roughly
   twice as fast; set `REPOLORE_DEVICE=cpu` to keep it on the CPU.

5. Search:

   ```sh
   repolore search "why did we drop the retry middleware?"
   repolore search "rate limit handling" --repo you/some-repo --kind pr -k 10
   ```

6. Search from Claude Code by registering the MCP server once:

   ```sh
   claude mcp add --scope user --transport stdio repolore -- repolore-mcp
   ```

   Claude Code then gets `search` and `list_repos` tools and can answer questions about your history with links to
   the sources. The first search in a session loads the model, which takes a little while.

To stay current, run `repolore refresh`: it ingests what changed and rebuilds the index with the settings it was
built with, and only one refresh runs at a time. Search keeps working meanwhile. Claude Code can do the same: the MCP
server's `refresh` tool starts one in the background, and `list_repos` says when the index was built and how the
last refresh went. Claude Code asks before the first call; to allow it for good, add `mcp__repolore__refresh` to
`permissions.allow` in your Claude Code settings. To refresh once at the start of every session instead, add a
`SessionStart` hook to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      { "matcher": "startup", "hooks": [{ "type": "command", "command": "repolore refresh", "async": true }] }
    ]
  }
}
```

`uv tool upgrade repolore` updates repolore itself. The index and the raw GitHub responses live in the data
directory that `repolore version` prints; they contain your private repos' content, so treat that folder like the
repos themselves.

## What the experiments found

Every retrieval choice was measured with `repolore eval` against 35 questions about the author's public repos, each
with the sources expected to answer it. [EXPERIMENTS.md](EXPERIMENTS.md) has the full log. In short:

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
