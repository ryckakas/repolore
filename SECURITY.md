# Security policy

## Supported versions

Only the latest release gets security fixes. repolore is pre-1.0, so a fix ships as a new release
rather than a backport; upgrade to pick it up.

## Reporting a vulnerability

Report it privately through GitHub:
**[Report a vulnerability](https://github.com/ryckakas/repolore/security/advisories/new)**.
Please don't open a public issue, pull request or discussion for it.

A useful report says which version you ran (`repolore version` prints it), how you installed it,
the command or MCP tool call, and what happened. The report stays private while it is
investigated and fixed. Once a fixed release is out, the advisory is published with credit to
you, unless you'd rather not be named.

## What counts

repolore reads a user's GitHub history with their own token, stores it in a local SQLite database,
and embeds it locally, or through Voyage's API if the user configured it. Anything that breaks
those boundaries is in scope, for example:

- A GitHub token, API key or other credential being logged, cached in the data directory, written
  into the index, or sent anywhere other than the API it belongs to.
- Indexed content (PR bodies, issues, commits, docs) reaching anything other than the configured
  embedding API or the local MCP client.
- Content from a repository the user can no longer access staying searchable after an ingest
  that should have removed it.
- Crafted repository content (a PR body, a markdown file) causing code execution, or a read or
  write outside the directories repolore is configured to use.
- A problem in a release artifact or the way it is built and published.

Prompt injection through indexed content is a known limitation of any RAG tool: retrieved text is
data, and the MCP server returns it to the client without acting on it. A way to make repolore
itself act on injected instructions is in scope.

A vulnerability in a dependency that repolore doesn't actually reach is better raised as a normal
issue, or left to Dependabot, which already watches the dependencies.
