SEARCH = (
    "Search the user's own engineering history on GitHub (PRs and their discussions, review threads, issues, commits, "
    "repo docs and local plans) for the passages that best match the query, best first, with permalinks to cite. The "
    "first search after the server starts can wait a minute or more while the embedding model loads."
)
INSTRUCTIONS = (
    "Searches the user's own engineering history on GitHub: pull request descriptions and discussions, review "
    "threads, issues, commits without a PR, and repository markdown (README, AGENTS.md, docs). Use it when the user "
    "asks why or how something was decided, built, fixed or rejected in their repos, or which repo already did "
    "something. Results are passages with GitHub permalinks; cite the permalinks. A passage cut short ends with …; "
    "read returns the rest of its source, which also works for private repos and local plans. A search can wait a "
    "minute or more while the server is still loading the embedding model. list_repos says when the index was built; "
    "when it looks out of date for the question, the refresh tool updates it in the background."
)
