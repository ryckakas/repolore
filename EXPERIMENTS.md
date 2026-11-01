# Experiments

Each retrieval change is measured with `repolore eval` against the same question set, one change at a time:
hypothesis, change, metric deltas, keep or drop. The question set lives outside this repository because it names
private repos, so this file records aggregate numbers only, never question text.

How to read the numbers:

- **recall@k**: of the sources a question expects, the share found in the top k chunks, averaged over questions.
- **MRR**: 1 / rank of the first expected source, averaged.
- **nDCG@10**: rewards finding every expected source near the top; only the first chunk of each source counts.
- Unanswerable questions are left out of these and shown as a mean top score instead, to see whether a score cutoff
  could answer "not found".
- With about 35 questions, one question moves recall by about 3%. Treat changes under 5% as noise.

## Baseline (2026-10-24)

Vector search only, with no reranking and no keyword search.

| Setting | Value |
|---|---|
| Corpus | 11 repos, 945 documents, 2,576 chunks |
| Chunking | a chunk per markdown section, sections over 400 estimated tokens split between blocks |
| Embedder | `voyage-4-nano` on the CPU, float32, 1024 dimensions |
| Questions | tuning set: 35 questions from public repos (32 answerable, 3 unanswerable) |

| Metric | Value |
|---|---|
| recall@5 | 0.818 |
| recall@10 | 0.919 |
| MRR | 0.809 |
| nDCG@10 | 0.804 |

| Question kind | recall@5 | Questions |
|---|---|---|
| rationale | 0.950 | 10 |
| factual | 0.844 | 15 |
| commit-only | 0.750 | 4 |
| cross-repo | 0.333 | 3 |

Mean top score: 0.649 on answerable questions, 0.480 on unanswerable ones.

Observations:

- Single-source questions mostly land at rank 1. Seven of the nine misses expect several sources: one is found
  near the top and another ranks lower (6th to beyond 10th).
- Cross-repo questions are the weakest. A question like "which repos use X" matches the tool's own README and
  docs better than the PRs that adopted it in other repos.
- Some misses retrieve a source that also answers the question but isn't listed as expected (a `SECURITY.md`
  rather than the PR that added it). The expected sources were written before any retrieval existed and are
  incomplete, which caps the scores. Fixing labels is a decision about the question set, not about retrieval.
- Top scores overlap: unanswerable questions reach up to 0.51, and 3 of the 32 answerable ones score below that. A
  score cutoff alone would turn away some answerable questions.

## Experiments (2026-10-25)

All runs use the same 35 tuning questions and compare with the baseline above (vector search, 400-token chunks,
voyage-4-nano). Changes under about 0.05 are noise at this size.

| # | Change | recall@5 | recall@10 | MRR | nDCG@10 | Verdict |
|---|---|---|---|---|---|---|
| 0 | Baseline: vector | 0.818 | 0.919 | 0.809 | 0.804 | |
| 1 | Keyword only (FTS5 BM25) | 0.719 | 0.844 | 0.753 | 0.741 | drop |
| 2 | Hybrid: vector + BM25, reciprocal rank fusion (k=60) | 0.833 | 0.919 | 0.774 | 0.782 | drop |
| 3a | Vector, then a cross-encoder reranks the top 50 | 0.841 | 0.888 | 0.758 | 0.770 | drop for ranking, keep for "not found" (see below) |
| 3b | Hybrid, then the cross-encoder reranks the top 50 | 0.841 | 0.888 | 0.758 | 0.770 | drop |
| 7 | Index every commit, not only those without a PR (+396 chunks) | 0.818 | 0.919 | 0.746 | 0.756 | drop |
| 4a | Chunks of at most 200 tokens (3,194 chunks) | 0.833 | 0.919 | 0.802 | 0.799 | drop |
| 4b | Chunks of at most 800 tokens (2,378 chunks) | 0.865 | 0.919 | 0.810 | 0.806 | |
| 4c | Whole sections, never split (2,316 chunks) | 0.865 | 0.919 | 0.814 | 0.810 | keep |
| 5 | bge-small-en-v1.5 instead of voyage-4-nano (400-token chunks) | 0.620 | 0.755 | 0.525 | 0.547 | drop |

**1. Keyword only.** Hypothesis: exact names (`zizmor`, `runAsNonRoot`, `hasPreviousPage`) are where vector search
fails. Result: worse on every metric. Questions are phrased in plain words, and BM25 over-rewards chunks that repeat
a common word. Commit-only questions fall from 0.75 to 0.50: commit messages are short, so they match few words.

**2. Hybrid.** Hypothesis: fusing the two rankings keeps vector search's hits and adds the exact-name matches.
Result: recall@5 +0.015 (noise), MRR -0.035. Keyword search mostly adds chunks that vector search already ranks
lower, and the fusion pushes some first-place hits down a rank. The rationale questions all reach the top 5
(1.000), which is the one consistent gain.

**3. Reranking** with `cross-encoder/ms-marco-MiniLM-L6-v2` (local, about 2 s per question for 50 candidates).
Hypothesis: reading query and passage together ranks better than comparing vectors. Result: recall@5 +0.023,
but recall@10 and MRR drop. Likely causes: the model was trained on web search questions, and it reads only the
first 256 tokens of each passage, which cuts off much of a long PR section. Reranking hybrid candidates gives the same numbers, so the top 50 already
contain the same passages either way.

What reranking does well is telling "found" from "not found". Its scores are comparable across questions, unlike
cosine scores:

| Top score | lowest answerable | highest unanswerable | answerable at or below that |
|---|---|---|---|
| vector (cosine) | 0.480 | 0.514 | 3 of 32 |
| keyword (BM25) | 10.6 | 14.1 | 2 of 32 |
| reranked (cross-encoder logit) | 0.35 | 2.09 | 1 of 32 |

With only 3 unanswerable questions this is a hint, not a result. It's the most promising lever for answering "I
don't know" instead of returning the nearest unrelated passages.

**7. All commits.** Hypothesis: commit messages add detail their squash-merged PRs leave out. Result: recall holds,
MRR falls by 0.063. The extra commits mostly restate their PR and push the PR's own passages down. This confirms
the rule of indexing only commits without a PR.

**4. Chunk size.** Hypothesis: the 400-token cap cuts long sections into pieces that lose their context, while
smaller chunks match narrow questions more precisely. Result: a steady trend toward bigger chunks: 200 tokens is
lowest, whole sections highest on every metric. The gain over 400 tokens is small (recall@5 +0.047, at the noise
line) but points the same way at 800 and whole sections. voyage-4-nano reads 32K tokens, so whole sections cost
nothing in truncation. Long sections are slow to embed on a CPU (the 41 not already cached took 10 minutes), but
only when they change. Keep whole sections.

*Later note (2026-10-31):* the caps in this experiment were estimated tokens (4 characters each), not the model's.
The truncation claim held for this corpus but not for a larger one, where 2 whole sections passed 32K tokens and
lost their tails (#17). Since then a chunk holds at most 8,192 real tokens by default (see "Ceiling" below).

**5. A smaller embedding model.** Hypothesis: `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions, a 512-token
limit) is close enough to voyage-4-nano (340M, 1024 dimensions) to be worth its speed. Result: far worse on every
metric (recall@5 -0.198, MRR -0.284); commit-only questions drop to 0.25. It embedded the whole corpus in 6.5 minutes
on 4 threads, about 15 times faster, which doesn't make up for losing a third of the first-place hits. Drop.

*Later note (2026-10-31):* this ran on 400-token chunks. A whole-section bge-small build would have been cut at its
512-token limit without a word (7.1% of chunks on the 2026-10-31 corpus); chunks are now kept inside the model's
window, so such a build splits those sections instead.

**6. Contextual retrieval** (a model-written sentence of context per chunk) was skipped: it needs a model API key.

## Decision (2026-10-25)

The default index is now **vector search over whole sections with voyage-4-nano**: the best or tied-best result on
every metric. Keyword search, hybrid fusion and reranking stay available (`--mode`, `--rerank`) but are off by
default, and only commits without a PR are indexed.

| Default config, tuning set (35) | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Baseline (400-token chunks) | 0.818 | 0.919 | 0.809 | 0.804 |
| Now (whole sections) | 0.865 | 0.919 | 0.814 | 0.810 |

**Held-out check**, run once on the final config: 5 questions, 4 answerable.

| Held-out set | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Whole sections, vector | 0.750 | 1.000 | 0.653 | 0.733 |

Every expected source is within the top 10. The one top-5 miss follows the tuning-set pattern: a doc section that
answers the question well outranks the commit listed as its source. Four questions can't confirm or refute the
tuning numbers; they show nothing broke.

**Search time**, measured later by `repolore eval` on the default index, which had grown to 2,587 chunks once
repo discovery added more repos: a median of 308 ms per question on a desktop CPU, query embedding included. The first search of a session is slower, because it loads the
model.

**Ceiling (2026-10-31).** Chunks now hold at most 8,192 tokens by default, counted with the model's tokenizer (#17).
On this corpus (now 1,431 documents, 3,309 chunks, the largest 2,903 tokens) no section reaches that, so the chunks
came out identical to whole sections, checked text by text, and nothing was re-embedded. The eval on that index:
recall@5 0.865, recall@10 0.919, MRR 0.814, nDCG@10 0.809, against 0.810 on the previous, smaller index; the
difference comes from the corpus having grown, not from the ceiling. Search took a median of 195 ms over 3,309
chunks. The ceiling's effect on very long sections is still unmeasured here: it needs a corpus that has them.

What would move the numbers next:

- **Labels.** Several "misses" are valid answers that aren't listed as expected sources. Completing the labels would
  measure retrieval more fairly than any change tried here.
- **Cross-repo questions** (0.333) need a different approach than ranking passages, for example grouping results by
  repo or a dedicated "which repos" query.
- **Not found.** The reranker's scores separate answerable from unanswerable questions; a cutoff on them is the
  most promising way to answer "nothing indexed covers this".

## Distinct sources (2026-11-01)

Hypothesis (#19): every section of a document is its own chunk, so a document that matches well fills several of
the top k slots, and other sources fall out. Keeping only the best-ranked chunk of each source URL (`--distinct`,
which fetches 50 candidates first) should cover more sources in the same k.

On the default index (3,574 chunks, vector search, whole sections), plain search returned 3.29 distinct sources in
the top 5 on average and 6.26 in the top 10; 26 of the 35 questions had a repeat in their top 5.

| Tuning set (35) | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Vector | 0.857 | 0.911 | 0.814 | 0.805 |
| Vector, one chunk per source | 0.904 | 0.935 | 0.818 | 0.822 |

| Held-out set (5) | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Vector | 0.750 | 1.000 | 0.653 | 0.733 |
| Vector, one chunk per source | 1.000 | 1.000 | 0.688 | 0.765 |

Six tuning questions ranked an expected source higher and none ranked one lower. Four of the six expect several
sources, and another of their sources moved up once the first one's other sections stopped taking slots (from rank
7 to 4, 3 to 2, beyond 10 to 6, and beyond 10 to 10). The other two moved up a rank or two as other documents'
repeats above them went. The gain in recall@5 (+0.047) is at the noise line, but it points the same way on every
metric and on the held-out set.

The eval only partly sees the cost. A question that needs a source's second-best section loses it, but only 1 of
the 43 expected sources names a section, so the set barely tests that. A review thread has its own URL, so it
still counts apart from its PR.

Decision: the MCP `search` tool returns one chunk per source by default, and Claude Code can pass
`distinct: false` for a document's other sections. The CLI keeps it as the `--distinct` option, so `search` and
`eval` without it keep measuring plain ranking.

## Headline-only commits (2026-11-01)

A PR without a description is indexed from its commits. Each commit was a `###` heading over its message, and a
heading with nothing under it gets no chunk, so every commit with only a headline was left out (#80): 140 of the
151 commits in the 99 PRs indexed this way. Those commits aren't indexed on their own either, because they belong to
a PR. The commits are now a list under `## Commits`, one `- **<headline>**` item each with its message body indented
under it, so every headline reaches a chunk. Each of the 99 PRs now has one `Commits` chunk, where they had 11 commit
chunks between them before; the index grew from 3,644 to 3,732 chunks, and 99 were embedded.

| Tuning set (35) | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Vector, before | 0.857 | 0.911 | 0.814 | 0.805 |
| Vector, after | 0.826 | 0.911 | 0.812 | 0.803 |
| Vector, one chunk per source, before | 0.904 | 0.935 | 0.818 | 0.822 |
| Vector, one chunk per source, after | 0.904 | 0.935 | 0.816 | 0.819 |

The held-out set (5) didn't change: recall@5 0.750 plain and 1.000 with one chunk per source.

Two questions each ranked one expected source a place lower, and nothing else moved. One of them fell from 5 to 6,
which is the whole recall@5 change (1 of 32 answerable questions). In both, what moved ahead was a newly indexed
commit list on the question's topic, not a labelled source. The eval can't show the gain, because no expected source
is a commit that was missing: those couldn't be found to be labelled. The question that found the bug shows it: a
keyword search now ranks the PR holding the commit 4th, where before no search could return that commit at all.

Decision: keep the list. A one-rank loss to on-topic chunks is the labelling gap noted above, not worse retrieval,
and a commit message no search can reach is a correctness bug.

## Parent headings in chunk headers (2026-11-01)

Hypothesis (#79): a chunk's header named only its own heading, so a nested section lost what its parents said. In
a CHANGELOG, every release's `### Changed` had the same header, and none said which release or date it belonged to.
Headers now carry the heading path, the nearest heading above at each shallower level:
`owner/repo · CHANGELOG.md · Changelog > [0.3.0] - 2026-09-24 > Changed`. The `section` stays the leaf heading.

1,147 of the 3,732 chunks have parent headings (826 doc, 268 plan, 31 PR and 22 issue chunks; 777 one level deep,
356 two, 14 three), and the build re-embedded 1,129 of them. The longest headers are still long commit titles, so
ancestor headings aren't truncated. The `Discussion` section appended to PRs and issues is now a top-level heading,
so a body with `#` sections can't make the comments read as part of its last one; that re-embedded 40 chunks and
changed no rank.

| Tuning set (35) | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Vector, before | 0.826 | 0.911 | 0.812 | 0.803 |
| Vector, after | 0.826 | 0.919 | 0.822 | 0.815 |
| Vector, one chunk per source, before | 0.904 | 0.935 | 0.816 | 0.819 |
| Vector, one chunk per source, after | 0.904 | 0.945 | 0.827 | 0.834 |

| Held-out set (5) | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| Vector, before | 0.750 | 1.000 | 0.653 | 0.733 |
| Vector, after | 0.750 | 1.000 | 0.656 | 0.737 |

With one chunk per source, five tuning questions ranked an expected source higher and one ranked one lower (2 to 3);
in plain vector search, four higher and two lower, each by one or two ranks. One held-out question moved up a rank.
The gains are small, but every metric moved up or held on both sets.

The eval set has no question that needs a version or a date, so it can't see the case the change is for. Five
release questions about bonsai-lint's CHANGELOG (a public repo), searched before and after:

- Asking what changed in 0.3.0, the top CHANGELOG hits were 0.4.0's and 0.2.1's `Changed`; now 0.3.0's `Changed` and
  `Fixed` rank 4th and 7th, and 0.2.1's falls below them.
- Asking what 0.5.0 fixed, 0.5.0's `Fixed` rose from 7th to 1st.
- Asking which release made one of two changes, the right `Changed` ranked first before and after both times, but
  only now does its header say it is 0.3.0, from 2026-09-24.
- Asking what 0.4.1 added found 0.4.0's `Changed` before and after, so a version number alone doesn't always win.

Decision: keep the heading path. A CHANGELOG or version question is worth adding to the eval set to measure this
case directly.
