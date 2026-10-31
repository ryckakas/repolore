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

**5. A smaller embedding model.** Hypothesis: `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions, a 512-token
limit) is close enough to voyage-4-nano (340M, 1024 dimensions) to be worth its speed. Result: far worse on every
metric (recall@5 -0.198, MRR -0.284); commit-only questions drop to 0.25. It embedded the whole corpus in 6.5 minutes
on 4 threads, about 15 times faster, which doesn't make up for losing a third of the first-place hits. Drop.

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

What would move the numbers next:

- **Labels.** Several "misses" are valid answers that aren't listed as expected sources. Completing the labels would
  measure retrieval more fairly than any change tried here.
- **Cross-repo questions** (0.333) need a different approach than ranking passages, for example grouping results by
  repo or a dedicated "which repos" query.
- **Not found.** The reranker's scores separate answerable from unanswerable questions; a cutoff on them is the
  most promising way to answer "nothing indexed covers this".
