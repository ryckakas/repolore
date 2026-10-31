import hashlib
import io
import logging
import random
import sys
import tomllib
from collections import Counter
from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from repolore import __version__
from repolore.chunking import Chunk, chunk_document, chunk_limit, count_split_sections
from repolore.config import Config, load_config
from repolore.discovery import discover
from repolore.documents import Document, DocumentKind
from repolore.embedding.local import MODELS, LocalEmbedder, ModelName, UnknownModelError, embedder_for
from repolore.evals.harness import (
    METRICS,
    SEARCH_DEPTH,
    QuestionResult,
    evaluate,
    mean_top_score,
    median_search_ms,
    recall_by_kind,
)
from repolore.evals.questions import QuestionKind, QuestionSet, load_questions
from repolore.evals.runs import previous_run, run_record, save_run
from repolore.hub import ModelUnavailableError
from repolore.indexing import BuildSettings, build_index
from repolore.ingest import cached_documents
from repolore.ingest import ingest as ingest_repos
from repolore.paths import AppPaths, resolve_paths
from repolore.reranking import CrossEncoderReranker
from repolore.retrieval import Mode
from repolore.retrieval import search as search_index
from repolore.sources.github import GitHubClient, GitHubError, resolve_token
from repolore.store import (
    EmbeddingCache,
    Filters,
    StoreError,
    changed_settings,
    connect,
    connect_read_only,
    index_model,
    index_settings,
    indexed_urls,
    repo_counts,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """RAG over your own engineering history."""
    _use_utf8_output()
    ctx.obj = resolve_paths(data_dir)


def _use_utf8_output() -> None:
    # Redirected output on Windows defaults to cp1252, which can't encode emoji and most non-Latin GitHub content.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _log_progress() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def version(ctx: typer.Context) -> None:
    """Print the version and the resolved config and data paths."""
    paths: AppPaths = ctx.obj
    typer.echo(f"repolore {__version__}")
    typer.echo(f"config: {paths.config_file}")
    typer.echo(f"data:   {paths.data_dir}")


@app.command()
def questions(
    ctx: typer.Context,
    file: Annotated[
        Path | None, typer.Option(help="Question file to check instead of the one in the config dir.")
    ] = None,
) -> None:
    """Validate the eval question set and print how it is made up."""
    paths: AppPaths = ctx.obj
    path = file or paths.questions_file
    question_set = _questions_or_exit(path)
    kinds = Counter(q.kind for q in question_set.questions)
    held_out = sum(q.holdout for q in question_set.questions)
    typer.echo(f"questions: {path}")
    typer.echo(f"total:     {len(question_set.questions)} ({held_out} held out)")
    for kind in QuestionKind:
        typer.echo(f"  {kind:<13} {kinds[kind]}")


@app.command()
def repos(
    ctx: typer.Context,
    excluded: Annotated[bool, typer.Option(help="Also list the repos repolore.toml excludes.")] = False,
) -> None:
    """List the repos ingest would fetch and why: owned, member, contributed to, or local plans only."""
    paths: AppPaths = ctx.obj
    config = _config_or_exit(paths.config_file)
    with _github() as client:
        discovery = discover(client, config)
    rows = [(source.name, source.scope.value, source.checkout) for source in discovery.included]
    if excluded:
        rows += [(source.name, "excluded", None) for source in discovery.excluded]
    width = max((len(name) for name, _, _ in rows), default=0)
    for name, reason, checkout in rows:
        plans = f"  + plans in {checkout}" if checkout else ""
        typer.echo(f"{name:<{width}}  {reason:<11}{plans}".rstrip())
    count = len(discovery.included)
    typer.echo(f"{count} repo{'' if count == 1 else 's'} for {discovery.login}")
    if discovery.excluded:
        typer.echo(f"{len(discovery.excluded)} excluded by {paths.config_file}")


@app.command()
def ingest(
    ctx: typer.Context,
    offline: Annotated[bool, typer.Option(help="Rebuild documents from the raw cache without calling GitHub.")] = False,
    full: Annotated[bool, typer.Option(help="Fetch everything again instead of only what changed.")] = False,
) -> None:
    """Fetch the discovered repos from GitHub into the raw cache and count the documents they yield."""
    _log_progress()
    paths: AppPaths = ctx.obj
    if offline:
        documents = _cached_documents(paths)
    else:
        config = _config_or_exit(paths.config_file)
        with _github() as client:
            documents = ingest_repos(discover(client, config), paths.raw_dir, client, full)
    _print_counts(documents)


@contextmanager
def _exit_on(*errors: type[Exception]) -> Generator[None]:
    try:
        yield
    except errors as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None


@contextmanager
def _github() -> Generator[GitHubClient]:
    with _exit_on(GitHubError), closing(GitHubClient.connect(resolve_token())) as client:
        yield client


def _cached_documents(paths: AppPaths, include_bot_reviews: bool = False) -> list[Document]:
    documents = cached_documents(paths.raw_dir, include_bot_reviews)
    if not documents:
        typer.echo(f"Nothing cached in {paths.raw_dir}; run `repolore ingest` first.", err=True)
        raise typer.Exit(1)
    return documents


def _config_or_exit(path: Path) -> Config:
    try:
        return load_config(path)
    except (tomllib.TOMLDecodeError, ValidationError) as error:
        typer.echo(f"{path} is invalid:\n{error}", err=True)
        raise typer.Exit(1) from None


def _print_counts(documents: list[Document]) -> None:
    repos = sorted({document.repo for document in documents}, key=str.lower)
    counts = Counter((document.repo.lower(), document.kind) for document in documents)
    width = max((len(repo) for repo in repos), default=len("total"))
    typer.echo(f"{'repo':<{width}}" + "".join(f"{kind:>8}" for kind in DocumentKind))
    for repo in repos:
        typer.echo(f"{repo:<{width}}" + "".join(f"{counts[repo.lower(), kind]:>8}" for kind in DocumentKind))
    totals = Counter(document.kind for document in documents)
    typer.echo(f"{'total':<{width}}" + "".join(f"{totals[kind]:>8}" for kind in DocumentKind))


TOKEN_BUCKETS = (50, 100, 200, 400, 800, 2048, 8192)
BAR_WIDTH = 40
SAMPLE_CHARS = 600


@app.command()
def inspect(
    ctx: typer.Context,
    samples: Annotated[int, typer.Option(help="How many random chunks to print.")] = 3,
    seed: Annotated[int | None, typer.Option(help="Seed for picking the samples, to see the same ones again.")] = None,
) -> None:
    """Chunk the cached documents as a default build would, and show counts, a token histogram and random samples."""
    documents = _cached_documents(ctx.obj)
    model = ModelName.VOYAGE_NANO
    embedder = LocalEmbedder(MODELS[model])
    limit = chunk_limit(None, embedder.max_tokens)
    with _exit_on(ModelUnavailableError):
        chunks = [chunk for document in documents for chunk in chunk_document(document, limit, embedder.count_tokens)]
        tokens = embedder.count_tokens([chunk.embedded_text for chunk in chunks])
    typer.echo(f"{len(documents)} documents, {len(chunks)} chunks")
    kinds = Counter(chunk.document.kind for chunk in chunks)
    for kind in DocumentKind:
        typer.echo(f"  {kind:<8} {kinds[kind]:>6}")
    _print_histogram(tokens, model)
    typer.echo(_limit_note(limit, chunks))
    for index in random.Random(seed).sample(range(len(chunks)), min(samples, len(chunks))):
        chunk = chunks[index]
        typer.echo(f"\n--- {chunk.id} ({tokens[index]} tokens)\n{chunk.embedded_text[:SAMPLE_CHARS]}")


def _limit_note(limit: int, chunks: list[Chunk]) -> str:
    return f"Chunks hold at most {limit} tokens; {count_split_sections(chunks)} sections were split to fit."


def _print_histogram(tokens: list[int], model: ModelName) -> None:
    counts = Counter(next((bucket for bucket in TOKEN_BUCKETS if count <= bucket), None) for count in tokens)
    largest = max(counts.values(), default=1)
    typer.echo(f"tokens ({model})")
    lower = 0
    for bucket in [*TOKEN_BUCKETS, None]:
        label = f"{lower + 1}-{bucket}" if bucket else f">{lower}"
        typer.echo(f"  {label:>9} {counts[bucket]:>6} {'#' * round(BAR_WIDTH * counts[bucket] / largest)}")
        lower = bucket or lower


DEFAULT_INDEX = "default"
SNIPPET_CHARS = 300
IndexOption = Annotated[str, typer.Option("--db", help="Name of the index to use, one per pipeline config.")]
ModeOption = Annotated[Mode, typer.Option(help="Vector (meaning), keyword (BM25) or hybrid (both, fused by rank).")]
RerankOption = Annotated[bool, typer.Option(help="Rerank the top 50 candidates with a local cross-encoder.")]
SEARCH_ERRORS = (StoreError, UnknownModelError, ModelUnavailableError)


@app.command()
def build(
    ctx: typer.Context,
    db: IndexOption = DEFAULT_INDEX,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Split sections longer than this many tokens; 0 goes back to the default, 8192 or the model's window.",
        ),
    ] = None,
    embedder: Annotated[ModelName | None, typer.Option(help="Local embedding model, voyage-4-nano at first.")] = None,
    bot_reviews: Annotated[
        bool | None, typer.Option(help="Index review threads started by bots such as CodeRabbit or Copilot.")
    ] = None,
) -> None:
    """Chunk the cached documents, embed what changed and write the search index.

    An option left out keeps the value the index was last built with.
    """
    _log_progress()
    paths: AppPaths = ctx.obj
    index_file = paths.index_file(db)
    settings = _recorded_settings(index_file).overridden(max_tokens, bot_reviews, embedder)
    model = LocalEmbedder(MODELS[settings.model])
    limit = chunk_limit(settings.max_tokens, model.max_tokens)
    typer.echo(f"Building with {settings.described(limit)}.")
    documents = _cached_documents(paths, settings.bot_reviews)
    with _exit_on(ModelUnavailableError):
        chunks = [chunk for document in documents for chunk in chunk_document(document, limit, model.count_tokens)]
    typer.echo(_limit_note(limit, chunks))
    with (
        _exit_on(ModelUnavailableError),
        closing(connect(index_file)) as index,
        closing(connect(paths.embedding_cache)) as cache,
    ):
        changes = changed_settings(index, settings.meta(limit) | {"model_id": model.model_id})
        if changes:
            details = ", ".join(f"{key}: {old} -> {new}" for key, (old, new) in changes.items())
            typer.echo(f"Warning: build settings changed ({details}); chunks may be re-embedded.", err=True)
        stats = build_index(chunks, model, index, EmbeddingCache(cache), settings.meta(limit))
    typer.echo(
        f"{stats.chunks} chunks indexed in {index_file}, {stats.embedded} embedded, "
        f"{stats.chunks - stats.embedded} reused"
    )


def _recorded_settings(index_file: Path) -> BuildSettings:
    if not index_file.exists():
        return BuildSettings()
    with closing(connect_read_only(index_file)) as index:
        try:
            return BuildSettings.recorded(index_settings(index))
        except StoreError:
            return BuildSettings()


@app.command()
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="What to look for, in plain words.")],
    k: Annotated[int, typer.Option("-k", min=1, help="How many chunks to return.")] = 5,
    repo: Annotated[str | None, typer.Option(help="Only this repo, as owner/name.")] = None,
    kind: Annotated[DocumentKind | None, typer.Option(help="Only this kind of source.")] = None,
    author: Annotated[str | None, typer.Option(help="Only this author's GitHub login.")] = None,
    db: IndexOption = DEFAULT_INDEX,
    mode: ModeOption = Mode.VECTOR,
    rerank: RerankOption = False,
    distinct: Annotated[bool, typer.Option(help="Return at most one chunk per source URL.")] = False,
) -> None:
    """Find the chunks that best match the query."""
    paths: AppPaths = ctx.obj
    index_file = _index_or_exit(paths, db)
    filters = Filters(repo=repo, kind=kind, author=author)
    with _exit_on(*SEARCH_ERRORS), closing(connect(index_file)) as index:
        reranker = CrossEncoderReranker() if rerank else None
        embedder = embedder_for(index_model(index)[0])
        hits = search_index(index, embedder, query, k, filters, mode, reranker, distinct=distinct)
    if not hits:
        typer.echo("No matches.")
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.text.split())[:SNIPPET_CHARS]
        typer.echo(f"{rank}. {hit.score:.3f}  {hit.header}\n   {hit.url}\n   {snippet}\n")


MISS_DEPTH = 5
MISS_HITS = 3


@app.command("eval")
def eval_(
    ctx: typer.Context,
    db: IndexOption = DEFAULT_INDEX,
    holdout: Annotated[bool, typer.Option(help="Score the held-out questions instead of the tuning set.")] = False,
    show_misses: Annotated[bool, typer.Option(help="List questions whose sources aren't all in the top 5.")] = False,
    mode: ModeOption = Mode.VECTOR,
    rerank: RerankOption = False,
    file: Annotated[
        Path | None, typer.Option(help="Question file to use instead of the one in the config dir.")
    ] = None,
) -> None:
    """Score retrieval on the eval questions, save the run and compare it with the previous one."""
    paths: AppPaths = ctx.obj
    questions_file = file or paths.questions_file
    question_set = _questions_or_exit(questions_file)
    questions = [question for question in question_set.questions if question.holdout is holdout]
    index_file = _index_or_exit(paths, db)
    with _exit_on(*SEARCH_ERRORS), closing(connect(index_file)) as index:
        settings = index_settings(index)
        embedder = embedder_for(settings["model_id"])
        missing = {source.url for q in questions for source in q.expected} - indexed_urls(index)
        chunks = sum(sum(kinds.values()) for kinds in repo_counts(index).values())
        reranker = CrossEncoderReranker() if rerank else None
        results = evaluate(
            questions, lambda text: search_index(index, embedder, text, SEARCH_DEPTH, Filters(), mode, reranker)
        )
    fingerprint = f"{hashlib.sha256(questions_file.read_bytes()).hexdigest()[:12]}:{'holdout' if holdout else 'tuning'}"
    config = {
        "index": db,
        "model_id": settings["model_id"],
        "max_tokens": settings.get("max_tokens"),
        "bot_reviews": settings.get("bot_reviews", "excluded"),
        "chunks": chunks,
        "retrieval": f"{mode}+rerank" if rerank else str(mode),
        "k": SEARCH_DEPTH,
        "questions": fingerprint,
    }
    runs_dir = paths.data_dir / "evals" / "runs"
    previous = previous_run(runs_dir, fingerprint)
    run = run_record(config, results)
    typer.echo(
        f"{len(questions)} {'held-out' if holdout else 'tuning'} questions on index {db} "
        f"({config['model_id']}, {config['retrieval']})"
    )
    for url in sorted(missing):
        typer.echo(f"  not in the index: {url}")
    _print_metrics(run["metrics"], previous["metrics"] if previous else None)
    _print_breakdown(results)
    if (search_ms := median_search_ms(results)) is not None:
        typer.echo(f"search: median {search_ms:.0f} ms per query over {chunks:,} chunks, query embedding included")
    if show_misses:
        _print_misses(results)
    typer.echo(f"run saved to {save_run(runs_dir, run)}")


def _questions_or_exit(path: Path) -> QuestionSet:
    try:
        return load_questions(path)
    except FileNotFoundError:
        typer.echo(f"No question file at {path}. Start from evals/questions.example.yaml.", err=True)
    except (yaml.YAMLError, ValidationError) as error:
        typer.echo(f"{path} is invalid:\n{error}", err=True)
    raise typer.Exit(1)


def _index_or_exit(paths: AppPaths, db: str) -> Path:
    index_file = paths.index_file(db)
    if not index_file.exists():
        typer.echo(f"No index at {index_file}; run `repolore build` first.", err=True)
        raise typer.Exit(1)
    return index_file


def _print_metrics(metrics: dict[str, float], previous: dict[str, float] | None) -> None:
    typer.echo(f"{'':<10} {'this run':>9} {'previous':>9} {'change':>8}")
    for name in METRICS:
        value = metrics.get(name, 0.0)
        before = previous.get(name) if previous else None
        change = f"{value - before:+8.3f}" if before is not None else f"{'':>8}"
        shown = f"{before:9.3f}" if before is not None else f"{'':>9}"
        typer.echo(f"{name:<10} {value:9.3f} {shown} {change}".rstrip())


def _print_breakdown(results: list[QuestionResult]) -> None:
    typer.echo("recall@5 by kind")
    for kind, (recall, count) in sorted(recall_by_kind(results).items()):
        typer.echo(f"  {kind:<13} {recall:.3f}  ({count})")
    answerable, unanswerable = mean_top_score(results, True), mean_top_score(results, False)
    if answerable is not None and unanswerable is not None:
        typer.echo(f"top score: {answerable:.3f} on answerable, {unanswerable:.3f} on unanswerable questions")


def _print_misses(results: list[QuestionResult]) -> None:
    for result in (result for result in results if result.missed):
        typer.echo(f"\nmiss: {result.question.id}: {result.question.question}")
        for source, rank in zip(result.question.expected, result.ranks, strict=True):
            typer.echo(f"  expected {source.url} (rank {rank or f'>{SEARCH_DEPTH}'})")
        for hit in result.hits[:MISS_HITS]:
            typer.echo(f"  got      {hit.url} ({hit.score:.3f}, {hit.section or '-'})")
