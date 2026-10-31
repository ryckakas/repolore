from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from repolore.chunking import (
    CEILING,
    Section,
    chunk_document,
    chunk_limit,
    count_split_sections,
    estimate_tokens,
    split_markdown,
)
from repolore.documents import Document, DocumentKind, Visibility

FENCE = "```"
LIMIT = 400


def document(kind: DocumentKind = DocumentKind.PR, body: str = "Body.", **changes: object) -> Document:
    fields: dict[str, object] = {
        "url": "https://github.com/octo/repo/pull/17",
        "repo": "octo/repo",
        "kind": kind,
        "visibility": Visibility.PUBLIC,
        "title": "Add the thing",
        "body": body,
        "author": "octocat",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    return Document.model_validate(fields | changes)


def paragraph(tokens: int, word: str = "word") -> str:
    return " ".join([word] * (tokens * 4 // (len(word) + 1)))


def test_sections_follow_headings() -> None:
    text = "Intro line.\n\n## Problem\n\nIt breaks.\n\n## Change\n\nFix it.\n\nAnd test it."

    assert split_markdown(text) == [
        Section(None, "Intro line."),
        Section("Problem", "## Problem\n\nIt breaks."),
        Section("Change", "## Change\n\nFix it.\n\nAnd test it."),
    ]


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_empty_text_has_no_sections(text: str) -> None:
    assert split_markdown(text) == []


def test_one_heading_is_one_section() -> None:
    assert split_markdown("# Title\n\nSome text.") == [Section("Title", "# Title\n\nSome text.")]


def test_heading_without_content_is_skipped() -> None:
    sections = split_markdown("## Change\n\n### Parser\n\nRewritten.")

    assert [section.heading for section in sections] == ["Parser"]


def test_oversized_section_is_split_between_blocks() -> None:
    blocks = [paragraph(150, word=f"w{n}") for n in range(5)]

    sections = split_markdown("## Notes\n\n" + "\n\n".join(blocks), LIMIT)

    assert len(sections) > 1
    assert {section.heading for section in sections} == {"Notes"}
    assert all(estimate_tokens(section.text) <= LIMIT for section in sections)
    assert "\n\n".join(section.text for section in sections).endswith(blocks[-1])


def test_section_exactly_at_the_limit_stays_whole() -> None:
    text = "a" * (LIMIT * 4)

    assert split_markdown(text, LIMIT) == [Section(None, text)]


def words(texts: Sequence[str]) -> list[int]:
    return [len(text.split()) for text in texts]


def test_a_fence_within_the_limit_is_one_block_and_never_read_as_headings() -> None:
    fence = f"{FENCE}sh\n# not a heading\n" + "\n".join(f"echo {n}" * 20 for n in range(10)) + f"\n{FENCE}"

    sections = split_markdown(f"## Usage\n\nRun:\n\n{fence}\n\nDone.", LIMIT)

    assert [section.heading for section in sections] == ["Usage"]
    assert fence in sections[0].text


def test_a_fence_over_the_limit_is_split_by_lines_into_closed_fences() -> None:
    body = [f"echo {n} " + "arg " * 8 for n in range(40)]
    fence = f"{FENCE}sh\n# not a heading\n" + "\n".join(body) + f"\n{FENCE}"

    sections = split_markdown(f"## Usage\n\n{fence}", 50, words)

    pieces = [section.text.removeprefix("## Usage\n\n") for section in sections]
    assert len(pieces) > 1
    assert {section.heading for section in sections} == {"Usage"}
    assert all(piece.startswith(f"{FENCE}sh\n") and piece.endswith(f"\n{FENCE}") for piece in pieces)
    assert [line for piece in pieces for line in piece.split("\n")[1:-1]] == ["# not a heading", *body]
    assert all(count <= 50 for count in words([section.text for section in sections]))


def test_an_unclosed_fence_over_the_limit_is_closed_in_every_piece() -> None:
    fence = "~~~\n" + "\n".join("line of code here" for _ in range(30))

    sections = split_markdown(fence, 20, words)

    assert len(sections) > 1
    assert all(section.text.startswith("~~~\n") and section.text.endswith("\n~~~") for section in sections)


def test_a_table_over_the_limit_repeats_its_header() -> None:
    rows = [f"| {n} | value number {n} |" for n in range(30)]
    table = "| id | value |\n|---|---|\n" + "\n".join(rows)

    sections = split_markdown(table, 40, words)

    assert len(sections) > 1
    assert all(section.text.startswith("| id | value |\n|---|---|\n") for section in sections)
    assert [line for section in sections for line in section.text.split("\n")[2:]] == rows


def test_a_line_longer_than_the_limit_is_cut_by_characters() -> None:
    line = " ".join(f"w{n}" for n in range(500))

    sections = split_markdown(line, 60, words)

    assert len(sections) > 1
    assert "".join(section.text.replace("\n", "") for section in sections) == line
    assert all(count <= 60 for count in words([section.text for section in sections]))


def test_a_lone_fence_line_over_the_limit_is_cut_not_dropped() -> None:
    line = f"{FENCE}" + " ".join(f"w{n}" for n in range(100))

    sections = split_markdown(line, 30, words)

    assert "".join(section.text.replace("\n", "") for section in sections) == line


def test_a_single_character_over_the_limit_is_kept_not_cut_forever() -> None:
    def three_per_char(texts: Sequence[str]) -> list[int]:
        return [3 * len(text) for text in texts]

    sections = split_markdown("abc", 2, three_per_char)
    chunks = chunk_document(document(body="hello world"), 1, words)

    assert [section.text for section in sections] == ["a", "b", "c"]
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == "helloworld"


def test_a_table_header_too_big_to_repeat_is_split_without_it() -> None:
    header = "| " + " | ".join(f"c{n}" for n in range(12)) + " |"
    rows = [f"| {n} | value {n} |" for n in range(20)]
    table = "\n".join([header, "|---|---|", *rows])

    sections = split_markdown(table, 40, words)

    assert sum(header in section.text for section in sections) == 1
    assert [line for section in sections for line in section.text.split("\n")] == table.split("\n")


def test_a_split_first_block_still_shares_a_chunk_with_its_heading() -> None:
    fence = f"{FENCE}sh\n" + "\n".join("a b c d" for _ in range(40)) + f"\n{FENCE}"

    for limit in (50, 51, 52):
        sections = split_markdown(f"## Usage\n\n{fence}", limit, words)

        assert sections[0].text.startswith(f"## Usage\n\n{FENCE}sh\na b c d")
        assert all(section.text != "## Usage" for section in sections)


def test_a_first_block_that_fits_alone_but_not_beside_its_heading_is_split() -> None:
    paragraph = " ".join(["w"] * 48)

    sections = split_markdown(f"## Usage\n\n{paragraph}\n\n{paragraph}", 49, words)

    assert sections[0].text.startswith("## Usage\n\nw")
    assert all(section.text != "## Usage" for section in sections)
    assert sections[-1].text.endswith(f"\n\n{paragraph}")


def test_a_document_that_fits_is_counted_once() -> None:
    calls: list[int] = []

    def counted(texts: Sequence[str]) -> list[int]:
        calls.append(len(texts))
        return words(texts)

    chunks = chunk_document(document(body="## A\n\nOne.\n\n## B\n\nTwo.\n\n## C\n\nThree."), 100, counted)

    assert len(chunks) == 3
    assert len(calls) == 1


def test_pieces_that_count_longer_once_joined_are_split_again() -> None:
    def seams_cost_more(texts: Sequence[str]) -> list[int]:
        return [len(text.split()) + text.count("\n\n") ** 2 for text in texts]

    blocks = [f"block {n} has five words" for n in range(8)]

    sections = split_markdown("\n\n".join(blocks), 30, seams_cost_more)

    assert [block for section in sections for block in section.text.split("\n\n")] == blocks
    assert all(count <= 30 for count in seams_cost_more([section.text for section in sections]))


def test_the_header_counts_toward_the_limit() -> None:
    body = "\n\n".join(f"para {n} of four" for n in range(5))
    [header] = words(["octo/repo · PR #17 · Add the thing"])

    whole = chunk_document(document(body=body), 20 + header + 1, words)
    split = chunk_document(document(body=body), 20 + header - 1, words)

    assert len(whole) == 1
    assert len(split) == 2
    assert all(count <= 20 + header - 1 for count in words([chunk.embedded_text for chunk in split]))


def test_a_title_longer_than_the_limit_still_leaves_half_of_it_for_the_text() -> None:
    long_title = " ".join(["word"] * 50)

    chunks = chunk_document(document(title=long_title, body=" ".join(["text"] * 20)), 40, words)

    assert [len(chunk.text.split()) for chunk in chunks] == [20]


@pytest.mark.parametrize(
    ("max_tokens", "window", "limit"),
    [(None, 32_752, CEILING), (500, 32_752, 500), (20_000, 32_752, 20_000), (20_000, 496, 496), (None, 496, 496)],
)
def test_the_limit_is_the_ceiling_or_max_tokens_within_the_window(
    max_tokens: int | None, window: int, limit: int
) -> None:
    assert chunk_limit(max_tokens, window) == limit


def test_split_sections_counts_sections_cut_into_several_chunks() -> None:
    chunks = chunk_document(document(body="## A\n\n" + " ".join(["a"] * 60) + "\n\n## B\n\nShort."), 30, words)

    assert len(chunks) > 2
    assert count_split_sections(chunks) == 1
    assert count_split_sections(chunk_document(document(body="## A\n\nShort."))) == 0
    assert count_split_sections(chunk_document(document(body="## A\n\nOne.\n\n## A\n\nTwo."), 30, words)) == 0


def test_sections_stay_whole_by_default() -> None:
    blocks = [paragraph(300, word=f"w{n}") for n in range(3)]

    assert len(split_markdown("## Notes\n\n" + "\n\n".join(blocks))) == 1


def test_table_stays_one_block() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 |"

    assert split_markdown(f"## Data\n\n{table}") == [Section("Data", f"## Data\n\n{table}")]


def test_front_matter_is_kept_as_yaml_not_a_heading() -> None:
    sections = split_markdown("---\nname: review\ndescription: Reviews code\n---\n\n# Review\n\nSteps.")

    assert sections[0] == Section(None, f"{FENCE}yaml\nname: review\ndescription: Reviews code\n{FENCE}")
    assert sections[1].heading == "Review"


def test_windows_line_endings_are_normalized() -> None:
    assert split_markdown("## A\r\n\r\nText.") == [Section("A", "## A\n\nText.")]


def test_pull_request_chunks_carry_a_header_per_section() -> None:
    chunks = chunk_document(document(body="## Problem\n\nSlow.\n\n## Change\n\nFaster."))

    assert [chunk.header for chunk in chunks] == [
        "octo/repo · PR #17 · Add the thing · Problem",
        "octo/repo · PR #17 · Add the thing · Change",
    ]
    assert [chunk.id for chunk in chunks] == [
        "https://github.com/octo/repo/pull/17::0",
        "https://github.com/octo/repo/pull/17::1",
    ]
    assert chunks[0].embedded_text == "octo/repo · PR #17 · Add the thing · Problem\n\n## Problem\n\nSlow."


@pytest.mark.parametrize(
    ("kind", "url", "title", "header"),
    [
        (DocumentKind.ISSUE, "https://github.com/octo/repo/issues/5", "Crash", "octo/repo · issue #5 · Crash"),
        (
            DocumentKind.COMMIT,
            "https://github.com/octo/repo/commit/0123456789abcdef",
            "Fix",
            "octo/repo · commit 0123456 · Fix",
        ),
        (DocumentKind.DOC, "https://github.com/octo/repo/blob/main/docs/a.md", "docs/a.md", "octo/repo · docs/a.md"),
        (DocumentKind.PLAN, "file:///D:/repo/.plans/a/plan.md", ".plans/a/plan.md", "octo/repo · .plans/a/plan.md"),
        (
            DocumentKind.REVIEW,
            "https://github.com/octo/repo/pull/1#discussion_r9",
            "PR #1 · a.go:3",
            "octo/repo · PR #1 · a.go:3",
        ),
    ],
)
def test_header_names_the_source(kind: DocumentKind, url: str, title: str, header: str) -> None:
    chunks = chunk_document(document(kind=kind, url=url, title=title, body="Plain text."))

    assert chunks[0].header == header


def test_document_without_body_is_one_chunk_of_its_title() -> None:
    chunks = chunk_document(document(kind=DocumentKind.COMMIT, url="https://github.com/octo/repo/commit/abc", body=""))

    assert len(chunks) == 1
    assert chunks[0].text == "Add the thing"
    assert chunks[0].section is None


def test_estimate_rounds_up() -> None:
    assert [estimate_tokens(text) for text in ["", "a", "abcd", "abcde"]] == [0, 1, 1, 2]


def test_max_tokens_controls_where_long_sections_split() -> None:
    body = "## Notes\n\n" + "\n\n".join(paragraph(60, word=f"w{n}") for n in range(4))

    assert len(chunk_document(document(body=body))) == 1
    assert len(chunk_document(document(body=body), max_tokens=100)) == 4
