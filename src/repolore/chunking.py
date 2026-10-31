import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict

from repolore.documents import Document, DocumentKind

CHARS_PER_TOKEN = 4
CEILING = 8192
FENCE_OPENING = re.compile(r"^ {0,3}(`{3,}|~{3,})")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
MARKDOWN = MarkdownIt("commonmark").enable("table")
SHORT_SHA = 7


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    document: Document
    section: str | None
    header: str
    text: str
    part: int = 0

    @property
    def embedded_text(self) -> str:
        return f"{self.header}\n\n{self.text}"


@dataclass(frozen=True)
class Section:
    heading: str | None
    text: str


@dataclass(frozen=True)
class _Block:
    text: str
    heading: str | None


TokenCounter = Callable[[Sequence[str]], list[int]]


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English prose; not an upper bound, so it only sizes chunks for tests."""
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate_counts(texts: Sequence[str]) -> list[int]:
    return [estimate_tokens(text) for text in texts]


def chunk_limit(max_tokens: int | None, window: int) -> int:
    return min(max_tokens or CEILING, window)


def chunk_document(
    document: Document, max_tokens: int | None = None, count: TokenCounter = estimate_counts
) -> list[Chunk]:
    """One chunk per section; with max_tokens, no chunk's embedded text (header included) is longer than that,
    unless the header alone takes over half of it."""
    sections = _sections(document.body)
    headers = [_header(document, heading) for heading, _ in sections]
    # Most sections fit whole, and one count per document is far cheaper than counting their blocks one by one.
    texts = [f"{header}\n\n" + "\n\n".join(blocks) for header, (_, blocks) in zip(headers, sections, strict=True)]
    whole = [0] * len(texts) if max_tokens is None else count(texts)
    pieces: list[tuple[str | None, int, str]] = []
    for (heading, blocks), header, tokens in zip(sections, headers, whole, strict=True):
        budget = None if max_tokens is None or tokens <= max_tokens else _text_budget(max_tokens, header, count)
        pieces += [(heading, part, text) for part, text in enumerate(_pack(heading, blocks, budget, count))]
    return [
        Chunk(
            id=f"{document.url}::{ordinal}",
            document=document,
            section=heading,
            header=_header(document, heading),
            text=text,
            part=part,
        )
        for ordinal, (heading, part, text) in enumerate(pieces or [(None, 0, document.title)])
    ]


def split_markdown(text: str, max_tokens: int | None = None, count: TokenCounter = estimate_counts) -> list[Section]:
    """One section per heading; with max_tokens, a longer section is split between, and if need be inside, blocks."""
    return [
        Section(heading, piece)
        for heading, blocks in _sections(text)
        for piece in _pack(heading, blocks, max_tokens, count)
    ]


def count_split_sections(chunks: Sequence[Chunk]) -> int:
    return sum(1 for chunk in chunks if chunk.part == 1)


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    blocks: list[str] = []
    for block in _blocks(_fence_front_matter(text.replace("\r\n", "\n"))):
        if block.heading is not None:
            sections.append((heading, blocks))
            heading, blocks = block.heading, [block.text]
        else:
            blocks.append(block.text)
    return [*sections, (heading, blocks)]


def _fence_front_matter(text: str) -> str:
    return FRONT_MATTER.sub(lambda match: f"```yaml\n{match.group(1)}\n```\n", text, count=1)


def _blocks(text: str) -> Iterator[_Block]:
    lines = text.split("\n")
    tokens = MARKDOWN.parse(text)
    for index, token in enumerate(tokens):
        if token.level != 0 or token.map is None or token.nesting < 0:
            continue
        start, end = token.map
        heading = tokens[index + 1].content.strip() if token.type == "heading_open" else None
        yield _Block(text="\n".join(lines[start:end]).strip(), heading=heading)


def _text_budget(max_tokens: int, header: str, count: TokenCounter) -> int:
    # A header over half the limit lets its chunks pass the limit rather than leave almost no room for text.
    [header_tokens] = count([f"{header}\n\n"])
    return max(max_tokens - header_tokens, max_tokens // 2, 1)


def _pack(heading: str | None, blocks: list[str], budget: int | None, count: TokenCounter) -> list[str]:
    content = blocks[1:] if heading is not None else blocks
    if not any(content):
        return []
    if budget is None:
        return ["\n\n".join(blocks)]
    limits = [budget] * len(blocks)
    if heading is not None and len(blocks) > 1:
        # The first content block shares a chunk with the heading, so it must fit beside it or split to.
        [heading_tokens] = count([f"{blocks[0]}\n\n"])
        limits[1] = max(budget - heading_tokens, budget // 2, 1)
    units = [
        unit
        for block, tokens, limit in zip(blocks, count(blocks), limits, strict=True)
        for unit in ([block] if tokens <= limit else _split_block(block, limit, count))
    ]
    return _joined(units, budget, count, "\n\n")


def _split_block(block: str, budget: int, count: TokenCounter) -> list[str]:
    head, body, tail = _frame(block.split("\n"))
    [frame_tokens] = count(["\n".join([*head, "", *tail])])
    if frame_tokens > budget // 2 or not body:
        head, body, tail, frame_tokens = [], block.split("\n"), [], 0
    line_budget = budget - frame_tokens
    lines = [
        piece
        for line, tokens in zip(body, count(body), strict=True)
        for piece in ([line] if tokens <= line_budget else _cut(line, line_budget, count))
    ]
    return _joined(lines, budget, count, "\n", lambda text: "\n".join([*head, text, *tail]))


def _frame(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """The lines every piece of a split block repeats: a fence's opening and closing, or a table's header."""
    if opening := FENCE_OPENING.match(lines[0]):
        marker = opening.group(1)
        if len(lines) > 1 and lines[-1].strip().startswith(marker):
            return lines[:1], lines[1:-1], lines[-1:]
        return lines[:1], lines[1:], [marker]
    if len(lines) > 2 and TABLE_SEPARATOR.match(lines[1]):
        return lines[:2], lines[2:], []
    return [], lines, []


def _cut(line: str, budget: int, count: TokenCounter) -> list[str]:
    [tokens] = count([line])
    if tokens <= budget or len(line) == 1:
        return [line]
    step = max(1, len(line) * budget // tokens)
    return [piece for start in range(0, len(line), step) for piece in _cut(line[start : start + step], budget, count)]


def _joined(
    units: list[str], budget: int, count: TokenCounter, separator: str, wrap: Callable[[str], str] = str
) -> list[str]:
    """Units joined greedily into pieces of at most budget tokens, each unit fitting on its own."""
    separator_tokens, frame_tokens = count([separator, wrap("")])
    groups: list[list[str]] = []
    size = budget + 1
    for unit, tokens in zip(units, count(units), strict=True):
        if size + separator_tokens + tokens > budget:
            groups.append([])
            size = frame_tokens - separator_tokens
        groups[-1].append(unit)
        size += separator_tokens + tokens
    texts = [wrap(separator.join(group)) for group in groups]
    # Counts summed per unit can drift from the joined text's at the seams, so each piece is counted once as a whole.
    pieces: list[str] = []
    for group, text, tokens in zip(groups, texts, count(texts), strict=True):
        if tokens <= budget or len(group) == 1:
            pieces.append(text)
        else:
            half = len(group) // 2
            pieces += _joined(group[:half], budget, count, separator, wrap)
            pieces += _joined(group[half:], budget, count, separator, wrap)
    return pieces


def _header(document: Document, section: str | None) -> str:
    parts = [document.repo, _reference(document), document.title, section]
    return " · ".join(part for part in parts if part)


def _reference(document: Document) -> str | None:
    last_segment = document.url.rsplit("/", 1)[-1]
    match document.kind:
        case DocumentKind.PR:
            return f"PR #{last_segment}"
        case DocumentKind.ISSUE:
            return f"issue #{last_segment}"
        case DocumentKind.COMMIT:
            return f"commit {last_segment[:SHORT_SHA]}"
        case _:
            return None
