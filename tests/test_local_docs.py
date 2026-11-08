import logging
import subprocess
import sys
from pathlib import Path

import pytest

from repolore.config import Config, LocalFolder
from repolore.ingest import cached_documents
from repolore.sources import local_docs
from repolore.sources.local_docs import (
    LocalFile,
    LocalFiles,
    TooManyFilesError,
    find_checkouts,
    find_folders,
    github_repo,
    load_local_files,
    read_local_files,
    save_local_files,
)

PLANS = ".plans/**/*.md"
ORIGIN = "https://github.com/octo/repo.git"


def checkout(root: Path, name: str, remote: str | None = None, plans: bool = True) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(directory)], check=True)
    if remote:
        subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", remote], check=True)
    if plans:
        write(directory / ".plans" / "feature" / "plan.md", "# Plan\n\nShip it. ✓")
    return directory


def write(path: Path, text: str = "text") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def link_folder(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def config(root: Path, depth: int = 2, patterns: tuple[str, ...] = (PLANS,)) -> Config:
    return Config(local_roots=[root], checkout_depth=depth, local_patterns=list(patterns))


def files(root: Path, *patterns: str, limit: int = 100) -> LocalFiles:
    return LocalFiles(root, patterns, limit)


def names(root: Path, *patterns: str) -> list[str]:
    return [path.relative_to(root).as_posix() for path in files(root, *patterns).matches()]


def test_checkouts_with_a_match_are_keyed_by_their_github_repo(tmp_path: Path) -> None:
    repo = checkout(tmp_path, "repo", ORIGIN)
    checkout(tmp_path, "no-plans", "https://github.com/octo/no-plans.git", plans=False)
    write(tmp_path / "loose-file.md", "not a checkout")

    assert find_checkouts(config(tmp_path)) == {"octo/repo": LocalFiles(repo, (PLANS,), 2000)}


def test_no_roots_means_no_checkouts() -> None:
    assert find_checkouts(Config(local_patterns=[PLANS])) == {}


def test_no_patterns_means_no_checkouts_even_with_plans_on_disk(tmp_path: Path) -> None:
    checkout(tmp_path, "repo", ORIGIN)

    assert find_checkouts(Config(local_roots=[tmp_path])) == {}


def test_no_patterns_means_the_roots_are_not_walked(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        find_checkouts(Config(local_roots=[tmp_path / "missing"]))

    assert caplog.text == ""


def test_missing_root_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert find_checkouts(config(tmp_path / "missing")) == {}

    assert "doesn't exist" in caplog.text


@pytest.mark.parametrize(
    ("depth", "found"),
    [
        (0, []),
        (1, ["flat"]),
        (2, ["flat", "org/nested"]),
        (3, ["flat", "org/nested", "a/b/deep"]),
    ],
)
def test_checkouts_are_found_down_to_the_depth(tmp_path: Path, depth: int, found: list[str]) -> None:
    for name in ["flat", "org/nested", "a/b/deep"]:
        checkout(tmp_path, name, f"https://github.com/octo/{Path(name).name}.git")

    result = find_checkouts(config(tmp_path, depth=depth))

    assert sorted(item.root.relative_to(tmp_path).as_posix() for item in result.values()) == sorted(found)


def test_a_root_that_is_itself_a_checkout_counts(tmp_path: Path) -> None:
    repo = checkout(tmp_path, "repo", ORIGIN)

    assert list(find_checkouts(config(repo, depth=0))) == ["octo/repo"]


def test_a_checkout_nested_in_a_checkout_is_not_searched(tmp_path: Path) -> None:
    outer = checkout(tmp_path, "outer", "https://github.com/octo/outer.git")
    checkout(outer, "inner", "https://github.com/octo/inner.git")

    assert list(find_checkouts(config(tmp_path))) == ["octo/outer"]


def test_a_worktree_is_not_a_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree = tmp_path / "worktrees" / "branch"
    write(worktree / ".git", "gitdir: /elsewhere")
    write(worktree / ".plans" / "plan.md")
    monkeypatch.setattr(local_docs, "github_repo", lambda _: "octo/repo")

    assert find_checkouts(config(tmp_path)) == {}


@pytest.mark.parametrize("folder", [".hidden", "node_modules", "vendor"])
def test_hidden_and_dependency_folders_hold_no_checkouts(tmp_path: Path, folder: str) -> None:
    checkout(tmp_path / folder, "repo", ORIGIN)

    assert find_checkouts(config(tmp_path)) == {}


def test_a_linked_folder_is_not_entered_when_looking_for_checkouts(tmp_path: Path) -> None:
    checkout(tmp_path / "outside", "repo", ORIGIN)
    (tmp_path / "root").mkdir()
    link_folder(tmp_path / "root" / "link", tmp_path / "outside")

    assert find_checkouts(config(tmp_path / "root")) == {}


def test_a_folder_that_cant_be_read_is_logged_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(_: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", refuse)

    with caplog.at_level(logging.INFO):
        assert find_checkouts(config(tmp_path)) == {}

    assert "can't look inside" in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="chmod can't lock a folder on Windows")
def test_a_locked_folder_does_not_stop_the_search(tmp_path: Path) -> None:
    checkout(tmp_path, "repo", ORIGIN)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert list(find_checkouts(config(tmp_path))) == ["octo/repo"]
    finally:
        locked.chmod(0o700)


def test_of_two_clones_the_first_wins_and_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    first = checkout(tmp_path, "a-clone", ORIGIN)
    second = checkout(tmp_path, "b-clone", ORIGIN)

    with caplog.at_level(logging.WARNING):
        result = find_checkouts(config(tmp_path))

    assert result["octo/repo"].root == first
    assert f"checked out in {first} and {second}" in caplog.text


def test_overlapping_roots_are_not_two_clones(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    checkout(tmp_path / "code", "repo", ORIGIN)
    overlapping = Config(local_roots=[tmp_path, tmp_path / "code"], local_patterns=[PLANS])

    with caplog.at_level(logging.WARNING):
        assert list(find_checkouts(overlapping)) == ["octo/repo"]

    assert caplog.text == ""


def test_git_is_only_asked_about_checkouts_with_a_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[Path] = []
    repo = checkout(tmp_path, "repo", ORIGIN)
    checkout(tmp_path, "no-plans", ORIGIN, plans=False)
    monkeypatch.setattr(local_docs, "github_repo", lambda path: asked.append(path) or "octo/repo")

    find_checkouts(config(tmp_path))

    assert asked == [repo]


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/octo/my.repo",
        "https://github.com/octo/my.repo.git",
        "https://github.com/octo/my.repo/",
        "git@github.com:octo/my.repo.git",
        "ssh://git@github.com/octo/my.repo.git",
    ],
)
def test_github_remote_forms_are_recognized(tmp_path: Path, remote: str) -> None:
    assert github_repo(checkout(tmp_path, "repo", remote)) == "octo/my.repo"


@pytest.mark.parametrize("remote", ["https://github.com/../..", "git@github.com:octo/..", "https://github.com/./x"])
def test_remote_that_would_leave_the_cache_dir_is_skipped(tmp_path: Path, remote: str) -> None:
    assert github_repo(checkout(tmp_path, "repo", remote)) is None


def test_root_under_the_home_directory_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    repo = checkout(tmp_path / "projects", "repo", ORIGIN)

    assert find_checkouts(config(Path("~/projects")))["octo/repo"].root == repo


def test_checkout_without_origin_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert github_repo(checkout(tmp_path, "repo")) is None

    assert "has no origin remote" in caplog.text


def test_checkout_hosted_elsewhere_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert github_repo(checkout(tmp_path, "repo", "https://gitlab.com/octo/repo.git")) is None

    assert "isn't a GitHub checkout" in caplog.text


def test_patterns_match_hidden_folders_and_several_patterns_add_up(tmp_path: Path) -> None:
    write(tmp_path / ".plans" / "a" / "plan.md")
    write(tmp_path / "docs" / "adr" / "one.md")
    write(tmp_path / "docs" / "other.md")

    assert names(tmp_path, PLANS, "docs/adr/*.md") == [".plans/a/plan.md", "docs/adr/one.md"]


def test_a_file_two_patterns_match_counts_once(tmp_path: Path) -> None:
    write(tmp_path / "docs" / "one.md")

    assert names(tmp_path, "docs/*.md", "**/*.md") == ["docs/one.md"]


def test_only_markdown_files_match(tmp_path: Path) -> None:
    write(tmp_path / "a.md")
    write(tmp_path / "b.MARKDOWN")
    write(tmp_path / "c.txt")
    write(tmp_path / "d.json")
    (tmp_path / "folder.md").mkdir()

    assert sorted(name.lower() for name in names(tmp_path, "*")) == ["a.md", "b.markdown"]


@pytest.mark.parametrize("folder", ["node_modules", "vendor", ".git"])
def test_dependency_and_git_folders_never_match(tmp_path: Path, folder: str) -> None:
    write(tmp_path / folder / "x" / "readme.md")
    write(tmp_path / "keep.md")

    assert names(tmp_path, "**/*.md") == ["keep.md"]


@pytest.mark.parametrize("pattern", ["**/*.md", "*/*.md", "link/*.md"])
def test_a_link_out_of_the_root_never_matches(tmp_path: Path, pattern: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write(tmp_path / "outside" / "secret.md")
    link_folder(root / "link", tmp_path / "outside")

    assert names(root, pattern) == []


def test_a_link_inside_the_root_gives_one_document_not_a_crash(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write(root / "real" / "a.md", "# A")
    link_folder(root / "link", root / "real")
    read = read_local_files(files(root, "*/*.md"))
    save_local_files(tmp_path / "raw" / "octo" / "repo", read)

    documents = cached_documents(tmp_path / "raw")

    assert sorted(file.path for file in read) == ["link/a.md", "real/a.md"]
    assert len(documents) == 1


def test_no_match_is_an_empty_list(tmp_path: Path) -> None:
    assert names(tmp_path, "docs/*.md") == []


def test_files_are_read_with_paths_and_file_urls(tmp_path: Path) -> None:
    repo = checkout(tmp_path, "repo")
    write(repo / ".plans" / "feature" / "resources" / "probe.md", "Probe.")
    write(repo / ".plans" / "feature" / "resources" / "probe.json", "{}")

    read = read_local_files(files(repo, PLANS))

    assert [file.path for file in read] == [".plans/feature/plan.md", ".plans/feature/resources/probe.md"]
    assert read[0].url == (repo / ".plans" / "feature" / "plan.md").resolve().as_uri()
    assert read[0].text == "# Plan\n\nShip it. ✓"


def test_exactly_the_limit_is_read(tmp_path: Path) -> None:
    for number in range(3):
        write(tmp_path / f"{number}.md")

    assert len(read_local_files(files(tmp_path, "*.md", limit=3))) == 3


def test_one_past_the_limit_is_refused_naming_the_folder_and_the_key(tmp_path: Path) -> None:
    for number in range(4):
        write(tmp_path / f"{number}.md")

    with pytest.raises(TooManyFilesError, match=r"more than 3 local files.*max_local_files") as refused:
        read_local_files(files(tmp_path, "*.md", limit=3))

    assert str(tmp_path) in str(refused.value)


def test_matching_stops_one_past_the_limit(tmp_path: Path) -> None:
    for number in range(10):
        write(tmp_path / f"{number}.md")

    assert len(files(tmp_path, "*.md", limit=3).matches()) == 4


def test_local_files_round_trip_through_the_cache(tmp_path: Path) -> None:
    cached = [LocalFile(path=".plans/a/plan.md", url="file:///a/plan.md", text="Grüße")]

    save_local_files(tmp_path / "octo" / "repo", cached)

    assert load_local_files(tmp_path / "octo" / "repo") == cached
    assert load_local_files(tmp_path / "octo" / "other") == []


def folders(*entries: LocalFolder, limit: int = 100) -> dict[str, LocalFiles]:
    return find_folders(Config(local_folders=list(entries), max_local_files=limit))


def test_a_folder_entry_matches_markdown_anywhere_by_default(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "top.md")
    write(tmp_path / "notes" / "deep" / "er.md")
    write(tmp_path / "notes" / "skip.pdf")

    found = folders(LocalFolder(name="notes", path=tmp_path / "notes"))

    assert list(found) == ["local/notes"]
    assert names(found["local/notes"].root, *found["local/notes"].patterns) == ["deep/er.md", "top.md"]


def test_a_folder_entry_uses_its_own_patterns_and_the_configured_limit(tmp_path: Path) -> None:
    write(tmp_path / "adr" / "one.md")
    write(tmp_path / "other.md")

    found = folders(LocalFolder(name="a", path=tmp_path, patterns=["adr/*.md"]), limit=7)

    assert found["local/a"] == LocalFiles(tmp_path.resolve(), ("adr/*.md",), 7)
    assert [file.path for file in read_local_files(found["local/a"])] == ["adr/one.md"]


def test_a_folder_entry_can_be_a_single_file_and_ignores_patterns(tmp_path: Path) -> None:
    file = write(tmp_path / "design.md", "# Design")
    write(tmp_path / "other.md")

    found = folders(LocalFolder(name="design", path=file, patterns=["nothing/*.md"]))

    read = read_local_files(found["local/design"])
    assert [(item.path, item.text) for item in read] == [("design.md", "# Design")]


def test_a_single_file_with_glob_characters_in_its_name_is_found(tmp_path: Path) -> None:
    file = write(tmp_path / "a[b].md")
    write(tmp_path / "ab.md")

    found = folders(LocalFolder(name="odd", path=file))

    assert [item.path for item in read_local_files(found["local/odd"])] == ["a[b].md"]


def test_a_single_file_that_isnt_markdown_matches_nothing(tmp_path: Path) -> None:
    found = folders(LocalFolder(name="data", path=write(tmp_path / "data.txt")))

    assert read_local_files(found["local/data"]) == []


def test_an_empty_folder_stays_a_source(tmp_path: Path) -> None:
    assert read_local_files(folders(LocalFolder(name="empty", path=tmp_path))["local/empty"]) == []


def test_no_folder_entries_means_no_folders() -> None:
    assert find_folders(Config()) == {}


def test_a_missing_path_is_logged_and_dropped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    present = LocalFolder(name="here", path=tmp_path)

    with caplog.at_level(logging.WARNING):
        found = folders(LocalFolder(name="gone", path=tmp_path / "gone"), present)

    assert list(found) == ["local/here"]
    assert "doesn't exist" in caplog.text


def test_a_folder_under_the_home_directory_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    write(tmp_path / "notes" / "a.md")

    found = folders(LocalFolder(name="notes", path=Path("~/notes")))

    assert found["local/notes"].root == (tmp_path / "notes").resolve()


def test_a_single_file_named_through_a_link_is_read(tmp_path: Path) -> None:
    target = write(tmp_path / "elsewhere" / "real.md", "# Real")
    link = tmp_path / "notes" / "named.md"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("creating a file symlink needs a privilege on this Windows account")

    found = folders(LocalFolder(name="linked", path=link))

    assert [item.text for item in read_local_files(found["local/linked"])] == ["# Real"]


def test_a_folder_path_that_is_a_link_is_followed_but_links_inside_are_not(tmp_path: Path) -> None:
    write(tmp_path / "real" / "a.md")
    write(tmp_path / "outside" / "secret.md")
    link_folder(tmp_path / "named", tmp_path / "real")
    link_folder(tmp_path / "real" / "inner", tmp_path / "outside")

    found = folders(LocalFolder(name="linked", path=tmp_path / "named", patterns=["*.md", "*/*.md"]))

    assert [item.path for item in read_local_files(found["local/linked"])] == ["a.md"]
