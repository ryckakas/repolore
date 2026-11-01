from pathlib import Path

from repolore.locks import hold, is_locked


def test_a_held_lock_is_locked_until_released(tmp_path: Path) -> None:
    lock_file = tmp_path / "x.lock"

    with hold(lock_file, timeout=0):
        assert is_locked(lock_file)

    assert not is_locked(lock_file)


def test_a_missing_lock_file_is_not_locked(tmp_path: Path) -> None:
    assert not is_locked(tmp_path / "never.lock")
