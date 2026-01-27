import pytest
from types import SimpleNamespace
from pyccea.utils import memory
from unittest.mock import MagicMock


def test_force_memory_release_non_posix(monkeypatch) -> None:
    """Test non-posix path skips malloc trim."""
    gc_mock = MagicMock()
    cdll_mock = MagicMock()
    monkeypatch.setattr(memory, "gc", MagicMock(collect=gc_mock))
    monkeypatch.setattr(memory.os, "name", "nt", raising=False)
    monkeypatch.setattr(memory.ctypes, "CDLL", cdll_mock)

    memory.force_memory_release()

    gc_mock.assert_called_once()
    cdll_mock.assert_not_called()


def test_force_memory_release_posix_calls_maloc_trim(monkeypatch) -> None:
    """Test posix path calls malloc trim when available."""
    gc_mock = MagicMock()
    malloc_trim = MagicMock()
    libc_mock = MagicMock(malloc_trim=malloc_trim)
    monkeypatch.setattr(memory, "gc", MagicMock(collect=gc_mock))
    monkeypatch.setattr(memory.os, "name", "posix", raising=False)
    monkeypatch.setattr(memory.ctypes, "CDLL", MagicMock(return_value=libc_mock))

    memory.force_memory_release()

    gc_mock.assert_called_once()
    malloc_trim.assert_called_once_with(0)


def test_force_memory_release_posix_handles_cdll_error(monkeypatch) -> None:
    """Test posix path ignores CDLL errors."""
    gc_mock = MagicMock()
    monkeypatch.setattr(memory, "gc", MagicMock(collect=gc_mock))
    monkeypatch.setattr(memory, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(memory.ctypes, "CDLL", MagicMock(sideeffect=Exception("boom")))

    memory.force_memory_release()

    gc_mock.assert_called_once()


def test_force_memory_release_posix_handles_malloc_trim_error(monkeypatch) -> None:
    """Test posix path ignores malloc trim errors."""
    gc_mock = MagicMock()
    libc_mock = MagicMock()
    libc_mock.malloc_trim.side_effect = OSError("boom")
    monkeypatch.setattr(memory, "gc", MagicMock(collect=gc_mock))
    monkeypatch.setattr(memory, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(memory.ctypes, "CDLL", MagicMock(sideeffect=libc_mock))

    memory.force_memory_release()

    gc_mock.assert_called_once()