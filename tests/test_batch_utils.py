import pytest
from _batch_utils import chunked, format_summary


def test_chunked_exact_divisor():
    assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_chunked_remainder():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty():
    assert list(chunked([], 3)) == []


def test_chunked_size_larger_than_input():
    assert list(chunked([1, 2], 5)) == [[1, 2]]


def test_chunked_size_zero_raises():
    with pytest.raises(ValueError):
        list(chunked([1, 2], 0))


def test_format_summary_basic():
    stats = {
        "qwen2_5_vl_7b": {"ok": 95, "error": 3, "skipped": 2, "elapsed_s": 412.0},
        "internvl2_5_8b": {"ok": 90, "error": 8, "skipped": 2, "elapsed_s": 530.5},
    }
    out = format_summary(stats)
    assert "qwen2_5_vl_7b" in out
    assert "95" in out and "3" in out and "412" in out
    assert "internvl2_5_8b" in out
    # Header row exists
    assert "model" in out.lower()


def test_format_summary_empty():
    assert "no models" in format_summary({}).lower()
