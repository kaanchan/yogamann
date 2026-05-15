import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

from vlm_inference import _parse_output, _validate, VLMSchemaError, annotate

VALID = {
    "rating": "good",
    "misaligned": ["left arm"],
    "unwanted_features": [],
    "fail_patterns": [],
    "notes": "Pose matches well.",
}


# ── _parse_output ─────────────────────────────────────────────────────────────

def test_parse_bare_json():
    assert _parse_output(json.dumps(VALID)) == VALID


def test_parse_strips_assistant_prefix():
    assert _parse_output("assistant:\n" + json.dumps(VALID)) == VALID


def test_parse_strips_im_start_prefix():
    assert _parse_output("<|im_start|>assistant\n" + json.dumps(VALID)) == VALID


def test_parse_strips_markdown_fence():
    raw = "```json\n" + json.dumps(VALID) + "\n```"
    assert _parse_output(raw) == VALID


def test_parse_raises_on_bad_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_output("not json at all")


# ── _validate ─────────────────────────────────────────────────────────────────

def test_validate_passes_complete_dict():
    _validate(VALID)  # no exception


def test_validate_raises_on_missing_rating():
    incomplete = {k: v for k, v in VALID.items() if k != "rating"}
    with pytest.raises(VLMSchemaError, match="rating"):
        _validate(incomplete)


def test_validate_raises_on_multiple_missing():
    with pytest.raises(VLMSchemaError):
        _validate({"rating": "good"})


# ── annotate ─────────────────────────────────────────────────────────────────

def _make_images(tmp_path):
    photo = tmp_path / "photo.jpg"
    render = tmp_path / "render.png"
    PILImage.new("RGB", (64, 64), color=(128, 64, 32)).save(photo)
    PILImage.new("RGB", (64, 64), color=(200, 200, 200)).save(render)
    return photo, render


@patch("vlm_inference._infer")
@patch("vlm_inference._load_model")
@patch("vlm_inference._load_config")
def test_annotate_returns_structured_dict(mock_cfg, mock_load, mock_infer, tmp_path):
    photo, render = _make_images(tmp_path)
    mock_cfg.return_value = {
        "model_key": "qwen2_5_vl_7b",
        "backend": "transformers",
        "max_new_tokens": 512,
        "prompt": {"system": "sys", "user": "usr"},
    }
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_infer.return_value = json.dumps(VALID)

    result = annotate(photo, render, model_key="qwen2_5_vl_7b")

    assert result["rating"] == "good"
    assert result["model_id"] == "qwen2_5_vl_7b"
    assert result["misaligned"] == ["left arm"]
    assert "latency_s" in result
    assert "raw_output" in result


@patch("vlm_inference._infer")
@patch("vlm_inference._load_model")
@patch("vlm_inference._load_config")
def test_annotate_retries_on_bad_json(mock_cfg, mock_load, mock_infer, tmp_path):
    photo, render = _make_images(tmp_path)
    mock_cfg.return_value = {
        "model_key": "qwen2_5_vl_7b",
        "backend": "transformers",
        "max_new_tokens": 512,
        "prompt": {"system": "sys", "user": "usr"},
    }
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_infer.side_effect = ["not valid json", json.dumps(VALID)]

    result = annotate(photo, render, model_key="qwen2_5_vl_7b")

    assert result["rating"] == "good"
    assert mock_infer.call_count == 2


@patch("vlm_inference._infer")
@patch("vlm_inference._load_model")
@patch("vlm_inference._load_config")
def test_annotate_raises_on_second_bad_json(mock_cfg, mock_load, mock_infer, tmp_path):
    photo, render = _make_images(tmp_path)
    mock_cfg.return_value = {
        "model_key": "qwen2_5_vl_7b",
        "backend": "transformers",
        "max_new_tokens": 512,
        "prompt": {"system": "sys", "user": "usr"},
    }
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_infer.side_effect = ["bad json", "still bad json"]

    with pytest.raises(json.JSONDecodeError):
        annotate(photo, render, model_key="qwen2_5_vl_7b")


def test_annotate_raises_on_missing_photo(tmp_path):
    render = tmp_path / "render.png"
    PILImage.new("RGB", (64, 64)).save(render)
    with pytest.raises(FileNotFoundError, match="[Pp]hoto|[Pp]hoto"):
        annotate(tmp_path / "missing.jpg", render, model_key="qwen2_5_vl_7b")


def test_annotate_raises_on_missing_render(tmp_path):
    photo = tmp_path / "photo.jpg"
    PILImage.new("RGB", (64, 64)).save(photo)
    with pytest.raises(FileNotFoundError, match="[Rr]ender"):
        annotate(photo, tmp_path / "missing.png", model_key="qwen2_5_vl_7b")
