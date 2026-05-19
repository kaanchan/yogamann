import json
import textwrap
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from PIL import Image as PILImage

from vlm_inference import (
    _parse_output, _validate, VLMSchemaError, annotate,
    _build_messages, _load_config,
    _infer_minicpm_v, _infer_internvl, _infer_molmo,
)

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


# ── _build_messages: system prompt inclusion (#38) ───────────────────────────

def test_build_messages_first_message_is_system_when_prompt_has_system():
    prompt = {"system": "You are a pose analyst.", "user": "Does the render match?"}
    messages = _build_messages(prompt, "transformers")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a pose analyst."


def test_build_messages_no_system_message_when_system_key_absent():
    prompt = {"user": "Does the render match?"}
    messages = _build_messages(prompt, "transformers")
    assert messages[0]["role"] == "user"
    assert len(messages) == 1


def test_build_messages_user_message_has_two_image_slots_and_text():
    prompt = {"system": "sys", "user": "check it"}
    messages = _build_messages(prompt, "transformers")
    user_msg = messages[-1]
    types = [c["type"] for c in user_msg["content"]]
    assert types == ["image", "image", "text"]
    assert user_msg["content"][-1]["text"] == "check it"


# ── _infer_minicpm_v: system_prompt kwarg (#38) ───────────────────────────────

@patch("vlm_inference._patch_minicpm_v_normalize_compat")
@patch("vlm_inference._get_tokenizer", return_value=MagicMock())
def test_infer_minicpm_v_passes_system_prompt_kwarg(mock_tok, mock_patch):
    model = MagicMock()
    model.chat.return_value = "{}"
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "check it"}]},
    ]
    _infer_minicpm_v(model, "openbmb/MiniCPM-V-2_6", messages, [], {"max_new_tokens": 64})
    _, kwargs = model.chat.call_args
    assert kwargs.get("system_prompt") == "Be precise."


@patch("vlm_inference._patch_minicpm_v_normalize_compat")
@patch("vlm_inference._get_tokenizer", return_value=MagicMock())
def test_infer_minicpm_v_no_system_prompt_kwarg_when_no_system_message(mock_tok, mock_patch):
    model = MagicMock()
    model.chat.return_value = "{}"
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "check it"}]},
    ]
    _infer_minicpm_v(model, "openbmb/MiniCPM-V-2_6", messages, [], {"max_new_tokens": 64})
    _, kwargs = model.chat.call_args
    assert "system_prompt" not in kwargs or kwargs.get("system_prompt") is None


# ── _infer_internvl: system text in question string (#38) ────────────────────

@patch("torch.cat", return_value=MagicMock())
@patch("vlm_inference._internvl_preprocess", return_value=MagicMock())
@patch("vlm_inference._patch_internvl_generation_mixin")
@patch("vlm_inference._get_tokenizer", return_value=MagicMock())
def test_infer_internvl_prepends_system_text_to_question(mock_tok, mock_patch, mock_prep, mock_cat):
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock()])
    model.chat.return_value = "{}"
    messages = [
        {"role": "system", "content": "System rule here."},
        {"role": "user", "content": [
            {"type": "image"}, {"type": "image"}, {"type": "text", "text": "does it match"},
        ]},
    ]
    _infer_internvl(model, "OpenGVLab/InternVL2_5-8B-MPO", messages, [MagicMock(), MagicMock()], {"max_new_tokens": 64})
    question = model.chat.call_args[0][2]
    assert "System rule here." in question
    assert "does it match" in question


# ── _infer_molmo: system text in user_text (#38) ─────────────────────────────

@patch("transformers.GenerationConfig")
@patch("transformers.AutoProcessor")
def test_infer_molmo_prepends_system_text_to_user_text(mock_proc_cls, mock_gen_cfg):
    mock_processor = MagicMock()
    mock_proc_cls.from_pretrained.return_value = mock_processor
    mock_tensor = MagicMock()
    mock_tensor.is_floating_point.return_value = False
    mock_processor.process.return_value = {"input_ids": mock_tensor}
    mock_processor.tokenizer.decode.return_value = "{}"
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock()])
    messages = [
        {"role": "system", "content": "Evaluate carefully."},
        {"role": "user", "content": [{"type": "text", "text": "describe the pose"}]},
    ]
    _infer_molmo(model, "allenai/Molmo-7B-D-0924", messages, [], {"max_new_tokens": 64})
    _, kwargs = mock_processor.process.call_args
    assert "Evaluate carefully." in kwargs["text"]
    assert "describe the pose" in kwargs["text"]


# ── _load_config: named prompt variants (#37) ─────────────────────────────────

_MULTI_PROMPT_YAML = textwrap.dedent("""\
    active_model: test_model
    active_prompt: v1
    models:
      test_model:
        repo: test/repo
        backend: transformers
    prompts:
      v1:
        system: "V1 system."
        user: "V1 user."
      v2_checklist:
        system: "V2 system with checklist."
        user: "V2 user with checklist."
    schema:
      rating: [good, acceptable, poor]
      required: [rating, misaligned, unwanted_features, fail_patterns, notes]
""")

_LEGACY_PROMPT_YAML = textwrap.dedent("""\
    active_model: test_model
    models:
      test_model:
        repo: test/repo
        backend: transformers
    prompt:
      system: "Legacy system."
      user: "Legacy user."
    schema:
      rating: [good, acceptable, poor]
      required: [rating, misaligned, unwanted_features, fail_patterns, notes]
""")


def test_load_config_named_prompts_uses_active_prompt(tmp_path):
    cfg = tmp_path / "vlm.yml"
    cfg.write_text(_MULTI_PROMPT_YAML)
    result = _load_config(model_key="test_model", config_path=cfg)
    assert result["prompt"]["system"] == "V1 system."
    assert result["prompt"]["user"] == "V1 user."


def test_load_config_named_prompts_prompt_key_override(tmp_path):
    cfg = tmp_path / "vlm.yml"
    cfg.write_text(_MULTI_PROMPT_YAML)
    result = _load_config(model_key="test_model", config_path=cfg, prompt_key="v2_checklist")
    assert result["prompt"]["system"] == "V2 system with checklist."
    assert result["prompt"]["user"] == "V2 user with checklist."


def test_load_config_legacy_prompt_field_backward_compat(tmp_path):
    cfg = tmp_path / "vlm.yml"
    cfg.write_text(_LEGACY_PROMPT_YAML)
    result = _load_config(model_key="test_model", config_path=cfg)
    assert result["prompt"]["system"] == "Legacy system."
    assert result["prompt"]["user"] == "Legacy user."


# ── annotate: prompt_key param (#37) ─────────────────────────────────────────

@patch("vlm_inference._infer")
@patch("vlm_inference._load_model")
@patch("vlm_inference._load_config")
def test_annotate_passes_prompt_key_to_load_config(mock_cfg, mock_load, mock_infer, tmp_path):
    photo, render = _make_images(tmp_path)
    mock_cfg.return_value = {
        "model_key": "qwen2_5_vl_7b",
        "backend": "transformers",
        "max_new_tokens": 512,
        "prompt": {"system": "sys", "user": "usr"},
    }
    mock_load.return_value = (MagicMock(), MagicMock())
    mock_infer.return_value = json.dumps(VALID)

    annotate(photo, render, model_key="qwen2_5_vl_7b", prompt_key="v2_checklist")

    mock_cfg.assert_called_once_with("qwen2_5_vl_7b", None, "v2_checklist")
