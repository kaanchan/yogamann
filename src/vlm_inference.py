"""src/vlm_inference.py — lazy-cached VLM inference core.

Entry point:
    result = annotate(photo_path, render_path, model_key="qwen2_5_vl_7b")

Returns:
    {"model_id", "rating", "misaligned", "unwanted_features",
     "fail_patterns", "notes", "raw_output", "latency_s"}
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml
from PIL import Image, ImageOps

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, tuple] = {}

REQUIRED_KEYS = ["rating", "misaligned", "unwanted_features", "fail_patterns", "notes"]

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Output ONLY a JSON object with exactly these keys: "
    "rating, misaligned, unwanted_features, fail_patterns, notes. No other text."
)


class VLMSchemaError(Exception):
    pass


# ── Config ────────────────────────────────────────────────────────────────────
def _load_config(model_key: str | None = None, config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "profiles" / "vlm.yml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    key = model_key or cfg["active_model"]
    result = dict(cfg["models"][key])
    result["model_key"] = key
    result["prompt"] = cfg["prompt"]
    return result


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model(model_key: str, config: dict) -> tuple:
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    repo = config["repo"]
    load_in_4bit = config.get("load_in_4bit", False)

    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM  # type: ignore[assignment]

    bnb = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
    model = _AutoVLM.from_pretrained(
        repo,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    _MODEL_CACHE[model_key] = (model, processor)
    return model, processor


# ── Prompt ───────────────────────────────────────────────────────────────────
def _build_messages(prompt: dict, backend: str, extra_user: str | None = None) -> list:
    user_text = extra_user if extra_user else prompt["user"]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]


# ── Image resizing ────────────────────────────────────────────────────────────
def _resize_for_vlm(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Downscale image so the longer side <= max_side, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── Inference ─────────────────────────────────────────────────────────────────
def _infer(model, processor, messages: list, images: list, config: dict) -> str:
    max_new_tokens = config.get("max_new_tokens", 512)
    max_image_side = config.get("max_image_side", 1024)
    images = [_resize_for_vlm(img, max_image_side) for img in images]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            import torch
            torch.cuda.empty_cache()
            raise MemoryError(f"GPU OOM during inference: {exc}") from exc
        raise
    generated = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


# ── Parsing + validation ──────────────────────────────────────────────────────
def _parse_output(raw: str) -> dict:
    text = raw.strip()
    for prefix in ("assistant:", "<|im_start|>assistant", "<|im_start|>"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _validate(data: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise VLMSchemaError(f"Missing required keys: {missing}")


# ── Public API ────────────────────────────────────────────────────────────────
def annotate(
    photo_path: Path,
    render_path: Path,
    model_key: str | None = None,
    config_path: Path | None = None,
) -> dict:
    photo_path = Path(photo_path)
    render_path = Path(render_path)
    if not photo_path.exists():
        raise FileNotFoundError(f"Photo not found: {photo_path}")
    if not render_path.exists():
        raise FileNotFoundError(f"Render not found: {render_path}")

    config = _load_config(model_key, config_path)
    key = config["model_key"]
    backend = config.get("backend", "transformers")

    model, processor = _load_model(key, config)

    photo = ImageOps.exif_transpose(Image.open(photo_path).convert("RGB"))
    render = Image.open(render_path).convert("RGB")
    images = [photo, render]

    messages = _build_messages(config["prompt"], backend)

    t0 = time.perf_counter()
    raw = _infer(model, processor, messages, images, config)

    try:
        data = _parse_output(raw)
    except json.JSONDecodeError:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _RETRY_PROMPT},
        ]
        raw = _infer(model, processor, retry_messages, images, config)
        data = _parse_output(raw)

    _validate(data)
    latency_s = time.perf_counter() - t0

    return {
        "model_id": key,
        "rating": data["rating"],
        "misaligned": data.get("misaligned", []),
        "unwanted_features": data.get("unwanted_features", []),
        "fail_patterns": data.get("fail_patterns", []),
        "notes": data.get("notes", ""),
        "raw_output": raw,
        "latency_s": latency_s,
    }
