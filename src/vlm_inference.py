"""src/vlm_inference.py — lazy-cached VLM inference core.

Entry point:
    result = annotate(photo_path, render_path, model_key="qwen2_5_vl_7b")

Returns:
    {"model_id", "rating", "misaligned", "unwanted_features",
     "fail_patterns", "notes", "raw_output", "latency_s"}
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import yaml
from PIL import Image, ImageOps

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, tuple] = {}
_TOKENIZER_CACHE: dict[str, object] = {}


# ── Module-load patches for trust_remote_code × bnb compatibility ────────────
def _patch_qwen2_init_weights_for_bnb() -> None:
    """transformers Qwen2 `_init_weights` calls `.normal_()` on Linear weights.
    When bnb 4-bit quantization is active, weights may already be uint8 (Byte)
    tensors — `.normal_()` has no kernel for that dtype and crashes with
    NotImplementedError. This matters specifically for MiniCPM-o-2.6 which
    embeds a Qwen2 audio-language backbone; the other Qwen2-based models in
    our suite (Qwen2.5-VL, MiniCPM-V-2.6) don't trigger this code path on
    quantized weights, so the patch is a no-op for them.

    The patch wraps `_init_weights` to early-return for uint8 weights,
    leaving bnb's quantization untouched."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
    if getattr(qwen2_mod.Qwen2PreTrainedModel._init_weights, "_bnb_safe", False):
        return
    _orig = qwen2_mod.Qwen2PreTrainedModel._init_weights

    def _safe_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dtype == torch.uint8:
                return  # already bnb-quantized; skip random re-init
        return _orig(self, module)

    _safe_init_weights._bnb_safe = True  # type: ignore[attr-defined]
    qwen2_mod.Qwen2PreTrainedModel._init_weights = _safe_init_weights


_patch_qwen2_init_weights_for_bnb()

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

    from transformers import (
        AutoProcessor, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
    )
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM  # type: ignore[assignment]

    bnb = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
    kwargs = dict(quantization_config=bnb, device_map="auto", trust_remote_code=True)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kwargs["local_files_only"] = True
    revision = config.get("revision")
    if revision:
        kwargs["revision"] = revision
    try:
        model = _AutoVLM.from_pretrained(repo, **kwargs)
    except ValueError:
        try:
            model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        except ValueError:
            model = AutoModel.from_pretrained(repo, **kwargs)
    # Custom-code VLMs (InternVL, MiniCPM-V/o, Molmo) don't expose a unified
    # processor with apply_chat_template — they're invoked via their own
    # model.chat(...) methods using a tokenizer directly. Skip AutoProcessor
    # entirely for those styles; _infer_<style> loads what it needs.
    if config.get("inference_style"):
        _MODEL_CACHE[model_key] = (model, None)
        return model, None
    proc_kwargs = {"trust_remote_code": True}
    if "local_files_only" in kwargs:
        proc_kwargs["local_files_only"] = kwargs["local_files_only"]
    if revision:
        proc_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(repo, **proc_kwargs)
    _MODEL_CACHE[model_key] = (model, processor)
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() // 1024 // 1024
            reserved = torch.cuda.memory_reserved() // 1024 // 1024
            total = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
            print(f"  [load] {model_key} loaded — VRAM alloc={used}MB reserved={reserved}MB total={total}MB", flush=True)
    except Exception:
        pass
    return model, processor


def _get_tokenizer(repo: str, revision: str | None = None, offline: bool = False) -> object:
    cache_key = f"{repo}@{revision or 'main'}"
    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]
    from transformers import AutoTokenizer
    tok_kwargs = {"trust_remote_code": True, "use_fast": False}
    if offline:
        tok_kwargs["local_files_only"] = True
    if revision:
        tok_kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(repo, **tok_kwargs)
    _TOKENIZER_CACHE[cache_key] = tok
    return tok


# ── Eviction (public — callers manage GPU memory between model phases) ───────
def evict_model(model_key: str) -> None:
    """Drop a cached model and free its GPU memory.

    Safe to call when model_key was never loaded — that's a no-op.
    Always clears the tokenizer entry too (tokenizers are small but
    keeping them around obscures which models are 'loaded').
    """
    import gc
    if model_key in _MODEL_CACHE:
        model, _ = _MODEL_CACHE.pop(model_key)
        del model
    # Also drop tokenizers cached under the model's repo (best-effort).
    _TOKENIZER_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evict_all() -> None:
    """Drop all cached models and tokenizers. Same semantics as
    evict_model called for every cached key."""
    for key in list(_MODEL_CACHE.keys()):
        evict_model(key)


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


def _extract_user_text(user_msg: dict) -> str:
    """Pull the user prompt out of a message. Handles both the structured form
    {"content": [{"type": "text", "text": "..."}]} and the plain-string retry
    form {"content": "..."}."""
    content = user_msg["content"]
    if isinstance(content, str):
        return content
    return next(c["text"] for c in content if c.get("type") == "text")


# ── InternVL-style inference (custom model.chat API) ─────────────────────────
def _internvl_preprocess(img: Image.Image, input_size: int = 448):
    """InternVL preprocessing: resize to square + ImageNet normalize -> fp32 CPU tensor.
    Caller casts to model.dtype before moving to GPU (bnb 4-bit residual modules
    are fp16 by default, so a static dtype here would mismatch the vision tower)."""
    import numpy as np
    import torch
    img = img.convert("RGB").resize((input_size, input_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def _patch_internvl_generation_mixin(model) -> None:
    """transformers >=4.50 stopped auto-inheriting GenerationMixin in PreTrainedModel.
    InternVL's hosted InternLM2ForCausalLM was authored against pre-v4.50 and assumes
    .generate() exists AND that GenerationMixin.__init__ initialized .generation_config.
    Inject GenerationMixin into the class bases at runtime (once) so .chat() works,
    and seed a default generation_config since the missed __init__ left it None."""
    from transformers import GenerationMixin, GenerationConfig
    cls = type(model.language_model)
    if GenerationMixin not in cls.__mro__:
        cls.__bases__ = cls.__bases__ + (GenerationMixin,)
    if getattr(model.language_model, "generation_config", None) is None:
        model.language_model.generation_config = GenerationConfig.from_model_config(
            model.language_model.config
        )


def _infer_internvl(model, repo: str, messages: list, images: list, config: dict) -> str:
    """InternVL uses model.chat(tokenizer, pixel_values, question, ...) — not generate()."""
    import torch
    _patch_internvl_generation_mixin(model)
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    tokenizer = _get_tokenizer(repo, revision=config.get("revision"), offline=offline)
    max_new_tokens = config.get("max_new_tokens", 512)
    input_size = config.get("internvl_input_size", 448)

    model_dtype = next(model.parameters()).dtype
    pixel_values_list = [
        _internvl_preprocess(img, input_size).unsqueeze(0).to(model_dtype).cuda()
        for img in images
    ]
    pixel_values = torch.cat(pixel_values_list, dim=0)
    num_patches_list = [1] * len(images)

    user_msg = messages[-1]
    user_text = _extract_user_text(user_msg)
    question = (
        "\n".join(f"Image-{i+1}: <image>" for i in range(len(images)))
        + "\n" + user_text
    )

    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    return model.chat(
        tokenizer, pixel_values, question, generation_config,
        num_patches_list=num_patches_list, history=None, return_history=False,
    )


# ── Molmo-style inference (processor.process + model.generate_from_batch) ────
def _infer_molmo(model, repo: str, messages: list, images: list, config: dict) -> str:
    """Molmo uses processor.process(images=, text=) + model.generate_from_batch().
    The processor returns a single batch dict; no chat template, plain text prompt."""
    import torch
    from transformers import AutoProcessor, GenerationConfig
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    proc_kwargs = {"trust_remote_code": True}
    if offline:
        proc_kwargs["local_files_only"] = True
    revision = config.get("revision")
    if revision:
        proc_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(repo, **proc_kwargs)
    max_new_tokens = config.get("max_new_tokens", 512)
    max_image_side = config.get("max_image_side", 1024)
    images = [_resize_for_vlm(img, max_image_side) for img in images]

    user_msg = messages[-1]
    user_text = _extract_user_text(user_msg)

    inputs = processor.process(images=images, text=user_text)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    # Cast floating-point tensors to model dtype (bnb residual modules are fp16);
    # processor.process returns fp32 which clashes with the model's Half params.
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(model_dtype) if v.is_floating_point() else v)
        for k, v in inputs.items()
    }
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>")
    output = model.generate_from_batch(inputs, gen_cfg, tokenizer=processor.tokenizer)
    new_tokens = output[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


_MINICPM_V_NORMALIZE_PATCHED = False


def _patch_minicpm_v_normalize_compat() -> None:
    """transformers >=4.49 normalize() requires mean/std to be collections.abc.Sequence;
    MiniCPM-V's custom image processor passes numpy arrays (set via np.array() in its
    __init__). np.ndarray fails the Sequence isinstance check, falls into the scalar
    branch which replicates a (3,) array 3 times giving a (3,3) mean — broadcast then
    fails against the (H, W, 3) image. Wrap normalize() to coerce ndarray -> list."""
    global _MINICPM_V_NORMALIZE_PATCHED
    if _MINICPM_V_NORMALIZE_PATCHED:
        return
    import numpy as np
    import transformers.image_transforms as it
    import transformers.image_processing_utils as ipu
    _orig = it.normalize
    def _patched_normalize(image, mean, std, *args, **kwargs):
        if isinstance(mean, np.ndarray):
            mean = mean.tolist()
        if isinstance(std, np.ndarray):
            std = std.tolist()
        return _orig(image, mean, std, *args, **kwargs)
    # Patch both the canonical module function AND the re-import inside
    # image_processing_utils (BaseImageProcessor.normalize calls the local name).
    it.normalize = _patched_normalize
    ipu.normalize = _patched_normalize
    _MINICPM_V_NORMALIZE_PATCHED = True


# ── MiniCPM-V-style inference (model.chat with msgs list) ────────────────────
def _infer_minicpm_v(model, repo: str, messages: list, images: list, config: dict) -> str:
    """MiniCPM-V uses model.chat(image=None, msgs=[...], tokenizer=) — multi-image
    is encoded by putting all PIL images directly into msg['content'] list. The
    model's internal preprocessor does its own slice-based tiling; do NOT pre-resize."""
    _patch_minicpm_v_normalize_compat()
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    tokenizer = _get_tokenizer(repo, revision=config.get("revision"), offline=offline)
    max_new_tokens = config.get("max_new_tokens", 512)

    user_msg = messages[-1]
    user_text = _extract_user_text(user_msg)

    msgs = [{"role": "user", "content": [*images, user_text]}]
    return model.chat(
        image=None, msgs=msgs, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )


# ── Inference (Qwen-style — processor.apply_chat_template + model.generate) ──
def _infer(model, processor, messages: list, images: list, config: dict) -> str:
    max_new_tokens = config.get("max_new_tokens", 512)
    max_image_side = config.get("max_image_side", 1024)
    images = [_resize_for_vlm(img, max_image_side) for img in images]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # max_pixels caps Qwen2.5-VL's dynamic patch count (28×28 px/patch).
    # Default 1M pixels → ~1000 patches/image × 2 images = ~2000 visual tokens → slow.
    # 401408 = 512 patches/image → ~1024 tokens total, ~2× faster with minimal quality loss.
    # Set attribute directly (guaranteed path) AND pass as kwarg (forwarded in newer transformers).
    max_px = config.get("max_pixels", 401_408)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = max_px
    inputs = processor(
        text=text, images=images, return_tensors="pt", max_pixels=max_px
    ).to("cuda")
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


# ── Batched inference (Qwen-style only) ───────────────────────────────────────
def _infer_batch(model, processor, messages_list: list, images_list: list, config: dict) -> list[str]:
    """Run a batched forward pass for multiple (photo, render) pairs.

    images_list must be flat-interleaved: [photo_0, render_0, photo_1, render_1, ...]
    The processor assigns images to each text string by counting <image> tokens, so
    order is critical. The caller (annotate_batch) is responsible for building this order.
    """
    import torch

    max_new_tokens = config.get("max_new_tokens", 256)
    max_image_side = config.get("max_image_side", 1024)
    max_px = config.get("max_pixels", 401_408)

    all_images = [_resize_for_vlm(img, max_image_side) for img in images_list]

    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = max_px

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    inputs = processor(
        text=texts, images=all_images, return_tensors="pt", padding=True, max_pixels=max_px
    ).to("cuda")

    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise MemoryError(f"GPU OOM in batch inference: {exc}") from exc
        raise
    except Exception as exc:
        # Catch torch.cuda.OutOfMemoryError if available (PyTorch >= 1.13)
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            torch.cuda.empty_cache()
            raise MemoryError(f"GPU OOM in batch inference: {exc}") from exc
        raise

    # All rows in inputs.input_ids are the same padded length — slice uniformly.
    generated = [output_ids[i][inputs.input_ids.shape[1]:] for i in range(len(texts))]
    return processor.batch_decode(generated, skip_special_tokens=True)


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

    style = config.get("inference_style", "qwen_style")
    if style == "internvl":
        infer_fn = lambda msgs: _infer_internvl(model, config["repo"], msgs, images, config)
    elif style == "molmo":
        infer_fn = lambda msgs: _infer_molmo(model, config["repo"], msgs, images, config)
    elif style == "minicpm_v":
        infer_fn = lambda msgs: _infer_minicpm_v(model, config["repo"], msgs, images, config)
    else:
        infer_fn = lambda msgs: _infer(model, processor, msgs, images, config)

    t0 = time.perf_counter()
    raw = infer_fn(messages)

    try:
        data = _parse_output(raw)
    except json.JSONDecodeError:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _RETRY_PROMPT},
        ]
        raw = infer_fn(retry_messages)
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


def annotate_batch(
    pairs: list[tuple],
    model_key: str | None = None,
    config_path: Path | None = None,
    _batch_size_override: int | None = None,
) -> list[dict | Exception]:
    """Annotate multiple (photo_path, render_path) pairs in a single batched GPU call.

    Returns a list of the same length as ``pairs``. Each element is either an
    annotation dict (same shape as ``annotate()`` returns) or an Exception instance.

    Batch path is Qwen-style only. Models with ``inference_style`` set fall back to
    sequential single-item ``annotate()`` calls automatically.

    ``images_list`` passed to ``_infer_batch`` is flat-interleaved:
    [photo_0, render_0, photo_1, render_1, ...]. The processor counts <image> tokens
    per text string to assign images — the order here is what makes that work.
    """
    pairs = [(Path(p), Path(r)) for p, r in pairs]

    # Pre-validate: store FileNotFoundError for any missing path immediately.
    results: list[dict | Exception | None] = [None] * len(pairs)
    valid_indices: list[int] = []
    for i, (photo_path, render_path) in enumerate(pairs):
        if not photo_path.exists():
            results[i] = FileNotFoundError(f"Photo not found: {photo_path}")
        elif not render_path.exists():
            results[i] = FileNotFoundError(f"Render not found: {render_path}")
        else:
            valid_indices.append(i)

    if not valid_indices:
        return results  # type: ignore[return-value]

    # Load config and model once for all valid pairs.
    config = _load_config(model_key, config_path)
    key = config["model_key"]
    model, processor = _load_model(key, config)

    # Compatibility gate: non-qwen_style models use their own chat APIs — fall back
    # to sequential single-item annotate() calls.
    if config.get("inference_style"):
        for i, (photo_path, render_path) in enumerate(pairs):
            if isinstance(results[i], Exception):
                continue
            try:
                results[i] = annotate(photo_path, render_path, model_key=model_key, config_path=config_path)
            except Exception as exc:
                results[i] = exc
        return results  # type: ignore[return-value]

    eff_batch = min(len(valid_indices), _batch_size_override or config.get("vlm_batch_size", 1))

    # Fall back to sequential if batch size is 1 or only one valid pair.
    if eff_batch <= 1 or len(valid_indices) <= 1:
        for idx in valid_indices:
            photo_path, render_path = pairs[idx]
            try:
                results[idx] = annotate(photo_path, render_path, model_key=model_key, config_path=config_path)
            except Exception as exc:
                results[idx] = exc
        return results  # type: ignore[return-value]

    # GPU batch path.
    messages = _build_messages(config["prompt"], config.get("backend", "transformers"))

    # Build flat-interleaved images_list and replicated messages_list.
    images_list: list = []
    messages_list: list = []
    for idx in valid_indices:
        photo_path, render_path = pairs[idx]
        photo = ImageOps.exif_transpose(Image.open(photo_path).convert("RGB"))
        render = Image.open(render_path).convert("RGB")
        images_list.extend([photo, render])   # interleaved: photo first, then render
        messages_list.append(messages)

    try:
        t0 = time.perf_counter()
        raw_outputs = _infer_batch(model, processor, messages_list, images_list, config)
        latency_s = (time.perf_counter() - t0) / len(valid_indices)

        for list_pos, idx in enumerate(valid_indices):
            raw = raw_outputs[list_pos]
            try:
                data = _parse_output(raw)
            except json.JSONDecodeError:
                # Per-item retry using single-item _infer() — NOT _infer_batch().
                photo_path, render_path = pairs[idx]
                photo = ImageOps.exif_transpose(Image.open(photo_path).convert("RGB"))
                render = Image.open(render_path).convert("RGB")
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _RETRY_PROMPT},
                ]
                raw = _infer(model, processor, retry_messages, [photo, render], config)
                data = _parse_output(raw)
            try:
                _validate(data)
            except VLMSchemaError as exc:
                results[idx] = exc
                continue
            results[idx] = {
                "model_id": key,
                "rating": data["rating"],
                "misaligned": data.get("misaligned", []),
                "unwanted_features": data.get("unwanted_features", []),
                "fail_patterns": data.get("fail_patterns", []),
                "notes": data.get("notes", ""),
                "raw_output": raw,
                "latency_s": latency_s,
            }

    except MemoryError:
        if len(valid_indices) == 1:
            raise
        # Halve the batch and recurse.
        mid = len(valid_indices) // 2
        first_half_pairs = [pairs[i] for i in valid_indices[:mid]]
        second_half_pairs = [pairs[i] for i in valid_indices[mid:]]
        new_batch_size = max(1, eff_batch // 2)
        first_results = annotate_batch(
            first_half_pairs, model_key=model_key, config_path=config_path,
            _batch_size_override=new_batch_size,
        )
        second_results = annotate_batch(
            second_half_pairs, model_key=model_key, config_path=config_path,
            _batch_size_override=new_batch_size,
        )
        for list_pos, idx in enumerate(valid_indices[:mid]):
            results[idx] = first_results[list_pos]
        for list_pos, idx in enumerate(valid_indices[mid:]):
            results[idx] = second_results[list_pos]

    return results  # type: ignore[return-value]
