# Fast Model Download

**3 nodes · Community 21 · Cohesion 1.00**

`docs/superpowers/plans/2026-05-11-fast-model-download.md` · `docs/superpowers/specs/2026-05-11-fast-model-download-design.md`

---

## What it is

A completed feature plan and spec for accelerating HuggingFace model downloads using `hf_transfer` — a Rust-based wheel that replaces Python's `requests`-based downloader with a multi-threaded implementation.

## The feature

`hf_transfer` is an optional HuggingFace library that uses multiple parallel HTTP connections to download model shards. For large models (7–8B parameter VLMs = 4–16 GB in 4-bit), download time drops from ~30 min to ~5 min on a fast connection.

**How to enable:** Set `HF_HUB_ENABLE_HF_TRANSFER=1` in the environment before running `download_models.py` (or any `transformers` model load). The library intercepts the HuggingFace download path automatically.

**Implementation:** `src/download_models.py` pre-fetches all registered model weights. With `hf_transfer` installed and the env var set, it uses the fast path automatically.

## Connects to

- [VLM Model Registry](vlm-model-registry.md) — the models being downloaded
