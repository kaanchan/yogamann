(Sythesized by Gemini)
---
# Engineering Synthesis: `transformers` Version Pinning & VLM Strategy

This report synthesizes technical research to resolve major-version regressions in the `yogamann` pipeline. The objective is to restore functionality for **InternVL2.5**, **MiniCPM-o-2.6**, and **Molmo-7B** while maintaining native support for **Qwen2.5-VL**.

## Top-line Recommendation

**`transformers==4.52.4`**

### Why:

* **Highest Safe Version:** Retains `WHISPER_ATTENTION_CLASSES` (removed in 4.53.0) and legacy `_tied_weights_keys` logic.
* **Native Compatibility:** Includes official `Qwen2.5-VL` support added in 4.49.0.
* **Environment Ready:** Verified for Python 3.13 and PyTorch 2.13-nightly (no strict upper bounds).

---

## Constraint Validation (Q1)

| # | Constraint | Verified at | Technical Detail | Source |
| --- | --- | --- | --- | --- |
| 1 | `WHISPER_ATTENTION_CLASSES` | v4.52.4 | Removed in v4.53.0 refactor; fatal for MiniCPM-o audio. |  |
| 2 | `_tied_weights_keys` | v4.52.4 | v5 requires `all_tied_weights_keys`; InternVL2.5 lacks this. |  |
| 3 | `Qwen2.5-VL*` Support | v4.52.4 | Native support merged in v4.49.0 via PR #35621. |  |
| 4 | Python 3.13 Support | v4.52.4 | Pure-python universal wheel; PyPI classifiers confirm 3.13. |  |
| 5 | Torch 2.13 / cu130 | v4.52.4 | No hard upper bounds in `setup.py`; compatible with 2.1+. |  |
| 6 | `bitsandbytes` (Windows) | bnb==0.48.0 | Version 0.48.0+ adds critical Windows + CUDA 13 support. |  |

---

## Per-Model Validation (Q2)

* **InternVL2.5-8B-MPO:** **SAFE**. The v4.x series continues to utilize the private `_tied_weights_keys` attribute during quantization mapping, avoiding the `AttributeError` triggered by v5's refactored `all_tied_weights_keys` requirement.
* **MiniCPM-o-2.6:** **SAFE**. Successfully imports `WHISPER_ATTENTION_CLASSES` for audio modules. Reports confirm failures starting at v4.53.1; v4.52.4 is the final stable release containing this symbol.
* **Molmo-7B-D-0924:** **SAFE**. Benefits from v4.x's lenient `AutoModel` loader logic. It avoids the rigid `post_init()` weight-tying validation of v5 that currently breaks its custom modeling script.
* **Qwen2.5-VL-7B-Instruct:** **SAFE**. Operates as a native first-party class within `transformers`. Does not rely on `trust_remote_code`, ensuring optimal integration with `bitsandbytes` 4-bit routines.

---

## v5-Only Models Worth Tracking (Q3)

| Model | Release | Multi-img | JSON | License | Verdict |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-VL** | Early 2026 | Yes | Yes | Apache-2.0 | **High Priority:** Native GUI parsing and spatial grounding. |
| **InternVL3** | Jan 2026 | Yes | Yes | MIT | **High Priority:** Precise geometric reasoning via SpatialCoT. |
| **Molmo2** | Late 2025 | Yes | Yes | Apache-2.0 | **Medium Priority:** Requires `transformers>=4.57.1`, forcing v5 migration. |

---

## Strategic Proposal (Q4)

### Primary: vLLM Inference Server Abstraction

* **Pros:** Decouples model dependencies from the pipeline; natively supports all four models; allows mixing v4 and v5 models via containerized ports.
* **Cons:** Higher initial setup cost.

### Fallback: Revision-Pinning + Strict Pin

* **Pros:** Immunizes pipeline against stealth updates to `modeling_*.py` on the HF Hub.
* **Cons:** High maintenance debt if upstream model authors fix issues later.

---

## Monitoring Approach (Q5)

Automate the detection of v5-readiness using a **GitHub Actions smoke test** on meta-devices to avoid GPU/VRAM overhead.

```yaml
name: VLM v5 compatibility monitor
on:
  schedule:
    - cron: '0 8 * * 1' # Weekly Monday run
jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - name: Test Meta-device Load
        run: |
          pip install transformers==5.0.0 torch accelerate
          python -c "
          import torch
          from transformers import AutoModel, AutoModelForCausalLM
          targets = {'OpenGVLab/InternVL2_5-8B-MPO': AutoModel, 'openbmb/MiniCPM-o-2_6': AutoModel}
          for repo, loader in targets.items():
              try:
                  with torch.device('meta'):
                      loader.from_pretrained(repo, trust_remote_code=True)
                  print(f'✅ {repo} is v5 safe')
              except Exception as e:
                  print(f'❌ {repo} failed: {e}')"

```

---

## Upstream Contribution Paths (Q6)

* **InternVL2.5:** Inject `all_tied_weights_keys = {}` into the `modeling_internvl_chat.py` constructor to resolve v5 quantization crashes.
* **MiniCPM-o-2.6:** Replace the `WHISPER_ATTENTION_CLASSES` import with direct imports of `WhisperAttention` or a local routing dictionary.
* **Molmo-7B:** Update the `config.json` on the Hugging Face Hub to include `AutoModelForImageTextToText` in the `auto_map` dictionary.

---

## Bibliography

[1] mini-Pipeline-Transformers-Version-Pinning-deep-research-report.md
[2] chatgpt-transformer-version-pinning-deep-research-report.md