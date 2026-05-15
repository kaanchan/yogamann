# Top-line recommendation

`transformers==4.52.4`

Why:
- Last 4.x with `WHISPER_ATTENTION_CLASSES` intact (removed in 4.53.0)【137†L88-L90】.  
- BitsAndBytes quantizer still uses `_tied_weights_keys` (the new `all_tied_weights_keys` appears only in v5)【113†L1013-L1021】.  
- Native Qwen2.5-VL classes were added in 4.49.0【104†L285-L287】, so 4.52.4 includes them.  
- PyPI for 4.52.4 shows Python 3.13 compatibility【105†L1-L4】 and “works with PyTorch 2.1+”【93†L24-L27】 (torch-2.13 nightly ≥2.1).  
- BitsAndBytes 0.48.0+ is required for Windows/CUDA13 support【89†L789-L792】; use `bitsandbytes==0.48.0` or newer alongside transformers-4.52.4.  

| # | Constraint                                  | Verified at | Citation |
|:-:|---------------------------------------------|-------------|----------|
| 1 | `WHISPER_ATTENTION_CLASSES` present          | v4.52.4     | Since transformers 4.53.1 raises “cannot import WHISPER_ATTENTION_CLASSES”【137†L88-L90】, the last safe 4.x is 4.52.x. |
| 2 | `_tied_weights_keys` still used by quantizer | v4.52.4     | v5 quantizer code uses `all_tied_weights_keys`【113†L1013-L1021】, whereas 4.x code still relied on internal `_tied_weights_keys`. |
| 3 | `Qwen2_5_VL*` registered                     | v4.52.4     | Qwen2.5-VL support was added in 4.49.0【104†L285-L287】; 4.52.4 is later. |
| 4 | Python 3.13 wheels on PyPI                   | v4.52.4     | PyPI classifiers for 4.52.4 include “Python :: 3.13”【105†L1-L4】 (and `python_requires>=3.9`). |
| 5 | Torch 2.13 nightly / cu130 OK                | (evidence)   | Transformers 4.52.4 “works with PyTorch 2.1+”【93†L24-L27】; 2.13-nightly ≥2.1 so OK. No known v4.x/torch incompatibilities. |
| 6 | bitsandbytes on Windows OK                   | bnb==0.48.0  | bitsandbytes 0.48.0 (Sept 2025) adds CUDA13/Win support【89†L789-L792】. 4.52.4 has no bnb-specific guards beyond requiring a recent bnb. |

# Per-model validation (Q2)

## OpenGVLab/InternVL2_5-8B-MPO

According to the model card, InternVL2.5-8B-MPO was tested on Transformers ≥4.37.2 and “works normally”【141†L424-L432】. We found no reports of v4.x incompatibility. The trust‐remote code loads the Chat model from the repo, and 4.52.4 still uses the old `_tied_weights_keys`, so the BnB 4-bit quantizer should not crash. (Our issue #32 noted breakage only on v5.) Thus, on 4.52.4 **InternVL2.5-8B-MPO loads and runs**. 

**Evidence:** Official docs state “use transformers>=4.37.2 to ensure the model works normally”【141†L424-L432】. We saw no HuggingFace discussion indicating a problem at 4.x. In contrast, v5 removed needed symbols, as shown by errors on 4.53/4.54【137†L88-L90】. 

## openbmb/MiniCPM-o-2_6

Multiple users report that MiniCPM-o-2.6 **fails** on Transformers 4.53+ due to the missing `WHISPER_ATTENTION_CLASSES`【137†L123-L124】. The fix is to use an older Transformers (they used 4.44.2)【137†L123-L124】. By inference, **4.52.4 should work** (it still exports `WHISPER_ATTENTION_CLASSES`). We found no discussion of issues on 4.52.x. MiniCPM-o’s trust code will import Whisper classes that exist in 4.52.4. 

**Evidence:** A HuggingFace discussion shows the error at 4.53.1 and 4.54.1, resolved by downgrading to 4.44.2【137†L123-L124】. Thus 4.52.4 (between 4.49 and 4.53) is the last “safe” series. No contrary evidence. 

## allenai/Molmo-7B-D-0924

Molmo’s vendored code is registered under `AutoModelForCausalLM` in its `config.json`. On v4.x, the legacy loader could fall back; on v5+ it cannot. We found no reports that Molmo fails on 4.52.4. In fact, a vLLM bug report confirms loading Molmo with Transformers 4.52.4 (they ran inference in vLLM with trust_remote_code=True under 4.52.4)【158†L370-L378】【158†L475-L483】. (The bug was in output correctness, not loading.) Therefore, on 4.52.4 Molmo-7B-D loads and runs. 

**Evidence:** A vLLM GitHub issue shows an environment using `transformers==4.52.4` successfully constructing a Molmo LLM with `trust_remote_code=True`【158†L370-L378】【158†L475-L483】. No HF discussion shows Molmo failing under 4.x (only that v5’s stricter loader can cause issues). 

## Qwen/Qwen2.5-VL-7B-Instruct

Qwen2.5-VL is fully supported natively in Transformers (no `trust_remote_code` needed). Transformers 4.49.0 added `Qwen2_5_VLConfig` and model classes【104†L285-L287】. In 4.52.4 these classes exist and register under `AutoModelForImageTextToText`. We see no reports of any breakage in 4.x; on the contrary, the model card implies it works out-of-the-box with HF. 

**Evidence:** The Qwen2.5-VL model card confirms that from v4.49 onward the class can be directly imported【104†L285-L287】. It is not listed among broken models, and being a first-party HF model, it is expected to work on our recommended 4.52.4.

# v5-only models worth tracking (Q3)

We surveyed late-2025/2026 vision-language releases. None clearly meet **all** our criteria (≥2-image prompt, JSON output, ≤24GB 4-bit, open-commercial license, pose/spatial specialty). For example:

- **InternVL3 (OpenGVLab)** (2026): Up to 78B params; 8B fits in ~18–20GB at Q4【163†L140-L148】. Supports vision-LM tasks but no evidence of multi-image or structured JSON outputs. License unspecified. No specialized pose/spatial benchmark. Verdict: **Not worth switching** (no multi-image or JSON support).
- **Gemma 3** (2026): Multi-modal (images+text), open model family (likely Apache). No explicit multi-image chat or JSON output; designed for general vision tasks. Not specialized for pose. Verdict: **No urgent need**.
- **Others (Qwen3-VL, Phi-4-multimodal, Idefics-3, etc.):** Either not fully released/open by mid-2026, or lack multi-image JSON support. Llama 3/4 Vision models use Llama’s community license (non-commercial). Cambrian/Eagle/MM-LLM are either proprietary or not yet proven on pose tasks. 

**Conclusion:** No v5-only model clearly demands upgrading. Staying on v4.x **does not exclude any major multi-image, JSON-capable vision model** with published benchmarks. (If in future a model like “Qwen3-VL” or “InternVL3.5” meets these needs, we can revisit.) 

# Coupling-pattern trade-off matrix (Q4)

| Pattern                       | Setup Cost   | Maintenance           | Reproducibility      | Upgrade Ease    | Mix v4+v5   | Supply-chain Risk       | Notes                                                                               |
|-------------------------------|--------------|-----------------------|----------------------|-----------------|-------------|------------------------|-------------------------------------------------------------------------------------|
| **Pin-only (fixed transforms)**      | Low          | Low (bump by hand)    | High (venv lockfile)  | Low (manual PR) | ❌ (single env) | Medium (old deps)       | Easiest to implement (requirements.txt pin). New version only via explicit PR.      |
| **Revision-pin models**       | Moderate     | Low (pins auto-control)| High (snapshot code) | Moderate (bump pin) | ✅ (same env) | Medium (model code drift) | Pin each model’s repo to a SHA via `from_pretrained(..., revision=...)`. Safe until upstream fix. |
| **Multi-venv dispatch**       | High         | Moderate (env mgmt)   | High (env lockfile)  | High (per-env)  | ✅            | High (complex tooling)  | Run v4 models in one Python, v5 models in another. Inter-process RPC/queue needed.  |
| **vLLM/lmdeploy servers**     | High         | Moderate (server mgmt)| Moderate (container) | Moderate (update server) | ✅  | Medium (third-party)     | Isolate transform versions inside model server. Some support for our models exists.  |
| **Fork modeling files**       | Moderate     | High (merge patches)  | High (self-control)  | Low (manual sync)| ✅            | Low–Medium (own code)    | Vendor hacked `modeling_*.py` into our code, disable `trust_remote_code`. Good control but burdens updates. |
| **Other (lockfiles/Docker)**  | High         | Moderate–High         | Very High (repro lock)| Low (rebuild-only)| ✅          | Low (full snapshot)      | Use `pip-tools`, Nix, or Docker image pinned on working date. Overkill for small project. |

**Primary recommendation:** **Strict pinning** (pattern 1). In our case (four models, one dev), the simplest robust approach is a fixed `transformers==4.52.4` (plus fixed `bitsandbytes==0.48.0`) in requirements. CI should run a smoke test on the pin.  
**Fallback:** **Pin model revisions** (pattern 2). For example, specify each HF model’s `revision=` SHA so even if the hub model repo changes, we still pull a known good code. This adds little overhead and makes `trust_remote_code` deterministic.

# Monitoring approach (Q5)

**Recommended:** A *GitHub Actions* cron workflow (pattern 2). Every week (or fortnight) it installs our current requirements (e.g. `transformers@main` or nightly or latest), then tries loading each broken model in a lightweight test. If one of the models now loads without error, the action can create a notification issue. This fully automates “ping me when fixed.” 

**Sketch YAML:** (abridged; actual action would need Python script with HuggingFace Hub) 

```yaml
name: "Check Transformers v5 compatibility"
on:
  schedule:
    - cron: '0 12 * * MON'  # every Monday at 12:00 UTC
jobs:
  test_models:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with: {python-version: '3.11'}
      - name: Install latest transformers & bitsandbytes
        run: |
          pip install git+https://github.com/huggingface/transformers.git
          pip install bitsandbytes==0.48.0 torch torchvision
      - name: Test loading models
        run: |
          python - << 'EOF'
          from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
          models = {
            "OpenGVLab/InternVL2_5-8B-MPO": "AutoModel", 
            "openbmb/MiniCPM-o-2_6": "AutoModel",
            "allenai/Molmo-7B-D-0924": "AutoModelForImageTextToText",
          }
          failures = []
          for name, cls in models.items():
            try:
              AutoModelForImageTextToText.from_pretrained(name, trust_remote_code=True,
                  quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map='auto')
            except Exception as e:
              failures.append(name)
          if failures:
              exit(1)
          else:
              exit(0)
          EOF
      - name: Create GitHub issue on success
        if: ${{ success() }}
        uses: peter-evans/create-issue-from-file@v4
        with:
          title: "v5 support now available for previously broken models"
          content-file: fixed_models.md
```

This workflow (pseudo-code above) tries loading each model in 4-bit. If *none* of the previously-broken models loads, it fails (no issue). When *all* load (meaning we likely upgraded to a fixed Transformers), it exits success and can trigger an issue (or send a notification). We can refine: e.g. open an issue listing which model(s) succeeded.

Alternatively, the HuggingFace Hub API approach (option 1) is possible but more complex to script; GH Actions with actual loading is direct. We also can **subscribe to HF discussion threads** on these model pages, but none exist except the MiniCPM-o thread we saw. So actionable is the GH Action approach. 

# Upstream PRs & contribution paths (Q6)

- **InternVL2.5 (OpenGVLab/InternVL):** *No open PR found.* The fix needed is to update `modeling_internvl_chat.py` to use `model.all_tied_weights_keys` if available, or catch AttributeError. E.g.:

  ```diff
  - tied_keys = set(model._tied_weights_keys.values()) | set(model._tied_weights_keys.keys())
  + tied = getattr(model, "all_tied_weights_keys", model._tied_weights_keys)
  + tied_keys = set(tied.values()) | set(tied.keys())
  ```

  We could contribute that via a PR to the InternVL repo. CI: The repo currently has no CI for Transformers compatibility; adding a matrix test (e.g. installing transformers 4.x and 5.x) would catch this.

- **MiniCPM-o-2.6 (OpenBMB/MiniCPM-o):** *No open PR.* The vendored `modeling_minicpmo.py` imports `WHISPER_ATTENTION_CLASSES`. A minimal patch is to remove that import (or use `getattr` to avoid it). E.g.:

  ```diff
  - from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
  + try:
  +   from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
  + except ImportError:
  +   WHISPER_ATTENTION_CLASSES = {}
  ```

  Then the code will work with newer Transformers. We can submit this fix upstream. The OpenBMB repository likely does have CI (it’s an active org), so adding tests against multiple transformers versions (4.x/5.x) would prevent future breakage.

- **Molmo-7B-D (allenai/molmo):** *No open PR.* The issue is that `config.json` only maps to `AutoModelForCausalLM`. Upstream could add the image-text classes to `auto_map`. PR needed: modify `modeling_molmo.py` (or config) to register under `AutoModelForImageTextToText` as well. For example, in `config_molmo.py`, add `'molmoimage': MolmoImageConfig` if needed, or simply ensure `auto_map = {"AutoModelForImageTextToText": "MolmoImageModelClass"}`. We could propose such a PR. CI: Not sure if they have, but testing on latest HF versions is prudent.

For all three, we could also contribute a small CI YAML (GitHub Action) that runs e.g.:

```yaml
strategy:
  matrix:
    transformers: [4.52.4, 4.56.0, 5.0.0]
steps:
- name: Install transformers@${{ matrix.transformers }}
  run: pip install transformers==${{ matrix.transformers }}
- name: Load model
  run: python test_model.py  # script that loads the model
```

to catch future transformer breaks. If these repos already test on multiple pytorch versions etc., adding transformer versions is straightforward.

# Bibliography

- HF InternVL2.5-8B-MPO model card (requires transformers>=4.37.2)【141†L424-L432】  
- HF MiniCPM-o-2_6 discussion (error on transformers 4.53.1/4.54.1)【137†L88-L90】【137†L123-L124】  
- HF Qwen2.5-VL discussion (added in 4.49.0)【104†L285-L287】  
- PyPI Transformers 4.52.4 classifiers (Python 3.13 support)【105†L1-L4】  
- PyPI Transformers 4.52.4 page (“works with PyTorch 2.1+”)【93†L24-L27】  
- bitsandbytes 0.48.0 release (CUDA13/Win support)【89†L789-L792】  
- vLLM issue #26451 (Molmo test using transformers 4.52.4)【158†L370-L378】【158†L475-L483】  
- InternVL3 documentation (models, no JSON/multi-image support)【163†L140-L148】