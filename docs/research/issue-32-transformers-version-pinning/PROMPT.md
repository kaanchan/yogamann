# Deep Research Prompt — `transformers` version pinning for our local VLM pipeline (issue #32)

> **For the research agent:** This is a focused, technical research request from a small open-source project (`yogamann`). The reader of your final report is the project maintainer, who will paste your output into the GitHub issue tracker. Cite primary sources (HuggingFace model cards and discussions, the `huggingface/transformers` GitHub repo — releases, commits, blame, issues — Python packaging metadata on PyPI, vLLM / lmdeploy / SGLang release notes) wherever possible. Prefer permalinks (commit-pinned URLs) over latest-`main` URLs. Match the inline-citation style of the project's prior synthesis report (`【ref†Lx-Ly】` markers tied to a numbered bibliography), or, if that is awkward, use plain Markdown footnotes — but every non-obvious factual claim must be cited.

---

## 1. Context (one-paragraph stage-set)

We maintain a local-only Vision-Language-Model annotation pipeline that scores yoga-pose images. The pipeline takes pairs of images — `(source_photo, mannequin_render)` — and asks a VLM to emit a structured JSON rating (alignment errors, missing/extra body parts, free-text notes). We use HuggingFace `transformers` with `trust_remote_code=True` for several VLMs whose modeling code lives in the model repo, not in `transformers` itself. Recently, a fresh `pip install transformers` into our project venv (`yogamann`) pulled **`transformers==5.8.0`** (the current latest on PyPI as of mid-2026, installed without a version pin). This single unintentional major-version bump silently broke **three of our four supported models**. Only Qwen2.5-VL still loads. We need a defensible, pinned, reproducible recipe — and a longer-term strategy for when transformers and `trust_remote_code` modeling files drift apart.

---

## 2. Environment (assume these are fixed unless you find a hard blocker)

- **OS:** Windows 11, Python **3.13.5** (CPython, 64-bit)
- **PyTorch:** **2.13.0.dev** nightly built against **CUDA 13.0** (`cu130` wheel)
- **GPU:** Single consumer-grade card, **15.92 GB VRAM** budget (RTX 5080 Laptop GPU — corrected from the original "24 GB" filter used in the first deep-research pass; revisit any Q3 model recommendations exceeding 16 GB peak with that constraint in mind)
- **Quantization:** `bitsandbytes` 4-bit (`load_in_4bit=True`) — essential, our 8B models do not fit in fp16 on 16 GB once activations + KV cache are included
- **Inference API in use:** plain `transformers.pipeline(...)` or `AutoModelForImageTextToText.from_pretrained(...)` + processor; `trust_remote_code=True` for non-native models; we are open to switching to `vLLM` / `lmdeploy` / `SGLang` if there is a clear win
- **Constraint:** local inference only — no API calls to hosted endpoints. Reproducibility matters: the venv must rebuild from the lockfile months from now and still work.

---

## 3. Failing-model details (with exact errors observed on `transformers==5.8.0`)

These are the four models in our `vlm_inference.py`. Three break on v5.

### 3.1 `OpenGVLab/InternVL2_5-8B-MPO` — **BROKEN on v5**

- **Trigger:** Loading with `BitsAndBytesConfig(load_in_4bit=True)` and `trust_remote_code=True`.
- **Error (verbatim):**
  ```
  AttributeError: 'InternVLChatModel' object has no attribute 'all_tied_weights_keys'
  ```
- **Root cause:** OpenGVLab's hosted `modeling_internvl_chat.py` (the `trust_remote_code` file pulled from the HF Hub model repo) references the **old private symbol** `_tied_weights_keys`. In `transformers` v5 the bitsandbytes quantizer path (`transformers.quantizers.base.get_keys_to_not_convert`) was refactored to use a new public-ish attribute named `all_tied_weights_keys`. The OpenGVLab modeling file has not been updated. The crash happens inside the bnb quantizer when it walks the model and asks for tied-weight key lists.
- **Implication:** This is not a `transformers` bug we can patch with a kwarg — the model repo's vendored code is out of sync with v5.

### 3.2 `openbmb/MiniCPM-o-2_6` — **BROKEN on v5**

- **Trigger:** Any load path (`AutoModel.from_pretrained`, `AutoModelForCausalLM`, etc.) with `trust_remote_code=True`.
- **Error (verbatim):**
  ```
  ImportError: cannot import name 'WHISPER_ATTENTION_CLASSES'
  from 'transformers.models.whisper.modeling_whisper'
  ```
- **Root cause:** The OpenBMB modeling file imports `WHISPER_ATTENTION_CLASSES` (used by their audio branch — MiniCPM-o is omni-modal, includes Whisper for audio). That symbol was **removed in `transformers` v5** as part of the attention-implementation refactor (the per-impl class registry was consolidated). MiniCPM-o has not been updated.

### 3.3 `allenai/Molmo-7B-D-0924` — **BROKEN on v5**

- **Trigger:** Auto-loading via `AutoModelForImageTextToText.from_pretrained(...)` or any of the canonical VLM loaders.
- **Symptom:** The model's `auto_map` (in `config.json`) registers it under `AutoModelForCausalLM`, **not** under `AutoModelForImageTextToText`. In our code we attempt `AutoModelForImageTextToText` → fall back to `AutoModel` → both fail because v5's auto loaders no longer perform the legacy "try every Auto class" fallback dance that v4 was lenient about.
- **Implication:** Even with a code change in our loader to try `AutoModelForCausalLM` explicitly, Molmo's vendored `modeling_molmo.py` may carry other v5 incompatibilities — please verify.

### 3.4 `Qwen/Qwen2.5-VL-7B-Instruct` — **WORKS on v5**

- Native `transformers` support (no `trust_remote_code` needed). `Qwen2_5_VLConfig` and `Qwen2_5_VLForConditionalGeneration` are first-party classes. Loads cleanly with bnb 4-bit on v5.8.0.
- This is the **only model currently working** in the broken venv, which is why we noticed the regression at all.

---

## 4. Research questions

Please answer **each** of these. They are in priority order: Q1 is the blocker, Q2 validates Q1, Q3–Q6 are forward-looking strategy.

### Q1. Exact `transformers` version pin recommendation

What is the **latest** `transformers` release in the **4.x** series that simultaneously satisfies **all** of the following constraints? Verify each constraint independently by inspecting the relevant source file at the corresponding release tag on GitHub (e.g. `https://github.com/huggingface/transformers/blob/v4.XX.0/...`) — do not infer.

1. **`WHISPER_ATTENTION_CLASSES` is still exported** from `transformers.models.whisper.modeling_whisper` (i.e. `from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES` succeeds). Cite the release tag and the file/line where the symbol is defined. Also identify the **first release that removed** this symbol so we know the upper bound of safety.
2. **The bitsandbytes quantizer base** (`transformers/quantizers/base.py`, function `get_keys_to_not_convert`) **still uses `_tied_weights_keys`** (the private name) and **does not** require `all_tied_weights_keys`. Cite the release tag and the file/line.
3. **Native `Qwen2.5-VL` support** exists — i.e. `Qwen2_5_VLConfig` and `Qwen2_5_VLForConditionalGeneration` are registered in the auto class mappings. (This was added in **4.49.0** per the release notes; confirm and link to the PR / commit.)
4. **Has Python 3.13 wheels on PyPI** — verify by checking the release's `setup.py`/`pyproject.toml` `python_requires` and the PyPI page's available wheels (`pip index versions transformers` or the PyPI JSON API). If the package is pure-Python (`py3-none-any.whl`), state that explicitly.
5. **Compatible with torch nightly 2.13.x / cu130.** Check `transformers`' `setup.py` / `pyproject.toml` for `torch` lower/upper bounds at that release, and scan the release's known-issue list on GitHub for any "requires torch >= X" hard guards in the bitsandbytes / generation paths.
6. **Compatible with current `bitsandbytes`.** Identify which `bitsandbytes` version pairs cleanly with the recommended `transformers` pin on Windows + CUDA 13. (bnb on Windows has historically been a pain point — flag this explicitly if there is a known-good combo.)

**Output for Q1:** a single line of the form
```
transformers==4.XX.Y
```
plus a short table justifying each of the six constraints with a citation.

### Q2. Per-model validation on the recommended pin

For **each of the four models below**, separately confirm that, on your recommended `transformers` version from Q1, the model:
- (a) imports its `trust_remote_code` modeling file without `ImportError`,
- (b) loads with `BitsAndBytesConfig(load_in_4bit=True)` without `AttributeError`,
- (c) processes a 2-image prompt and emits text output.

Models:
- `OpenGVLab/InternVL2_5-8B-MPO`
- `openbmb/MiniCPM-o-2_6`
- `allenai/Molmo-7B-D-0924`
- `Qwen/Qwen2.5-VL-7B-Instruct`

Evidence that counts: HF Hub community-tab discussions where someone reports success on a specific `transformers` version; HF Hub `discussions` filed against the model repo confirming/denying compatibility; GitHub issues against the model's upstream repo (e.g. `OpenGVLab/InternVL`, `OpenBMB/MiniCPM-o`, `allenai/molmo`); blog posts; Reddit `r/LocalLLaMA` threads with explicit pip-freeze output. **Do not lump models together — give each its own paragraph and citations.** If you find a model that is *not* confirmed working on the recommended pin, say so plainly and suggest the closest version that is.

### Q3. Trending v5-only VLMs we would be excluding by staying on v4

What new (2025-late / 2026) open-weight vision-language models have been released that **require `transformers` v5+** and are relevant to our workload? Filter to models that meet **all** of these:

- Accept **≥2 images per prompt** natively (no chunked-inference workarounds)
- Reliably emit **structured JSON** (cite benchmark or model card evidence — `JSONSchemaBench`, IFEval-multimodal, model-card examples, etc.)
- Run locally on **≤16 GB VRAM** in 4-bit quantization (RTX 5080 Laptop budget — see §2 note about the corrected hardware constraint)
- Open weights with **permissive license** (Apache-2.0, MIT, Llama Community License with commercial OK — flag any non-commercial clauses)
- Show **competitive performance on pose / spatial / fine-grained-visual benchmarks** — e.g. `MMVP`, `RealWorldQA`, `BLINK`, `SpatialBench`, `BodyHands`, or anything pose-specific

Examples to investigate (non-exhaustive — find others if you can): `Qwen3-VL`, `InternVL3` / `InternVL3.5`, `Gemma 3` vision variants, `Pixtral`, `Llama 3.3 / 4 Vision` variants, `Phi-4-multimodal`, `Idefics3`, `Cambrian-1`, `Eagle`, `MM-LLM` family. For each one you surface, give a one-row table entry: name, release date, params, VRAM-Q4 estimate, multi-image support, JSON ability, license, pose/spatial score (if any), and one-sentence verdict on whether it's worth our switching cost. **If staying on v4.x means we miss nothing significant, say so plainly.**

### Q4. Community-verified patterns for the `trust_remote_code` × `transformers`-version coupling problem

This is a recurring pain point across the HF ecosystem: a model's hosted modeling code imports private/internal symbols from `transformers`, and any major-version bump silently breaks it for everyone. How are **production / serious-hobbyist teams handling this**? Evaluate each pattern below — give pros, cons, and at least one real-world example of a project using it (link to repo / blog / discussion):

1. **Strict pinning + manual upgrade gates** (`transformers==4.X.Y`, `bitsandbytes==Z`, etc. in `requirements.txt` or `pyproject.toml`; CI tests on the pin; upgrades are explicit PRs).
2. **HF Hub `revision=` pinning** of the model repo to a specific commit hash, so that `trust_remote_code` always pulls the same modeling file even if the model authors push a regression.
3. **Two (or N) venvs, dispatch by model** — separate Python environments for legacy-v4 models vs. v5-native models, with a thin process-boundary RPC between them.
4. **Inference-server abstraction** — running models behind **vLLM**, **lmdeploy** (TurboMind), or **SGLang** so that `transformers` version coupling is hidden inside the server's container/conda env. Note which servers support which of our four models as of 2026.
5. **Forking the modeling file** — vendor a known-good `modeling_*.py` into our own repo, patch it for v5 compat, and load with `trust_remote_code=False` against the local copy. (Trade-off: maintenance burden vs. supply-chain control.)
6. **Anything else** the agent finds in the wild — e.g. `huggingface_hub` snapshot caching with content-hash verification, `uv` lockfiles, Nix-style reproducible builds, Docker base images frozen at a known-good day, etc.

Output for Q4: a **trade-off comparison table** (pattern × dimensions: setup cost, ongoing maintenance, reproducibility guarantee, ease of upgrade, ability to mix v4-only + v5-only models, supply-chain risk) plus one recommended **primary** pattern and one **fallback** pattern for our specific situation (single-dev project, four models, mixed compat).

### Q5. Programmatic monitoring strategy

We want a "ping me when X is fixed" workflow so we are not manually re-testing every two weeks. Specifically, **when does each of our three broken models become v5-compatible upstream?**

Evaluate these monitoring approaches and recommend one (or a small combination):

1. **HF Hub API watcher** — periodically poll `huggingface_hub.HfApi.list_repo_commits(repo_id)` for each model and look for new commits touching `modeling_*.py` since a baseline date. Sample script structure welcome.
2. **GitHub Actions cron** — a scheduled workflow in our repo that pip-installs `transformers @ main` (or latest stable) plus the three broken model repos, tries `AutoModelForImageTextToText.from_pretrained(..., load_in_4bit=True)` on each (dummy GPU or CPU-meta-tensor smoke test), and opens a GitHub issue / writes to the repo when one of them starts passing.
3. **HF Hub community-tab subscription** — subscribe to discussion threads on each model repo where users are tracking the v5 issue. (Find the relevant threads if they exist — give URLs.)
4. **Anthropic / OpenAI / Gemini deep-research triggers** — out of scope, ignore.
5. **The `transformers` repo's own labels** — e.g. `Vision`, `bitsandbytes`, `trust_remote_code` — and a label-watcher Action.

For the recommended approach, give a concrete implementation sketch (workflow YAML or Python snippet — 10–30 lines), with the specific HF Hub / GitHub API endpoints called out.

### Q6. Upstream contribution opportunities

For each of our three broken models, search for **open PRs** (against the model's upstream code repo — typically the GitHub repo that mirrors the HF Hub modeling files) that fix the v5 compat issues we hit in §3. URLs and PR statuses please.

- `OpenGVLab/InternVL` (or wherever `modeling_internvl_chat.py` is canonically maintained) — any PR replacing `_tied_weights_keys` with `all_tied_weights_keys`?
- `OpenBMB/MiniCPM-o` — any PR replacing the `WHISPER_ATTENTION_CLASSES` import with the v5 equivalent (likely `WhisperAttention` directly + the new `attn_implementation` plumbing)?
- `allenai/molmo` — any PR registering the model under `AutoModelForImageTextToText` in addition to `AutoModelForCausalLM`?

If PRs exist but are stale, are they mergeable? If none exist, what is the minimal patch we (the project maintainer) could submit upstream? Could we contribute a **CI matrix** to each repo that tests against the last three `transformers` releases on every push, so this class of regression is caught at the source? Identify any existing CI in those repos to avoid duplication.

---

## 5. Output format I want back from you

A single self-contained Markdown report, structured as follows:

```markdown
# Top-line recommendation

`transformers==4.XX.Y` (one-line answer, no hedging in this section)

Why: <three-bullet summary, ≤60 words total>

# Constraint validation (Q1)

| # | Constraint                                  | Verified at | Citation |
|---|---------------------------------------------|-------------|----------|
| 1 | `WHISPER_ATTENTION_CLASSES` present          | v4.XX.Y      | [link]   |
| 2 | `_tied_weights_keys` still used by quantizer | v4.XX.Y      | [link]   |
| 3 | `Qwen2_5_VL*` registered                     | v4.XX.Y      | [link]   |
| 4 | Python 3.13 wheels on PyPI                   | v4.XX.Y      | [link]   |
| 5 | Torch 2.13 nightly / cu130 OK                | (evidence)   | [link]   |
| 6 | bitsandbytes on Windows OK                   | bnb==A.B.C   | [link]   |

# Per-model validation (Q2)

## InternVL2.5-8B-MPO
…verdict, evidence, citations…

## MiniCPM-o-2.6
…verdict, evidence, citations…

## Molmo-7B-D
…verdict, evidence, citations…

## Qwen2.5-VL-7B
…verdict, evidence, citations…

# v5-only models worth tracking (Q3)

| Model | Released | Params | VRAM-Q4 | Multi-img | JSON | License | Pose score | Verdict |
|-------|----------|-------:|--------:|:---------:|:----:|---------|-----------:|---------|
| …     | …        | …      | …       | …         | …    | …       | …          | …       |

# Coupling-pattern trade-off matrix (Q4)

| Pattern | Setup cost | Maintenance | Reproducibility | Upgrade ease | Mix v4+v5 | Supply-chain risk | Notes |
|---------|------------|-------------|-----------------|--------------|-----------|-------------------|-------|
| Pin only | …         | …           | …               | …            | …         | …                 | …     |
| Revision-pin model repos | … | … | …       | …            | …         | …                 | …     |
| Multi-venv dispatch | …  | …           | …               | …            | …         | …                 | …     |
| vLLM / lmdeploy server | … | …          | …               | …            | …         | …                 | …     |
| Fork modeling files | …  | …           | …               | …            | …         | …                 | …     |

Primary recommendation: …
Fallback: …

# Monitoring approach (Q5)

Recommended approach + implementation sketch (YAML or Python, 10–30 lines).

# Upstream PRs & contribution paths (Q6)

- InternVL2.5-MPO: [PR link or "no open PR — minimal patch sketch:"]
- MiniCPM-o-2.6: …
- Molmo-7B-D: …

# Bibliography

[1] …
[2] …
…
```

---

## 6. Acceptance criteria for a useful answer

The maintainer will judge your report against these:

1. **The recommendation in §"Top-line recommendation" is a specific, copy-pasteable version pin string** — `transformers==4.X.Y`, not a range, not "latest 4.x".
2. **Each of the four models gets its own verdict paragraph** in §"Per-model validation". No lumping.
3. **Every non-obvious factual claim has a citation** to a permalink — GitHub commit-pinned URL, PyPI release page, HF Hub commit SHA, dated blog post. "Latest" or "main" URLs are insufficient because they will drift.
4. **The trade-off matrix in §Q4 is filled in completely** — no empty cells, no "TBD". If a cell is genuinely unknowable, say "unknown — needs benchmark" and explain why.
5. **The monitoring sketch in §Q5 is runnable** — if it's a GitHub Actions YAML, it should at minimum parse as valid YAML and reference real action names. If it's Python, it should reference real `huggingface_hub` / `github` API methods.
6. **The agent flags genuine uncertainty.** If you cannot verify a constraint (e.g. you cannot find a known-good `bitsandbytes` version for Windows + CUDA 13 + transformers 4.X), say so — do not fabricate.
7. **The agent does not recommend upgrades we did not ask about.** Do not suggest replacing torch, dropping bitsandbytes for `quanto`, or switching to `mlx-vlm`, unless it is directly tied to one of the six numbered constraints in Q1.
8. **The report is ≤2,500 words** in total (excluding the bibliography). We are optimizing for signal density, not length.

---

## 7. Non-goals (do not spend tokens on these)

- Do **not** re-evaluate the four models against each other for accuracy on pose tasks — that work was already done in issue #29 (synthesis report at `docs/research/issue-29-vlm-pose-analysis/FINAL-synthesis.md`). Take their selection as given.
- Do **not** propose moving inference off the local machine.
- Do **not** propose CPU-only fallbacks. We have GPU, we will use it.
- Do **not** propose fine-tuning or LoRA work.
- Do **not** propose schema-enforcement libraries (`outlines`, `lm-format-enforcer`, `xgrammar`) — orthogonal to the version-pin question.
- Do **not** rewrite our pipeline architecture. Stay focused on the dependency-version problem.

---

## 8. Tone & format reminders

- Direct, technical, no marketing language.
- When you cite a HuggingFace model card or a GitHub release, give the **release tag / commit SHA**, not the floating `main`/`master` URL.
- When you cite a Reddit / X / forum post, include the **post date** in `YYYY-MM-DD` form and the author handle.
- If two sources contradict, **say so explicitly** and pick one — explain why.
- Code blocks should be **runnable as-is** wherever possible (i.e. complete imports, no `...` placeholders in critical paths).

---

## 9. One-shot summary of what success looks like

After the maintainer reads your report, they should be able to:
1. Run **one `pip install` command** with a pinned `transformers==4.X.Y` and have all four models work.
2. Add **one watcher** (GitHub Action or local cron) that pings them when InternVL / MiniCPM-o / Molmo become v5-safe.
3. Know **which v5-only models** (if any) they are leaving on the table by staying on v4.x — and have a clean decision criterion for when to migrate.
4. Have a **link** to at least one open upstream PR or a copy-pasteable patch for each broken model, so they can either +1 the PR or submit the fix themselves.

That is the deliverable.
