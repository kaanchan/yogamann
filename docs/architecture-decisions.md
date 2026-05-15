# Architecture Decisions and Lessons Learned

Living log of architectural decisions, ML-library version pinning, and trade-offs.
Newest entries at top. Each entry: what was decided, the reasoning, and what to do
differently next time.

---

## 2026-05-15

### ADR-002 — Per-model inference dispatchers: pattern works, but `trust_remote_code` rot is orthogonal

**Decision:** `src/vlm_inference.py` supports a per-model `inference_style` config field
in `profiles/vlm.yml`. Each style routes through its own `_infer_<style>` function and
optionally its own auxiliary loader (e.g. tokenizer vs. processor). Adding a new style
is a closed, ~30 LoC change. The original "one `_infer` for everything" assumption was
wrong: HuggingFace `trust_remote_code` VLMs each ship their own non-unified inference
API and cannot be served by a single Qwen-shaped code path.

**Implemented styles as of this commit:**

- `qwen_style` (default) — `processor.apply_chat_template(...)` + `model.generate(...)`
- `internvl` — `model.chat(tokenizer, pixel_values, question, ..., num_patches_list=...)`
- `molmo` — `processor.process(images=, text=)` + `model.generate_from_batch(...)`

**Why:** During the issue #32 v4-downgrade investigation we discovered that fixing the
transformers version (v5 → v4.57.6) was necessary but not sufficient. Even on v4,
InternVL and Molmo could not be invoked via the Qwen-style `processor + generate` API
because their `AutoProcessor` returns a bare `CLIPImageProcessor` with no chat template,
and the models expose their text-conditioning through their own `chat()` / `process()`
methods. A dispatcher was unavoidable.

**The harder finding (orthogonal to dispatch):** Once the dispatcher pattern was in
place, each `trust_remote_code` model failed for *its own* reason that has nothing to
do with our code path:

- **InternVL2.5-MPO on transformers 4.57.6:** Four cascading v4.50+ rot points in its
  hosted `modeling_internlm2.py` — missing `GenerationMixin` inheritance, missing
  `generation_config` initialisation (because `GenerationMixin.__init__` never ran),
  legacy `past_key_values[0][0].shape[2]` indexing that v4.50 replaced with
  `DynamicCache.get_seq_length()`. We added monkeypatches for the first two; the
  third would need to patch the hosted modeling file directly.
- **Molmo-7B-D on transformers 4.57.6:** Its hosted modeling/preprocessing file has
  an unconditional top-level `import tensorflow`. `transformers.dynamic_module_utils.
  check_imports` scans for any top-level `import` and demands the package even if the
  code path is dead; installing TF for one model is a ~500 MB cost we declined.
- **MiniCPM-o-2.6:** Omni-modal — its modeling file unconditionally imports `soundfile`,
  `torchaudio`, and `librosa` because it can ingest audio. We installed all three
  during the v5 investigation before discovering the deeper `WHISPER_ATTENTION_CLASSES`
  removal rot. On v4 the symbol exists, but other v4.50+ rot is likely lurking.

**Lesson:** `trust_remote_code` models carry **multiple independent dependency contracts**
that are not surfaced by any package manifest:

1. transformers minor version (renamed/removed symbols, e.g. `WHISPER_ATTENTION_CLASSES`)
2. transformers major version refactors (GenerationMixin inheritance, DynamicCache)
3. Python runtime deps that look optional but aren't (tensorflow scanned even when unused)
4. Heavy auxiliary deps for capabilities we don't use (Molmo's TF, MiniCPM-o's audio stack)
5. The model's own per-instance API shape (no `AutoProcessor` standard for these)

**Considerations for current model choices:**

- The dispatcher pattern is now landed and the `qwen_style` path is the proven working
  one. Treat Qwen2.5-VL as the primary annotator until further notice.
- InternVL, MiniCPM-o, and Molmo dispatchers are scaffolded but not in production use.
  They are *learning artifacts* and reference implementations for when (a) upstream
  publishes v4.50+-compatible modeling files or (b) we decide to vendor-fork their
  modeling files and patch them ourselves.
- Re-test all three on every upstream model-card commit hash bump — old rot may fix,
  new rot may appear.

**Considerations for future choices (issue #29, #33, future VLM swaps):**

- A model card showing `model.chat(...)` calls in its inference example is a signal
  of `trust_remote_code` non-uniformity. Budget at least a half-day per such model
  for: dispatcher implementation, dependency archaeology, monkeypatch hunting,
  documentation. Compare against the half-hour cost of adopting a native-transformers
  VLM (which uses `AutoProcessor.apply_chat_template` out of the box).
- The "comparison study across N VLMs" framing in our original issue #29 plan
  assumed model interchangeability that doesn't exist for `trust_remote_code` models.
  Future multi-model studies should either (a) restrict candidates to native-transformers
  VLMs, or (b) accept the dispatcher per model as a first-class cost line item.
- Inference servers (vLLM, lmdeploy/TurboMind, SGLang) swallow this per-model API
  fragmentation at the cost of a process boundary. Re-evaluate them when issue #32
  deep research returns — see Q4 of the research prompt.

**The dispatcher code is itself reusable** — even if we drop these three models, the
pattern (config-driven `inference_style` + lambda dispatch in `annotate`) is the right
shape for the next time we add a non-Qwen-style VLM. Keep it.

**Related:** #32 (transformers pinning + per-model rot), #29 (VLM annotation pipeline),
[`src/vlm_inference.py`](../src/vlm_inference.py) — `_infer_internvl`, `_infer_molmo`,
`_patch_internvl_generation_mixin`

---

### ADR-001 — Pin ML libraries explicitly; treat them like production DB drivers

**Decision:** All ML library dependencies (starting with `transformers`, `torch`,
`accelerate`, `bitsandbytes`) are pinned in `pyproject.toml` / `requirements.txt` to a
specific major version with patch and minor updates allowed:

```
transformers>=4.49,<5
```

A major-version bump is never a routine `pip install` — it is a planned migration with
a full VLM/pose comparison re-run before it lands on `main`.

**Why:** This week a routine `pip install transformers` brought in `transformers==5.8.0`
(the v5 major released early 2026) and silently broke three of the four VLMs in the
issue #29 comparison suite. Every breakage was in a `trust_remote_code` model whose
hosted modeling file reached into transformers internals that v5 had reorganised:

- **InternVL2.5-8B-MPO** — bnb 4-bit quantiser crashed because OpenGVLab's hosted
  modeling code references `_tied_weights_keys`, which v5 renamed to
  `all_tied_weights_keys`.
- **MiniCPM-o-2.6** — import-time failure: the modeling file imports
  `WHISPER_ATTENTION_CLASSES` from `transformers.models.whisper.modeling_whisper`,
  a symbol v5 deleted as part of the attention-class refactor.
- **Molmo-7B-D** — registers only under `AutoModelForCausalLM`; v5's tighter
  auto-class resolution dropped the implicit fallback path the model relied on.

Only **Qwen2.5-VL-7B-Instruct** — the one model with native, non-`trust_remote_code`
support and an entry in the transformers auto registry — survived.

The core insight: `trust_remote_code` models are **silently coupled to transformers
minor versions**. They import private and internal symbols (`_tied_weights_keys`,
`WHISPER_ATTENTION_CLASSES`, internal mixins) with no version pin, no `requirements.txt`
inside the model repo, and no compatibility metadata. When a transformers release
reorganises internals, every custom-code model that wasn't updated by its maintainer
breaks. We have **no contract** with these models beyond "it worked once on the
transformers version we happened to have."

ML libraries are not casual dev dependencies — they are the equivalent of a production
database driver. A `psycopg2` major bump is never an absent-minded install; a
`transformers` major bump shouldn't be either.

**Considerations for current model choices (issue #29 VLM suite):**

- Until upstream maintainers republish v5-compatible modeling files, we stay on
  `transformers>=4.49,<5`. This keeps all four VLMs runnable today.
- Of the four, only Qwen2.5-VL is safe under arbitrary transformers minor bumps.
  Treat the other three as "works on the pin, may not survive the next major."
- Where two models are otherwise comparable, prefer the one with `config_class` in
  the transformers auto registry over the `trust_remote_code` one. This is now an
  explicit model-selection criterion alongside latency, VRAM, and quality.

**Considerations for future choices (issue #33 pose pipeline alternatives, and beyond):**

- When evaluating any new HF model, check whether it requires `trust_remote_code=True`
  before adding it to the candidate list. If yes, it is structurally less stable than
  a native-transformers alternative of similar quality — weight that into selection.
- A model + transformers version is a *joint* pin. Pinning `revision=<commit-hash>` in
  `from_pretrained` locks weights and modeling file, but does not lock the transformers
  version that file expects. Both must be pinned together; neither alone is sufficient.
- Same pattern applies to future major-version pivots (transformers v6, torch 3.x,
  CUDA major bumps): lag the ecosystem by 6–12 months for production inference.
  Bleeding-edge is for research scripts, not the pipeline.

**Upgrade protocol for any major-version bump:**

1. Branch from `main` and bump the pin in isolation.
2. Run the full VLM comparison: `compare_vlm.py --limit 5 --output-root ...`.
3. Run the pose pipeline smoke test on the same five images.
4. For each model that breaks: file an issue, decide whether to (a) drop the model,
   (b) patch the breakage locally and pin to a fork, or (c) wait for upstream.
5. Only land the upgrade on `main` once the decision is recorded for every model in
   the suite. Never let an unattended `pip install` decide for us.

**Monitoring obligation:** While pinned to v4, we want to learn when upstream model
code becomes v5-compatible so we can re-evaluate. Concrete mechanism deferred to
issue #32 (deep research on transformers pinning + upstream-watch tooling), but this
is now an accepted maintenance task, not a "we'll notice eventually."

**Trade-off we are accepting by pinning to v4:** Any VLM or auxiliary model released
after early 2026 that requires `transformers>=5` is off the table until we plan a
v5 migration. We do not enumerate those models here — that survey lives in issue #32
— but we acknowledge the trade-off explicitly rather than discovering it through a
broken install six months from now.

**Related:** #32 (transformers pinning + upstream-watch), #29 (VLM selection),
#33 (pose pipeline alternatives)

---

## How to use this file

- **Before adding any new ML dependency** → check the pinning rule above and confirm
  the library is pinned to a major version in `pyproject.toml` / `requirements.txt`.
- **Before adding any new HF model** → check whether it requires `trust_remote_code`;
  if yes, document why the native-transformers alternatives were rejected.
- **Before upgrading a pinned major version** → follow the upgrade protocol in
  ADR-001. No silent `pip install --upgrade`.

When adding a new ADR, use this structure:

```
### ADR-NNN — Short title

**Decision:** The rule or choice, stated concretely.
**Why:** What went wrong (or what would go wrong) without this decision.
**Considerations for current choices:** How it affects what we run today.
**Considerations for future choices:** How it affects what we evaluate next.
**Related:** Issue numbers and prior ADRs.
```
