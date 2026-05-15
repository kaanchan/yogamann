# Architecture Decisions and Lessons Learned

Living log of architectural decisions, ML-library version pinning, and trade-offs.
Newest entries at top. Each entry: what was decided, the reasoning, and what to do
differently next time.

---

## 2026-05-15 (late eve, post-orchestration)

### ADR-004 — Model-major batch orchestration with DB-driven resumability

**Decision:** `src/compare_vlm.py` runs N images × M models in a model-major
loop: outer iterates models (one resident at a time), inner iterates images
in micro-batches of K (default 25). Each (run_id, model_id) annotation is
committed to `vlm_annotations` immediately via `save_vlm_annotation`.
Resumption uses `get_unanalyzed_runs(conn, model_id)` to skip already-done
pairs unless `--force` is passed.

**Why:** RTX 5080 Laptop has 15.92 GB VRAM. Each VLM in bnb 4-bit peaks at
~7-9 GB during inference (weights + KV cache + activations on a 2-image
prompt). No two models fit simultaneously. The previous image-major loop
with a never-evicting `_MODEL_CACHE` would OOM on the second image of any
multi-model run — it only worked in our testing because each test was a
fresh Python process.

Model-major also has the right cost shape: total cold-load cost is
**N_models × 1**, not **N_models × N_images**. For a 1000-image run that's
~2 minutes of swap overhead vs. **hours**.

**Why not image-major with LRU eviction:** Same swap cost as model-major
(must evict every model on every image anyway) with worse cache locality.
The only place image-major wins is when you need *all model responses for
image i before moving to image i+1* (e.g., cross-model voting). Our DB
records each (image, model) pair independently, so this isn't a constraint.

**Micro-batch boundary purpose:** Not for commit cadence (each save auto-
commits). The boundary exists to (a) call `torch.cuda.empty_cache()` to
release transient activation memory between iterations, and (b) emit a
progress line every K images. K=25 was picked as a balance between
progress visibility and overhead — adjust via `--batch-size` if a model
turns out to have unusual memory characteristics.

**Resumability invariant:** "If a row exists in `vlm_annotations` for
`(run_id, model_id)`, that pair is done." This is the only state the
orchestrator reads — there's no separate progress file, no journal, no
crash-recovery code, no session/batch ID. Kill the process at any point;
on next invocation it filters out the completed rows and continues. The
`UNIQUE(run_id, model_id)` constraint on the table makes this trivially
correct. Today's run and last week's run are indistinguishable to the
filter — the state is pair-atomic, not session-tracked.

**Failure handling:** Per-image `try/except` increments an error counter
and continues to the next image. Three exception classes:
- `FileNotFoundError` — source photo or render missing on disk
- `VLMSchemaError` — model emitted unparseable JSON twice (after retry)
- `Exception` — anything else (OOM, hardware fault, etc.) — full traceback
  is printed but the run continues.

Errors do NOT cause re-annotation on the next run — if a particular
`(run_id, model_id)` always fails, it will be retried every time. To
mark a pair as "permanently broken," insert a stub annotation manually
or use `--force` deliberately. Acceptable for our scale; revisit if we
see frequent persistent failures.

**Empirical validation:** Confirmed working at this hardware (RTX 5080
Laptop, 15.92 GB) on commit `aec7524`:
- Smoke 1 (1 model, 1 image, all done): correctly identifies all-done state, 0s elapsed
- Smoke 2 (4 models, 3 images, mixed done/todo): 6 ok / 0 err / 6 skipped, 113.6s, no OOM
- Smoke 3 (re-run Smoke 2 immediately): 0 ok / 0 err / 12 skipped, 0s elapsed (no model loads)
- Smoke 4 (`--force` on 3 Qwen-done images): 3 ok / 0 err / 0 skipped, 54.6s (full re-annotation)

**Considerations for future choices:**

- The micro-batch boundary is also the right place to add per-batch
  metrics writes (mean latency, retry rate, etc.) if we want timeseries
  data later. Currently we only emit a progress print — adding metrics
  is a small change inside `_run_model_phase`.
- The `--force` flag is "ignore the DB filter, redo everything." A
  cheaper version would be "redo only failed annotations" — if we want
  that, add an `--errors-only` flag that filters to `(run_id, model_id)`
  pairs where the last annotation has an `[error]` rating. Easy follow-up.
- For multi-GPU workstations: this orchestrator is single-GPU. Adding
  GPU-parallel model phases (e.g., one model per GPU running concurrently)
  is a separate change in scope to a future ADR.
- If/when we need batch identity (for "show me yesterday's batch results"
  or "rerun batch X with current model code"), add a `batch_runs` table
  with `batch_id`, `started_at`, `args`, and tag each `vlm_annotations`
  row with the creating `batch_id`. Currently out of scope.

**Related:** #32 (transformers pinning + this orchestration),
[`src/compare_vlm.py`](../src/compare_vlm.py),
[`src/_batch_utils.py`](../src/_batch_utils.py),
[`src/vlm_inference.py`](../src/vlm_inference.py) (`evict_model`, `evict_all`)

---

## 2026-05-15 (eve, post-research)

### ADR-003 — Final empirical pin: `transformers==4.49.0`, 4 of 5 VLMs working

**Decision:** `transformers==4.49.0` pinned exactly (not a range) in `pyproject.toml`
and `requirements.txt`. This is simultaneously the **floor** (Qwen2.5-VL was added
in 4.49.0) and the **practical ceiling** (4.50.0+ refactors break the three
`trust_remote_code` VLMs we care about). 4.49.0 has no patch releases — it's a
singleton, hence the `==` rather than `>=4.49,<4.50`.

**Empirical results on this pin:**

| Model                  | Status | Notes |
|------------------------|--------|-------|
| Qwen2.5-VL-7B-Instruct | working | native; rating="good" |
| InternVL2.5-8B-MPO     | working | bnb 4-bit works; rating="poor" |
| Molmo-7B-D-0924        | working | needed fp16 dtype cast in `_infer_molmo`; rating="acceptable" |
| MiniCPM-V-2.6          | working | needed numpy-array normalize() monkeypatch; rating parsed |
| MiniCPM-o-2.6          | **working** (resolved 2026-05-15 late-eve) | needed `_patch_qwen2_init_weights_for_bnb`; rating="good", `misaligned=["torso"]`, 10.3s inference, 6.25 GB VRAM |

**Why earlier candidate pins failed:**

- `transformers==4.57.6` (initial `>=4.49,<5` guess): InternVL hit
  `'InternLM2ForCausalLM' has no attribute 'generate'` because v4.50 stopped
  auto-inheriting `GenerationMixin` from `PreTrainedModel`. Also DynamicCache
  refactor broke InternVL's `past_key_values[0][0].shape[2]` legacy indexing.
- `transformers==4.52.4` (Gemini synthesis): MiniCPM-V/o hit
  `'Resampler' object has no attribute '_initialize_weights'` because v4.50
  introduced `smart_apply` walking submodules calling a method their custom
  Resampler class never defined.

**Why 4.49.0 specifically:** Both refactors above happened in 4.50. Sticking at
the last pre-refactor release sidesteps them in one move.

**Three patches in `_infer_*` to get the trust_remote_code models running:**

1. **`_infer_molmo`** — bnb-quantized residual modules are fp16; processor
   returns fp32 inputs. Cast pixel_values to `model.dtype` before
   `generate_from_batch`. ~3 LoC.
2. **`_infer_minicpm_v`** — MiniCPM-V's `MiniCPMVImageProcessor` passes
   `np.ndarray` mean/std to `transformers.image_transforms.normalize`. v4.49's
   normalize() does `isinstance(mean, Sequence)` — ndarray fails this check,
   so mean gets replicated 3x into a (3,3) tensor that won't broadcast.
   Monkeypatch normalize() (both at `image_transforms.normalize` and the
   re-import in `image_processing_utils`) to coerce ndarray → list. ~10 LoC.
3. **`_extract_user_text` helper** — bug we'd been carrying: the retry path
   passes `content=<string>` rather than `content=[{"type": "text", ...}]`.
   The original extraction crashed. Made all dispatchers robust to both shapes.

**MiniCPM-o-2.6 — the one that refused to play, now resolved:**

MiniCPM-o uses a Qwen2 audio-language module as part of its omni-modal stack.
When loading with `BitsAndBytesConfig(load_in_4bit=True)`, the Qwen2 backbone's
native `_init_weights` (in `transformers.models.qwen2.modeling_qwen2`) calls
`.normal_(mean=0, std=...)` on already-bnb-quantized weights, which are
uint8/Byte tensors. PyTorch's `normal_kernel_cpu` is not implemented for Byte
dtype.

**Resolution (2026-05-15 late-eve):** Added `_patch_qwen2_init_weights_for_bnb()`
in `src/vlm_inference.py` — applied at module import. The patch wraps
`Qwen2PreTrainedModel._init_weights` with an early-return for uint8 weights:

```python
def _safe_init_weights(self, module):
    if hasattr(module, "weight") and module.weight is not None:
        if module.weight.dtype == torch.uint8:
            return  # already bnb-quantized; skip random re-init
    return _orig(self, module)
```

This is benign for the other Qwen2-based models in our suite — Qwen2.5-VL
and MiniCPM-V-2.6 don't trigger `_init_weights` on quantized weights in
their load path, so the patch is a no-op for them. Verified empirically:
Qwen2.5-VL regression after the patch lands still produces `rating="good"`
in 27.2s, identical to pre-patch behavior.

MiniCPM-o now loads in 19.5s at 6.25 GB VRAM with bnb 4-bit. Single-image
inference completes in 10.3s and emits valid JSON (`rating="good"`,
`misaligned=["torso"]`). Re-enabled in `profiles/vlm.yml` with
`inference_style: minicpm_v` since its `model.chat(image=None, msgs=[...],
tokenizer=)` API is identical to MiniCPM-V-2.6's.

The "Some weights were not initialized: tts.head_code.*" warnings during
load are harmless for vision use — those are TTS-specific parameters
we don't exercise.

**Alternative paths considered (now historical):**
- ~~(a) Disable bnb 4-bit and load in bf16~~ — not viable: 8B bf16 = ~16 GB,
  no KV cache room on 15.92 GB GPU.
- ~~(b) Find an even older transformers pin~~ — would have broken Qwen2.5-VL
  (added in 4.49.0).
- ~~(c) Vendor-fork MiniCPM-o's modeling file~~ — superseded by the simpler
  global Qwen2 patch.
- ~~(d) Use the vision-only sibling MiniCPM-V-2.6 only~~ — we kept both;
  now we have a 5-model suite.

**Considerations for current model choices (#29 VLM suite):**

- The dispatcher pattern (`inference_style:` in `vlm.yml`) is the right shape;
  adding a new VLM family is ~30 LoC + a `_infer_<style>` function.
- All three custom-code models needed *small* per-model patches to actually
  work, not just the loader fallbacks. Plan for "patch budget" when picking
  a new `trust_remote_code` VLM in the future — assume 1-3 small workarounds
  to get past version-coupling rot, beyond just loading.
- Schema compliance varies wildly across the four working models. Qwen and
  InternVL emit the exact JSON we ask for; MiniCPM-V invents its own rating
  scale; Molmo emits the right keys but populates them differently. This is
  a prompt-engineering issue, not an infrastructure one — handle in a future
  iteration.

**Considerations for future choices (#33 pose pipeline alternatives, and beyond):**

- The "dependency closure" criterion from ADR-001 is now sharpened: read every
  modeling file in the candidate model's HF cache *before* committing to it.
  Look for: unconditional `import` of heavy deps (tensorflow, deepspeed,
  flash_attn), references to private transformers symbols (anything starting
  with `_`), bnb-incompatible weight-init patterns.
- Prefer the "vision-only" variant of any model family that ships both vision
  and omni versions. The omni variant pulls in a whole audio stack (Whisper,
  TTS, vector quantization) plus an additional language backbone that doubles
  the surface area for version-pinning conflicts.
- A model that takes more than 2 monkeypatches to make work on the pinned
  transformers version is signalling that it will rot fast. Document the
  patches, run the comparison, then plan to drop or replace within a quarter.

**Related:** #32 (transformers pinning — this completes the empirical work),
#29 (VLM annotation pipeline — now has 4 working comparators),
[`src/vlm_inference.py`](../src/vlm_inference.py),
[`profiles/vlm.yml`](../profiles/vlm.yml)

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
