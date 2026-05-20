# Architecture Decisions & ADRs

**18 nodes · Community 7 · Cohesion 0.21**

`docs/architecture-decisions.md` · GitHub issues #29, #32, #37, #38

---

## What it is

The design memory of the project. Four Architecture Decision Records (ADRs) plus the GitHub issues that prompted them. High cohesion (0.21) because these nodes are densely interconnected — each issue links to an ADR, each ADR references the issue that caused it.

## The four ADRs

### ADR-001: Pin ML Libraries Like Production DB Drivers
**Decision:** Treat `transformers`, `diffusers`, `torch`, and other ML libraries as if they were database drivers — pin exact versions in `requirements.txt`, never float them.

**Why:** ML library APIs break silently between minor versions in ways that only surface at runtime (wrong output shapes, missing kwargs, changed default behavior). The pain is the same as a DB schema change but with no migration system.

### ADR-002: Per-Model Inference Dispatchers Pattern
**Decision:** Each VLM model gets its own private `_infer_*()` function. No shared inference path.

**Why:** The 5 models have incompatible APIs. A shared path would require a tangle of conditionals or an abstraction layer that leaks. Per-dispatcher keeps each backend isolated and testable.

### ADR-003: Final Empirical Pin — transformers==4.49.0
**Decision:** Pin `transformers` at exactly `4.49.0`.

**Why:** 4.49.0 is the last version before a breaking change in the Qwen2 `_init_weights()` method that causes a `NotImplementedError` with BitsAndBytes 4-bit quantization on uint8 tensors. This was diagnosed in Issue #32 after MiniCPM-o-2.6 started crashing. The fix was `_patch_qwen2_init_weights_for_bnb()` applied at module load in `vlm_inference.py`. ADR-003 documents both the diagnosis and the pin. **This is the most-referenced node in the graph (11 edges).**

### ADR-004: Model-Major Batch Orchestration with DB-Driven Resumability
**Decision:** In `compare_vlm.py`, iterate models in the outer loop and images in the inner loop.

**Why:** Consumer GPU VRAM cannot hold two models at once. Loading each model once per batch pass (rather than once per image) minimises total load time. DB-driven resumability (checking `vlm_annotations` before each run) makes the process safe to interrupt.

## GitHub issues tracked here

| Issue | Summary |
|-------|---------|
| #29 | VLM pose analysis feature — original feature request |
| #32 | Pin transformers version — led to ADR-003 |
| #37 | VLM prompt tightening — v2b promotion |
| #38 | MiniCPM-V JSON structure non-compliance |

## Connects to

- [VLM Inference Core](vlm-inference-core.md) — ADR-002 and ADR-003 directly shape its architecture
- [Batch Orchestration](batch-orchestration.md) — ADR-004 defines the model-major loop
- [VLM Model Registry](vlm-model-registry.md) — ADR-001 explains the version pins in `vlm.yml`
- [Batch Utilities & Model-Major Design](batch-utilities.md) — ADR-004 also lives there
