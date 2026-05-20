# GPU Stability & Memory Management

**11 nodes · Community 13 · Cohesion 0.31**

`docs/research/issue-34-gpu-management/` · `Plan — VLM Batch Orchestration`

---

## What it is

The research corpus and design decisions for running 12–14 hour VLM inference sessions on a consumer GPU without crashes or memory leaks. High cohesion (0.31) — this community is tightly self-referential because all the concepts feed directly into the batch orchestration design.

## The core problem

Consumer GPUs on Windows run in **WDDM mode** (not TCC). WDDM shares VRAM with the Windows display compositor — VRAM is not exclusively owned by CUDA. This means:
- Available VRAM fluctuates (other applications, OS chrome, browser)
- OOM errors can appear even when model size suggests headroom
- Long inference sessions accumulate VRAM fragmentation

## The recommended GPU stability triad

From the Issue #34 SYNTHESIS (captured as a hyperedge in the graph):

| Tool | Role |
|------|------|
| **pynvml (nvidia-ml-py)** | Python wrapper around NVIDIA Management Library. Used to read GPU power draw and set power limits programmatically. |
| **nvitop** | Multi-threaded GPU profiling engine built on NVML. More accurate than `nvidia-smi` for per-process VRAM tracking. |
| **PyTorch CUDA Memory API** | `torch.cuda.empty_cache()`, `memory_reserved()`, `memory_allocated()` — used between model loads to ensure VRAM is actually released. |

## The 80% power cap decision

Community sweet spot identified by research: capping GPU TGP at 80% (e.g. ~320W for RTX 5080 at 400W TGP) reduces thermal throttling incidents during long runs without meaningful throughput loss. Lower cap = more stable temps = fewer mid-run crashes.

## WDDM VRAM constraint → model-major design

The graph has an INFERRED `rationale_for` edge: `Model-Major Batch Loop → WDDM VRAM Constraints`. This captures the causal link: the reason yogamann loads one model, annotates everything, then evicts — rather than keeping all models loaded — is specifically because WDDM mode makes multi-model residency unreliable on 12–16 GB cards.

## PM2 process supervisor

The Issue #34 research also identified PM2 (Node.js process manager) as a viable watchdog for very long inference runs — exponential backoff restart if the Python process crashes. Not currently integrated but documented as a fallback for overnight runs.

## Connects to

- [Batch Orchestration](batch-orchestration.md) — the 80% power cap and evict-between-models pattern are implemented there
- [VLM Inference Core](vlm-inference-core.md) — `evict_all()` / `evict_model()` are the VRAM management hooks
- [Architecture Decisions & ADRs](architecture-decisions.md) — ADR-004 was partly motivated by WDDM constraints
