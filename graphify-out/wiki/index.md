# yogamann — Knowledge Graph Wiki

370 nodes · 536 edges · 36 communities · built 2026-05-18

This wiki is generated from the semantic knowledge graph of the yogamann codebase.
Each article describes one community: what it does, what files live in it, and how it connects to the rest of the system.

---

## Core Pipeline

These communities form the main runtime path from pose input to VLM annotation.

| Community | What it is | Article |
|-----------|-----------|---------|
| VLM Inference Core | The heart of the system — loads models, dispatches inference across 5 backends | [vlm-inference-core.md](vlm-inference-core.md) |
| Batch Orchestration | Multi-model batch runner — model-major loop, resumability, ETA reporting | [batch-orchestration.md](batch-orchestration.md) |
| Single-Run Analyzer | Lightweight poller — annotates one run at a time using the active VLM | [single-run-analyzer.md](single-run-analyzer.md) |
| Mannequin Generator | Batch-plans and drives `sd_make.py` to render 3D mannequin poses | [mannequin-generator.md](mannequin-generator.md) |
| Contact Sheet Builder | Assembles side-by-side comparison images (2-up and 4-up layouts) | [contact-sheet-builder.md](contact-sheet-builder.md) |

## Data & Storage

| Community | What it is | Article |
|-----------|-----------|---------|
| Database — Run & Annotation Queries | High-level DB queries: ingest, annotate, paginate, compare | [database-queries.md](database-queries.md) |
| DB Thumbnail & Test Fixtures | Low-level DB plumbing: connections, thumbnails, test helpers | [db-thumbnail-fixtures.md](db-thumbnail-fixtures.md) |

## User Interfaces

| Community | What it is | Article |
|-----------|-----------|---------|
| Review Gallery UI | Streamlit live gallery — VLM table, agreement matrix, inline inference trigger | [review-gallery-ui.md](review-gallery-ui.md) |
| Static Gallery Builder | Generates a static `output/index.html` from rendered PNG comparisons | [static-gallery-builder.md](static-gallery-builder.md) |

## Research & Design

| Community | What it is | Article |
|-----------|-----------|---------|
| Architecture Decisions & ADRs | ADR-001 through ADR-004, GitHub issues, and the reasoning behind key choices | [architecture-decisions.md](architecture-decisions.md) |
| VLM Model Registry | The 5 registered models, their VRAM tradeoffs, and the prompt variant system | [vlm-model-registry.md](vlm-model-registry.md) |
| Pose Pipeline Research | Issue #33 research corpus — pose estimators, renderers, observations | [pose-pipeline-research.md](pose-pipeline-research.md) |
| Pose Estimator Alternatives | Evaluated alternatives: DWPose, Sapiens, RTMPose, OpenPose, FLUX, SMPL-X | [pose-estimator-alternatives.md](pose-estimator-alternatives.md) |
| GPU Stability & Memory Management | Issue #34 research — power caps, WDDM constraints, long-run survival | [gpu-stability.md](gpu-stability.md) |

## Infrastructure & Utilities

| Community | What it is | Article |
|-----------|-----------|---------|
| Batch Utilities & Model-Major Design | `_batch_utils.py` helpers — `chunked()`, `format_summary()`, ADR-004 | [batch-utilities.md](batch-utilities.md) |
| Batch Lock (Process Safety) | File-based GPU lock — prevents concurrent batch runs | [batch-lock.md](batch-lock.md) |
| Windows Job Objects | Auto-kill wrapper for child processes using Win32 Job Objects | [windows-job-objects.md](windows-job-objects.md) |
| Progress Monitor Script | PowerShell `monitor.ps1` — live GPU/CPU dashboard in the terminal | [progress-monitor.md](progress-monitor.md) |
| Batch Runner Script | PowerShell `batch.ps1` — outer driver for per-directory render batches | [batch-runner-script.md](batch-runner-script.md) |
| RTX 5080 Diagnostics | One-off benchmark script validating GPU and yogamann imports | [rtx5080-diagnostics.md](rtx5080-diagnostics.md) |

## Feature Plans (tracked in docs/)

| Community | What it is | Article |
|-----------|-----------|---------|
| Fast Model Download | Plan + spec for `hf_transfer` Rust-based HuggingFace download acceleration | [fast-model-download.md](fast-model-download.md) |
| Gallery Missing Thumbnails Fix | Plan + spec for `get_or_create_thumbnail` lazy disk fallback | [gallery-thumbnails-fix.md](gallery-thumbnails-fix.md) |
| ControlNet Strength Sweep | `yoga_asana_cs_sweep` and `yoga_asana_strong` profiles — cond_scale 1.0–2.0 | [controlnet-strength-sweep.md](controlnet-strength-sweep.md) |

---

## God Nodes (most connected — core abstractions)

These are the highest-betweenness nodes. If you touch one of these, ripples spread wide.

1. `annotate()` — 17 edges — the single-image VLM entry point; everything feeds into or out of it
2. `main()` — 13 edges — the batch orchestrator's outer loop
3. `_run_model_phase()` — 12 edges — per-model inner loop; bridges batch ↔ inference ↔ DB
4. `annotate_batch()` — 11 edges — batched wrapper around `annotate()`
5. `ADR-003` — 11 edges — the transformers==4.49.0 pin; most referenced design decision
6. `render()` — 10 edges — mannequin render entry point
7. `Issue #33 SYNTHESIS` — 9 edges — the consolidated pose pipeline research doc
8. `open_db()` — 8 edges — DB connection entry point; bridges almost every subsystem
9. `get_unanalyzed_runs()` — 8 edges — the queue query; connects batch and single-run paths
10. `_parse_output()` — 8 edges — VLM JSON parser; handles schema validation and fallbacks

## Surprising Cross-File Connections

These were inferred by semantic analysis — not visible from imports alone.

- `Sapiens (Meta)` ↔ `DWPose` — both solve whole-body pose estimation; research doc ↔ README
- `FLUX.1-dev ControlNet` ↔ `SDXL` — parallel renderer architectures; research doc ↔ README
- `MediaPipe (v1 baseline)` ↔ `OpenPose (current baseline)` — superseded ↔ active pose detectors
- `research32_pin_4490` → `ADR-003` — the GitHub issue that *caused* the transformers pin decision
- `Model-Major Batch Loop` → `WDDM VRAM Constraints` — the design choice is explained by the GPU constraint

## Group Relationships (Hyperedges)

- **Five-Model VLM Suite** — Qwen2.5-VL · InternVL2.5 · Molmo · MiniCPM-V · MiniCPM-o
- **Core Pose Pipeline Stack** — DWPose → ControlNet OpenPose SDXL → SDXL render
- **VLM Batch Orchestration** — `compare_vlm.py` + `vlm_inference.py` + `_batch_utils.py`
- **Recommended Future Stack** — DWPose + HSMR/Blender + VLM evaluation
- **GPU Stability Triad** — pynvml power cap + nvitop monitoring + PyTorch cache management
