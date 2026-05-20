# Graph Report - .  (2026-05-18)

## Corpus Check
- 72 files · ~155,626 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 370 nodes · 536 edges · 36 communities (24 shown, 12 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 93 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_VLM Inference Core|VLM Inference Core]]
- [[_COMMUNITY_Batch Orchestration|Batch Orchestration]]
- [[_COMMUNITY_Contact Sheet Builder|Contact Sheet Builder]]
- [[_COMMUNITY_Pose Pipeline Research|Pose Pipeline Research]]
- [[_COMMUNITY_Database — Run & Annotation Queries|Database — Run & Annotation Queries]]
- [[_COMMUNITY_Single-Run Analyzer|Single-Run Analyzer]]
- [[_COMMUNITY_Review Gallery UI|Review Gallery UI]]
- [[_COMMUNITY_Architecture Decisions & ADRs|Architecture Decisions & ADRs]]
- [[_COMMUNITY_DB Thumbnail & Test Fixtures|DB Thumbnail & Test Fixtures]]
- [[_COMMUNITY_VLM Model Registry|VLM Model Registry]]
- [[_COMMUNITY_Pose Estimator Alternatives|Pose Estimator Alternatives]]
- [[_COMMUNITY_Batch Utilities & Model-Major Design|Batch Utilities & Model-Major Design]]
- [[_COMMUNITY_Batch Lock (Process Safety)|Batch Lock (Process Safety)]]
- [[_COMMUNITY_GPU Stability & Memory Management|GPU Stability & Memory Management]]
- [[_COMMUNITY_Mannequin Generator|Mannequin Generator]]
- [[_COMMUNITY_Windows Job Objects|Windows Job Objects]]
- [[_COMMUNITY_Progress Monitor Script|Progress Monitor Script]]
- [[_COMMUNITY_Batch Runner Script|Batch Runner Script]]
- [[_COMMUNITY_Static Gallery Builder|Static Gallery Builder]]
- [[_COMMUNITY_RTX 5080 Diagnostics|RTX 5080 Diagnostics]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]

## God Nodes (most connected - your core abstractions)
1. `annotate()` - 17 edges
2. `main()` - 13 edges
3. `_run_model_phase()` - 12 edges
4. `annotate_batch()` - 11 edges
5. `ADR-003: Final Empirical Pin transformers==4.49.0` - 11 edges
6. `render()` - 10 edges
7. `Issue #33 SYNTHESIS — Yoga Pose Alignment Pipeline Consolidated Engineering Summary` - 9 edges
8. `open_db()` - 8 edges
9. `get_unanalyzed_runs()` - 8 edges
10. `_parse_output()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Sapiens (Meta) Pose Analyzer` --semantically_similar_to--> `DWPose Skeleton Detection`  [INFERRED] [semantically similar]
  docs/research/issue-33-pose-pipeline-evaluation/chatgpt-deep-research-response.md → README.md
- `FLUX.1-dev ControlNet Renderer` --semantically_similar_to--> `Stable Diffusion XL (SDXL)`  [INFERRED] [semantically similar]
  docs/research/issue-33-pose-pipeline-evaluation/chatgpt-deep-research-response.md → README.md
- `MediaPipe Pose — v1 Baseline Pose Detector (33 Landmarks, Superseded)` --semantically_similar_to--> `OpenPose — Current Baseline Pose Analyzer (2D Skeleton)`  [INFERRED] [semantically similar]
  profiles/yoga_asana_v1_archival.yml → docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md
- `conn()` --calls--> `open_db()`  [INFERRED]
  tests/test_db_thumbnail.py → src/db.py
- `conn()` --calls--> `open_db()`  [INFERRED]
  tests/test_db_vlm.py → src/db.py

## Hyperedges (group relationships)
- **Five-Model VLM Annotation Suite** — vlm_qwen25_vl_7b, vlm_internvl25_8b, vlm_molmo_7b, vlm_minicpm_v, vlm_minicpm_o [EXTRACTED 1.00]
- **Core Pose-to-Mannequin Pipeline Stack** — readme_dwpose, readme_controlnet_openpose_sdxl, readme_sdxl [EXTRACTED 1.00]
- **VLM Batch Orchestration Components** — src_compare_vlm_py, src_vlm_inference_py, src_batch_utils_py [EXTRACTED 1.00]
- **Recommended Pose Pipeline Stack: DWPose + HSMR/Blender + VLM Evaluation** — concept_dwpose, concept_hsmr_blender, concept_vlm_annotations_table [EXTRACTED 0.95]
- **VLM Model Registry: Qwen2.5-VL + InternVL2.5 + MiniCPM-o + MiniCPM-V + Molmo** — concept_qwen2_5_vl, concept_internvl2_5, concept_minicpm_o, concept_minicpm_v, concept_molmo_7b [EXTRACTED 1.00]
- **GPU Stability Triad: pynvml Power Cap + nvitop Monitoring + PyTorch Cache Management** — concept_pynvml, concept_nvitop, concept_torch_cuda_api [EXTRACTED 0.95]

## Communities (36 total, 12 thin omitted)

### Community 0 - "VLM Inference Core"
Cohesion: 0.07
Nodes (51): Exception, annotate(), annotate_batch(), _build_messages(), evict_all(), evict_model(), _extract_user_text(), _get_tokenizer() (+43 more)

### Community 1 - "Batch Orchestration"
Cohesion: 0.12
Nodes (25): _compute_slice_size(), _count_completed(), _fetch_run_candidates(), _fmt_eta(), _historical_avg_latency(), main(), src/compare_vlm.py — multi-model VLM annotation orchestrator.  Model-major loo, Mean latency_s for this model from prior annotations.     Returns (avg_seconds, (+17 more)

### Community 2 - "Contact Sheet Builder"
Cohesion: 0.11
Nodes (18): build_comparison_2(), build_comparison_4(), _depth(), _meta(), create_contact_sheet.py -- build side-by-side comparison images.  Two layouts:, DWposeDetector, ONNX-based DWPose detector — drop-in for controlnet_aux.DWposeDetector.  Uses rt, _render() (+10 more)

### Community 3 - "Pose Pipeline Research"
Cohesion: 0.12
Nodes (23): docs/ideal-targets Visual Target References, Issue #33: Alternative Pose Analyzers and Renderers, OBS-001: Outdoor Complex Backgrounds Produce Artifacts, OBS-002: Prone Back-Bends Not Captured Correctly, observations.json Machine-Readable Observations, RTMPose (OpenMMLab), Sapiens (Meta) Pose Analyzer, scripts/batch.ps1 Batch Runner (+15 more)

### Community 4 - "Database — Run & Annotation Queries"
Cohesion: 0.10
Nodes (16): count_vlm_annotations(), get_all_annotations(), get_or_create_user(), get_vlm_comparison_page(), ingest_all(), ingest_json(), make_thumbnail(), db.py — SQLite revision-history store for yogamann.  Schema ------   source_ (+8 more)

### Community 5 - "Single-Run Analyzer"
Cohesion: 0.21
Nodes (15): main(), _process_run(), src/analyze.py — poll DB and annotate unanalyzed runs with active VLM model., get_unanalyzed_runs(), get_vlm_annotations(), Returns runs with no vlm_annotations entry for model_id.     Each row exposes:, save_rating(), save_vlm_annotation() (+7 more)

### Community 6 - "Review Gallery UI"
Cohesion: 0.13
Nodes (11): get_thumbnail(), _human_badge_html(), _inference_terminal_dialog(), _launch_vlm_inference(), gallery.py — live review gallery for yogamann outputs (DB-backed).  Usage:, Spawn a compare_vlm.py subprocess for one (model, run) and store in session_stat, Modal showing live stdout from the active inference subprocess., _render_vlm_table() (+3 more)

### Community 7 - "Architecture Decisions & ADRs"
Cohesion: 0.21
Nodes (18): ADR-001: Pin ML Libraries Like Production DB Drivers, ADR-002: Per-Model Inference Dispatchers Pattern, ADR-003: Final Empirical Pin transformers==4.49.0, Issue #29: VLM Pose Analysis Feature, Issue #32: Pin Transformers Version, Issue #37: VLM Prompt Tightening, Issue #38: MiniCPM-V JSON Structure Non-Compliance, profiles/vlm.yml VLM Profile (+10 more)

### Community 8 - "DB Thumbnail & Test Fixtures"
Cohesion: 0.15
Nodes (17): get_or_create_thumbnail(), open_db(), _get_conn(), conn(), conn_with_source(), _make_jpeg(), When thumbnails table already has the sha256, return that blob directly., When thumbnails table misses, load from disk, return bytes, and write back. (+9 more)

### Community 9 - "VLM Model Registry"
Cohesion: 0.17
Nodes (17): BitsAndBytes 4-bit Quantization — VRAM Reduction for VLM Loading, InternVL2.5-8B-MPO — Secondary VLM Pose Analyzer Model, MiniCPM-o-2.6 — Omni-Modal VLM (Qwen2 Backbone BNB Patch Required), MiniCPM-V-2.6 — Lightweight VLM Pose Analyzer, Molmo-7B-D — Fallback VLM (Partial Multi-Image Support), prompt_tag — Named Prompt Variant Key for vlm_annotations, Qwen2.5-VL-7B-Instruct — Primary VLM Pose Analyzer Model, vlm_annotations DB Table — (run_id, model_id, prompt_tag) Unique Rows (+9 more)

### Community 10 - "Pose Estimator Alternatives"
Cohesion: 0.17
Nodes (15): DWPose — Recommended Pose Analyzer (133-keypoint whole-body), FLUX.1-dev + ControlNet Union — Best Diffusion Renderer, HSMR + Headless Blender — Recommended Parametric Renderer, MediaPipe Pose — v1 Baseline Pose Detector (33 Landmarks, Superseded), OpenPose — Current Baseline Pose Analyzer (2D Skeleton), RTMPose — Real-Time Pose Estimator (Exercise Tracking Accuracy), Sapiens — Meta Foundation Pose Model (High Accuracy, NC License), SDXL + ControlNet OpenPose — Current Baseline Renderer (+7 more)

### Community 11 - "Batch Utilities & Model-Major Design"
Cohesion: 0.19
Nodes (13): ADR-004: Model-Major Batch Orchestration with DB-Driven Resumability, chunked(), format_summary(), src/_batch_utils.py — pure-Python helpers for VLM batch orchestration.  No tor, Yield successive lists of up to `size` items from `items`.     Empty iterable y, Format a per-model batch-run summary as a table string., test_chunked_empty(), test_chunked_exact_divisor() (+5 more)

### Community 12 - "Batch Lock (Process Safety)"
Cohesion: 0.21
Nodes (13): acquire(), check(), is_locked(), _lock_path(), _pid_alive(), src/batch_lock.py — GPU-busy lock file for yogamann batch inference.  Prevents, Return True if the process with pid is currently running., Write the lock file and return its path.      Args:         output_root: Dire (+5 more)

### Community 13 - "GPU Stability & Memory Management"
Cohesion: 0.31
Nodes (11): 80% TGP Power Cap — Community Sweet Spot for Long Inference Runs, Model-Major Batch Loop — Load One Model, Annotate All Images, Evict, nvitop — Multi-threaded GPU Profiling Engine (NVML-based), PM2 Process Supervisor — Daemon Watchdog with Exponential Backoff Restart, pynvml (nvidia-ml-py) — NVML Python Wrapper for GPU Telemetry and Power Control, PyTorch CUDA Memory API (empty_cache, memory_reserved, memory_allocated), WDDM Mode VRAM Constraints — Consumer GPU Memory Sharing on Windows, Issue #34 ChatGPT Deep Research Report — GPU Stability (+3 more)

### Community 14 - "Mannequin Generator"
Cohesion: 0.31
Nodes (9): build_tasks(), _detect_format(), expand_value(), linspace(), next_free_stem(), pad_version(), make_mannequin.py – batch planner / driver 2025‑07‑20  ·  smarter versions  ·, Find & read a profile, guessing extension + format when omitted. (+1 more)

### Community 15 - "Windows Job Objects"
Cohesion: 0.25
Nodes (8): assign_process(), create_job(), _IO_COUNTERS, _JOBOBJECT_BASIC_LIMIT_INFORMATION, _JOBOBJECT_EXTENDED_LIMIT_INFORMATION, src/win_job.py — Windows Job Object wrapper for auto-kill of child processes., Create (or open) a named Windows Job Object with KILL_ON_JOB_CLOSE.      Retur, Assign the process identified by pid to the job object.      Returns True on s

### Community 17 - "Batch Runner Script"
Cohesion: 0.43
Nodes (4): Get-ImageFiles(), Get-OutputDir(), Invoke-DirBatch(), Invoke-Python()

### Community 18 - "Static Gallery Builder"
Cohesion: 0.53
Nodes (5): build_gallery(), _card(), _cfg_badges(), make_gallery.py — scan output/ for comparison PNGs and write output/index.html., _read_meta()

### Community 21 - "Community 21"
Cohesion: 1.00
Nodes (3): hf_transfer — Rust-based Fast HuggingFace Model Download Wheel, Plan — Fast Model Download (hf_transfer), Spec — Fast Model Download Design

### Community 23 - "Community 23"
Cohesion: 1.00
Nodes (3): get_or_create_thumbnail — Lazy Disk Fallback for Missing DB Thumbnails, Plan — Gallery Missing Thumbnails Fix, Spec — Gallery Missing Thumbnails Design

### Community 24 - "Community 24"
Cohesion: 0.67
Nodes (3): cond_scale Sweep — ControlNet Conditioning Strength Optimization (1.0-2.0), Profile — yoga_asana_cs_sweep (cond_scale Sweep), Profile — yoga_asana_strong (High cond_scale 2.0)

## Knowledge Gaps
- **25 isolated node(s):** `PreToolUse`, `allow`, `observations`, `_JOBOBJECT_BASIC_LIMIT_INFORMATION`, `_IO_COUNTERS` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `open_db()` connect `DB Thumbnail & Test Fixtures` to `Batch Orchestration`, `Contact Sheet Builder`, `Database — Run & Annotation Queries`, `Single-Run Analyzer`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `main()` connect `Batch Orchestration` to `DB Thumbnail & Test Fixtures`, `VLM Inference Core`, `Batch Utilities & Model-Major Design`, `Single-Run Analyzer`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `_run_model_phase()` connect `Batch Orchestration` to `VLM Inference Core`, `Batch Utilities & Model-Major Design`, `Single-Run Analyzer`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `annotate()` (e.g. with `_process_run()` and `test_annotate_returns_structured_dict()`) actually correct?**
  _`annotate()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `main()` (e.g. with `open_db()` and `get_unanalyzed_runs()`) actually correct?**
  _`main()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `_run_model_phase()` (e.g. with `get_unanalyzed_runs()` and `chunked()`) actually correct?**
  _`_run_model_phase()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ADR-003: Final Empirical Pin transformers==4.49.0` (e.g. with `transformers==4.49.0 Empirical Pin Decision` and `ADR-001: Pin ML Libraries Like Production DB Drivers`) actually correct?**
  _`ADR-003: Final Empirical Pin transformers==4.49.0` has 2 INFERRED edges - model-reasoned connections that need verification._