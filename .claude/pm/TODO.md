# TODO

## Open Issues

| # | Title | Priority |
|---|-------|----------|
| TBD | perf: batched GPU inference for Qwen2.5-VL (vlm_batch_size) — `feature/batched-inference` | HIGH (active) |
| [#34](https://github.com/kaanchan/yogamann/issues/34) | research: GPU resource management for sustained inference runs | MEDIUM (impl pending) |
| [#32](https://github.com/kaanchan/yogamann/issues/32) | tech-debt: pin transformers version — 5-model annotation run in progress | MEDIUM |
| [#29](https://github.com/kaanchan/yogamann/issues/29) | Feature: VLM pose analysis — local vision-language model integration | HIGH |
| [#33](https://github.com/kaanchan/yogamann/issues/33) | research: alternative pose analyzers and renderers beyond OpenPose + SDXL ControlNet | LOW (deferred) |
| [#19](https://github.com/kaanchan/yogamann/issues/19) | Feature: mark.ps1 / gold index | MEDIUM |
| [#18](https://github.com/kaanchan/yogamann/issues/18) | Baseline: standard test set | MEDIUM |
| [#16](https://github.com/kaanchan/yogamann/issues/16) | Feature: OKS self-eval | MEDIUM |
| [#14](https://github.com/kaanchan/yogamann/issues/14) | Perf: torch.compile cudagraphs | LOW |

## Recently Closed (for reference)

| # | Title | Commit |
|---|-------|--------|
| #28 | Feature: patterns catalogue | ccc8570 |
| #27 | Feature: multi-user annotations + gallery UX | ccc8570 |
| #26 | Feature: multi-batch.ps1 overnight queue | d428c63 |
| #25 | Docs: README usage guide | 3c6b800 |
| #23 | Feature: skip-existing flag + ingest/review targets | b87b52a |
| #17 | Fix: head direction — cond_scale sweep | e4633ff |

## Backlog (no issue yet)

- Replace nvidia-smi subprocess in gpu_monitor.py with pynvml (zero overhead, no zombie procs)
- nvitop integration for throttle bitmask + VRAM-by-PID visibility (issue #34 recommendation)
- PM2 process supervisor for unattended overnight runs (issue #34 recommendation)
- MiniCPM-V schema compliance: normalise numeric ratings ("3") → "good"/"acceptable"/"poor"
- Qwen first-84-image re-annotation with new prompt (--force --models qwen2_5_vl_7b)
- Folder-based scheduling strategy for compare_vlm.py (--strategy folder)
- Investigate LoRA wiring referenced in `yoga_asana_batch_2.yml` (not implemented in sd_make.py)
- Add `.avif` support to `--folder` photo collector in `make_mannequin.py`
- OBS-NNN pattern tagging on gallery interface
