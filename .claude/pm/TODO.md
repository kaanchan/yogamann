# TODO

## Open Issues

| # | Title | Priority |
|---|-------|----------|
| [#32](https://github.com/kaanchan/yogamann/issues/32) | tech-debt: pin transformers version — v5 breaks trust_remote_code VLMs | HIGH (active) |
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

- Investigate LoRA wiring referenced in `yoga_asana_batch_2.yml` (not implemented in sd_make.py)
- Add `.avif` support to `--folder` photo collector in `make_mannequin.py`
- Thumbnail storage in DB (discussed, deferred)
- OBS-NNN pattern tagging on gallery interface
