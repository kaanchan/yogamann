# PENDING TASK

## Overnight annotation run — in progress

**Branch:** main  
**DB:** `D:\Temp\yogamann-output\yogamann.db`  
**Started:** 2026-05-19 ~23:51

### What's running
```
uv run python src/compare_vlm.py \
  --limit 0 --force --strategy round-robin --slice 20 \
  --output-root "D:/Temp/yogamann-output" \
  --models qwen2_5_vl_7b internvl2_5_8b minicpm_o_2_6 minicpm_v_2_6 molmo_7b_d
```

### Pre-run state
- Cleared 550 stale VLM annotations (retained satchid's 55 human reviews)
- 981 candidate runs × 5 models = 4,905 target annotations

### When done
- [ ] Verify annotation counts per model in DB
- [ ] Spot-check Molmo annotations for naturalness leakage (v2b_molmo compliance)
- [ ] Update PROGRESS.md and clear this file
- [ ] Create GH issue for baseline metrics / Blender next-phase planning
