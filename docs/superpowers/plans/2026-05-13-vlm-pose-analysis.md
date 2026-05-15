# VLM Pose Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous VLM annotation pipeline that loads (source photo, mannequin render) pairs, compares poses with a local vision-language model, and writes structured feedback into `vlm_annotations` in the DB.

**Architecture:** Four components — a config file (`profiles/vlm.yml`), a lazy-cached inference core (`src/vlm_inference.py`), a multi-model comparison harness (`src/compare_vlm.py`), and a polling daemon (`src/analyze.py`). The DB is extended with a single new table. All three models (Qwen2.5-VL, InternVL2.5, MiniCPM-V) use the transformers backend; lmdeploy is stubbed for later.

**Tech Stack:** Python 3.13, transformers ≥ 4.41, bitsandbytes (4-bit quant), Pillow, PyYAML, SQLite, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/db.py` | Add `vlm_annotations` DDL + 3 helper functions |
| Create | `profiles/vlm.yml` | Active model, all model configs, prompt template |
| Create | `tests/conftest.py` | Add `src/` to sys.path for all tests |
| Create | `tests/test_db_vlm.py` | DB schema + helper unit tests |
| Create | `tests/test_vlm_inference.py` | Inference core unit tests (mocked model) |
| Create | `src/vlm_inference.py` | Lazy-cached inference: load → prompt → infer → parse → validate |
| Create | `src/compare_vlm.py` | CLI: run N models over DB runs, print table, write annotations |
| Create | `src/analyze.py` | Polling daemon / `--once` batch runner |
| Modify | `src/download_models.py` | Add 3 VLM entries + `--vlm-only` flag |
| Modify | `make.ps1` | Add `vlm-download` and `analyze` targets |
| Modify | `pyproject.toml` | Add `bitsandbytes`, pytest dev dependency |

---

## Task 1: DB Schema + Helper Functions

**Files:**
- Modify: `src/db.py`

- [ ] **Step 1: Add `vlm_annotations` DDL to `_DDL` string**

  In `src/db.py`, find the end of `_DDL` (just before the closing `"""`). Insert after the `idx_ann_rating` index line:

  ```python
  CREATE TABLE IF NOT EXISTS vlm_annotations (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id            INTEGER NOT NULL REFERENCES runs(id),
      model_id          TEXT    NOT NULL,
      timestamp         TEXT    NOT NULL,
      rating            TEXT,
      misaligned        TEXT,
      unwanted_features TEXT,
      fail_patterns     TEXT,
      notes             TEXT,
      raw_output        TEXT,
      latency_s         REAL,
      UNIQUE(run_id, model_id)
  );

  CREATE INDEX IF NOT EXISTS idx_vlm_run   ON vlm_annotations(run_id);
  CREATE INDEX IF NOT EXISTS idx_vlm_model ON vlm_annotations(model_id);
  ```

- [ ] **Step 2: Add helper functions at the end of `src/db.py`**

  Append after the last function in the file:

  ```python
  # ── VLM annotation helpers ─────────────────────────────────────────────────────
  def save_vlm_annotation(
      conn: sqlite3.Connection,
      run_id: int,
      model_id: str,
      rating: str,
      misaligned: list[str],
      unwanted_features: list[str],
      fail_patterns: list[str],
      notes: str,
      raw_output: str,
      latency_s: float,
  ) -> None:
      now = datetime.now(timezone.utc).isoformat()
      conn.execute("""
          INSERT INTO vlm_annotations
              (run_id, model_id, timestamp, rating, misaligned, unwanted_features,
               fail_patterns, notes, raw_output, latency_s)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(run_id, model_id) DO UPDATE SET
              timestamp=excluded.timestamp,
              rating=excluded.rating,
              misaligned=excluded.misaligned,
              unwanted_features=excluded.unwanted_features,
              fail_patterns=excluded.fail_patterns,
              notes=excluded.notes,
              raw_output=excluded.raw_output,
              latency_s=excluded.latency_s
      """, (
          run_id, model_id, now, rating,
          json.dumps(misaligned),
          json.dumps(unwanted_features),
          json.dumps(fail_patterns),
          notes, raw_output, latency_s,
      ))


  def get_vlm_annotations(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
      return conn.execute(
          "SELECT * FROM vlm_annotations WHERE run_id=? ORDER BY model_id",
          (run_id,),
      ).fetchall()


  def get_unanalyzed_runs(
      conn: sqlite3.Connection,
      model_id: str,
      limit: int = 0,
  ) -> list[sqlite3.Row]:
      """Returns runs with no vlm_annotations entry for model_id.
      Each row exposes: id, output_png, source_path (from source_images.path).
      """
      q = """
          SELECT r.id, r.output_png, si.path as source_path
          FROM runs r
          JOIN source_images si ON r.source_sha256 = si.sha256
          WHERE NOT EXISTS (
              SELECT 1 FROM vlm_annotations va
              WHERE va.run_id = r.id AND va.model_id = ?
          )
          ORDER BY r.timestamp DESC
      """
      if limit:
          q += f" LIMIT {limit}"
      return conn.execute(q, (model_id,)).fetchall()
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add src/db.py
  git commit -m "feat: add vlm_annotations table + save/get/unanalyzed helpers (#29)"
  ```

---

## Task 2: Config File + Dependencies

**Files:**
- Create: `profiles/vlm.yml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create `profiles/vlm.yml`**

  ```yaml
  active_model: qwen2_5_vl_7b

  models:
    qwen2_5_vl_7b:
      repo: Qwen/Qwen2.5-VL-7B-Instruct
      backend: transformers
      load_in_4bit: true
      max_new_tokens: 512

    internvl2_5_8b:
      repo: OpenGVLab/InternVL2_5-8B-MPO
      backend: transformers
      load_in_4bit: true
      max_new_tokens: 512

    minicpm_v_4_6:
      repo: openbmb/MiniCPM-V-2_6
      backend: transformers
      load_in_4bit: true
      max_new_tokens: 512

  prompt:
    system: |
      You are a yoga pose alignment analyst. You receive two images:
      Image 1 is a source photo of a person in a yoga pose.
      Image 2 is a rendered wooden mannequin that should match that pose.
      Compare the poses and output ONLY valid JSON — no prose, no markdown.
    user: |
      Compare the pose in Image 1 (source) with Image 2 (mannequin render).
      Output JSON with exactly these keys:
        rating            — string: "good" | "acceptable" | "poor"
        misaligned        — list of strings: body parts where pose diverges
        unwanted_features — list of strings: artifacts in the render
        fail_patterns     — list of strings: systematic issues (e.g. "head always forward")
        notes             — string: one sentence of free-form observation

  schema:
    rating: [good, acceptable, poor]
    required: [rating, misaligned, unwanted_features, fail_patterns, notes]
  ```

- [ ] **Step 2: Add `bitsandbytes` to `pyproject.toml` dependencies**

  In the `dependencies` list, after `"streamlit>=1.37"`, add:

  ```toml
  "bitsandbytes>=0.45",
  ```

- [ ] **Step 3: Add pytest dev dependency to `pyproject.toml`**

  After the `[tool.uv.sources]` block, append:

  ```toml
  [dependency-groups]
  dev = ["pytest>=8.0"]
  ```

- [ ] **Step 4: Install new dependency**

  ```bash
  .\.venv\Scripts\python.exe -m pip install bitsandbytes pytest
  ```

  Expected: both install without error.

- [ ] **Step 5: Commit**

  ```bash
  git add profiles/vlm.yml pyproject.toml
  git commit -m "feat: vlm.yml config + bitsandbytes + pytest deps (#29)"
  ```

---

## Task 3: Test Infrastructure + DB Tests

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_db_vlm.py`

- [ ] **Step 1: Create `tests/conftest.py`**

  ```python
  import sys
  from pathlib import Path

  sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
  ```

- [ ] **Step 2: Create `tests/test_db_vlm.py`**

  ```python
  import json
  import pytest
  from pathlib import Path
  from db import open_db, save_vlm_annotation, get_vlm_annotations, get_unanalyzed_runs


  @pytest.fixture
  def conn(tmp_path):
      return open_db(tmp_path / "test.db")


  def _seed_run(conn):
      conn.execute("""
          INSERT INTO source_images (sha256, path, filename, added_at)
          VALUES ('abc123', '/tmp/photo.jpg', 'photo.jpg', '2026-01-01T00:00:00+00:00')
      """)
      conn.execute("""
          INSERT INTO runs (source_sha256, timestamp, output_png, metrics_json)
          VALUES ('abc123', '2026-01-01T00:00:00+00:00', '/tmp/render.png', '{}')
      """)
      conn.commit()
      return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


  def test_save_and_get_vlm_annotation(conn):
      run_id = _seed_run(conn)
      save_vlm_annotation(
          conn, run_id, "qwen2_5_vl_7b",
          rating="good",
          misaligned=["left arm"],
          unwanted_features=[],
          fail_patterns=[],
          notes="Pose matches well.",
          raw_output='{"rating":"good"}',
          latency_s=2.3,
      )
      conn.commit()
      rows = get_vlm_annotations(conn, run_id)
      assert len(rows) == 1
      assert rows[0]["rating"] == "good"
      assert json.loads(rows[0]["misaligned"]) == ["left arm"]
      assert rows[0]["latency_s"] == pytest.approx(2.3)


  def test_upsert_replaces_existing(conn):
      run_id = _seed_run(conn)
      for rating in ("poor", "good"):
          save_vlm_annotation(
              conn, run_id, "qwen2_5_vl_7b",
              rating=rating, misaligned=[], unwanted_features=[],
              fail_patterns=[], notes="", raw_output="", latency_s=1.0,
          )
          conn.commit()
      rows = get_vlm_annotations(conn, run_id)
      assert len(rows) == 1
      assert rows[0]["rating"] == "good"


  def test_two_models_two_rows(conn):
      run_id = _seed_run(conn)
      for model in ("qwen2_5_vl_7b", "minicpm_v_4_6"):
          save_vlm_annotation(
              conn, run_id, model,
              rating="acceptable", misaligned=[], unwanted_features=[],
              fail_patterns=[], notes="", raw_output="", latency_s=1.0,
          )
      conn.commit()
      rows = get_vlm_annotations(conn, run_id)
      assert len(rows) == 2
      assert {r["model_id"] for r in rows} == {"qwen2_5_vl_7b", "minicpm_v_4_6"}


  def test_get_unanalyzed_runs_includes_unannotated(conn):
      run_id = _seed_run(conn)
      runs = get_unanalyzed_runs(conn, "qwen2_5_vl_7b")
      assert any(r["id"] == run_id for r in runs)


  def test_get_unanalyzed_runs_excludes_annotated(conn):
      run_id = _seed_run(conn)
      save_vlm_annotation(
          conn, run_id, "qwen2_5_vl_7b",
          rating="good", misaligned=[], unwanted_features=[],
          fail_patterns=[], notes="", raw_output="", latency_s=1.0,
      )
      conn.commit()
      runs = get_unanalyzed_runs(conn, "qwen2_5_vl_7b")
      assert not any(r["id"] == run_id for r in runs)


  def test_get_unanalyzed_runs_exposes_paths(conn):
      run_id = _seed_run(conn)
      runs = get_unanalyzed_runs(conn, "qwen2_5_vl_7b")
      row = next(r for r in runs if r["id"] == run_id)
      assert row["output_png"] == "/tmp/render.png"
      assert row["source_path"] == "/tmp/photo.jpg"
  ```

- [ ] **Step 3: Run tests — expect PASS**

  ```bash
  .\.venv\Scripts\python.exe -m pytest tests/test_db_vlm.py -v
  ```

  Expected output: `6 passed`

- [ ] **Step 4: Commit**

  ```bash
  git add tests/conftest.py tests/test_db_vlm.py
  git commit -m "test: vlm_annotations DB schema + helper coverage (#29)"
  ```

---

## Task 4: `vlm_inference.py` — Write Tests First, Then Implement

**Files:**
- Create: `tests/test_vlm_inference.py`
- Create: `src/vlm_inference.py`

- [ ] **Step 1: Create `tests/test_vlm_inference.py` (failing)**

  ```python
  import json
  import pytest
  from pathlib import Path
  from unittest.mock import patch, MagicMock
  from PIL import Image as PILImage

  from vlm_inference import _parse_output, _validate, VLMSchemaError, annotate

  VALID = {
      "rating": "good",
      "misaligned": ["left arm"],
      "unwanted_features": [],
      "fail_patterns": [],
      "notes": "Pose matches well.",
  }


  # ── _parse_output ─────────────────────────────────────────────────────────────

  def test_parse_bare_json():
      assert _parse_output(json.dumps(VALID)) == VALID


  def test_parse_strips_assistant_prefix():
      assert _parse_output("assistant:\n" + json.dumps(VALID)) == VALID


  def test_parse_strips_im_start_prefix():
      assert _parse_output("<|im_start|>assistant\n" + json.dumps(VALID)) == VALID


  def test_parse_strips_markdown_fence():
      raw = "```json\n" + json.dumps(VALID) + "\n```"
      assert _parse_output(raw) == VALID


  def test_parse_raises_on_bad_json():
      with pytest.raises(json.JSONDecodeError):
          _parse_output("not json at all")


  # ── _validate ─────────────────────────────────────────────────────────────────

  def test_validate_passes_complete_dict():
      _validate(VALID)  # no exception


  def test_validate_raises_on_missing_rating():
      incomplete = {k: v for k, v in VALID.items() if k != "rating"}
      with pytest.raises(VLMSchemaError, match="rating"):
          _validate(incomplete)


  def test_validate_raises_on_multiple_missing():
      with pytest.raises(VLMSchemaError):
          _validate({"rating": "good"})


  # ── annotate ─────────────────────────────────────────────────────────────────

  def _make_images(tmp_path):
      photo = tmp_path / "photo.jpg"
      render = tmp_path / "render.png"
      PILImage.new("RGB", (64, 64), color=(128, 64, 32)).save(photo)
      PILImage.new("RGB", (64, 64), color=(200, 200, 200)).save(render)
      return photo, render


  @patch("vlm_inference._infer")
  @patch("vlm_inference._load_model")
  @patch("vlm_inference._load_config")
  def test_annotate_returns_structured_dict(mock_cfg, mock_load, mock_infer, tmp_path):
      photo, render = _make_images(tmp_path)
      mock_cfg.return_value = {
          "model_key": "qwen2_5_vl_7b",
          "backend": "transformers",
          "max_new_tokens": 512,
          "prompt": {"system": "sys", "user": "usr"},
      }
      mock_load.return_value = (MagicMock(), MagicMock())
      mock_infer.return_value = json.dumps(VALID)

      result = annotate(photo, render, model_key="qwen2_5_vl_7b")

      assert result["rating"] == "good"
      assert result["model_id"] == "qwen2_5_vl_7b"
      assert result["misaligned"] == ["left arm"]
      assert "latency_s" in result
      assert "raw_output" in result


  @patch("vlm_inference._infer")
  @patch("vlm_inference._load_model")
  @patch("vlm_inference._load_config")
  def test_annotate_retries_on_bad_json(mock_cfg, mock_load, mock_infer, tmp_path):
      photo, render = _make_images(tmp_path)
      mock_cfg.return_value = {
          "model_key": "qwen2_5_vl_7b",
          "backend": "transformers",
          "max_new_tokens": 512,
          "prompt": {"system": "sys", "user": "usr"},
      }
      mock_load.return_value = (MagicMock(), MagicMock())
      mock_infer.side_effect = ["not valid json", json.dumps(VALID)]

      result = annotate(photo, render, model_key="qwen2_5_vl_7b")

      assert result["rating"] == "good"
      assert mock_infer.call_count == 2


  @patch("vlm_inference._infer")
  @patch("vlm_inference._load_model")
  @patch("vlm_inference._load_config")
  def test_annotate_raises_on_second_bad_json(mock_cfg, mock_load, mock_infer, tmp_path):
      photo, render = _make_images(tmp_path)
      mock_cfg.return_value = {
          "model_key": "qwen2_5_vl_7b",
          "backend": "transformers",
          "max_new_tokens": 512,
          "prompt": {"system": "sys", "user": "usr"},
      }
      mock_load.return_value = (MagicMock(), MagicMock())
      mock_infer.side_effect = ["bad json", "still bad json"]

      with pytest.raises(json.JSONDecodeError):
          annotate(photo, render, model_key="qwen2_5_vl_7b")


  def test_annotate_raises_on_missing_photo(tmp_path):
      render = tmp_path / "render.png"
      PILImage.new("RGB", (64, 64)).save(render)
      with pytest.raises(FileNotFoundError, match="photo"):
          annotate(tmp_path / "missing.jpg", render, model_key="qwen2_5_vl_7b")


  def test_annotate_raises_on_missing_render(tmp_path):
      photo = tmp_path / "photo.jpg"
      PILImage.new("RGB", (64, 64)).save(photo)
      with pytest.raises(FileNotFoundError, match="render"):
          annotate(photo, tmp_path / "missing.png", model_key="qwen2_5_vl_7b")
  ```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

  ```bash
  .\.venv\Scripts\python.exe -m pytest tests/test_vlm_inference.py -v
  ```

  Expected: `ModuleNotFoundError: No module named 'vlm_inference'`

- [ ] **Step 3: Create `src/vlm_inference.py`**

  ```python
  """src/vlm_inference.py — lazy-cached VLM inference core.

  Entry point:
      result = annotate(photo_path, render_path, model_key="qwen2_5_vl_7b")

  Returns:
      {"model_id", "rating", "misaligned", "unwanted_features",
       "fail_patterns", "notes", "raw_output", "latency_s"}
  """
  from __future__ import annotations

  import json
  import re
  import time
  from pathlib import Path

  import yaml
  from PIL import Image, ImageOps

  # ── Constants ─────────────────────────────────────────────────────────────────
  _MODEL_CACHE: dict[str, tuple] = {}

  REQUIRED_KEYS = ["rating", "misaligned", "unwanted_features", "fail_patterns", "notes"]

  _RETRY_PROMPT = (
      "Your previous response was not valid JSON. "
      "Output ONLY a JSON object with exactly these keys: "
      "rating, misaligned, unwanted_features, fail_patterns, notes. No other text."
  )


  class VLMSchemaError(Exception):
      pass


  # ── Config ────────────────────────────────────────────────────────────────────
  def _load_config(model_key: str | None = None, config_path: Path | None = None) -> dict:
      if config_path is None:
          config_path = Path(__file__).parent.parent / "profiles" / "vlm.yml"
      with open(config_path, encoding="utf-8") as f:
          cfg = yaml.safe_load(f)
      key = model_key or cfg["active_model"]
      result = dict(cfg["models"][key])
      result["model_key"] = key
      result["prompt"] = cfg["prompt"]
      return result


  # ── Model loading ─────────────────────────────────────────────────────────────
  def _load_model(model_key: str, config: dict) -> tuple:
      if model_key in _MODEL_CACHE:
          return _MODEL_CACHE[model_key]

      repo = config["repo"]
      load_in_4bit = config.get("load_in_4bit", False)

      from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

      bnb = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
      model = AutoModelForCausalLM.from_pretrained(
          repo,
          quantization_config=bnb,
          device_map="auto",
          trust_remote_code=True,
      )
      processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
      _MODEL_CACHE[model_key] = (model, processor)
      return model, processor


  # ── Prompt ───────────────────────────────────────────────────────────────────
  def _build_messages(prompt: dict, backend: str, extra_user: str | None = None) -> list:
      user_text = prompt["user"]
      if extra_user:
          user_text = extra_user
      return [
          {
              "role": "user",
              "content": [
                  {"type": "image"},
                  {"type": "image"},
                  {"type": "text", "text": user_text},
              ],
          }
      ]


  # ── Inference ─────────────────────────────────────────────────────────────────
  def _infer(model, processor, messages: list, images: list, config: dict) -> str:
      max_new_tokens = config.get("max_new_tokens", 512)
      text = processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
      output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
      generated = [
          out_ids[len(in_ids):]
          for in_ids, out_ids in zip(inputs.input_ids, output_ids)
      ]
      return processor.batch_decode(generated, skip_special_tokens=True)[0]


  # ── Parsing + validation ──────────────────────────────────────────────────────
  def _parse_output(raw: str) -> dict:
      text = raw.strip()
      for prefix in ("assistant:", "<|im_start|>assistant", "<|im_start|>"):
          if text.lower().startswith(prefix.lower()):
              text = text[len(prefix):].strip()
      text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.MULTILINE)
      text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE)
      return json.loads(text.strip())


  def _validate(data: dict) -> None:
      missing = [k for k in REQUIRED_KEYS if k not in data]
      if missing:
          raise VLMSchemaError(f"Missing required keys: {missing}")


  # ── Public API ────────────────────────────────────────────────────────────────
  def annotate(
      photo_path: Path,
      render_path: Path,
      model_key: str | None = None,
      config_path: Path | None = None,
  ) -> dict:
      photo_path = Path(photo_path)
      render_path = Path(render_path)
      if not photo_path.exists():
          raise FileNotFoundError(f"Photo not found: {photo_path}")
      if not render_path.exists():
          raise FileNotFoundError(f"Render not found: {render_path}")

      config = _load_config(model_key, config_path)
      key = config["model_key"]
      backend = config.get("backend", "transformers")

      model, processor = _load_model(key, config)

      photo = ImageOps.exif_transpose(Image.open(photo_path).convert("RGB"))
      render = Image.open(render_path).convert("RGB")
      images = [photo, render]

      messages = _build_messages(config["prompt"], backend)

      t0 = time.perf_counter()
      raw = _infer(model, processor, messages, images, config)

      try:
          data = _parse_output(raw)
      except json.JSONDecodeError:
          retry_messages = messages + [
              {"role": "assistant", "content": raw},
              {"role": "user", "content": _RETRY_PROMPT},
          ]
          raw = _infer(model, processor, retry_messages, images, config)
          data = _parse_output(raw)

      _validate(data)
      latency_s = time.perf_counter() - t0

      return {
          "model_id": key,
          "rating": data["rating"],
          "misaligned": data.get("misaligned", []),
          "unwanted_features": data.get("unwanted_features", []),
          "fail_patterns": data.get("fail_patterns", []),
          "notes": data.get("notes", ""),
          "raw_output": raw,
          "latency_s": latency_s,
      }
  ```

- [ ] **Step 4: Run all tests — expect PASS**

  ```bash
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```

  Expected: all 17 tests pass (6 DB + 11 inference)

- [ ] **Step 5: Commit**

  ```bash
  git add src/vlm_inference.py tests/test_vlm_inference.py
  git commit -m "feat: vlm_inference.py — lazy-cached inference core with retry + tests (#29)"
  ```

---

## Task 5: `compare_vlm.py` — Multi-Model Comparison Harness

**Files:**
- Create: `src/compare_vlm.py`

- [ ] **Step 1: Create `src/compare_vlm.py`**

  ```python
  """src/compare_vlm.py — run N models over DB runs, print side-by-side table.

  Usage:
      python src/compare_vlm.py --run-ids 1 2 3 --output-root D:\\Temp\\yogamann-output
      python src/compare_vlm.py --limit 5 --output-root D:\\Temp\\yogamann-output
      python src/compare_vlm.py --limit 3 --models qwen2_5_vl_7b minicpm_v_4_6
  """
  from __future__ import annotations

  import argparse
  from pathlib import Path

  import yaml

  from db import open_db, save_vlm_annotation
  from vlm_inference import annotate, VLMSchemaError


  def _fetch_runs(conn, run_ids: list[int] | None, limit: int) -> list:
      if run_ids:
          placeholders = ",".join("?" * len(run_ids))
          return conn.execute(
              f"""SELECT r.id, r.output_png, si.path as source_path
                  FROM runs r JOIN source_images si ON r.source_sha256 = si.sha256
                  WHERE r.id IN ({placeholders})""",
              run_ids,
          ).fetchall()
      q = """SELECT r.id, r.output_png, si.path as source_path
             FROM runs r JOIN source_images si ON r.source_sha256 = si.sha256
             ORDER BY r.timestamp DESC"""
      if limit:
          q += f" LIMIT {limit}"
      return conn.execute(q).fetchall()


  def _print_comparison(run_id: int, source_name: str, results: list[dict]) -> None:
      if not results:
          return
      model_ids = [r["model_id"] for r in results]
      col_w = max(16, max(len(m) for m in model_ids) + 2)
      label_w = 22
      header = f"{'':>{label_w}}" + "".join(m.center(col_w) for m in model_ids)
      sep = "─" * len(header)
      print(f"\nRun {run_id}  {source_name}")
      print(sep)
      print(header)
      for field in ("rating", "misaligned", "unwanted_features", "fail_patterns", "notes"):
          row = f"  {field:<{label_w - 2}}"
          for r in results:
              val = r.get(field, "")
              if isinstance(val, list):
                  val = repr(val)
              row += str(val)[: col_w - 2].center(col_w)
          print(row)
      print(sep)


  def main() -> None:
      parser = argparse.ArgumentParser(description="Compare VLM pose analysis across models")
      parser.add_argument("--run-ids", nargs="+", type=int, help="Specific run IDs to analyze")
      parser.add_argument("--limit", type=int, default=5, help="N most recent runs (default 5)")
      parser.add_argument("--output-root", default=r"D:\Temp\yogamann-output")
      parser.add_argument("--models", nargs="+", help="Model keys to use (default: all in vlm.yml)")
      args = parser.parse_args()

      db_path = Path(args.output_root) / "yogamann.db"
      conn = open_db(db_path)

      vlm_cfg = yaml.safe_load(
          (Path(__file__).parent.parent / "profiles" / "vlm.yml").read_text(encoding="utf-8")
      )
      model_keys = args.models or list(vlm_cfg["models"].keys())

      runs = _fetch_runs(conn, args.run_ids, args.limit)
      if not runs:
          print("No runs found.")
          return

      for run in runs:
          run_id = run["id"]
          source_name = Path(run["source_path"]).name
          results: list[dict] = []

          for model_key in model_keys:
              try:
                  result = annotate(
                      Path(run["source_path"]),
                      Path(run["output_png"]),
                      model_key=model_key,
                  )
                  save_vlm_annotation(
                      conn, run_id, model_key,
                      rating=result["rating"],
                      misaligned=result["misaligned"],
                      unwanted_features=result["unwanted_features"],
                      fail_patterns=result["fail_patterns"],
                      notes=result["notes"],
                      raw_output=result["raw_output"],
                      latency_s=result["latency_s"],
                  )
                  conn.commit()
                  results.append(result)
              except FileNotFoundError as exc:
                  print(f"[skip] run {run_id}: {exc}")
              except VLMSchemaError as exc:
                  print(f"[skip] run {run_id} {model_key}: schema error — {exc}")
              except Exception as exc:  # noqa: BLE001
                  import traceback
                  print(f"[error] run {run_id} {model_key}: {exc}")
                  traceback.print_exc()

          _print_comparison(run_id, source_name, results)


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2: Smoke-test the CLI help (no GPU needed)**

  ```bash
  .\.venv\Scripts\python.exe src/compare_vlm.py --help
  ```

  Expected: prints usage without error.

- [ ] **Step 3: Commit**

  ```bash
  git add src/compare_vlm.py
  git commit -m "feat: compare_vlm.py — multi-model comparison harness (#29)"
  ```

---

## Task 6: `analyze.py` — Polling Daemon

**Files:**
- Create: `src/analyze.py`

- [ ] **Step 1: Create `src/analyze.py`**

  ```python
  """src/analyze.py — poll DB and annotate unanalyzed runs with active VLM model.

  Usage:
      # Annotate all pending runs and exit:
      .\.venv\Scripts\python.exe src/analyze.py --once --output-root D:\\Temp\\yogamann-output

      # Run as persistent daemon (checks every 30 s):
      .\.venv\Scripts\python.exe src/analyze.py --poll-interval 30 --output-root D:\\Temp\\yogamann-output
  """
  from __future__ import annotations

  import argparse
  import signal
  import time
  import traceback
  from pathlib import Path

  import yaml

  from db import open_db, save_vlm_annotation, get_unanalyzed_runs, save_rating
  from vlm_inference import annotate, VLMSchemaError

  _RUNNING = True


  def _handle_sigint(sig, frame):
      global _RUNNING
      print("\n[analyze] Ctrl-C received — shutting down after current run...")
      _RUNNING = False


  def _process_run(conn, run, model_key: str, promote_to_runs: bool) -> None:
      run_id = run["id"]
      source = Path(run["source_path"])
      render = Path(run["output_png"])

      if not source.exists():
          print(f"[skip] run {run_id}: source photo missing: {source}")
          return
      if not render.exists():
          print(f"[skip] run {run_id}: render missing: {render}")
          return

      try:
          result = annotate(source, render, model_key=model_key)
          save_vlm_annotation(
              conn, run_id, model_key,
              rating=result["rating"],
              misaligned=result["misaligned"],
              unwanted_features=result["unwanted_features"],
              fail_patterns=result["fail_patterns"],
              notes=result["notes"],
              raw_output=result["raw_output"],
              latency_s=result["latency_s"],
          )
          if promote_to_runs:
              save_rating(
                  conn, run_id,
                  result["rating"],
                  result["notes"],
                  misaligned=result["misaligned"],
                  unwanted_features=result["unwanted_features"],
              )
          conn.commit()
          print(
              f"[ok] run {run_id} | {model_key} | {result['rating']}"
              f" | {result['latency_s']:.1f}s"
          )
      except VLMSchemaError as exc:
          print(f"[warn] run {run_id} schema error: {exc}")
      except Exception as exc:  # noqa: BLE001
          print(f"[error] run {run_id}: {exc}")
          traceback.print_exc()


  def main() -> None:
      global _RUNNING
      signal.signal(signal.SIGINT, _handle_sigint)

      parser = argparse.ArgumentParser(description="Annotate DB runs with active VLM model")
      parser.add_argument("--once", action="store_true", help="Process pending runs and exit")
      parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls")
      parser.add_argument("--output-root", default=r"D:\Temp\yogamann-output")
      args = parser.parse_args()

      db_path = Path(args.output_root) / "yogamann.db"
      conn = open_db(db_path)

      vlm_cfg = yaml.safe_load(
          (Path(__file__).parent.parent / "profiles" / "vlm.yml").read_text(encoding="utf-8")
      )
      active_model = vlm_cfg["active_model"]

      mode = "once" if args.once else f"poll every {args.poll_interval}s"
      print(f"[analyze] active model : {active_model}")
      print(f"[analyze] mode         : {mode}")
      print(f"[analyze] db           : {db_path}")

      while _RUNNING:
          runs = get_unanalyzed_runs(conn, active_model)
          if runs:
              print(f"[analyze] {len(runs)} unanalyzed run(s)")
              for run in runs:
                  if not _RUNNING:
                      break
                  _process_run(conn, run, active_model, promote_to_runs=True)
          elif args.once:
              print("[analyze] No pending runs.")

          if args.once:
              break
          time.sleep(args.poll_interval)

      print("[analyze] Done.")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2: Smoke-test the CLI help**

  ```bash
  .\.venv\Scripts\python.exe src/analyze.py --help
  ```

  Expected: prints usage without error.

- [ ] **Step 3: Commit**

  ```bash
  git add src/analyze.py
  git commit -m "feat: analyze.py — VLM polling daemon + --once batch mode (#29)"
  ```

---

## Task 7: Download Extension + `make.ps1` Targets

**Files:**
- Modify: `src/download_models.py`
- Modify: `make.ps1`

- [ ] **Step 1: Refactor `src/download_models.py` to add VLM entries + `--vlm-only` flag**

  Replace the entire file content with:

  ```python
  """src/download_models.py — pre-fetch all HF model weights.

  Usage:
      python src/download_models.py           # download all models
      python src/download_models.py --vlm-only  # download VLM models only
  """
  import argparse
  import os

  os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

  from huggingface_hub import snapshot_download

  # (model_id, allow_patterns, ignore_patterns, reason)
  PIPELINE_MODELS = [
      (
          "xinsir/controlnet-openpose-sdxl-1.0",
          None, None,
          "ControlNet OpenPose weights for SDXL (~2.5 GB)",
      ),
      (
          "stabilityai/stable-diffusion-xl-base-1.0",
          [
              "model_index.json",
              "scheduler/**",
              "tokenizer/**",
              "tokenizer_2/**",
              "text_encoder/config.json",
              "text_encoder/model.fp16.safetensors",
              "text_encoder_2/config.json",
              "text_encoder_2/model.fp16.safetensors",
              "unet/config.json",
              "unet/diffusion_pytorch_model.fp16.safetensors",
              "vae/config.json",
              "vae/diffusion_pytorch_model.fp16.safetensors",
          ],
          None,
          "SDXL fp16 safetensors only (~7 GB)",
      ),
      (
          "depth-anything/Depth-Anything-V2-Small-hf",
          None, None,
          "Depth estimation model (~100 MB)",
      ),
  ]

  VLM_MODELS = [
      (
          "Qwen/Qwen2.5-VL-7B-Instruct",
          None, None,
          "VLM pose analyzer — primary model (~15 GB)",
      ),
      (
          "OpenGVLab/InternVL2_5-8B-MPO",
          None, None,
          "VLM pose analyzer — secondary model (~16 GB)",
      ),
      (
          "openbmb/MiniCPM-V-2_6",
          None, None,
          "VLM pose analyzer — lightweight model (~8 GB)",
      ),
  ]


  def _download(models: list, token: str | None, cache_dir: str) -> None:
      print(f"hf_transfer active : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1'}")
      print(f"cache              : {cache_dir}")
      print(f"token              : {'set' if token else 'NOT SET (will be slower)'}")
      print(f"models to fetch    : {len(models)}\n")

      for model_id, allow_patterns, ignore_patterns, reason in models:
          print(f"==> {model_id}")
          print(f"    why    : {reason}")
          print(f"    allow  : {', '.join(allow_patterns) if allow_patterns else 'all'}")
          print(f"    ignore : {', '.join(ignore_patterns) if ignore_patterns else 'none'}")
          snapshot_download(
              model_id,
              cache_dir=cache_dir,
              token=token,
              max_workers=16,
              allow_patterns=allow_patterns,
              ignore_patterns=ignore_patterns,
          )
          print(f"    done\n")

      print("All models cached.")


  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--vlm-only", action="store_true", help="Download VLM models only")
      args = parser.parse_args()

      token = os.environ.get("HF_TOKEN")
      cache_dir = os.environ.get("HF_HUB_CACHE", r"D:\models\hub")
      models = VLM_MODELS if args.vlm_only else PIPELINE_MODELS + VLM_MODELS
      _download(models, token, cache_dir)
  ```

- [ ] **Step 2: Add `vlm-download` and `analyze` targets to `make.ps1`**

  In `make.ps1`, find the `"download"` case block. Insert two new cases immediately before the `default` block:

  ```powershell
      "vlm-download" {
          if (-not $env:HF_TOKEN) {
              Write-Host "`n[WARN] HF_TOKEN not set — downloads will be slower (unauthenticated)." -ForegroundColor Yellow
              Write-Host "       Get a free read token at https://huggingface.co/settings/tokens" -ForegroundColor Yellow
          }
          Invoke-Step "Download VLM models (hf_transfer)" {
              & $Python src/download_models.py --vlm-only
          }
      }
      "analyze" {
          $env:HF_HUB_OFFLINE = "1"
          Invoke-Step "Annotate pending runs with active VLM model" {
              & $Python src/analyze.py --once --output-root $OutputRoot
          }
      }
  ```

  Also add entries to the help text in the `default` block:
  ```powershell
          Write-Host "  vlm-download  download VLM model weights (~15-40 GB depending on models)"
          Write-Host "  analyze       annotate pending DB runs with active VLM model (--once)"
  ```

- [ ] **Step 3: Verify CLI help still works**

  ```bash
  .\.venv\Scripts\python.exe src/download_models.py --help
  ```

  Expected: shows `--vlm-only` option.

  ```powershell
  .\make.ps1 -Target help
  ```

  Expected: `vlm-download` and `analyze` appear in target list.

- [ ] **Step 4: Run full test suite — all pass**

  ```bash
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```

  Expected: all 17 tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/download_models.py make.ps1
  git commit -m "feat: vlm-download + analyze make.ps1 targets, download_models.py refactor (#29)"
  ```

- [ ] **Step 6: Push branch**

  ```bash
  git push origin feature/vlm-analysis
  ```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `vlm_annotations` table with `UNIQUE(run_id, model_id)` | Task 1 |
| `profiles/vlm.yml` with 3 models + prompt | Task 2 |
| `vlm_inference.py`: lazy load, multi-image prompt, retry, schema validate | Task 4 |
| `compare_vlm.py`: `--run-ids` / `--limit`, side-by-side table, writes annotations | Task 5 |
| `analyze.py`: poll loop, `--once`, SIGINT, promote to runs | Task 6 |
| `download_models.py` extended, `vlm-download` make target | Task 7 |
| Unit tests: parse, retry, validate, file-not-found | Task 4 |
| DB tests: save, upsert, two-model, unanalyzed query | Task 3 |
| Error handling: missing files → skip, schema fail → warn | Tasks 4, 6 |
| OOM handling | **Gap — see below** |

**Gap: OOM handling**

The spec requires catching `torch.cuda.OutOfMemoryError` in `_infer`. Add this to `_infer` in `src/vlm_inference.py`:

```python
def _infer(model, processor, messages, images, config):
    ...
    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            import torch
            torch.cuda.empty_cache()
            raise MemoryError(f"GPU OOM during inference: {exc}") from exc
        raise
    ...
```

And in `analyze.py`'s `_process_run`, catch `MemoryError`:

```python
    except MemoryError as exc:
        print(f"[oom] run {run_id}: {exc} — skipping")
```

Add this fix as an amendment to Task 4 Step 3 (before committing).

**Placeholder scan:** None found.

**Type consistency:** `save_vlm_annotation` signature used identically in Tasks 1, 5, and 6. `get_unanalyzed_runs` returns rows with `id`, `output_png`, `source_path` — consumed correctly in Tasks 5 and 6.
