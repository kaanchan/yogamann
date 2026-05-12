# ── Cross-platform detection ─────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    PYTHON   := .venv/Scripts/python.exe
    STREAMLIT := .venv/Scripts/streamlit.exe
    OPEN     := start
else
    PYTHON   := .venv/bin/python
    STREAMLIT := .venv/bin/streamlit
    OPEN     := xdg-open
endif

SAMPLE       := input/yoga-pose-sample-4.jpg
PROFILE      := yoga_asana
OUTPUT_ROOT  := D:/Temp/yogamann-output

.PHONY: test test-all diag gallery open ingest review download

# Quick smoke test — single image + open gallery
test:
	HF_HUB_OFFLINE=1 $(PYTHON) src/make_mannequin.py $(SAMPLE) \
	    --profile $(PROFILE)
	$(PYTHON) src/make_gallery.py
	$(OPEN) out/index.html

# All reference images + gallery
test-all:
	HF_HUB_OFFLINE=1 $(PYTHON) src/make_mannequin.py --folder input \
	    --profile $(PROFILE)
	$(PYTHON) src/make_gallery.py
	$(OPEN) out/index.html

# Hardware + import diagnostics
diag:
	$(PYTHON) src/diagnostics/rtx5080-test.py

# Regenerate gallery HTML and open browser
gallery:
	$(PYTHON) src/make_gallery.py
	$(OPEN) out/index.html

# Open existing gallery without regenerating
open:
	$(OPEN) out/index.html

# Import .metrics.json files into yogamann.db
ingest:
	$(PYTHON) src/db.py --ingest --output-root $(OUTPUT_ROOT)
	$(PYTHON) src/db.py --stats  --output-root $(OUTPUT_ROOT)

# Launch Streamlit review gallery
review:
	HF_HUB_OFFLINE=1 $(STREAMLIT) run src/gallery.py -- \
	    --output-root $(OUTPUT_ROOT)

# First-time setup: pre-fetch all HuggingFace model weights (~10 GB)
download:
	$(PYTHON) src/download_models.py
