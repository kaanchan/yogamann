# On this Windows setup, use run.ps1 instead:
#   .\run.ps1 test        single image + open gallery
#   .\run.ps1 test-all    all input images + open gallery
#   .\run.ps1 diag        hardware + import diagnostics
#   .\run.ps1 gallery     regenerate HTML gallery + open browser
#   .\run.ps1 open        open existing gallery
#
# If make is available (WSL / Linux):
PYTHON  := .venv/Scripts/python.exe
SAMPLE  := input/yoga-pose-sample-4.jpg
PROFILE := yoga_asana

.PHONY: test test-all diag gallery open

test:
	$(PYTHON) src/make_mannequin.py $(SAMPLE) --profile $(PROFILE) && \
	$(PYTHON) src/make_gallery.py && \
	start out/index.html

test-all:
	$(PYTHON) src/make_mannequin.py --folder input --profile $(PROFILE) && \
	$(PYTHON) src/make_gallery.py && \
	start out/index.html

diag:
	$(PYTHON) src/diagnostics/rtx5080-test.py

gallery:
	$(PYTHON) src/make_gallery.py && start out/index.html

open:
	start out/index.html
