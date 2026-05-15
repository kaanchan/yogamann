Deep Research Prompt: GPU Resource Management for Sustained 12-14 Hour VLM Inference Runs

---

RESPONSE FORMAT REQUIRED

Begin your response with a header block in exactly this format:

  Date: YYYY-MM-DD HH:MM UTC
  Model: [your model name and version, e.g. "GPT-4o (2024-11-20)" or "Gemini 2.0 Flash Deep Research"]
  Task: GPU Resource Management — Issue #34

Then follow with your full findings below that header.

---

Context

I am running sustained 12-14 hour GPU inference jobs (PyTorch, CUDA, 4-bit quantized VLMs via
bitsandbytes) on a Windows 11 machine with an NVIDIA RTX 5080 (16 GB VRAM). Other GPU-consuming
applications run simultaneously — LM Studio, Ollama, and standard desktop rendering. The GPU has
already caused system shutdowns mid-run, likely from thermal buildup combined with sustained high
power draw. The batch job is resumable (DB-backed), so no work is lost on crash — but the goal is
to complete the full run without interruption.

What is already in place:

  - src/gpu_monitor.py — polls nvidia-smi subprocess every ~8-30 seconds, provides sample()
    returning gpu_temp_c / gpu_util_pct / gpu_mem_used_mb / gpu_power_w / cpu_util_pct, and
    cooldown_if_hot(threshold_c, sleep_s) which blocks inference until GPU cools.
  - src/compare_vlm.py — --gpu-temp-limit (°C) and --cooldown-secs flags; per-image JSONL log
    at batch_metrics.jsonl recording all hardware metrics alongside inference latency.
  - Hardware: RTX 5080, 16 GB VRAM, Windows 11, Python 3.13, PyTorch + bitsandbytes + HF
    Transformers (pinned to 4.49.0).

---

Research Scope

Investigate and report only on validated, community-proven, implementable solutions for the
following. Cite sources (GitHub URLs, paper titles, documentation links) and version numbers
where possible. Do not report on experimental or unmaintained tools.

1. Thermal and Power Budget Management

   - Validated techniques to cap GPU power draw via nvidia-smi --power-limit or NVML calls.
     Does setting a power limit require administrator elevation on Windows 11? What is the
     minimum required privilege level, and is there a workaround?
   - Whether pynvml can set power limits programmatically in Python on Windows without
     running as admin.
   - Any open-source scheduler or watchdog that enforces cooling pauses based on live GPU
     temperature, rather than relying on the OS thermal governor.
   - What power limit percentage has the community found effective for long-run RTX cards
     (e.g., 80% TDP) to reduce heat without significant throughput loss?

2. VRAM and Process Priority Management

   - Does the RTX 5080 support CUDA MPS (Multi-Process Service) or CUDA MIG (Multi-Instance
     GPU) on Windows? MIG is documented for data-center cards — what is the consumer-card
     equivalent, if any?
   - How to yield the GPU cooperatively to other processes between inference batches, without
     terminating the process. Is torch.cuda.set_per_process_memory_fraction() the right
     primitive, or are there better-documented alternatives?
   - Any Windows-specific GPU scheduling settings (Hardware-accelerated GPU scheduling /
     WDDM vs TCC mode) that affect how multiple GPU processes share resources.

3. Long-Running PyTorch Job Resilience

   - Production-grade patterns for babysitting long inference loops: watchdog processes,
     automatic restart-on-OOM or crash, process supervisors (e.g., supervisord, PM2,
     Windows Task Scheduler).
   - How Hugging Face Accelerate, Ray (single-node), or Celery are used — if at all — for
     single-GPU long-running inference management (not distributed training).
   - Any community tooling specifically designed for "run N jobs over 12+ hours, survive a
     crash, resume cleanly" on a single consumer GPU.

4. OS-Level Signals and Monitoring

   - On Windows, are there WMI events, kernel events, or ACPI thermal zone notifications that
     fire before a thermal shutdown (not just Kernel-Power Event ID 41, which is post-hoc)?
     If so, how do you subscribe to them from Python?
   - nvidia-smi dmon vs pynvml vs gputil: which is most reliable for sustained 1-5 second
     polling over a 12-hour run on Windows? Are there known issues with high-frequency
     nvidia-smi subprocess calls?
   - Does nvidia-smi report GPU throttle state (P-state, TGP throttle reason)? How to detect
     active thermal throttling before it causes a crash.

5. Open-Source Tools Audit

For each tool below, provide: (a) active maintenance status and last release date, (b) Windows
11 compatibility confirmed or not, (c) PyTorch process integration — does it see PyTorch's
VRAM usage, (d) a 1-2 sentence honest verdict on whether to use it here.

   - gputil (GitHub: anderskm/gputil)
   - pynvml / nvidia-ml-py (GitHub: gpuopenanalytics/pynvml)
   - nvitop (GitHub: XuehaiPan/nvitop)
   - py3nvml
   - torch.cuda.memory_stats() and torch.cuda.memory_reserved()
   - Any other tool with >500 GitHub stars that meaningfully addresses this use case

---

Output Format

Return exactly top 3-5 implementable recommendations, ranked by:
  1. Minimal code change to integrate with an existing Python batch loop
  2. Windows 11 confirmed compatibility
  3. Proven community adoption (not experimental or abandoned)

For each recommendation provide:
  - One paragraph description of what it does and why it fits
  - A Python implementation sketch (5-15 lines) showing how it integrates
  - Citation: GitHub URL, docs link, or paper
  - Limitations: an honest 2-3 sentence assessment of what it does not handle
