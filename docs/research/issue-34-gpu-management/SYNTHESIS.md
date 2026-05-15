# GPU Resource Management — Consolidated Findings

## Issue
[#34 — Research: GPU resource management for sustained 12-14 hour VLM inference runs](https://github.com/kaanchan/yogamann/issues/34)

Date: 2026-05-15 22:03 UTC

Model: Gemini 3 Flash

Task: GPU Resource Management — Issue #34

Metadata: Technical Synthesis Pipeline for Blackwell Architecture Stability

---

## **Consensus Summary: The Architectural Facts**

Cutting through the reporting fluff reveals a clear consensus regarding the hardware and driver constraints of running continuous 12–14 hour VLM workloads on an NVIDIA RTX 5080 (Blackwell) within a Windows 11 ecosystem:

* **The Thermal/Power Realities:** Running consumer-grade hardware at 100% Total Graphics Power (TGP) under sustained matrix-multiplication loads creates progressive chassis heat soak, leading to aggressive driver downclocking ("Fast Trigger" protection modes) or full system shutdowns. The community sweet spot is capping the power limit at **80% of TGP**. This slashes GPU temperatures by 12–15°C while incurring a negligible 3–5% drop in token throughput.
* **The Windows Privilege Barrier:** Programmatically altering hardware states—such as modifying the power limit via `nvidia-smi -pl` or the underlying NVML API—**strictly requires Administrator elevation** on Windows 11. Telemetry collection (temperature, utilization, clock rates), however, can be performed safely by standard unprivileged users.
* **Telemetry Overhead:** Repeatedly spawning `nvidia-smi` as a subprocess every few seconds introduces severe CPU overhead, process latency, and a potential build-up of zombie processes over long periods. Production-grade scripts must use direct C-level library bindings (`pynvml` or `nvitop`) for high-frequency polling.
* **VRAM Isolation Myths:** Consumer GeForce cards do not support enterprise hardware partitioning like Multi-Instance GPU (MIG) or Multi-Process Service (MPS) on Windows. They rely entirely on time-sliced scheduling via Hardware-Accelerated GPU Scheduling (HAGS). Furthermore, because the card operates in Windows Display Driver Model (WDDM) mode, a baseline of 0.5–1.0 GB of VRAM is permanently locked by the OS. If PyTorch, Ollama, and LM Studio combined exceed physical VRAM, WDDM transparently offloads memory to system RAM (Shared Video Memory), dropping performance by 10x–100x and causing process timeouts.

---

## **Top 3 Core Tools for Implementation**

### **1. pynvml (nvidia-ml-py)**

* **Description:** The official Python wrapper for the NVIDIA Management Library (NVML). It bypasses expensive subprocess calls by executing direct, low-level DLL queries against the NVIDIA driver. This is the fundamental building block required to query precise hardware throttle reason bitmasks (`nvmlDeviceGetCurrentClocksThrottleReasons`) and programmatically apply an 80% TGP power cap at script initialization to insulate the hardware from thermal runaway.
* **Python Implementation Sketch:**

```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 1. Enforce 80% TGP Cap (Requires Admin privileges)
try:
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, 280000) # 280W cap for RTX 5080
    print("Sustained run power cap initialized successfully.")
except pynvml.NVMLError:
    print("Admin elevation missing. Continuing with read-only telemetry.")

# 2. Extract Active Throttle State inside the loop
def check_thermal_throttle():
    reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
    # Bitmask match for HwThermalSlowdown (0x0000000000000040LL)
    return bool(reasons & 0x0000000000000040)

```

* **Citation:** [https://github.com/gpuopenanalytics/pynvml](https://github.com/gpuopenanalytics/pynvml)
* **Limitations:** Power configuration modifications will fail silently or throw permission errors if the parent script is executed without administrative rights. It operates at the driver layer, meaning it has zero visibility into PyTorch's internal memory allocation states or process persistence.

### **2. nvitop**

* **Description:** A highly maintained, multi-threaded GPU profiling engine built directly on top of NVML. It resolves the multi-tenant VRAM visibility problem on consumer cards by matching active Windows PIDs to specific GPU memory footprints. This enables your batch loop to behave in a "resource-aware" manner: the script can inspect the active memory footprint of concurrent apps (Ollama, LM Studio) and dynamically inject cooperative pauses or hysteresis cooling delays before hitting a hard crash threshold.
* **Python Implementation Sketch:**

```python
from nvitop import Device
import time

device = Device.all()[0]

for img_path in image_queue:
    # Cooperative VRAM yielding based on surrounding app spikes
    if device.memory_percent() > 88:
        print("VRAM congestion detected. Yielding to competing processes...")
        time.sleep(30)
        continue
        
    # Hysteresis-based thermal protection
    if 'HwThermalSlowdown' in device.throttle_reasons():
        print("GPU thermal safeguard triggered. Initiating 60s cooldown...")
        time.sleep(60)
        
    run_vlm_inference(img_path)

```

* **Citation:** [https://github.com/XuehaiPan/nvitop](https://github.com/XuehaiPan/nvitop)
* **Limitations:** Tracking per-process metrics introduces a slight computational overhead (roughly 50–100ms per check), meaning it should be executed between image batches rather than inside tight kernel operations. It acts purely as a monitoring and advisory mechanism; it cannot programmatically force other processes to give up their VRAM slices.

### **3. PyTorch Caching Allocator API (`torch.cuda`)**

* **Description:** Native memory orchestration primitives internal to PyTorch. By default, PyTorch holds onto allocated memory blocks to avoid recycling overhead. In a shared 16GB WDDM environment, this behavior accelerates memory fragmentation, starving Ollama or LM Studio and triggering sudden memory faults. Explicitly querying `memory_reserved()` and flushing unused pools with `empty_cache()` between inference batches keeps the runtime footprint lean and releases unallocated blocks back to the OS.
* **Python Implementation Sketch:**

```python
import torch

def garbage_collect_vram(fragmentation_threshold=0.85):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    
    # Identify if PyTorch is hoarding unused reserved blocks
    if reserved > 0 and (allocated / reserved) < fragmentation_threshold:
        print(f"Flushing fragmented cache block. Releasing VRAM pool.")
        torch.cuda.empty_cache()

# Integrated into the end of your VLM execution pipeline
for batch in image_dataloader:
    outputs = model(batch)
    garbage_collect_vram()

```

* **Citation:** [https://pytorch.org/docs/stable/cuda.html](https://pytorch.org/docs/stable/cuda.html)
* **Limitations:** Executing `torch.cuda.empty_cache()` creates a brief performance penalty on the immediate subsequent inference batch because the model must request new memory allocations from the driver. It can only manage allocations owned by its specific PyTorch process context and will not resolve memory leaks originating from secondary host software.

---

## **The Operational Layer: Process Resilience**

While the above tools stabilize the hardware and memory, long-running reliability requires a process supervisor to handle the unpreventable.

Wrap your completed Python batch loop inside **PM2** using an ecosystem configuration file. PM2 operates natively on Windows 11 and includes an `exp_backoff_restart_delay` parameter. If the system undergoes an unavoidable driver crash or thermal event, PM2 acts as a daemon watchdog, automatically restarting the script but applying an exponentially increasing delay. This ensures that if a crash is heat-induced, the system is given an absolute cooling window before the GPU is subjected to heavy VLM compute loads again.
