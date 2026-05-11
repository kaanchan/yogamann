import torch
import time
import sys

# Attempt to import Blackwell-specific scaling tools if available
try:
    from torchao.prototype.mx_formats import NVFP4Tensor
    HAS_AO = True
except ImportError:
    HAS_AO = False

def run_benchmark(precision_name, dtype, size=10000, iterations=50):
    print(f"\n[Testing {precision_name}]...")
    try:
        if dtype is None:
            raise AttributeError("Dtype not available in this build.")

        # Blackwell Fix: randn doesn't support 8-bit, so we create and convert
        if "float8" in str(dtype):
            a = torch.randn(size, size, device='cuda', dtype=torch.float16).to(dtype)
            b = torch.randn(size, size, device='cuda', dtype=torch.float16).to(dtype)
        else:
            a = torch.randn(size, size, device='cuda', dtype=dtype)
            b = torch.randn(size, size, device='cuda', dtype=dtype)

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a, b)

        torch.cuda.synchronize()
        end_time = time.time()

        print(f"  SUCCESS: {iterations} iterations in {end_time - start_time:.4f}s")
        print(f"  Avg: {(end_time - start_time) / iterations:.6f}s")

    except Exception as e:
        print(f"  NOT WORKING: {precision_name} failed.")
        print(f"  Reason: {str(e)[:100]}")

def main():
    print("="*50)
    print("RTX 5080 BLACKWELL COMPREHENSIVE BENCHMARK")
    print("="*50)

    # 1. Hardware & Capabilities Probe
    prop = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {prop.name} (sm_{cc[0]}{cc[1]})")

    # Explicit FP4/FP8 Feature Check
    # On Blackwell (12.0), these should technically be True if the driver is cu130
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    print(f"Native FP8 Support: {has_fp8}")
    print(f"Native NVFP4 Support: {HAS_AO} (via torchao)")
    print("-" * 50)

    N = 10000

    # Standard Precisions
    run_benchmark("FP16 (Half)", torch.float16, size=N)
    run_benchmark("BF16 (BFloat16)", torch.bfloat16, size=N)

    # Blackwell 8-bit (FP8)
    run_benchmark("FP8 (E4M3FN)", getattr(torch, 'float8_e4m3fn', None), size=N)
    run_benchmark("FP8 (E5M2)", getattr(torch, 'float8_e5m2', None), size=N)

    # Blackwell 4-bit (NVFP4) Reporting
    print("\n[Testing NVFP4 (4-bit)]...")
    if not HAS_AO:
        print("  SKIPPED: NVFP4 requires 'pip install torchao'.")
        print("  Note: Blackwell hardware supports this, but it requires a specialized quantizer.")
    else:
        print("  Status: Hardware ready. Run 'torchao' examples to see 4-bit speedups.")

    run_benchmark("FP32 (Full Legacy)", torch.float32, size=N)

    print("\n" + "="*50)
    print("BENCHMARK COMPLETE")
    print("="*50)


# ── yogamann pipeline import checks ─────────────────────────────────────────
def check_yogamann_imports():
    print("\n" + "="*50)
    print("YOGAMANN PIPELINE IMPORT CHECKS")
    print("="*50)

    checks = [
        ("DWposeDetector (no mediapipe)",
         "from controlnet_aux import DWposeDetector"),
        ("StableDiffusionXLControlNetPipeline",
         "from diffusers import StableDiffusionXLControlNetPipeline"),
        ("ControlNetModel",
         "from diffusers import ControlNetModel"),
        ("Depth Anything V2 via transformers",
         "from transformers import pipeline as hf_pipeline; hf_pipeline('depth-estimation')"),
    ]

    all_ok = True
    for name, stmt in checks:
        try:
            exec(stmt)
            print(f"  [OK]  {name}")
        except Exception as e:
            print(f"  [FAIL] {name}")
            print(f"         {str(e)[:120]}")
            all_ok = False

    print()
    if all_ok:
        print("All yogamann imports OK -- pipeline ready.")
    else:
        print("Some imports failed. Run 'uv sync' to install missing packages.")

    # VRAM headroom report
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"\nVRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
        print("SDXL in bfloat16 uses ~8 GB -- headroom looks",
              "good." if free > 8e9 else "tight -- close other GPU apps.")


if __name__ == "__main__":
    main()
    check_yogamann_imports()
