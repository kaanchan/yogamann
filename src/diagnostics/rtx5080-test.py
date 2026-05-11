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

        is_fp8 = "float8" in str(dtype)
        if is_fp8:
            # FP8 requires _scaled_mm — plain matmul rejects these types by design.
            # cuBLAS FP8 GEMM requires a = row-major [M,K], b = column-major [K,N].
            # b.T (the view) has stride=[1,N] — column-major — without making a copy.
            # b.t().contiguous() would create a NEW row-major tensor (wrong layout).
            # E5M2×E5M2 is not a valid cuBLAS config; use E4M3FN for both operands.
            fp8_size = min(size, 4096)  # large FP8 GEMM may exceed cuBLAS heuristic range
            fp8_dtype = torch.float8_e4m3fn
            a = torch.randn(fp8_size, fp8_size, device='cuda', dtype=torch.float16).to(fp8_dtype)
            b = torch.randn(fp8_size, fp8_size, device='cuda', dtype=torch.float16).to(fp8_dtype).T
            scale = torch.tensor(1.0, device='cuda')
        else:
            a = torch.randn(size, size, device='cuda', dtype=dtype)
            b = torch.randn(size, size, device='cuda', dtype=dtype)

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            if is_fp8:
                _ = torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                     out_dtype=torch.bfloat16)
            else:
                _ = torch.matmul(a, b)

        torch.cuda.synchronize()
        end_time = time.time()

        print(f"  SUCCESS: {iterations} iterations in {end_time - start_time:.4f}s")
        print(f"  Avg: {(end_time - start_time) / iterations:.6f}s")

    except AttributeError as e:
        print(f"  SKIPPED: {precision_name} ({e})")
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

    # Blackwell 8-bit (FP8) — capped at 4096² (cuBLAS heuristic limit); BF16/FP16 run at 10k²
    # E4M3FN x E4M3FN: standard FP8 GEMM path
    run_benchmark("FP8 (E4M3FN x E4M3FN, 4096²)", getattr(torch, 'float8_e4m3fn', None), size=N)
    # E5M2 x E5M2 is NOT a valid cuBLAS config (only E4M3FN x E4M3FN or mixed E4M3FN/E5M2)
    run_benchmark("FP8 (E5M2 x E5M2 — not a valid cuBLAS config)", None, size=N)

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
        compile_available = hasattr(torch, 'compile')
        compile_note = "torch.compile available -- UNet will be compiled on first run (~2-5 min warmup)" \
                       if compile_available else "torch.compile NOT available -- running in eager mode"
        print("All yogamann imports OK -- pipeline ready.")
        print(f"Compile: {compile_note}")
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
