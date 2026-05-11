import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download, list_repo_files

# (model_id, allow_patterns or None, reason)
# (model_id, allow_patterns, ignore_patterns, reason)
MODELS = [
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
        "SDXL fp16 safetensors only (~7 GB) — excludes full-precision, ONNX, Flax, OpenVINO variants",
    ),
    (
        "depth-anything/Depth-Anything-V2-Small-hf",
        None, None,
        "Depth estimation model (~100 MB)",
    ),
]
# Note: DWposeDetector fetches its weights (YOLO + DWPose .pth) from hardcoded
# URLs at runtime — they are not pre-cacheable via snapshot_download.

token = os.environ.get("HF_TOKEN")
cache_dir = os.environ.get("HF_HUB_CACHE", r"D:\models\hub")

print(f"hf_transfer active : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1'}")
print(f"cache              : {cache_dir}")
print(f"token              : {'set' if token else 'NOT SET (will be slower)'}")
print(f"models to fetch    : {len(MODELS)}\n")

for model_id, allow_patterns, ignore_patterns, reason in MODELS:
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

print("All models cached. Run .\\make.ps1 to generate.")
