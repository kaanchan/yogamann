"""src/download_models.py — pre-fetch all HF model weights.

Usage:
    python src/download_models.py             # download all models
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
