# Deep Research Prompt — Local Vision Models for Yoga Pose Annotation

## Context

I am building a local image annotation pipeline for yoga pose images. The pipeline generates mannequin-style renders from source yoga photos using Stable Diffusion XL + DWPose ControlNet. A Streamlit gallery app lets human reviewers (and eventually AI reviewers) rate each output and tag issues such as:

- Overall quality rating (Bad / OK / Good / Gold)
- Misaligned body parts (face, neck, shoulders, arms, hands, hips, legs, feet)
- Unwanted visual features (deformities, props, artifacts, textured surfaces, facial expressions)
- Known failure pattern tags (e.g. "prone back-bend pose detection failure", "outdoor background bleed")
- Free-text notes

The AI reviewer will be given a pair of images (source yoga photo + generated mannequin output) and asked to return a structured JSON annotation matching the schema above.

---

## Hardware

- **Laptop:** Lenovo Legion 7 Gen 9
- **GPU:** NVIDIA GeForce RTX 5080 Laptop (16 GB GDDR7 VRAM)
- **CPU:** Intel Core Ultra 9 275HX (Arrow Lake-HX, 24 cores, 2.7 GHz base)
- **System RAM:** 32 GB DDR5
- **Storage:** NVMe SSD (assume ample space)
- **Runtime:** Windows 11, LM Studio or Ollama for local inference

---

## Candidate Models (already identified — please supplement and evaluate)

The following models were identified as initial candidates. Please research their current state, best available quantizations, and suitability for this task:

1. **LLaMA 3.2 Vision 11B** (Meta) — strong pose/body understanding, JSON output
2. **Qwen2-VL 7B** (Alibaba) — strong spatial reasoning, strict schema following
3. **Phi-3.5-Vision 4B** (Microsoft) — fastest inference, good structured output, weaker anatomy
4. **InternVL2 8B** (Shanghai AI Lab) — dense visual detail, less mature LM Studio support

---

## Research Questions

Please research and compare all viable options across these dimensions:

### 1. Model Catalogue (as of your knowledge cutoff)
List **all publicly available offline-capable vision-language models** that:
- Can be run locally (GGUF via llama.cpp / Ollama / LM Studio, or similar)
- Fit within **16 GB VRAM** at a reasonable quantization (Q4_K_M or better preferred)
- Accept image input (multi-modal, not text-only)

Include any newer releases not in the candidate list above (e.g. LLaMA 3.3 Vision, Gemma 3 Vision, Mistral Pixtral, MiniCPM-V, Moondream, InternVL3, etc.)

### 2. Pose and Human Body Awareness
**Primary:** Quality assessment of a generated render against a source pose
- Identifying individual body parts and their relative positions in a rendered image
- Detecting misalignment between a source photo pose and a generated mannequin output
- Distinguishing left vs right limbs, occluded limbs, unusual body orientations (prone, inverted)
- Recognising unwanted visual features: deformities, artifacts, background bleed, props

**Secondary (useful for future automation):** Anatomical correctness assessment
- Describing alignment issues in anatomical terms (e.g. "left arm bent at incorrect angle", "hip rotated past neutral")
- Understanding yoga / athletic poses by name or family
- Flagging biomechanically impossible or implausible joint positions

### 3. Structured / JSON Output Reliability
Which models reliably follow a strict JSON schema when prompted? Rate each on:
- Does it require grammar-constrained decoding (llama.cpp `--json-schema`) to be reliable?
- Does it hallucinate fields outside the schema?
- Does it handle enum fields cleanly (e.g. rating must be one of: Bad, OK, Good, Gold)?

### 4. Dual-Image Input (Comparison Tasks)
The annotation task requires comparing **two images** side by side:
- Source photo (original yoga practitioner)
- Generated output (mannequin render)

Which models handle multi-image prompts well? Which are limited to single-image input?

### 5. VRAM Fit and Quantization
For each recommended model, provide:
- Recommended quantization for 16 GB VRAM (leave headroom for KV cache)
- Estimated VRAM usage at that quantization
- Whether it fits with system RAM offload fallback if needed
- LM Studio / Ollama availability (direct model ID if known)

### 6. Inference Speed
Approximate tokens/sec for image + ~200 token response on RTX 5080 16 GB at recommended quant.
Speed matters: the pipeline may auto-annotate hundreds of images in a batch overnight.

### 7. Prompt Engineering Notes
For the top 2-3 models, provide:
- A sample system prompt that reliably produces structured JSON annotation
- Any known prompt patterns that improve body-part spatial accuracy
- Whether chain-of-thought reasoning before the JSON output improves accuracy

---

## Desired Output Format

Please structure your response as:

1. **Ranked shortlist** (top 3-5 models for this use case) with a one-paragraph justification each
2. **Comparison table** covering all candidates: model, params, VRAM, quant, dual-image, JSON reliability, pose awareness rating (1-5)
3. **Download / setup instructions** for the top recommendation (LM Studio model ID or Ollama pull command)
4. **Sample prompt** for the annotation task (system + user turn, targeting the schema described above)
5. **Any caveats** — licensing restrictions, models that are "open weight" but not truly open, models requiring internet calls despite appearing local

## Out of Scope
- Cloud APIs (OpenAI, Gemini, Claude) — must run fully offline
- Models requiring >16 GB VRAM even at Q4 quantization
- Fine-tuning advice (inference only for now)
- Any model requiring a paid license or API key even for local use
