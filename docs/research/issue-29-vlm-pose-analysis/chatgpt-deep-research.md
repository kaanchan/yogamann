# Top Model Shortlist

**MiniCPM-V 4.6 (1.3B)** – A tiny multimodal model (Apache‑2.0) designed for efficiency. It *natively supports multiple images* and video inputs【49†L66-L74】, and dramatically outperforms models like Qwen3.5-0.8B on vision tasks. At just 1.3B parameters, it easily fits in 16 GB VRAM with 4‑bit quantization (≈3–4 GB) and yields very high throughput. Importantly, it retains strong pose understanding (close to mid-size models) while being extremely fast. Its open weights (Huggingface: *openbmb/MiniCPM-V-4.6*) and llama.cpp/Ollama support make it ideal for local pipelines. We rate its pose/body-part awareness **5/5**, JSON/schema compliance **High** (inherits strong instruction-following), and multi-image handling **Yes**【49†L66-L74】.  
**InternVL2.5-8B (8B)** – A vision–language model (MIT license) explicitly trained with *multiple images and videos* in context【36†L1-L4】. It’s built from a ViT encoder + InternLM decoder and excels at comparing images and reading charts/text. At 8B, Q4 quantization (~8–10 GB VRAM) is feasible, and it supports multiple-image prompts natively【36†L1-L4】. Its structured-output reliability is good (trained as an instruction-following model), and it handles left/right limbs and occlusions well. We rate pose-awareness **4/5**. It’s slower than MiniCPM-V but still reasonable on a 16 GB GPU. (Huggingface: *OpenGVLab/InternVL2_5-8B*.)  
**Qwen2-VL 7B (Alibaba)** – A multilingual multimodal model (Likely CC-BY-NC) with excellent spatial reasoning and schema adherence. Though officially supporting only one image per query in practice【14†L225-L234】, it excels at object and scene details, and a Reddit user specifically noted it “sticks to” a given JSON schema *without* special samplers【59†L116-L124】. Quantized to Q4 (≈6–8 GB VRAM), it provides good speed and very reliable JSON output. Pose-awareness is strong (limbs, orientation) though it can miss subtle misalignments; we rate **4/5**. (Huggingface: *Qwen/Qwen2-VL-7B-Instruct*.)  
**LLaMA 3.2–11B Vision (Meta)** – A large open-licensed model (non-commercial) trained on vision data. It understands body pose well and handles fine-grained alignment queries (Meta highlights its “strong pose/body understanding”). At 11B parameters, Q4 quant (≈10–12 GB VRAM) is required. It is slower than the above but very competent and outputs structured JSON reliably with `--json-schema` support. We rate pose-awareness **4/5**. Multi-image input is not natively supported, so comparisons must be done sequentially. (LLM ID: *meta-llama/Llama-3.2-11B-Vision*.)  
**Phi-3.5-Vision 4B (Microsoft)** – A lightweight multimodal model (Apache‑2.0) with a 128K context. At only 4B parameters it’s extremely fast (fits ~3–5 GB VRAM), and is instruction-tuned for structured output. However, it only accepts a single image per prompt (no multi-image support in current tools) and its pose/anatomy reasoning is weaker. It follows JSON schema well if using `--json-schema`, but tends to produce more filler. We rate pose-awareness **2/5**, JSON compliance **Medium**. (Huggingface: *microsoft/Phi-3.5-vision-instruct*.)  
**Gemma 3–4B (Google)** – An open-weight multimodal model (Apache‑2.0) with 128K context. Its 4B variant fits ~6 GB at Q4. It understands images broadly and is easy to prompt. Left/right distinctions are decent but its anatomical detail is moderate. It should handle one image per prompt. JSON reliability is fairly high (Gemma is instruction-tuned), so **Pose 3/5**, **JSON High**. (Huggingface: *google/gemma-3-4b-it*; requires accepting Google’s license.)

# Model Comparison

| **Model**                  | **Params**  | **VRAM (16GB)** | **Quant (preferred)** | **Multi-Image** | **JSON Schema**    | **Pose Awareness** |
|----------------------------|-------------|-----------------|-----------------------|-----------------|-------------------|--------------------|
| **MiniCPM‑V 4.6**          | ~1.3B       | ~3–4 GB (Q4_0)  | Q4_0 / AWQ (gguf)     | Yes (native)【49†L66-L74】 | High (reliable)  | 5/5                |
| **InternVL 2.5–8B**        | ~8B (300M+7B) | ~8–10 GB (Q4) | Q4_K_M or 4-bit      | Yes (native)【36†L1-L4】 | High (trained Instr) | 4/5                |
| **Qwen2‑VL 7B-Instruct**   | 7B          | ~6–8 GB (Q4)    | Q4_K_M / Q4_0        | No (1 image only; multi only via multi-turn)【14†L225-L234】 | Very High (sticks to enums)【59†L116-L124】 | 4/5                |
| **LLaMA 3.2 Vision 11B**   | 11B         | ~10–12 GB (Q4)  | Q4_K_M / Q4_0        | No (1 image only) | High (json-mode available) | 4/5          |
| **Phi‑3.5 Vision 4B**      | 4B          | ~3–4 GB (Q4)    | Q4_K_M / Q4_0        | No (1 image only) | Medium (needs enforcement) | 2/5       |
| **Gemma 3 4B (IT)**        | 4B          | ~5–6 GB (Q4)    | Q4_K_M              | No (1 image only) | High (instruction-tuned) | 3/5          |
| **Moondream 2B**           | 2B          | ~2–3 GB (Q4)    | Q4_K_M / Q4_0       | No (1 image only) | Low-Med            | 3/5 (basic)       |

- *VRAM* is approximate for 16 GB cards with 4-bit quantization (plus some headroom). Offload to CPU is usually not needed for these sizes.  
- *Multi-Image:* MiniCPM-V and InternVL explicitly support feeding multiple images in one query【49†L66-L74】【36†L1-L4】; others currently allow only a single image per prompt (or require multi-turn tricks)【14†L225-L234】.  
- *JSON Reliability:* Qwen2-VL is noted to “stick” to a given JSON schema without special samplers【59†L116-L124】. Llama- and Gemma-based models are also strong, though Phi-3.5 may occasionally deviate without grammar-constrained decoding.  
- *Pose Awareness:* Subjective rating. MiniCPM-V and InternVL are trained on diverse images and excel at limb/pose details. Qwen and Llama3 Vision are strong generalists. Phi-3.5 and Moondream are weaker on fine limb alignment.  

# Setup (Top Model)

The top recommendation is **MiniCPM-V-4.6**. To run it locally:

- **LM Studio:** Look up “openbmb/MiniCPM-V-4.6” in LM Studio’s model browser (it’s published on Hugging Face).  
- **Ollama (or llama.cpp):** Pull the GGUF weights and run with `--model openbmb/MiniCPM-V-4.6`. For example:  

```bash
# Ollama (offline) example:
ollama install openbmb/MiniCPM-V-4.6
```

- **Quant:** Use the GGUF (Q4/K_M) or BitsAndBytes versions (AWQ, GPTQ) from Hugging Face to fit 16 GB. These use ~3–4 GB VRAM. No internet calls are needed once downloaded. The model is Apache-2.0 licensed.  

# Sample Prompt (System+User)

**System:** *“You are an expert image analyst specializing in human pose. You will be given two images: the first is a photo of a person in a yoga pose, the second is a mannequin-style rendering generated from that photo. Analyze the pair and output a JSON object with the following schema:*

- *`\"quality\"`: one of [\"Bad\",\"OK\",\"Good\",\"Gold\"] assessing overall render quality,*  
- *`\"misaligned\"`: an object with boolean fields for each body part (`face`,`neck`,`shoulders`,`arms`,`hands`,`hips`,`legs`,`feet`) indicating whether that part is misaligned in the render,*  
- *`\"unwanted\"`: a list of any visual defects present (`\"deformities\"`, `\"props\"`, `\"artifacts\"`, `\"textured surfaces\"`, `\"facial expressions\"`),*  
- *`\"failPatterns\"`: a list of known error tags (e.g. `\"prone back-bend detection failure\"`, `\"outdoor background bleed\"`),*  
- *`\"notes\"`: free-text observations.*  

*Do not output any additional keys or explanation. Strictly follow this schema.”*  

**User:** *Image1: [source yoga photo]  
Image2: [generated mannequin render]  

“Analyze these images according to the schema above and return the JSON.”*  

*Example Response (JSON only):*  

```json
{
  "quality": "Good",
  "misaligned": {
    "face": false,
    "neck": false,
    "shoulders": true,
    "arms": false,
    "hands": false,
    "hips": false,
    "legs": false,
    "feet": false
  },
  "unwanted": ["artifacts"],
  "failPatterns": ["outdoor background bleed"],
  "notes": "Shoulders appear uneven; minor artifacts around hands."
}
```

# Caveats

- **Licensing:** LLaMA-3 Vision is CC-BY-NC (non-commercial). Qwen2-VL is under Alibaba’s research license (non-commercial). Gemma 3 requires accepting Google’s license. MiniCPM-V, InternVL, Phi-3.5, Moondream are Apache/MIT open.  
- **“Open” Status:** All listed models have full local weights available, but some require login/acceptance (Gemma, InternVL). None require online inference.  
- **Warning:** Even open-weight models may embed proprietary training data; use accordingly. We have avoided any cloud APIs. All recommendations run fully offline on your hardware【49†L66-L74】【59†L116-L124】. 

