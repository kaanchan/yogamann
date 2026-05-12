For your yoga pose annotation pipeline, the hardware constraints (16 GB VRAM) and the need for spatial awareness make this a high-precision task. As of early 2026, the local vision model landscape has shifted toward **Qwen2.5-VL** and **InternVL2.5**, which significantly outperform the original Llama 3.2 Vision in spatial grounding.

### 1. Ranked Shortlist

1. **Qwen2.5-VL 7B (Instruct)**
* **Justification:** The gold standard for this task. It was specifically trained for visual grounding (bounding boxes/points) and supports high-resolution input. It natively understands complex layouts and is extremely reliable for JSON extraction. Its spatial reasoning allows it to "see" a bent elbow versus a straight one more accurately than its peers.


2. **InternVL2.5 8B**
* **Justification:** Excellent for dense visual details. It uses a "dynamic resolution" approach that preserves small details (like finger positions or foot rotation) that smaller models often blur. It is slightly more "verbose" than Qwen but very strong at anatomical descriptions.


3. **Llama 3.2 Vision 11B**
* **Justification:** A solid "all-rounder." While its spatial precision is slightly lower than Qwen2.5-VL, its instruction following is top-tier. Use this if you find the other models struggling with the complexity of your "failure pattern" logic.


4. **Molmo 7B-D (Allen Institute)**
* **Justification:** A specialized grounding model. It excels at pointing to specific pixels. If your goal is to locate exactly *where* a deformity occurs, Molmo is superior, though its JSON schema following is slightly more brittle than Qwen’s.



---

### 2. Comparison Table (16 GB VRAM Context)

| Model | Params | VRAM (Q4_K_M) | Dual-Image | JSON Reliability | Pose Rating |
| --- | --- | --- | --- | --- | --- |
| **Qwen2.5-VL 7B** | 7.3B | ~6.5 GB | **Yes (Native)** | 5/5 | 5/5 |
| **InternVL2.5 8B** | 8.2B | ~7.2 GB | Yes | 4/5 | 5/5 |
| **Llama 3.2 11B** | 11B | ~9.5 GB | Yes | 5/5 | 4/5 |
| **Molmo 7B-D** | 7B | ~6.0 GB | Partial | 3/5 | 5/5 |
| **Phi-3.5 Vision** | 4.2B | ~3.8 GB | No | 3/5 | 2/5 |

---

### 3. Setup Instructions (Top Recommendation)

For the **RTX 5080**, use **Ollama** for the best balance of speed and easy multi-image handling.

* **Ollama Command:**
```bash
ollama pull qwen2.5-vl:7b

```


* **LM Studio Path:** Search for `Qwen2.5-VL-7B-Instruct-GGUF` by `bartowski` or `MaziyarPanahi`. Select the **Q4_K_M** or **Q5_K_M** version.

---

### 4. Sample Prompt (Structured Annotation)

**System Prompt:**

> You are a professional yoga instructor and anatomical movement specialist. You will be provided with a source photo and a generated mannequin render. Your task is to audit the render for anatomical accuracy and visual quality.
> Return ONLY a JSON object following this schema:
> {
> "rating": "Bad" | "OK" | "Good" | "Gold",
> "alignment_issues": [{"part": string, "issue": string}],
> "artifacts": [string],
> "failure_patterns": [string],
> "notes": string
> }

**User Prompt:**

> [Image 1: Source] [Image 2: Render]
> Compare the mannequin render to the source yoga photo. Check for "prone back-bend pose detection failure" and specific misalignment in the shoulders and neck.

---

### 5. Technical Insights for RTX 5080 (16 GB)

* **Inference Speed:** On your RTX 5080, Qwen2.5-VL 7B (Q4) will likely hit **70-90 tokens/sec** for text generation. The "prefill" (image processing) phase will take ~200-400ms per pair of images.
* **Dual-Image Logic:** Qwen2.5-VL treats multiple images as a sequence. It can differentiate between them by index (e.g., "In the first image, the leg is straight; in the second image, it is bent").
* **Quantization Tip:** With 16 GB VRAM, you have plenty of room. Use **Q6_K** or even **Q8_0** for the 7B/8B models. The precision gain in "Pose Awareness" is noticeable when moving from Q4 to Q8 in vision tasks.

### 6. Caveats

* **VRAM Fragmentation:** When running via Streamlit, ensure your Python environment isn't hogging VRAM with the SDXL/ControlNet weights. Use `torch.cuda.empty_cache()` or run the annotation script as a separate process.
* **Licensing:** Qwen2.5-VL and Llama 3.2 are highly permissive (Apache 2.0 / Llama Community License), making them safe for most local dev projects. InternVL2.5 is also commercially friendly but check the specific "OpenBV" license terms.

```

```