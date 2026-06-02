# Censor Module (algorithm v2)

Multi-layer image guardrail for `text2image`, `img2img stylization`, and `img2img editing`.

> **v2** is image-centric. The text guard and OCR are intentionally deferred
> (see [PLAN.md](PLAN.md), P0.1) — their adapter files stay in the tree for later.

## What is implemented

- unified moderation API for input and output checks
- visual multi-label image classifier adapter (zero-shot)
- explicit-content (NSFW) detector adapter
- judge layer: real **ShieldGemma-2** multimodal judge (optional) + a consensus
  heuristic that always runs and gates soft-category blocks
- rule-based decision engine with `allow / review / block`, where soft categories
  block only when a judge (`role="judge"`) confirms them
- CLI for local moderation runs

## Architecture

```text
request
  -> load image (no image -> allow)
  -> sensors (role="sensor")
       -> visual classifier (zero-shot)
       -> explicit content detector (NSFW)
  -> judges (role="judge")
       -> ShieldGemma-2 (optional, strong signal on hard categories)
       -> heuristic consensus (>=2 sensors agree -> confirm)
  -> decision engine
  -> verdict + category + rationale
```

## Quick start

1. Create an environment with Python 3.11+.
2. Install the base dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn censor_guard.app:app --reload
```

4. Open the docs:

`http://127.0.0.1:8000/docs`

## CLI example

```bash
python -m censor_guard.cli --scenario output --image-path path/to/image.png
```

## Environment variables

- `CENSOR_ENABLE_OCR=true|false`
- `CENSOR_ENABLE_VISUAL_CLASSIFIER=true|false`
- `CENSOR_ENABLE_EXPLICIT_DETECTOR=true|false`
- `CENSOR_ENABLE_POLICY_JUDGE=true|false`
- `CENSOR_VISUAL_MODEL_ID=openai/clip-vit-base-patch32`
- `CENSOR_EXPLICIT_MODEL_ID=Falconsai/nsfw_image_detection`
- `CENSOR_POLICY_JUDGE_MODEL_ID=google/shieldgemma-2-4b-it`
- `CENSOR_BLOCK_THRESHOLD=0.85`
- `CENSOR_REVIEW_THRESHOLD=0.55`
- `CENSOR_HF_CACHE_DIR=D:/alpha_siirius/censor_module/.cache/huggingface`
- `CENSOR_TESSERACT_CMD=D:/alpha_siirius/censor_module/tools/Tesseract-OCR/tesseract.exe`

OCR lookup order:
- `CENSOR_TESSERACT_CMD`, if set
- `tesseract` from `PATH`
- Windows defaults under `C:\Program Files\Tesseract-OCR`
- macOS defaults such as `/opt/homebrew/bin/tesseract` and `/usr/local/bin/tesseract`
- Linux default `/usr/bin/tesseract`

## Notes

- v2 moderates the image only. The text guard and OCR adapters exist but are not
  wired into the pipeline yet (see [PLAN.md](PLAN.md), P0.1).
- The visual classifier, explicit detector, and ShieldGemma judge are optional. If
  a backend is unavailable, the service returns a structured `skipped`/`error`
  status instead of failing the whole request.
- ShieldGemma-2 (`google/shieldgemma-2-4b-it`) is a gated model: enable it with
  `CENSOR_ENABLE_POLICY_JUDGE=true`, install `requirements-ml.txt`, and
  `huggingface-cli login`. Until then the consensus heuristic keeps the pipeline
  operational on its own.
