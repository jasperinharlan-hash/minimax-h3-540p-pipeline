# MiniMax H3 540P Ref2VA + FL2VA Production Pipeline (RTX 5060 8GB + 16GB RAM)

Verified, zero-melting, clear-audio, seamless shot-to-shot continuity pipeline for **ComfyUI + MiniMax H3** on **8GB VRAM / 16GB System RAM** hardware.

## Verified Benchmarks (RTX 5060 Laptop GPU 8GB + 16GB RAM)

| Stage | Model & LoRA | Resolution | Time / Shot | Key Result |
| :--- | :--- | :--- | :--- | :--- |
| **Anchor / Scene Cut Shot** | `ref2va` FP8 + `ref2v_turbo_4step_v0.1` | `540P (960×544)` 24fps 5.0s | `186.1s` | 0 melting, Whisper verified `'秘密转学生'` (`no_speech_prob=0.299`) |
| **Seamless Continuation Shot** | `fl2va` FP8 + `fl2v_turbo_4step_v1.0_768p` | `540P (960×544)` 24fps 5.0s | `165.4s` | Frame-0 MAE vs previous end frame: `5.00 / 255`, Whisper `''` (no hallucinated speech) |

> **Why `540P (960×544)` instead of `768P (1344×768)`?**  
> On an 8GB VRAM + 16GB RAM machine, `768P` overflows both VRAM and physical RAM into Windows pagefile thrashing (`15.34 GB` commit / `0.19 GB` working set, GPU stalled at `34.55 W`). `540P` fits cleanly in memory and finishes in ~165–186s per 5s shot.

---

## 1. Required Models (ComfyUI `models/`)

- **Diffusion Models** (`models/diffusion_models/`):
  - `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`
  - `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- **Text Encoder**:
  - `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- **VAEs**:
  - `minimax_h3_video_vae_int8_convrot.safetensors`
  - `minimax_h3_audio_vae_fp32.safetensors`
- **Turbo 4-Step LoRAs** (`models/loras/`):
  - `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`
  - `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`

## 2. Recommended ComfyUI Launch Flags

```text
python main.py --windows-standalone-build --disable-auto-launch --listen 127.0.0.1 --port 8188 --disable-cuda-malloc --disable-async-offload --disable-cuda-graphs --disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-GGUF ComfyUI-MiniMaxH3-Easy
```

## 3. Core Node & Sampling Configuration

- **Attention Backend**: `ModelAttentionBackend` -> `comfy kitchen attention`
- **LoRA**: `LoraLoaderModelOnly` (`strength_model = 1.0`)
- **Sigma Shift**: `MiniMaxH3SigmaShift` -> `shift_video = 12.0`, `shift_audio = 3.0`
- **Scheduler**: `BasicScheduler` -> `scheduler = "beta"`, `steps = 4`, `denoise = 1.0`
- **Sampler**: `KSamplerSelect` -> `sampler_name = "euler"` with `MiniMaxH3EasySampler`

## 4. Prompt & Native Audio Rules (Anti-Corruption)

1. Keep `retention_analysis:` present as a structural anchor.
2. Write `detailed_description:` in pure English (with `[CLEAN SCREEN LOCK]` to prevent subtitles/watermarks).
3. In `overall_soundscape:`, **do NOT** use `<Subject X>` tags, `whispers`, or Chinese narration prefixes—keep `overall_soundscape` in clean English foley/voice description, and wrap spoken Chinese lines only in `<d>[Chinese] ... </d>`.
4. For `fl2va` continuation shots (`mode="image"`, `keyframe_role="first"`), extract the exact last frame of the previous shot to `ComfyUI/input/` and set `media_1` to that frame.

## 5. Repository Layout

- `workflows/ref2va_540p_4step.py` — Parameterized Ref2VA 540P builder & ComfyUI runner
- `workflows/fl2va_540p_4step.py` — Parameterized FL2VA 540P first-frame continuation builder & runner
- `workflows/ref2va_540p_4step_api.json` — Raw ComfyUI API JSON template (Ref2VA)
- `workflows/fl2va_540p_4step_api.json` — Raw ComfyUI API JSON template (FL2VA)
- `prompts/prompt_validator.py` — Prompt & continuity lock validator
- `docs/AGENTS.md` & `docs/SKILL.md` — Production rules and skill specification
