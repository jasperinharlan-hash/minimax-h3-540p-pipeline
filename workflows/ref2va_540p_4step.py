"""Verified MiniMax H3 Ref2VA 540P (960x544) 4-Step Turbo ComfyUI Workflow Builder."""

import json
import time
import urllib.request
from pathlib import Path

COMFY_HOST = "http://127.0.0.1:8188"


def build_ref2va_540p_workflow(
    prompt_text: str,
    image_1: str,
    image_2: str | None = None,
    seed: int = 20262104,
    filename_prefix: str = "h3_ref2va_540p",
    seconds: float = 5.0,
    fps: float = 24.0,
) -> dict:
    """Build a verified 540P Ref2VA + 4-Step Turbo + Kitchen Attention + SigmaShift workflow."""
    wf = {
        "101": {
            "inputs": {"image": image_1, "upload": "image"},
            "class_type": "LoadImage",
        },
        "1": {
            "inputs": {
                "fl2va_model": "none",
                "ref2va_model": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
                "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            },
            "class_type": "MiniMaxH3EasyLoader",
        },
        "2": {
            "inputs": {
                "h3_bundle": ["1", 0],
                "mode": "reference",
                "resolution": "540P",
                "aspect_ratio": "16:9",
                "width": 960,
                "height": 544,
                "seconds": seconds,
                "fps": fps,
                "advanced": True,
                "keyframe_role": "first",
                "ref_image_size": "match",
                "reference_mention_mode": "index",
                "prompt_optimizer_settings": False,
                "prompt_optimizer_scene_guide": "none",
                "prompt": prompt_text,
                "media_1": ["101", 0],
                "media_type_1": "image",
            },
            "class_type": "MiniMaxH3Easy",
        },
        "3": {
            "inputs": {
                "model": ["2", 0],
                "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "strength_model": 1.0,
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "4": {
            "inputs": {
                "model": ["3", 0],
                "attention": "comfy kitchen attention",
            },
            "class_type": "ModelAttentionBackend",
        },
        "13": {
            "inputs": {
                "model": ["4", 0],
                "shift_video": 12.0,
                "shift_audio": 3.0,
            },
            "class_type": "MiniMaxH3SigmaShift",
        },
        "5": {
            "inputs": {
                "model": ["13", 0],
                "scheduler": "beta",
                "steps": 4,
                "denoise": 1.0,
            },
            "class_type": "BasicScheduler",
        },
        "6": {
            "inputs": {"sampler_name": "euler"},
            "class_type": "KSamplerSelect",
        },
        "7": {
            "inputs": {
                "model": ["13", 0],
                "h3_context": ["2", 1],
                "sampler": ["6", 0],
                "sigmas": ["5", 0],
                "seed": seed,
            },
            "class_type": "MiniMaxH3EasySampler",
        },
        "8": {
            "inputs": {"h3_context": ["2", 1]},
            "class_type": "MiniMaxH3EasyOutput",
        },
        "9": {
            "inputs": {"samples": ["7", 0], "vae": ["8", 2]},
            "class_type": "VAEDecode",
        },
        "10": {
            "inputs": {"samples": ["7", 0], "vae": ["8", 3]},
            "class_type": "VAEDecodeAudio",
        },
        "11": {
            "inputs": {"images": ["9", 0], "audio": ["10", 0], "fps": ["8", 4]},
            "class_type": "CreateVideo",
        },
        "12": {
            "inputs": {
                "video": ["11", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
            },
            "class_type": "SaveVideo",
        },
    }
    if image_2:
        wf["102"] = {
            "inputs": {"image": image_2, "upload": "image"},
            "class_type": "LoadImage",
        }
        wf["2"]["inputs"]["media_2"] = ["102", 0]
        wf["2"]["inputs"]["media_type_2"] = "image"
    return wf


def wait_for_queue_idle(comfy_host: str = COMFY_HOST, timeout: int = 900) -> None:
    t0 = time.time()
    while True:
        q = json.loads(urllib.request.urlopen(f"{comfy_host}/queue", timeout=3).read().decode("utf-8"))
        if not q.get("queue_running") and not q.get("queue_pending"):
            return
        if time.time() - t0 > timeout:
            raise TimeoutError("ComfyUI queue did not become idle in time.")
        time.sleep(5.0)


def submit_and_wait(workflow: dict, comfy_host: str = COMFY_HOST, comfy_output_root: str = r"D:\ComfyUI_H3_isolated_20261021\ComfyUI\output") -> Path:
    wait_for_queue_idle(comfy_host)
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"{comfy_host}/prompt", data=data, headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))["prompt_id"]
    while True:
        time.sleep(2.5)
        hist = json.loads(urllib.request.urlopen(f"{comfy_host}/history/{pid}").read().decode("utf-8"))
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(entry["status"], ensure_ascii=False))
            out = entry.get("outputs", {}).get("12", {})
            items = out.get("images") or out.get("videos") or out.get("gifs") or []
            if items:
                fn = items[0]["filename"]
                sub = items[0].get("subfolder", "")
                root = Path(comfy_output_root)
                return root / sub / fn if sub else root / fn
