"""Verified MiniMax H3 FL2VA 540P (960x544) 4-Step First-Frame Continuation Workflow Builder."""

from pathlib import Path
from ref2va_540p_4step import COMFY_HOST, submit_and_wait


def build_fl2va_540p_workflow(
    prompt_text: str,
    first_frame_image: str,
    seed: int = 20262105,
    filename_prefix: str = "h3_fl2va_540p",
    seconds: float = 5.0,
    fps: float = 24.0,
) -> dict:
    """Build a verified 540P FL2VA first-frame continuation workflow."""
    return {
        "101": {
            "inputs": {"image": first_frame_image, "upload": "image"},
            "class_type": "LoadImage",
        },
        "1": {
            "inputs": {
                "fl2va_model": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                "ref2va_model": "none",
                "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
                "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            },
            "class_type": "MiniMaxH3EasyLoader",
        },
        "2": {
            "inputs": {
                "h3_bundle": ["1", 0],
                "mode": "image",
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
                "lora_name": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
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
