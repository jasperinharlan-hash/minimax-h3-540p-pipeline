---
name: comfyui-minimax-h3
description: "Authoritative production standard and workflow for MiniMax H3 (RTX 5060 8GB VRAM) on local ComfyUI. Combines the verified Ref2VA + Turbo 4-Step + Kitchen Attention acceleration and H3 native soundscape synthesis with the clean, lightweight 9.20 positive prompt architecture (zero redundant negative constraints)."
---

# ComfyUI MiniMax H3 轻量化制作规范（保留加速与原生音效 · 回归 9.20 纯净提示词架构）

> **核心原则**：在**牢牢保留现有显存加速（Turbo 4步 + Kitchen Attention + 4K无缝超分）**与**原生同步声音（H3 Native Audio）**的基础上，彻底移除后期堆叠的冗余负面诅咒词（`Strictly NO...`）与形式主义表格（`retention_analysis`），回归 9 月 20 日轻盈、灵动、纯正向的**三段式结构化提示词**。

---

## 0. 核心制作准则

1. **黄金两秒与叙事节奏**：
   - 开篇直接见角色、见动作、见冲突或萌点；动作自然流畅、富有弹性与卡通趣味。
2. **100% 纯正向肯定句描述（严禁负面词与禁止式表述）**：
   - 彻底废除 `Strictly NO...`、`NEVER...`、`PROHIBIT...` 等负面描述，避免引发 DiT 交叉注意力逆火。
   - 仅用简明、自然的肯定句描述角色外观、动作与环境（例如直接写 `smooth round tumbler body`，无需附加 `no tail`）。
3. **官方正版资产目录**：
   - 角色立绘统一取自：[`C:\Users\32503\Desktop\立绘-皮克斯\`](file:///C:/Users/32503/Desktop/立绘-皮克斯/)
     - **笨梨（菊花梨）**：[`菊花梨异色.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/菊花梨异色.png) — 圆滚滚果绿毛绒身躯、浅米色圆面盘、棕色弯月笑眼、右上角单颗可爱小尖牙、头顶五瓣白菊花带两根弯曲金黄小花蕊、右下侧两颗深绿斑点、纯圆不倒翁底座。
     - **毛豆（豆豆）**：[`毛豆异色.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/毛豆异色.png) — 3D 软萌毛绒、戴白兔连帽衫（帽身带两块青色圆斑）、粉色齐刘海与粉色眉毛、深蓝大眼睛、温柔微笑。
     - **大耳（帽兜）**：[`大耳异色.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/大耳异色.png) — 纯白软萌毛绒玩偶、绿松石大眼睛、粉色渐变下垂长垂耳（内侧带黄色星纹）、连指肉球手、圆润短腿。
     - **雪影 / 伙伴**：[`雪影异色.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/雪影异色.png)、[`雪影.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/雪影.png)、[`小夜异色.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/小夜异色.png)、[`睡玲.png`](file:///C:/Users/32503/Desktop/立绘-皮克斯/睡玲.png)
   - 背景参考图内保持纯环境与道具，不混入无关人物。
4. **尊重共享 ComfyUI 环境与逐镜抽帧抽检**：
   - 提交前确认队列与显存空闲，不中断他人任务；
   - 渲染完成后抽帧核验关键帧画面、角色数量及音画同步度。

---

## 1. 保留的现有核心加速与模型配置（RTX 5060 8GB 黄金参数）

### 1.1 模型与算子配置（Ref2VA + Turbo + Kitchen Attention）
| 算子/模型槽位 | 权威权重 / 参数设置 | 作用说明 |
|:---|:---|:---|
| **fl2va_model** | `none` | 保持为 none（统一使用 Ref2VA 多图参考） |
| **ref2va_model** | `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` | FP8 缩放版图生视频音频主干 |
| **text_encoder** | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | NVFP4 量化版 Qwen3-VL 编码器 |
| **video_vae** | `minimax_h3_video_vae_int8_convrot.safetensors` | INT8 ConvRot 视频 VAE |
| **audio_vae** | `minimax_h3_audio_vae_fp32.safetensors` | FP32 音频 VAE |
| **turbo_lora** | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | Ref2V 专用 Turbo LoRA（强度 1.0） |
| **attention** | `comfy kitchen attention` | 显存优化与推理加速（RTX 5060 8GB 必备） |
| **sigma_shift** | `shift_video = 12.0`, `shift_audio = 3.0` | Ref2VA 扩散去噪标准时序偏移 |
| **scheduler** | `BasicScheduler` (`scheduler: "beta"`, `steps: 4`, `denoise: 1.0`) | 4 步极速抗融化调度锁定 |
| **sampler** | `KSamplerSelect` (`sampler_name: "euler"`) | 欧拉采样器 |
| **resolution** | 540P (`960×544`) 或 768P (`1344×768`)，24fps | 兼顾速度与显存安全 |

### 1.2 4K AI 超分后处理管道（按需启用）
* **模型**：`RealESRGAN_x4plus_anime_6B.pth`（PyTorch FP16 CUDA，`spandrel` 加载）。
* **防换页策略**：上下 2 分块推理（`pad = 16px`），峰值显存 **3.78 GB**，4x 重建后在 GPU 端 `bicubic` 降采样对齐 `3840×2160`。
* **流式封装**：FFmpeg `h264_nvenc` (`preset: p4`, `cq: 18`) + `-c:a copy` 无损复制原生音轨。

### 1.3 标准 API 工作流拓扑
```text
MiniMaxH3EasyLoader (1)
  ├── ref2va_model, text_encoder, video_vae, audio_vae
  └── [h3_bundle] ──────► MiniMaxH3Easy (2) ◄── [media_1~4] (LoadImage: 立绘与背景)
                             ├── [model] ──────► ModelAttentionBackend (4) [comfy kitchen attention]
                             │                     └── [model] ──► LoraLoaderModelOnly (35) [Turbo 4-step]
                             │                                       └── [model] ──► MiniMaxH3SigmaShift (13)
                             │                                                         └── [model] ──► BasicScheduler (5) [beta, 4 steps]
                             │                                                                         BasicGuider (15)
                             │                                                                         SamplerCustomAdvanced (7)
                             └── [h3_context] ──► MiniMaxH3EasyOutput (8)
                                                    ├── [conditioning] ──► BasicGuider (15)
                                                    ├── [latent] ────────► SamplerCustomAdvanced (7)
                                                    ├── [video_vae] ─────► VAEDecode (9)
                                                    └── [audio_vae] ─────► VAEDecodeAudio (10)
                                                                            └──► CreateVideo (11) ──► SaveVideo (12)
```

---

## 2. 保留的现有声音工程规范（H3 原生同步音效）

* **模式锁定**：`audio_mode: "generated"`，**`ref_audios` 保持为空**（不外挂 `LoadAudio` 节点，避免合成器和弦杂音）。
* **声音触发写法**（直接在提示词末尾 `[Synchronized Audio]` / `Audio:` 中用简洁自然语言描述）：
  * **纯物理 ASMR 拟音**：微距脚步声、揉搓摩擦声、雪地踩踏声、风声与温暖空间空气感。
  * **角色短萌叫与口癖**（仅限 1~2 字叹词或短口癖，严禁多字长句）：
    * 萌叫：`Da-Er makes a brief soft high-pitched baby-like effort sound, “嗯呀”.`
    * 短口癖：`Ben Li lets out a cheerful tiny baby call: <d>[Chinese] 柿饼！</d>.`
* **严禁引入外部 TTS 念白与长句对话**，杜绝模型将提示词念出。

---

## 3. 回归 9 月 20 日经典轻量化提示词标准（去冗余三段式）

彻底摆脱冗长的负面词堆砌与 `retention_analysis` 表格，采用直观、干净、正向的**三段式 + 原生音效**结构：

```text
[Characters Identity]:
Two adorable 3D plush mascot friends:
- Da-Er <Picture 1>: fluffy white bunny mascot with giant sparkling turquoise starry eyes, long pink-tipped floppy ears with yellow star patterns inside, soft mitten paws.
- Ben Li <Picture 2>: chubby pear-shaped chartreuse plush mascot with a cream-beige round face, happy crescent squinting eyes, a single tiny milk tooth at upper right corner, blooming white daisy with two golden bobbing antennas on head, and two dark green flank spots.

[Environment Lock]:
Fixed eye-level 16:9 warm shot on the sunlit wooden porch overlooking a peaceful autumn courtyard identical to <Picture 3>, warm golden sunlight.

[Cinematic Action & Dynamics]:
Ben Li waddles happily forward with brisk toddler steps toward a wooden tray of orange persimmons, tiny paws reaching out excitedly as its golden flower antennas bounce with each step. Beside him, Da-Er hops lightly and playfully, long soft ears swaying in the breeze, tilting its head with bright sparkling curiosity. Lively cartoon mascot timing, bouncy squash-and-stretch physics, rich cozy plush textures.

[Synchronized Audio]:
Gentle autumn breeze, brisk soft toddler pitter-patter footsteps on the wooden porch, playful soft plush rustling, a happy tiny baby chirp from Ben Li: <d>[Chinese] 柿饼！</d>. Pure environmental ASMR foley.
```

---

## 4. 交付路径规范
* 所有公共交付成果统一保存至：`C:\Users\32503\Desktop\codex\<项目名称>\`；
* 同步镜像一份至谷歌云盘：`H:\我的云端硬盘\`。
