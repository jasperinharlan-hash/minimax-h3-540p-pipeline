# 已验收的 540P 双镜转身接力配置

这是 2026-11-14 制作人看过并回复“还可以，略微脱相但是绝对可以接受”“完美”的那一组测试的完整保存配置。

| 项目 | 第一镜：普通智能参考 | 第二镜：真实尾帧接力 |
|---|---|---|
| 模型 | Ref2VA FP8 | FL2VA FP8 |
| LoRA | Ref turbo4 v0.1 | FL turbo4 v1.0_768p |
| 模式 | reference | image，keyframe_role=first |
| 输入 | 原角色图＋空场景图 | 第一镜真实背身最后一帧 |
| 实际本地生成至落盘核验 | 197.573秒 | 175.166秒 |

共同参数：960×544、24fps、请求6秒、beta调度4步、Euler、Kitchen attention、video shift12/audio shift3、LoRA强度1。实际每镜141帧、5.875秒。

- [配置.json](配置.json)：完整参数、模型名、种子及验收范围。
- [第一镜 API 工作流](01_普通智能参考_API.json) 和 [第二镜 API 工作流](02_真实尾帧接力_API.json)：保留测试实际节点连接和提示词；API图不是前端布局JSON。
- [input](input/)：两张原始普通参考和该验收第一镜的真实尾帧；路径与API图相同。新跑第一镜后，应提取新片真实末帧供第二镜使用。
- [素材清单](素材清单.json)：输入SHA256。
- [公开模型清单](public_models.json)：全部7个公开权重、固定仓库修订、字节数和SHA256，合计64.932GB。模型权重不在此Git仓库中。
- [本地环境快照](本地环境快照.json)：记录本地运行配置，Windows依赖快照不等于Linux安装锁。

在 Kaggle 准备公开模型（无需个人 Hugging Face token）：

```python
%pip install huggingface_hub
```

```bash
python download_public_models.py --models-dir /kaggle/temp/h3_shared_models --profile all --search-root /kaggle/temp/h3_t4x2/ComfyUI/models --search-root /kaggle/input
```

从本目录执行。只需要普通参考路线可使用 `--profile reference`。脚本会核验并复用同SHA256的现有文件，不会提交渲染或重启服务。

两条保存样片使用本地 RTX 5060 Laptop 8GB；Kaggle T4适配和速度尚未实测。两镜原生音频为动作/环境声，本次接受记录不代表新对白测试。接力帧参考仅对应本次明确授权的方案；其他项目继续使用各自制作约束。
