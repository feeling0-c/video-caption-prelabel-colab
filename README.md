# Video Caption 预标注（全 Colab）

这个项目将“已切分的 MP4 视频片段”直接处理为可导入标注工具的双语 Caption 包。准备、切镜、抽帧、ASR、Qwen 视觉预标注、融合、中文翻译、视频分包和 ZIP 下载都在同一个 Colab 运行时完成；本地电脑不保存中间数据，也不持有 API Key。

## Colab 入口

在 Colab 打开 [notebooks/Video_Caption_Prelabel_All_in_Colab.ipynb](notebooks/Video_Caption_Prelabel_All_in_Colab.ipynb)。首次运行会：

1. 克隆本仓库并安装运行依赖；
2. 上传私有 `input_videos.zip`；
3. 从 Colab Secret `DASHSCOPE_API_KEY` 取得临时 API Key；
4. 先试跑 2 条；
5. 人工检查后，重新完整跑 20 条；
6. 下载 `video_caption_delivery.zip`。

运行前，请在百炼控制台为所选模型启用“免费额度用完即停”。代码不会传入任何付费开关；若服务端返回 403，任务会停止，不会自动换付费模型。

## 输入约定

`input_videos.zip` 内只放 1–60 秒的 MP4 切片，可保持任意相对目录结构。例如：

```text
input_videos.zip
└── mt_movie_2k/video_clips/<group>/<clip>.mp4
```

如果需要保留已有的 `_id` 和最终包中的 `video_path`，同时上传可选的 `source_index.jsonl`。每行只需要视频定位字段，绝不放旧 Caption：

```json
{"_id":"6a37...","source_file":"mt_movie_2k/video_clips/g01/clip.mp4","video_path":"mt_movie_2k/video_clips/g01/clip.mp4"}
```

没有索引也可以运行：代码会从相对视频路径生成稳定的匿名 ID，并把该相对路径作为 `video_path`。

## 同一 Colab 运行时内的流水线

| 阶段 | 运行内容 | 关键产物 |
|---|---|---|
| 输入准备 | PyAV 解码、PySceneDetect 切镜候选、真实 PTS 抽帧、16 kHz 音频 | `inputs/manifest.json` |
| 语音 | faster-whisper `large-v3`，词级/句级时间戳与 VAD | `asr/<id>/asr.json` |
| 画面 | Qwen 视觉模型读抽帧，返回人物、故事线、画面描述 | `visual/<id>/visual.json` |
| 融合 | 程序拥有 ID 与时间；Storyline 时间来自镜头，Speech 时间来自 ASR | `fused_preannotations.jsonl` |
| 翻译 | Qwen 只翻译语义英文文本；时间和原始 Speech 不交给翻译模型 | `translations.jsonl` |
| 交付 | 校验双语章节、复制对应 MP4、输出 JSONL 与 ZIP | `delivery/`、`video_caption_delivery.zip` |

模型输出即使偶尔把 `storyline` 返成字符串而非数组，融合程序也会规范化为数组并记录到 `fuse_warnings.jsonl`；不会因一条格式偏差中断整批任务。

## 最终包

下载的 ZIP 中包含：

```text
delivery/
├── 0001-0020/
│   ├── caption_pilot_0001_0020_final_caption_zh.jsonl
│   └── <与 video_path 一致的 MP4 相对目录>
├── delivery_manifest.jsonl
├── pending_review.jsonl
└── delivery_summary.json
```

JSONL 的每条正式导入记录仅含：

```json
{"_id":"...","video_path":"...","caption_en":"...","caption_zh":"..."}
```

## 数据与安全边界

- `.gitignore` 明确排除视频、帧、音频、运行输出、ZIP、`.env` 与密钥；仓库只提交代码和文档。
- Key 只在 Colab Secret 注入到当前 Python 进程，不作为命令行参数、不写入 JSONL 或 Notebook 输出。
- Qwen 视觉模型不能听音频。因此音量、语气、情绪和说话人与画面人物的关联会明确标为待人工确认；在试产时使用 `--allow-unresolved`，让标注员完成这些字段，而不是伪造模型判断。
- 切镜是工程生成的候选边界，决定 Storyline 可以跳转的区间；它不是由 Caption 模型自行编造的时间戳。

## 源码结构

```text
prepare_raw_videos.py      # 原始 MP4 -> manifest、镜头、帧、音频
run_asr.py                 # faster-whisper ASR
qwen_api.py                # Qwen 视觉预标注
fuse_results.py            # 结构、ID、时间戳的确定性融合
translate_fields.py        # 字段级中文翻译
finalize_delivery.py       # 严格校验和导入包生成
run_colab_pipeline.py      # 以上各步骤的一键编排
notebooks/                 # 可直接在 Colab 打开的入口
```
