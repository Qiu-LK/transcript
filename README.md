# 微信视频号文字稿提取器

## 使用

1. 首次使用双击 `install.bat` 安装依赖。
2. 在项目根目录 `.env` 中填写阿里云配置，可参照 `.env.example`。密钥只保存在本机，不要上传或分享此文件。
3. 双击 `start.bat`，浏览器会打开当前版本页面。已有端口被占用时自动选择空闲端口，不会关闭其他进程。
4. 选择“新版：阿里云转写 + 独立声纹校正”，粘贴分享链接；已知说话人数可选择人数，然后生成文字稿。人数是模型参考，不能保证正确分离。

首次在新电脑运行时，请先安装 Python 3.12 和 Git，运行 `install.bat`，再运行 `install_model.bat` 下载转写、声纹和视觉模型。模型体积较大，下载时间取决于网络；模型文件不包含在 Git 仓库里。复制 `.env.example` 为 `.env` 后填写自己的阿里云配置，再运行 `start.bat`。项目仅在本机监听，启动窗口显示的地址才是当前页面地址。`.env`、视频、文字稿、模型、虚拟环境和本地评测数据均不上传到仓库。旧浏览器扩展目录仅作历史参考。

填写 API Key 后默认选择新版。阿里云负责转写和词级时间戳，本地 WeSpeaker 独立聚类并按首次出现编号；中文模糊短句额外用英文声纹模型交叉核对。新版还运行独立 Pyannote 音频分段 + ERes2NetV2 多语言声纹复核，以及 LR-ASD 主动说话人检测。后两者只在证据充分时修复“未判定”，不会强行用其聚类编号覆盖现有说话人。此流程不使用人工参考稿或剧情文本给人物贴标签，也没有独立强制对齐。未确定的短片段仍标为未判定。阿里云或校正失败会明确报错。页面仍可选择阿里云原始标注作对比。

新版需 `models/speaker-diarization/` 内的 WeSpeaker、`segmentation.onnx` 和 `embedding-eres2netv2.onnx`，以及 `models/visual/lr_asd_pretrain_ava.model`。这台电脑已有这些文件。LR-ASD 的 MIT 许可代码位于 `third_party/lr_asd`。视觉支路使用 CPU，仅在存在低置信/未判定片段时检查其附近画面。本地 Whisper 文字复核默认最多检查 10 段符合现有修正规则的中文长句，并在 120 秒后停止，保留阿里云原文；页面会显示已检查段数。可用 `WX_TEXT_REVIEW_MAX_CLIPS` 和 `WX_TEXT_REVIEW_TIMEOUT` 调整数量与秒数，或用 `WX_LOCAL_TEXT_REVIEW=0` 关闭此阶段。`/api/health` 可核对当前服务版本是否为 `audio-visual-v4`。

## 云端配置

需要 `DASHSCOPE_API_KEY`、`ALIYUN_ASR_MODEL=paraformer-v2`、`ALIYUN_SPEAKER_DIARIZATION=true`，以及 `ALIYUN_OSS_ACCESS_KEY_ID`、`ALIYUN_OSS_ACCESS_KEY_SECRET`、`ALIYUN_OSS_BUCKET`、`ALIYUN_OSS_ENDPOINT`。Endpoint 填公网域名，例如 `oss-cn-guangzhou.aliyuncs.com`。

若使用业务空间专属 API Key，可将控制台对应地址填入 `DASHSCOPE_BASE_URL`。程序会使用该域名的 `/api/v1` 接口。

OSS 使用私有 Bucket，保持禁止公共访问即可。RAM 凭据需要对该 Bucket 的 `wx-transcript-temp/*` 对象具有 `oss:PutObject`、`oss:GetObject`、`oss:DeleteObject` 权限，不需要公开 Bucket。

流程为下载视频 → 提取 16kHz 单声道 WAV → 上传私有 OSS → 生成有效期 1 小时的签名 URL → 提交 Paraformer 异步转写和说话人分离 → 下载结果 → 删除 OSS 临时文件。本地 MP4 和 TXT 保留在 `media`，临时 WAV 自动删除。异常情况下若 OSS 删除失败，会提示检查删除权限；建议为临时前缀设置 1 天到期的生命周期规则作为补充。

音频会发送到阿里云；转写、OSS 存储和网络流量可能产生费用，以账户实际计费为准。云端说话人分离限 2 小时以内音频。短句、重叠发言、音乐和变声仍可能影响准确率。模型未返回说话人编号时标为“说话人未判定”，不会沿用上一人的标签。A/B/C 按首次出现顺序编号，不代表真实姓名。

## 本地选项

可主动选择本地 Whisper + 声纹引擎。此时另需运行 `install_model.bat` 下载本地模型。原有本地链路和实验性剧情校正代码保留，后者默认关闭（`WX_ROLE_CORRECTION=0`），不作为准确性保证。

## 分享链接解析

分享链接发送到第三方 `api.bugpk.com` 解析，视频从腾讯媒体域名下载。公开解析接口可能限流、失效或不支持某些链接；这与云端语音识别是两个步骤。可通过 `WX_PARSE_API` 更换含 `{url}` 的接口模板，多模板用 `|` 分隔。

当前流程不需要浏览器扩展、微信扫码、系统代理或根证书。旧的 `extension` 文件夹保留但不参与此流程。
