"""Private OSS staging and DashScope asynchronous transcription. No credentials in results."""
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen
import json
import os
import re
import time
import uuid
import wave


class CloudASRError(RuntimeError):
    pass


def settings():
    names = {
        "api_key": "DASHSCOPE_API_KEY",
        "access_id": "ALIYUN_OSS_ACCESS_KEY_ID",
        "access_secret": "ALIYUN_OSS_ACCESS_KEY_SECRET",
        "bucket": "ALIYUN_OSS_BUCKET",
        "endpoint": "ALIYUN_OSS_ENDPOINT",
    }
    values = {key: os.getenv(env, "").strip() for key, env in names.items()}
    missing = [names[key] for key, value in values.items() if not value]
    if missing:
        raise CloudASRError("阿里云配置缺少：" + "、".join(missing))
    endpoint = values["endpoint"].removeprefix("https://").rstrip("/")
    if not re.fullmatch(r"oss-[a-z0-9-]+\.aliyuncs\.com", endpoint):
        raise CloudASRError("OSS Endpoint 应为 oss-cn-guangzhou.aliyuncs.com 这样的公网域名。")
    values["endpoint"] = "https://" + endpoint
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", values["bucket"]):
        raise CloudASRError("OSS Bucket 名称格式不正确。")
    base = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").strip().rstrip("/")
    parsed = urlparse(base)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.port not in (None, 443)
            or not (parsed.hostname == "dashscope.aliyuncs.com"
                    or str(parsed.hostname).endswith(".maas.aliyuncs.com")
                    or parsed.hostname == "dashscope-intl.aliyuncs.com")):
        raise CloudASRError("DASHSCOPE_BASE_URL 必须为阿里云 HTTPS API 地址。")
    # Console copy buttons may supply an OpenAI-compatible URL, while ASR uses /api/v1.
    values["base"] = f"https://{parsed.netloc}/api/v1"
    values["model"] = os.getenv("ALIYUN_ASR_MODEL", "paraformer-v2").strip() or "paraformer-v2"
    if os.getenv("ALIYUN_SPEAKER_DIARIZATION", "true").lower() not in {"1", "true", "yes", "on"}:
        raise CloudASRError("请将 ALIYUN_SPEAKER_DIARIZATION 设为 true。")
    return values


def public_error(stage, error):
    """Do not stringify SDK/network errors: they can include signed URLs or credentials."""
    code = str(getattr(error, "code", ""))
    code = code if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", code) else type(error).__name__
    status = getattr(error, "status", None)
    status_text = f"，HTTP {status}" if isinstance(status, int) else ""
    return CloudASRError(f"{stage}失败：{code}{status_text}。请检查密钥、地域和权限。")


def request_json(url, data=None, api_key=None, asynchronous=False, method=None):
    headers = {"Accept": "application/json", "User-Agent": "WeChatTranscript/2"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    if asynchronous:
        headers["X-DashScope-Async"] = "enable"
    body = None if data is None else json.dumps(data).encode("utf-8")
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=45) as response:
            return json.load(response)
    except HTTPError as exc:
        code = "HTTPError"
        try:
            payload = json.loads(exc.read(32768))
            candidate = str(payload.get("code", ""))
            if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", candidate):
                code = candidate
        except (ValueError, AttributeError):
            pass
        hints = {
            401: "API Key 无效或与接口业务空间不匹配；业务空间 Key 请配置 DASHSCOPE_BASE_URL。",
            403: "请检查模型权限、业务空间和账户欠费状态。",
            429: "请求限流或额度不足，请稍后重试。",
        }
        raise CloudASRError(f"阿里云请求失败：HTTP {exc.code} / {code}。" + hints.get(exc.code, "请检查接口配置。")) from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise CloudASRError("阿里云连接超时、网络不可达或返回无效数据，请稍后重试。") from None


def parse_result(payload):
    sentences = []
    for transcript in payload.get("transcripts", []):
        sentences.extend(transcript.get("sentences", []))
    sentences.sort(key=lambda item: float(item.get("begin_time") or 0))
    labels, lines = {}, []
    unknown = 0
    for sentence in sentences:
        text = str(sentence.get("text") or "").strip()
        if not text:
            continue
        speaker_id = sentence.get("speaker_id")
        if speaker_id is None or speaker_id == -1:
            speaker = "说话人未判定"
            unknown += 1
        else:
            identifier = str(speaker_id)
            if identifier not in labels:
                n = len(labels)
                labels[identifier] = chr(65 + n) if n < 26 else str(n + 1)
            speaker = "说话人 " + labels[identifier]
        start = max(0.0, float(sentence.get("begin_time") or 0) / 1000)
        end = max(start, float(sentence.get("end_time") or 0) / 1000)
        lines.append({"start": round(start, 2), "end": round(end, 2), "text": text, "speaker": speaker})
    if not lines:
        raise CloudASRError("阿里云没有返回有效语音文字，请检查音轨。")
    return lines, len(labels), unknown


def submit_and_wait(config, signed_url, num_speakers, progress, timeout=900):
    parameters = {"channel_id": [0], "diarization_enabled": True,
                  "timestamp_alignment_enabled": True}
    if num_speakers >= 2:
        parameters["speaker_count"] = num_speakers
    response = request_json(
        config["base"] + "/services/audio/asr/transcription",
        {"model": config["model"], "input": {"file_urls": [signed_url]}, "parameters": parameters},
        api_key=config["api_key"], asynchronous=True,
    )
    task_id = response.get("output", {}).get("task_id")
    if not task_id:
        raise CloudASRError("阿里云未返回任务 ID，请检查模型和接口是否匹配。")
    progress("diarizing", "音频已提交，正在等待阿里云转写和说话人分离。")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = request_json(config["base"] + "/tasks/" + quote(str(task_id), safe=""),
                              api_key=config["api_key"], method="POST")
        output = result.get("output", {})
        status = output.get("task_status")
        if status == "SUCCEEDED":
            files = output.get("results") or []
            item = next((row for row in files if row.get("subtask_status") == "SUCCEEDED"), None)
            if item is None:
                code = str((files[0] if files else {}).get("code") or "SubtaskFailed")
                code = code if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", code) else "SubtaskFailed"
                raise CloudASRError(f"阿里云音频子任务失败：{code}。请检查 OSS 文件下载权限和格式。")
            url = item.get("transcription_url", "")
            parsed = urlparse(url)
            if parsed.scheme != "https" or not str(parsed.hostname).endswith(".aliyuncs.com"):
                raise CloudASRError("阿里云返回的结果链接域名无效。")
            # Signed result URL authenticates itself; never forward the DashScope Key to OSS.
            return request_json(url), str(task_id)
        if status not in {"PENDING", "RUNNING"}:
            raise CloudASRError("阿里云识别任务失败或被取消，请检查模型权限和账户状态。")
        time.sleep(3)
    raise CloudASRError("等待阿里云识别超过 15 分钟，请稍后重试。")


def transcribe(path, num_speakers, decode_audio, progress=None, refine_speakers=False):
    config = settings()
    progress = progress or (lambda status, message: None)
    try:
        import oss2
        import numpy as np
    except ImportError:
        raise CloudASRError("缺少 OSS 上传组件，请运行 install.bat。") from None
    audio_path = Path(path).with_name(Path(path).stem + ".cloud.wav")
    object_key = "wx-transcript-temp/" + uuid.uuid4().hex + ".wav"
    bucket = None
    upload_attempted = False
    cleanup_warning = ""
    try:
        progress("transcribing", "正在提取单声道音频并上传私有 OSS。")
        samples = decode_audio(path)
        if len(samples) > 16000 * 7200:
            raise CloudASRError("阿里云说话人分离仅支持 2 小时以内的音频。")
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(pcm.tobytes())
        try:
            auth = oss2.Auth(config["access_id"], config["access_secret"])
            bucket = oss2.Bucket(auth, config["endpoint"], config["bucket"], connect_timeout=30)
            upload_attempted = True
            bucket.put_object_from_file(object_key, str(audio_path), headers={"Content-Type": "audio/wav"})
            signed_url = bucket.sign_url("GET", object_key, 3600)
        except Exception as exc:
            raise public_error("OSS 上传或签名", exc) from None
        payload, provider_task_id = submit_and_wait(config, signed_url, num_speakers, progress)
        lines, count, unknown = parse_result(payload)
        refined = None
        if refine_speakers:
            from speaker_refinement import refine
            try:
                refined = refine(payload, samples, num_speakers, progress)
            except Exception as exc:
                raise CloudASRError('独立说话人校正失败：' + type(exc).__name__ + '。请检查本地 WeSpeaker 模型和词级时间戳。') from None
    finally:
        if bucket is not None and upload_attempted:
            try:
                bucket.delete_object(object_key)
            except Exception:
                cleanup_warning = "OSS 临时文件删除失败，请检查 RAM DeleteObject 权限或生命周期规则。"
                progress("diarizing", cleanup_warning)
        audio_path.unlink(missing_ok=True)
    notices = []
    if unknown:
        notices.append(f"有 {unknown} 段未返回说话人编号，已标为未判定。")
    if num_speakers > 0 and count != num_speakers:
        notices.append(f"指定 {num_speakers} 人，阿里云实际标注 {count} 人；人数设置不能保证聚类准确。")
    if cleanup_warning:
        notices.append(cleanup_warning)
    result = {"segments": lines, "speaker_count": count, "speaker_notice": " ".join(notices),
            "language": "auto", "role_method": "aliyun", "engine": "aliyun",
            "provider_model": config["model"], "provider_task_id": provider_task_id}
    if refined is not None:
        result.update(refined)
        result['engine'] = 'aliyun-refined'
        if cleanup_warning:
            result['speaker_notice'] += ' ' + cleanup_warning
    return result
