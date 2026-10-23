from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import gc
import os
import re
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent


def load_local_env():
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_local_env()
HOST, PORT = "127.0.0.1", int(os.getenv("WX_PORT", "8765"))
API_TEMPLATES = os.getenv("WX_PARSE_API", "https://api.bugpk.com/api/wxsph?url={url}")
MODEL = os.getenv("WX_WHISPER_MODEL", "medium")
MAX_MEDIA_BYTES = 500 * 1024 * 1024
DIARIZATION_DIR = ROOT / "models" / "speaker-diarization"
TASKS = {}
LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
WHISPER_MODEL = None
DIARIZERS = {}
DIARIZER_LOCK = threading.Lock()
ROLE_MODEL_LOCK = threading.Lock()
ROLE_MODEL = None
ROLE_MODEL_PATH = ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
ROLE_EMBEDDING_PATH = DIARIZATION_DIR / "embedding-zh-en-campplus.onnx"
ROLE_CORRECTION_ENABLED = os.getenv("WX_ROLE_CORRECTION", "0") == "1"
ALIYUN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
ALIYUN_ASR_MODEL = os.getenv("ALIYUN_ASR_MODEL", "paraformer-v2").strip() or "paraformer-v2"
ALIYUN_SPEAKER_DIARIZATION = os.getenv("ALIYUN_SPEAKER_DIARIZATION", "true").lower() in {"1", "true", "yes", "on"}
ALIYUN_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")


def send_json(handler, status, data):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def extract_share_url(value):
    match = re.search(r"https?://[^\s<>\"'，。！？)）]+", str(value or ""))
    if not match:
        raise ValueError("请粘贴视频号分享链接。")
    parsed = urlparse(match.group(0).rstrip(".,;:，。！？"))
    if parsed.scheme != "https" or parsed.hostname != "weixin.qq.com" or not re.fullmatch(r"/sph/[A-Za-z0-9_-]+/?", parsed.path):
        raise ValueError("目前只支持 https://weixin.qq.com/sph/... 分享链接。")
    return f"https://weixin.qq.com{parsed.path}"


def update(task_id, **values):
    with LOCK:
        TASKS[task_id].update(values)


def validate_media_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "finder.video.qq.com":
        raise RuntimeError("解析接口返回的媒体域名不符合预期，已拒绝下载。")
    return url


def parse_video(share_url):
    errors = []
    for template in API_TEMPLATES.split("|"):
        template = template.strip()
        if not template:
            continue
        if "{url}" not in template or urlparse(template).scheme != "https":
            errors.append("解析接口配置无效")
            continue
        # BugPk 在第一次请求新链接时可能返回 429/cache_build_in_progress，
        # 需要按照 retry_after 等待缓存构建完成后再取一次。
        for attempt in range(4):
            try:
                endpoint = template.format(url=quote(share_url, safe=""), name="wxsph")
                req = Request(endpoint, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                try:
                    with urlopen(req, timeout=30) as response:
                        payload = json.load(response)
                except HTTPError as http_error:
                    # 429 响应通常仍带有 JSON 的 retry_after；不要把它直接变成最终失败。
                    if http_error.code == 429:
                        try:
                            payload = json.loads(http_error.read().decode("utf-8", "replace"))
                        except Exception:
                            payload = {}
                        retry_after = (payload.get("data") or {}).get("retry_after") or 1
                        try:
                            delay = min(max(float(retry_after), 0.5), 8.0)
                        except (TypeError, ValueError):
                            delay = 1.0
                        if attempt < 3:
                            time.sleep(delay)
                            continue
                        raise RuntimeError("解析服务暂时限流，请稍后重试") from http_error
                    raise
                code = str(payload.get("code"))
                if code == "429":
                    details = payload.get("data") or {}
                    retry_after = details.get("retry_after") or 1
                    try:
                        delay = min(max(float(retry_after), 0.5), 8.0)
                    except (TypeError, ValueError):
                        delay = 1.0
                    if attempt < 3:
                        time.sleep(delay)
                        continue
                    raise RuntimeError("解析服务暂时限流，请稍后重试")
                if code not in {"200", "0", "1"}:
                    raise RuntimeError(str(payload.get("msg") or "解析失败")[:200])
                data = payload.get("data") or {}
                if isinstance(data, list):
                    data = data[0] if data else {}
                if not isinstance(data, dict):
                    raise RuntimeError("解析接口返回格式异常")
                media_url = data.get("url") or data.get("video") or data.get("video_url")
                if not media_url:
                    raise RuntimeError("解析接口没有返回视频地址")
                validate_media_url(media_url)
                author = data.get("author") or ""
                if isinstance(author, dict):
                    author = author.get("name") or author.get("nickname") or ""
                metadata = {
                    "title": data.get("title") or data.get("desc") or "视频号视频",
                    "description": data.get("desc") or "",
                    "author": author,
                    "cover_url": data.get("cover") or "",
                    "quality": data.get("quality") or "",
                    "expires_at": (data.get("extra") or {}).get("expire_time") or "",
                    "source_url": share_url,
                }
                return media_url, metadata
            except Exception as exc:
                timed_out = isinstance(exc, TimeoutError) or (
                    isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError)
                )
                server_error = isinstance(exc, HTTPError) and 500 <= exc.code < 600
                if attempt < 3 and (timed_out or server_error):
                    time.sleep(0.8)
                    continue
                if timed_out:
                    errors.append("第三方解析接口连接超时（重试后仍未响应）")
                else:
                    errors.append(str(exc)[:200])
                break
    raise RuntimeError("解析失败：" + "；".join(errors))


def download_media(url, destination):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://channels.weixin.qq.com/"})
    with urlopen(req, timeout=45) as response, destination.open("wb") as output:
        validate_media_url(response.url)
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > MAX_MEDIA_BYTES:
            raise RuntimeError("视频超过 500 MB，已停止下载。")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MEDIA_BYTES:
                raise RuntimeError("视频超过 500 MB，已停止下载。")
            output.write(chunk)
    with destination.open("rb") as media:
        head = media.read(12)
    if total < 1024 or head[4:8] != b"ftyp":
        raise RuntimeError("下载结果不是有效 MP4；视频地址可能已过期。")
    return total


def build_transcription_prompt(metadata):
    parts = []
    for label, key in (("视频标题", "title"), ("视频描述", "description")):
        value = re.sub(r"\s+", " ", str(metadata.get(key) or "")).strip()
        if value and value not in parts:
            parts.append(value)
    return " ".join(parts)[:240] or None


def transcribe(path, metadata):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("缺少语音识别组件。请运行 install.bat 安装依赖。") from exc
    model_dir = ROOT / "models"
    local_model = model_dir / f"faster-whisper-{MODEL}"
    if not (local_model / "model.bin").is_file() or not (local_model / "config.json").is_file():
        raise RuntimeError(f"缺少本地 {MODEL} 语音模型。请运行 install_model.bat 下载模型。")
    global WHISPER_MODEL
    with MODEL_LOCK:
        if WHISPER_MODEL is None:
            WHISPER_MODEL = WhisperModel(str(local_model), device="cpu", compute_type="int8", local_files_only=True)
        model = WHISPER_MODEL
    segments, info = model.transcribe(
        str(path),
        task="transcribe",
        initial_prompt=build_transcription_prompt(metadata),
        beam_size=5,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=True,
        word_timestamps=True,
    )
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        words = []
        for word in segment.words or []:
            if word.start is None or word.end is None or not word.word.strip():
                continue
            words.append({"start": float(word.start), "end": float(word.end), "text": word.word})
        lines.append({"start": float(segment.start), "end": float(segment.end), "text": text, "words": words})
    return lines, info.language


def decode_audio(path, sample_rate=16000):
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("缺少音频解码组件。请运行 install.bat 安装依赖。") from exc

    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError("视频没有可用于说话人识别的音轨。")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError("没有解码出有效音频。")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def diarize(path, num_speakers):
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError("缺少说话人识别组件。请运行 install.bat 安装依赖。") from exc

    segmentation_model = DIARIZATION_DIR / "segmentation.onnx"
    embedding_model = DIARIZATION_DIR / "embedding.onnx"
    if not segmentation_model.is_file() or not embedding_model.is_file():
        raise RuntimeError("缺少本地说话人识别模型。请运行 install_model.bat 下载模型。")

    threads = min(4, os.cpu_count() or 1)
    with DIARIZER_LOCK:
        diarizer = DIARIZERS.get(num_speakers)
        if diarizer is None:
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(segmentation_model)
                    ),
                    num_threads=threads,
                    provider="cpu",
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(embedding_model),
                    num_threads=threads,
                    provider="cpu",
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=num_speakers,
                    threshold=0.7,
                ),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not config.validate():
                raise RuntimeError("说话人识别模型配置无效。")
            diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
            DIARIZERS[num_speakers] = diarizer
        result = diarizer.process(decode_audio(path))
    turns = [
        {"speaker_id": int(segment.speaker), "start": float(segment.start), "end": float(segment.end)}
        for segment in result.sort_by_start_time()
    ]
    return turns, int(result.num_speakers)


def interval_overlap(start, end, other_start, other_end):
    return max(0.0, min(end, other_end) - max(start, other_start))


def speaker_for_interval(start, end, turns, max_gap=0.5):
    scores = {}
    for turn in turns:
        overlap = interval_overlap(start, end, turn["start"], turn["end"])
        if overlap:
            speaker_id = turn["speaker_id"]
            scores[speaker_id] = scores.get(speaker_id, 0.0) + overlap
    if scores:
        return max(scores, key=scores.get)
    if not turns:
        return None
    midpoint = (start + end) / 2
    nearest = min(turns, key=lambda turn: min(abs(midpoint - turn["start"]), abs(midpoint - turn["end"])))
    distance = min(abs(midpoint - nearest["start"]), abs(midpoint - nearest["end"]))
    return nearest["speaker_id"] if distance <= max_gap else None


def align_speakers(transcript_segments, turns):
    grouped = []
    for segment in transcript_segments:
        words = segment.get("words") or []
        units = words or [{"start": segment["start"], "end": segment["end"], "text": segment["text"]}]
        for unit in units:
            speaker_id = speaker_for_interval(unit["start"], unit["end"], turns)
            same_speaker = grouped and grouped[-1]["speaker_id"] == speaker_id
            gap = unit["start"] - grouped[-1]["end"] if same_speaker else 0
            previous_text = grouped[-1]["text"] if same_speaker else ""
            sentence_ended = bool(re.search(r"[。！？!?；;.]$", previous_text.rstrip()))
            too_long = same_speaker and (
                len(previous_text) >= 140 or unit["end"] - grouped[-1]["start"] >= 45
            )
            if same_speaker and gap <= 0.8 and not sentence_ended and not too_long:
                grouped[-1]["end"] = unit["end"]
                grouped[-1]["text"] += unit["text"]
            else:
                grouped.append({
                    "speaker_id": speaker_id,
                    "start": unit["start"],
                    "end": unit["end"],
                    "text": unit["text"],
                })

    label_by_id = {}
    for item in grouped:
        speaker_id = item.pop("speaker_id")
        if speaker_id is None:
            item["speaker"] = "说话人未判定"
        else:
            if speaker_id not in label_by_id:
                index = len(label_by_id)
                label_by_id[speaker_id] = chr(ord("A") + index) if index < 26 else str(index + 1)
            item["speaker"] = f"说话人 {label_by_id[speaker_id]}"
        item["start"] = round(item["start"], 1)
        item["end"] = round(item["end"], 1)
        item["text"] = item["text"].strip()
    return [item for item in grouped if item["text"]], len(label_by_id)


def normalize_role_text(value):
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or ""))


def parse_role_script(script, transcript_segments):
    """Map an LLM-formatted script back to the untouched ASR segments in order."""
    parsed = []
    for raw_line in str(script or "").splitlines():
        match = re.match(r"\s*([ABC])\s*[：:]\s*(.+?)\s*$", raw_line)
        if match:
            parsed.append((match.group(1), normalize_role_text(match.group(2))))
    labels = [None] * len(transcript_segments)
    source_index = 0
    for speaker, line_text in parsed:
        if not line_text:
            continue
        matched_any = False
        while source_index < len(transcript_segments):
            source_text = normalize_role_text(transcript_segments[source_index].get("text"))
            if source_text and source_text in line_text:
                labels[source_index] = speaker
                source_index += 1
                matched_any = True
                continue
            if matched_any:
                break
            # Allow the formatter to omit an uncertain fragment without shifting all later labels.
            later_match = next((
                offset for offset in range(source_index + 1, min(source_index + 4, len(transcript_segments)))
                if normalize_role_text(transcript_segments[offset].get("text")) in line_text
            ), None)
            if later_match is None:
                break
            source_index = later_match
    return labels


def parse_dialogue_candidates(value, segment_count):
    match = re.search(r"\{[\s\S]*\}", str(value or ""))
    if not match:
        return set()
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    candidates = set()
    for item in payload.get("dialogue", []):
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < segment_count:
            candidates.add(index)
    return candidates


def is_narration_fragment(text):
    compact = normalize_role_text(text)
    narration_markers = (
        "男主", "女主", "岳父", "岳母", "此时", "这时", "发现", "上前",
        "进屋", "饭桌上", "旁边的", "独自在", "招呼", "表现一下",
    )
    return any(marker in compact for marker in narration_markers)


def is_standalone_address(text):
    compact = normalize_role_text(text)
    address_words = {
        "叔叔", "叔叔好", "阿姨", "阿姨好", "哥哥", "姐姐", "爸爸", "妈妈",
        "爸", "妈", "爷爷", "奶奶", "外公", "外婆", "老师", "师傅", "先生",
    }
    return compact in address_words


def load_role_model():
    if not ROLE_MODEL_PATH.is_file():
        raise RuntimeError("缺少本地角色校正模型。请重新运行 install_model.bat。")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("缺少角色校正组件。请重新运行 install.bat。") from exc
    global ROLE_MODEL
    if ROLE_MODEL is None:
        ROLE_MODEL = Llama(
            model_path=str(ROLE_MODEL_PATH), n_ctx=4096,
            n_threads=min(6, os.cpu_count() or 1), verbose=False,
        )
    return ROLE_MODEL


def role_completion(model, system, user, max_tokens, json_mode=False):
    options = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    result = model.create_chat_completion(**options)
    return result["choices"][0]["message"]["content"]


def extract_segment_embeddings(path, transcript_segments):
    if not ROLE_EMBEDDING_PATH.is_file():
        return None
    try:
        import numpy as np
        import sherpa_onnx
    except ImportError:
        return None
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(ROLE_EMBEDDING_PATH), num_threads=min(4, os.cpu_count() or 1), provider="cpu"
    )
    if not config.validate():
        return None
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    audio = decode_audio(path)
    embeddings = []
    for segment in transcript_segments:
        start = max(0, int((float(segment["start"]) - 0.08) * 16000))
        end = min(len(audio), int((float(segment["end"]) + 0.08) * 16000))
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=audio[start:end])
        stream.input_finished()
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        embeddings.append(vector)
    return np.asarray(embeddings)


def complete_role_labels(labels, candidates, transcript_segments, path):
    """Promote filtered dialogue candidates and identify their role from known voice anchors."""
    candidates = {
        index for index in candidates
        if not is_narration_fragment(transcript_segments[index].get("text"))
    }
    unresolved = [index for index in candidates if labels[index] in (None, "A")]
    if not unresolved:
        return [label or "A" for label in labels]
    embeddings = extract_segment_embeddings(path, transcript_segments)
    if embeddings is None:
        return [label or "A" for label in labels]
    try:
        import numpy as np
    except ImportError:
        return [label or "A" for label in labels]
    centers = {}
    for speaker in ("B", "C"):
        indexes = [index for index, label in enumerate(labels) if label == speaker]
        if indexes:
            center = embeddings[indexes].mean(axis=0)
            centers[speaker] = center / (np.linalg.norm(center) + 1e-12)
    for index in unresolved:
        if centers:
            labels[index] = max(centers, key=lambda speaker: float(embeddings[index] @ centers[speaker]))
    return [label or "A" for label in labels]


def group_role_segments(transcript_segments, labels, max_gap=1.0):
    grouped = []
    for segment, label in zip(transcript_segments, labels):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if grouped and grouped[-1]["speaker"] == label and start - grouped[-1]["end"] <= max_gap:
            grouped[-1]["end"] = end
            grouped[-1]["text"] += text
        else:
            grouped.append({"speaker": label, "start": start, "end": end, "text": text})
    for item in grouped:
        item["speaker"] = f'说话人 {item["speaker"]}'
        item["start"] = round(item["start"], 1)
        item["end"] = round(item["end"], 1)
    return grouped


def correct_roles(path, transcript_segments):
    """Use story semantics for roles, then use voice embeddings only for missed short addresses."""
    source_text = "\n".join(str(segment.get("text") or "").strip() for segment in transcript_segments)
    script_system = (
        "你是短剧字幕编辑。把无标点、被切碎的ASR原文整理成带角色的逐句剧本。"
        "A固定表示贯穿全片、描述动作并吐槽画面的旁白/解说；B表示第一个开口的画面人物；"
        "C表示第二个开口的画面人物。旁白即使对人物说你、使用问句或模拟对话，仍标A。"
        "人物对现场另一人打招呼、称呼或交谈才标B/C。合并属于同一句的相邻碎片，"
        "不能改写、增删或调整原文先后。每行严格为标签、中文冒号、原文。不要说明。"
    )
    with ROLE_MODEL_LOCK:
        model = load_role_model()
        script = role_completion(
            model, script_system,
            "这段视频有3个角色。请整理以下按时间排列的ASR原文：\n" + source_text,
            1200,
        )
        labels = parse_role_script(script, transcript_segments)
        coverage = sum(label is not None for label in labels) / max(1, len(labels))
        if coverage < 0.8 or not any(label == "B" for label in labels) or not any(label == "C" for label in labels):
            raise RuntimeError("角色校正模型没有返回可可靠对齐的剧本。")
        candidates = {
            index for index, segment in enumerate(transcript_segments)
            if is_standalone_address(segment.get("text"))
        }
        labels = complete_role_labels(labels, candidates, transcript_segments, path)
    return group_role_segments(transcript_segments, labels), len(set(labels))


def process(task_id):
    with LOCK:
        source_url = TASKS[task_id]["source_url"]
    media_dir = ROOT / "media"
    media_dir.mkdir(exist_ok=True)
    media_path = media_dir / f"{task_id}.mp4"
    try:
        url, metadata = parse_video(source_url)
        update(task_id, status="downloading", metadata=metadata)
        size = download_media(url, media_path)
        update(task_id, status="transcribing", media_bytes=size)
        visual_dir = ROOT / "evals" / "visual_evidence"
        visual_dir.mkdir(parents=True, exist_ok=True)
        visual_evidence = {"enabled": False, "active_speaker_available": False,
                           "reason": "正在等待音频结果后运行 LR-ASD。"}
        # Keep the visual result as an artifact for inspection as well as
        # feeding it to the conservative fusion step below.
        try:
            visual_dir = ROOT / "evals" / "visual_evidence"
            visual_dir.mkdir(parents=True, exist_ok=True)
            (visual_dir / f"{task_id}.json").write_text(
                json.dumps(visual_evidence, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        update(task_id, visual_evidence=visual_evidence)
        with LOCK:
            selected_engine = TASKS[task_id].get("engine", "aliyun-refined" if ALIYUN_API_KEY else "local")
            requested_speakers = TASKS[task_id]["requested_speakers"]
        if selected_engine in {"aliyun", "aliyun-refined"}:
            import aliyun_asr
            result = aliyun_asr.transcribe(
                media_path, requested_speakers, decode_audio,
                lambda status, message: update(task_id, status=status, progress_message=message),
                refine_speakers=selected_engine == "aliyun-refined",
            )
            lines = result["segments"]
            audio_samples = None
            update(task_id, status="diarizing", progress_message="正在复核独立音频分段与说话人。")
            # Independent acoustic segmentation is a second opinion. Its
            # cluster IDs are not copied into the transcript because they may
            # over-split a short edited video.
            if selected_engine == "aliyun-refined":
              try:
                from independent_diarization import inspect_audio, align_audio_labels
                audio_samples = decode_audio(media_path)
                audio_evidence = inspect_audio(audio_samples, requested_speakers)
                lines, audio_fusion = align_audio_labels(lines, audio_evidence)
                (visual_dir / f"{task_id}.audio.json").write_text(
                    json.dumps(audio_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
              except Exception:
                audio_fusion = {"applied": False, "reason": "独立音频分段失败，保留原声纹结果。"}
            else:
                audio_fusion = {"applied": False, "reason": "阿里云原始模式未启用独立音频复核。"}
            update(task_id, status="diarizing", progress_message="正在运行 LR-ASD 音视频说话人检测。")
            try:
                from active_speaker import inspect
                if audio_samples is None:
                    audio_samples = decode_audio(media_path)
                uncertain = [line for line in lines if "未判定" in str(line.get("speaker"))
                             or line.get("speaker_confidence") == "low"]
                if uncertain:
                    intervals = [(max(0.0, float(line["start"]) - 4.0),
                                  float(line["end"]) + 4.0) for line in uncertain]
                    visual_evidence = inspect(media_path, audio_samples, intervals=intervals)
                else:
                    visual_evidence = {"enabled": True, "method": "LR-ASD-pretrain-AVA",
                                       "active_speaker_available": False, "active_windows": [],
                                       "reason": "没有低置信/未判定片段，本次无需视觉仲裁。"}
                (visual_dir / f"{task_id}.json").write_text(
                    json.dumps(visual_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
                update(task_id, visual_evidence=visual_evidence)
            except Exception:
                visual_evidence = {"enabled": False, "active_speaker_available": False,
                                   "reason": "LR-ASD 检测失败，保留音频结果。"}
            # Visual evidence is used only as a conservative second opinion:
            # reliable audio labels remain authoritative, while short or
            # unknown fragments may be repaired when face/audio mapping is
            # unambiguous.
            try:
                from visual_fusion import fuse
                lines, fusion = fuse(lines, visual_evidence)
            except Exception:
                fusion = {"applied": False, "reason": "视觉融合失败，保留音频结果。"}
            try:
                from text_refinement import review_with_time_limit
                update(task_id, status="diarizing", progress_message="正在准备限时文字复核。")
                lines, text_review = review_with_time_limit(
                    media_path, lines, result.get("language"),
                    lambda done, total: update(
                        task_id, progress_message=f"正在复核文字：{done}/{total} 段。"),
                )
            except Exception:
                text_review = {"enabled": False, "changed_segments": 0,
                               "reason": "本地文字复核失败，保留阿里云文本。"}
            if fusion.get("reason"):
                result["speaker_notice"] = (str(result.get("speaker_notice") or "") + " "
                                              + "画面融合：" + fusion["reason"]).strip()
            if audio_fusion.get("reason"):
                result["speaker_notice"] = (str(result.get("speaker_notice") or "") + " "
                                              + "音频复核：" + audio_fusion["reason"]).strip()
            if text_review.get("reason"):
                result["speaker_notice"] = (str(result.get("speaker_notice") or "") + " "
                                              + "文字复核：" + text_review["reason"]).strip()
            result["segments"] = lines
            transcript = "\n".join(f'{line["speaker"]}：{line["text"]}' for line in lines)
            (media_dir / f"{task_id}.txt").write_text(transcript, encoding="utf-8")
            update(task_id, status="complete", transcript=transcript, visual_fusion=fusion,
                   audio_fusion=audio_fusion,
                   text_review=text_review,
                   progress_message="阿里云文字稿已生成。", **result)
            return
        raw_lines, language = transcribe(media_path, metadata)
        with LOCK:
            requested_speakers = TASKS[task_id]["requested_speakers"]
        update(task_id, status="diarizing")
        speaker_notice = ""
        role_method = "voice"
        if requested_speakers == 3 and ROLE_CORRECTION_ENABLED and ROLE_MODEL_PATH.is_file():
            try:
                lines, speaker_count = correct_roles(media_path, raw_lines)
                role_method = "story-and-voice"
                speaker_notice = "已结合剧情语义与声纹校正三位角色；处理时间会比纯声纹模式更长。"
            except Exception as role_exc:
                turns, detected_speakers = diarize(media_path, requested_speakers)
                lines, labelled_speakers = align_speakers(raw_lines, turns)
                speaker_count = labelled_speakers or detected_speakers
                speaker_notice = f"剧情角色校正未完成，已回退到声纹模式：{role_exc}"
        else:
            turns, detected_speakers = diarize(media_path, requested_speakers)
            lines, labelled_speakers = align_speakers(raw_lines, turns)
            speaker_count = labelled_speakers or detected_speakers
            if requested_speakers > 0 and speaker_count < requested_speakers:
                speaker_notice = (
                    f"你选择了 {requested_speakers} 位说话人，音频中实际分离出 {speaker_count} 位。"
                    "人数设置是聚类目标，短句、多人重叠、背景音乐或声音相似都会影响结果。"
                )
        transcript = "\n".join(f'{line["speaker"]}：{line["text"]}' for line in lines)
        (media_dir / f"{task_id}.txt").write_text(transcript, encoding="utf-8")
        update(task_id, status="complete", segments=lines, transcript=transcript,
               language=language, speaker_count=speaker_count, speaker_notice=speaker_notice,
               role_method=role_method)
    except Exception as exc:
        update(task_id, status="error", error=str(exc))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            from text_refinement import TEXT_REVIEW_VERSION
            from lr_asd import available as asd_available
            return send_json(self, 200, {"ok": True, "version": "audio-visual-v4",
                                         "text_review_version": TEXT_REVIEW_VERSION,
                                         "lr_asd_available": asd_available(),
                                         "cloud_configured": bool(ALIYUN_API_KEY),
                                         "default_engine": "aliyun-refined" if ALIYUN_API_KEY else "local"})
        if parsed.path == "/api/task":
            task_id = (parse_qs(parsed.query).get("id") or [""])[0]
            with LOCK:
                task = TASKS.get(task_id)
                result = task.copy() if task else None
            return send_json(self, 200 if result else 404, result or {"error": "任务不存在"})
        if parsed.path.startswith("/api/transcript/"):
            name = parsed.path.removeprefix("/api/transcript/")
            if not re.fullmatch(r"[0-9a-f]{32}\.txt", name):
                return self.send_error(404)
            return self.file(ROOT / "media" / name, "text/plain; charset=utf-8", download=True)
        paths = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if parsed.path in paths:
            name, content_type = paths[parsed.path]
            return self.file(ROOT / name, content_type)
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/analyze":
            return send_json(self, 404, {"ok": False, "error": "接口不存在"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 16 * 1024:
                raise ValueError("请求内容过大或为空。")
            data = json.loads(self.rfile.read(length))
            source_url = extract_share_url(data.get("url"))
            requested_speakers = data.get("num_speakers", -1)
            if isinstance(requested_speakers, bool) or not isinstance(requested_speakers, int):
                raise ValueError("说话人数格式无效。")
            if requested_speakers != -1 and not 1 <= requested_speakers <= 8:
                raise ValueError("说话人数必须选择自动或 1 至 8 人。")
            engine = data.get("engine", "aliyun-refined" if ALIYUN_API_KEY else "local")
            if engine not in {"aliyun-refined", "aliyun", "local"}:
                raise ValueError("请选择阿里云或本地引擎。")
            if engine in {"aliyun", "aliyun-refined"}:
                import aliyun_asr
                aliyun_asr.settings()
            task_id = uuid.uuid4().hex
            with LOCK:
                TASKS[task_id] = {"task_id": task_id, "source_url": source_url,
                                  "status": "parsing", "metadata": {}, "media_bytes": 0,
                                  "segments": [], "transcript": "", "error": "", "created_at": time.time(),
                                  "requested_speakers": requested_speakers, "speaker_count": 0,
                                  "engine": engine, "progress_message": ""}
            threading.Thread(target=process, args=(task_id,), daemon=True).start()
            return send_json(self, 202, {"ok": True, "task_id": task_id})
        except (ValueError, json.JSONDecodeError) as exc:
            return send_json(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            return send_json(self, 500, {"ok": False, "error": f"创建任务失败：{exc}"})

    def file(self, path, content_type, download=False):
        if not path.is_file():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"视频号文字稿提取器：http://{HOST}:{PORT}")
    print("按 Ctrl+C 停止服务")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
