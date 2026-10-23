"""Optional local ASR review for obvious cloud transcription mistakes."""

from difflib import SequenceMatcher
from pathlib import Path
import os
import re
import multiprocessing as mp
import queue
import time


ROOT = Path(__file__).resolve().parent


def _compact(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _review_enabled():
    return os.getenv("WX_LOCAL_TEXT_REVIEW", "1").lower() in {"1", "true", "yes", "on"}


TEXT_REVIEW_VERSION = "bounded-opening-v4"


def _review_candidates(lines, language):
    """Only substantial Chinese lines can be changed by _repair_opening."""
    if language != "zh":
        return []
    limit = max(0, min(30, int(os.getenv("WX_TEXT_REVIEW_MAX_CLIPS", "10"))))
    return [i for i, line in enumerate(lines)
            if float(line.get("end", 0)) - float(line.get("start", 0)) >= 1.5
            and len(_compact(line.get("text"))) >= 10][:limit]


def _local_segments(path, language=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, "本地 Whisper 组件未安装。"
    model_name = os.getenv("WX_TEXT_REVIEW_MODEL", "medium").strip() or "medium"
    model_dir = ROOT / "models" / f"faster-whisper-{model_name}"
    if not (model_dir / "model.bin").is_file():
        return None, f"本地文字复核模型不存在：{model_name}。"
    try:
        model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)
        segments, _ = model.transcribe(
            str(path),
            task="transcribe",
            language=language if language in {"zh", "en", "ja", "ko"} else None,
            beam_size=5,
            temperature=0,
            vad_filter=True,
            # Do not let a previous hallucinated phrase bias the next clip.
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        rows = []
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            rows.append({"start": float(segment.start), "end": float(segment.end),
                         "text": text, "avg_logprob": float(getattr(segment, "avg_logprob", -9.0))})
        return rows, None
    except Exception as exc:
        return None, f"本地文字复核失败：{type(exc).__name__}。"


def _decode_audio(path):
    import av
    import numpy as np
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.zeros(0, dtype=np.float32)


def _local_clip_texts(path, lines, language=None, progress=None):
    """Transcribe each target line in a padded short clip with word times."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {}, "本地 Whisper 组件未安装。"
    model_name = os.getenv("WX_TEXT_REVIEW_MODEL", "medium").strip() or "medium"
    model_dir = ROOT / "models" / f"faster-whisper-{model_name}"
    if not (model_dir / "model.bin").is_file():
        return {}, f"本地文字复核模型不存在：{model_name}。"
    try:
        model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)
        audio = _decode_audio(path)
        if not len(audio):
            return {}, "视频没有可复核的音轨。"
        output = {}
        candidates = _review_candidates(lines, language)
        for position, index in enumerate(candidates, 1):
            line = lines[index]
            start, end = float(line.get("start", 0)), float(line.get("end", 0))
            if end <= start:
                continue
            pad = 0.35
            clip_start = max(0.0, start - pad)
            clip_end = min(len(audio) / 16000.0, end + pad)
            clip = audio[int(clip_start * 16000):int(clip_end * 16000)]
            if len(clip) < 1600:
                continue
            segments, _ = model.transcribe(
                clip,
                task="transcribe",
                language=language if language in {"zh", "en", "ja", "ko"} else None,
                beam_size=2,
                temperature=0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt="中文对白。" if language == "zh" else None,
                word_timestamps=True,
            )
            words = []
            logprobs = []
            for segment in segments:
                logprobs.append(float(getattr(segment, "avg_logprob", -9.0)))
                for word in segment.words or []:
                    if word.start is None or word.end is None:
                        continue
                    absolute_start = clip_start + float(word.start)
                    absolute_end = clip_start + float(word.end)
                    midpoint = (absolute_start + absolute_end) / 2
                    if start - 0.08 <= midpoint <= end + 0.08 and str(word.word).strip():
                        words.append(str(word.word).strip())
            if words and logprobs and max(logprobs) >= -0.85:
                row = {"text": "".join(words), "avg_logprob": max(logprobs)}
                output[index] = row
            if progress:
                progress(position, len(candidates))
        return output, None
    except Exception as exc:
        return {}, f"本地短窗口复核失败：{type(exc).__name__}。"


def _repair_opening(cloud_text, prefix_text):
    """Replace only a disputed opening before a shared three-character anchor."""
    cloud = _compact(cloud_text)
    prefix = _compact(prefix_text)
    if not cloud or not prefix:
        return None
    # Proper names and very short questions have too little redundant speech
    # evidence for an automatic spelling change.
    if len(cloud) < 10:
        return None
    matches = SequenceMatcher(None, cloud[:12], prefix[:12], autojunk=False).get_matching_blocks()
    anchors = [m for m in matches if m.size >= 3 and 2 <= m.a <= 3
               and 2 <= m.b <= 3 and cloud[:m.a] != prefix[:m.b]]
    if not anchors:
        return None
    anchor = max(anchors, key=lambda m: m.size)
    indices = [i for i, char in enumerate(str(cloud_text)) if _compact(char)]
    if anchor.a >= len(indices):
        return None
    opening = prefix[:anchor.b]
    if len(opening) > 3 or not opening:
        return None
    if SequenceMatcher(None, cloud[:anchor.a], opening, autojunk=False).ratio() >= 0.4:
        return None
    return opening + "，" + str(cloud_text)[indices[anchor.a]:]


def review_with_local(path, lines, language=None, progress=None):
    """Review cloud text by time-aligned local ASR, preserving speaker labels.

    A local line must have reasonable log probability and overlap the target
    line. Known speaker labels and timestamps are never changed here.
    """
    if not _review_enabled() or not lines:
        return lines, {"enabled": False, "changed_segments": 0, "reason": "本地文字复核未启用。"}
    candidates = _review_candidates(lines, language)
    if not candidates:
        return lines, {"enabled": True, "changed_segments": 0,
                       "reason": "无符合当前规则的中文长句，已跳过本地文字复核。"}
    if progress:
        progress(0, len(candidates))
    # Keep the original three-argument call so existing integrations remain compatible.
    local, error = _local_clip_texts(path, lines, language, progress) if progress else _local_clip_texts(path, lines, language)
    if not local:
        return lines, {"enabled": True, "changed_segments": 0,
                       "reason": error or f"已检查 {len(candidates)} 段，未找到可安全替换的文字。"}
    reviewed = []
    changed = 0
    for index, line in enumerate(lines):
        item = dict(line)
        row = local.get(index)
        if row:
            local_text = row["text"].strip()
            similarity = SequenceMatcher(None, _compact(item.get("text")), _compact(local_text),
                                         autojunk=False).ratio()
            opening = _repair_opening(item.get("text", ""),
                                      row.get("prefix_text") or local_text)
            if opening and similarity < 0.9:
                item["text"] = opening
                changed += 1
                reviewed.append(item)
                continue
            # A single Whisper pass can hallucinate on a short reply or name.
            # Keep the cloud text when the two recognizers disagree elsewhere;
            # a shared anchor is required before changing words automatically.
        reviewed.append(item)
    return reviewed, {"enabled": True, "changed_segments": changed,
                       "reason": f"本地 Whisper 检查了 {len(candidates)} 段，替换了 {changed} 段明显不一致的文字。"}


def _review_worker(path, lines, language, messages):
    def report(done, total):
        messages.put(("progress", done, total))
    try:
        messages.put(("result", *review_with_local(path, lines, language, report)))
    except Exception as exc:
        messages.put(("error", f"{type(exc).__name__}: {exc}"))


def review_with_time_limit(path, lines, language=None, progress=None):
    """Run CPU review separately so the server can enforce a real wall-clock limit."""
    if not _review_enabled() or not _review_candidates(lines, language):
        return review_with_local(path, lines, language)
    seconds = max(10, min(600, int(os.getenv("WX_TEXT_REVIEW_TIMEOUT", "120"))))
    context = mp.get_context("spawn")
    messages = context.Queue()
    worker = context.Process(target=_review_worker, args=(str(path), lines, language, messages))
    worker.start()
    deadline = time.monotonic() + seconds
    result = None
    try:
        while time.monotonic() < deadline:
            try:
                message = messages.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if message[0] == "progress":
                if progress:
                    progress(message[1], message[2])
            elif message[0] == "result":
                result = message[1], message[2]
                break
            elif message[0] == "error":
                break
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=5)
        messages.close()
    if result:
        return result
    return lines, {"enabled": True, "changed_segments": 0,
                   "reason": f"本地文字复核超时或提前退出（上限 {seconds} 秒），保留阿里云原文。"}
