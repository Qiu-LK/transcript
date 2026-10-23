"""Conservative fusion of visual active-speaker candidates with audio labels.

The visual detector only produces anonymous face IDs.  This module learns a
face-to-audio-speaker mapping from time ranges where the audio label is
already known, then uses that mapping to repair unknown or extremely short
audio lines.  It never overwrites a known audio label.
"""

from collections import defaultdict


def _overlap(left_start, left_end, right_start, right_end):
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _speaker_name(value):
    value = str(value or "")
    if "未判定" in value:
        return None
    if "说话人" in value:
        return value.split("说话人", 1)[1].strip() or None
    return value.strip() or None


def _face_mapping(segments, windows):
    scores = defaultdict(lambda: defaultdict(float))
    for window in windows:
        face_id = window.get("face_id")
        if face_id is None:
            continue
        ws, we = float(window.get("start", 0)), float(window.get("end", 0))
        confidence = float(window.get("confidence", 0) or 0)
        if confidence < 0.55:
            continue
        for segment in segments:
            speaker = _speaker_name(segment.get("speaker"))
            if speaker is None:
                continue
            if float(segment.get("end", 0)) - float(segment.get("start", 0)) < 0.7:
                continue
            overlap = _overlap(ws, we, float(segment.get("start", 0)),
                               float(segment.get("end", 0)))
            if overlap > 0:
                scores[face_id][speaker] += overlap * confidence
    mapping = {}
    for face_id, speaker_scores in scores.items():
        ranked = sorted(speaker_scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            continue
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(speaker_scores.values())
        # Require enough evidence and a meaningful margin.  This prevents a
        # face visible during a cutaway from being assigned arbitrarily.
        if best_score >= 0.45 and best_score >= max(0.12, second_score * 1.25) and best_score / max(total, 1e-6) >= 0.58:
            mapping[face_id] = {"speaker": best_name,
                                "confidence": round(min(1.0, best_score / max(total, 1e-6)), 3),
                                "score": round(best_score, 3)}
    return mapping


def _best_visual_speaker(segment, windows, mapping):
    start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
    candidates = defaultdict(float)
    for window in windows:
        face_id = window.get("face_id")
        mapped = mapping.get(face_id)
        if not mapped:
            continue
        overlap = _overlap(start, end, float(window.get("start", 0)),
                           float(window.get("end", 0)))
        if overlap <= 0:
            continue
        confidence = float(window.get("confidence", 0) or 0)
        candidates[mapped["speaker"]] += overlap * confidence * mapped["confidence"]
    if not candidates:
        return None
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * 1.2:
        return None
    return ranked[0][0]


def _merge_adjacent(segments):
    merged = []
    for segment in segments:
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        gap = float(segment.get("start", 0)) - float(previous.get("end", 0))
        if (previous.get("speaker") == segment.get("speaker") and gap <= 0.8
                and not str(previous.get("text", "")).rstrip().endswith(("。", "！", "？", ".", "!", "?"))):
            previous["text"] = str(previous.get("text", "")) + str(segment.get("text", ""))
            previous["end"] = segment.get("end", previous.get("end"))
        else:
            merged.append(dict(segment))
    return merged


def fuse(segments, visual_evidence):
    """Repair only unknown/very short labels using high-confidence visual evidence."""
    if not isinstance(visual_evidence, dict) or not visual_evidence.get("active_speaker_available"):
        return segments, {"applied": False, "reason": "没有可用的主动说话人视觉候选。"}
    if visual_evidence.get("method") != "LR-ASD-pretrain-AVA":
        return segments, {"applied": False, "reason": "嘴部运动估计不具备主动说话人证据，未修改说话人。"}
    windows = visual_evidence.get("active_windows") or []
    mapping = _face_mapping(segments, windows)
    if not mapping:
        return segments, {"applied": False, "reason": "无法建立人脸与音频说话人的可靠映射。"}
    repaired = []
    changed = 0
    for segment in segments:
        item = dict(segment)
        speaker = _speaker_name(item.get("speaker"))
        duration = float(item.get("end", 0)) - float(item.get("start", 0))
        if speaker is None or item.get("speaker_confidence") == "low":
            visual_speaker = _best_visual_speaker(item, windows, mapping)
            support = sum(_overlap(float(item.get("start", 0)), float(item.get("end", 0)),
                                   float(w.get("start", 0)), float(w.get("end", 0)))
                          for w in windows if mapping.get(w.get("face_id"), {}).get("speaker") == visual_speaker
                          and float(w.get("confidence", 0)) >= 0.8)
            if visual_speaker and support >= min(0.2, duration * 0.5):
                item["speaker"] = "说话人 " + visual_speaker
                changed += 1
        repaired.append(item)
    repaired = _merge_adjacent(repaired)
    return repaired, {"applied": changed > 0, "changed_segments": changed,
                      "mapped_faces": len(mapping),
                      "reason": (f"视觉证据修复了 {changed} 个低置信/未判定片段，"
                                 f"建立了 {len(mapping)} 条人脸-声纹映射。" if changed else
                                 f"已建立 {len(mapping)} 条人脸-声纹映射，未发现可安全修复的片段。")}
