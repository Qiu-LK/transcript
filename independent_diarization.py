"""Independent audio segmentation used as a second opinion for speaker turns."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "speaker-diarization"


def inspect_audio(samples, speaker_count=-1):
    import sherpa_onnx
    segmentation = MODELS / "segmentation.onnx"
    embedding = MODELS / "embedding-eres2netv2.onnx"
    if not segmentation.is_file() or not embedding.is_file():
        return {"available": False, "reason": "缺少独立分段或多语言声纹模型。", "turns": []}
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
            num_threads=4),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speaker_count if speaker_count > 0 else -1,
            threshold=0.65, compute_confidence=True))
    result = sherpa_onnx.OfflineSpeakerDiarization(config).process(
        np.asarray(samples, dtype=np.float32))
    turns = [{"start": round(float(s.start), 3), "end": round(float(s.end), 3),
              "speaker_id": int(s.speaker), "confidence": round(float(s.confidence), 3)}
             for s in result.sort_by_start_time()]
    return {"available": True, "method": "pyannote-segmentation + eres2netv2",
            "speaker_count": int(result.num_speakers), "turns": turns}


def _overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def align_audio_labels(lines, evidence):
    """Map independent IDs globally; only fill unknown, well-supported turns."""
    turns = evidence.get("turns") or []
    if not evidence.get("available") or not turns:
        return lines, {"applied": False, "changed_segments": 0, "reason": "独立音频分段不可用。"}
    scores = {}
    for turn in turns:
        if turn["confidence"] < 0.45:
            continue
        ident = turn["speaker_id"]
        for line in lines:
            speaker = str(line.get("speaker") or "")
            if "未判定" in speaker:
                continue
            weight = _overlap(turn["start"], turn["end"], line["start"], line["end"])
            if weight > 0:
                scores[(ident, speaker)] = scores.get((ident, speaker), 0.0) + weight * turn["confidence"]
    mapping = {}
    for ident in {t["speaker_id"] for t in turns}:
        ranked = sorted(((weight, speaker) for (i, speaker), weight in scores.items() if i == ident), reverse=True)
        if not ranked:
            continue
        best, speaker = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else 0.0
        if best >= 1.5 and best >= runner * 2.0:
            mapping[ident] = speaker
    changed = 0
    output = []
    for line in lines:
        item = dict(line)
        if "未判定" in str(item.get("speaker")):
            ranked = sorted(((_overlap(item["start"], item["end"], t["start"], t["end"]), t)
                             for t in turns), key=lambda x: x[0], reverse=True)
            if ranked:
                overlap, turn = ranked[0]
                duration = max(0.01, item["end"] - item["start"])
                if (turn["speaker_id"] in mapping and turn["confidence"] >= 0.55
                        and overlap / duration >= 0.8 and
                        (len(ranked) == 1 or overlap >= ranked[1][0] * 2.0)):
                    item["speaker"] = mapping[turn["speaker_id"]]
                    changed += 1
        output.append(item)
    return output, {"applied": bool(changed), "changed_segments": changed,
                    "mapped_speakers": len(mapping),
                    "reason": f"独立音频分段修复了 {changed} 段未判定说话人。"}
