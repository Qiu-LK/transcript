"""Sample continuous face tracks and score audio-visual speech with LR-ASD."""
from pathlib import Path
import math
import numpy as np

from visual_evidence import _opencv_path


def _crop(gray, box):
    import cv2
    x, y, w, h = box
    side = max(w, h) * 1.25
    cx, cy = x + w / 2, y + h / 2
    left, top = int(cx - side / 2), int(cy - side / 2)
    right, bottom = int(cx + side / 2), int(cy + side / 2)
    pad = int(side) + 3
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=110)
    cut = padded[top+pad:bottom+pad, left+pad:right+pad]
    if cut.size == 0:
        return None
    face = cv2.resize(cut, (224, 224), interpolation=cv2.INTER_AREA)
    return face[56:168, 56:168]


def _match(previous, box):
    x, y, w, h = box
    best, distance = None, float("inf")
    for ident, old in previous.items():
        ox, oy, ow, oh = old
        d = math.hypot(x + w/2 - ox - ow/2, y + h/2 - oy - oh/2)
        if d < distance and d < max(w, h) * 0.65:
            best, distance = ident, d
    return best


def inspect(path, samples, max_seconds=240, intervals=None):
    import cv2
    import lr_asd
    if not lr_asd.available():
        return {"enabled": False, "active_speaker_available": False,
                "reason": "LR-ASD 或其 CPU 依赖未安装。"}
    capture = cv2.VideoCapture(_opencv_path(path))
    if not capture.isOpened():
        return {"enabled": True, "active_speaker_available": False,
                "reason": "LR-ASD 无法读取视频画面。"}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25)
    duration = min(len(samples) / 16000, max_seconds)
    cascade_candidates = [Path(r"C:\Users\Public\haarcascade_frontalface_default.xml"),
                          Path(__file__).resolve().parent / "models" / "visual" /
                          "haarcascade_frontalface_default.xml",
                          Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"]
    cascade_path = next((p for p in cascade_candidates if p.is_file()), None)
    if cascade_path is None:
        capture.release()
        return {"enabled": True, "active_speaker_available": False,
                "reason": "LR-ASD 缺少人脸检测模型。"}
    cascade = cv2.CascadeClassifier(_opencv_path(cascade_path))
    if cascade.empty():
        capture.release()
        return {"enabled": True, "active_speaker_available": False,
                "reason": "LR-ASD 人脸检测模型无法加载。"}
    tracks = {}
    previous = {}
    next_id = 0
    count = int(duration * 25)
    for i in range(count):
        current_time = i / 25.0
        if intervals is not None and not any(left <= current_time <= right for left, right in intervals):
            previous = {}
            continue
        capture.set(cv2.CAP_PROP_POS_MSEC, i * 40.0)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=.5, fy=.5)
        detected = cascade.detectMultiScale(small, scaleFactor=1.12, minNeighbors=5,
                                            minSize=(28, 28))
        current = {}
        for sx, sy, sw, sh in detected:
            box = (int(sx*2), int(sy*2), int(sw*2), int(sh*2))
            ident = _match(previous, box)
            if ident is None or ident in current:
                ident = next_id
                next_id += 1
            face = _crop(gray, box)
            if face is not None:
                tracks.setdefault(ident, []).append((i, face))
                current[ident] = box
        previous = current
    capture.release()
    windows = []
    scored_tracks = 0
    for ident, frames in tracks.items():
        runs = []
        current = []
        for item in frames:
            if current and item[0] - current[-1][0] > 2:
                runs.append(current)
                current = []
            current.append(item)
        if current:
            runs.append(current)
        for run in runs:
            if len(run) < 10:
                continue
            start, end = run[0][0], run[-1][0] + 1
            by_frame = {i: face for i, face in run}
            ordered = []
            last = run[0][1]
            for i in range(start, end):
                last = by_frame.get(i, last)
                ordered.append(last)
            audio = samples[int(start * 640):int(end * 640)]
            if len(audio) < 16000 * .4:
                continue
            logits = lr_asd.score(audio, ordered)
            if not len(logits):
                continue
            scored_tracks += 1
            for j, logit in enumerate(logits):
                if logit <= 0.5:
                    continue
                confidence = float(1 / (1 + np.exp(-float(logit))))
                windows.append({"start": round((start+j)/25, 3),
                                "end": round((start+j+1)/25, 3),
                                "face_id": ident, "confidence": round(confidence, 3),
                                "score": round(float(logit), 3)})
    from visual_evidence import _merge_active_windows
    windows = _merge_active_windows(windows, max_gap=0.08)
    return {"enabled": True, "method": "LR-ASD-pretrain-AVA",
            "active_speaker_available": bool(windows), "active_windows": windows,
            "track_count": scored_tracks,
            "targeted": intervals is not None,
            "reason": f"LR-ASD 检查了 {scored_tracks} 段连续人脸轨迹，找到 {len(windows)} 个主动说话候选。"}
