"""Conservative visual evidence extraction for speaker attribution.

This is deliberately a *candidate* active-speaker detector, not a face
recognition system.  It tracks faces between sampled frames and measures
motion in the lower-face (mouth) region.  The result is useful for validating
whether a real ASD model can help a clip, but it is not allowed to overwrite
the audio speaker labels on its own.
"""
from pathlib import Path
import math


def _opencv_path(path):
    """Use an ASCII Windows short path when OpenCV cannot open Unicode paths."""
    path = str(path)
    try:
        import ctypes
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        size = get_short(path, None, 0)
        if size:
            buffer = ctypes.create_unicode_buffer(size + 1)
            if get_short(path, buffer, size + 1):
                return buffer.value
    except Exception:
        pass
    return path


def _track_faces(previous, detections, next_id, max_distance):
    """Greedily associate current detections with prior face centroids."""
    assigned = []
    used = set()
    for x, y, w, h in detections:
        cx, cy = x + w / 2, y + h / 2
        best = None
        best_dist = float("inf")
        for item in previous:
            if item["id"] in used:
                continue
            dist = math.hypot(cx - item["cx"], cy - item["cy"])
            limit = max_distance * max(1.0, (w + h) / 160.0)
            if dist < best_dist and dist <= limit:
                best, best_dist = item, dist
        if best is None:
            face_id = next_id
            next_id += 1
        else:
            face_id = best["id"]
            used.add(face_id)
        assigned.append({"id": face_id, "x": int(x), "y": int(y),
                         "w": int(w), "h": int(h), "cx": cx, "cy": cy})
    return assigned, next_id


def _mouth_motion(gray, face):
    """Return a normalized mouth-region frame for temporal motion scoring."""
    x, y, w, h = face["x"], face["y"], face["w"], face["h"]
    # Lower half of a face is a useful cheap proxy for lip motion.  Padding is
    # intentionally small so nearby faces do not contaminate the measurement.
    top = y + int(h * 0.48)
    bottom = y + int(h * 0.92)
    left = x + int(w * 0.18)
    right = x + int(w * 0.82)
    if bottom <= top or right <= left:
        return None
    crop = gray[top:bottom, left:right]
    if crop.size == 0:
        return None
    import cv2
    crop = cv2.resize(crop, (32, 20), interpolation=cv2.INTER_AREA)
    return crop


def _motion_score(previous, current):
    if previous is None or current is None:
        return 0.0
    import cv2
    delta = cv2.absdiff(previous, current)
    return float(delta.mean()) / 255.0


def _merge_active_windows(windows, max_gap=0.5):
    """Merge adjacent evidence windows for one stable face track."""
    merged = []
    for item in sorted(windows, key=lambda value: (value["face_id"], value["start"])):
        if merged and merged[-1]["face_id"] == item["face_id"] and item["start"] - merged[-1]["end"] <= max_gap:
            previous = merged[-1]
            previous["end"] = item["end"]
            previous["score"] = round(max(previous["score"], item["score"]), 4)
            previous["confidence"] = round(max(previous["confidence"], item["confidence"]), 3)
        else:
            merged.append(dict(item))
    return sorted(merged, key=lambda value: value["start"])


def inspect_video(path, max_frames=240):
    try:
        import cv2
    except ImportError:
        return {"enabled": False, "usable": False,
                "reason": "未安装 OpenCV，未启用画面证据。"}
    candidates = [
        Path(r"C:\Users\Public\haarcascade_frontalface_default.xml"),
        Path(__file__).resolve().parent / "models" / "visual" / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
    ]
    cascade_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if cascade_path is None:
        return {"enabled": False, "usable": False,
                "reason": "缺少人脸检测模型，未启用画面证据。"}
    capture = cv2.VideoCapture(_opencv_path(path))
    if not capture.isOpened():
        return {"enabled": True, "usable": False, "reason": "无法读取视频画面。"}
    detector = cv2.CascadeClassifier()
    loaded = False
    for candidate in candidates:
        if candidate.is_file() and detector.load(_opencv_path(candidate)):
            loaded = True
            break
    if not loaded or detector.empty():
        return {"enabled": True, "usable": False,
                "reason": "人脸检测模型无法加载，未启用画面证据。"}
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    duration = total / fps if total else 0.0
    # Four visual samples per second are enough for a first-pass mouth-motion
    # signal while keeping CPU use bounded on long videos.
    step = max(1, int(round(fps / 4.0)))
    if total and total // step > max_frames:
        step = max(step, int(math.ceil(total / max_frames)))
    indexes = range(0, total, step) if total else range(max_frames)
    frames = faces = 0
    face_counts = []
    previous_faces = []
    previous_mouth = {}
    next_id = 0
    tracks = {}
    active_windows = []
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(48, 48))
        current_faces, next_id = _track_faces(previous_faces, found, next_id, max_distance=90)
        count = len(current_faces)
        face_counts.append(count)
        if count:
            faces += 1
        frame_time = index / fps
        frame_motions = []
        for face in current_faces:
            face_id = face["id"]
            mouth = _mouth_motion(gray, face)
            score = _motion_score(previous_mouth.get(face_id), mouth)
            previous_mouth[face_id] = mouth
            frame_motions.append((face_id, score))
            track = tracks.setdefault(face_id, {"face_id": face_id,
                                                "first_seen": frame_time,
                                                "last_seen": frame_time,
                                                "samples": 0,
                                                "motion_sum": 0.0,
                                                "motion_peaks": 0})
            track["last_seen"] = frame_time
            track["samples"] += 1
            track["motion_sum"] += score
            if score >= 0.045:
                track["motion_peaks"] += 1
        if frame_motions:
            frame_motions.sort(key=lambda item: item[1], reverse=True)
            best_id, best_score = frame_motions[0]
            second_score = frame_motions[1][1] if len(frame_motions) > 1 else 0.0
            # A candidate is useful only when motion is clearly above the
            # noise floor and separated from another face's motion.
            confidence = max(0.0, min(1.0, (best_score - 0.012) / 0.08))
            if best_score >= 0.045 and (len(frame_motions) == 1 or best_score >= second_score * 1.15):
                active_windows.append({"start": round(frame_time, 3),
                                       "end": round(frame_time + step / fps, 3),
                                       "face_id": best_id,
                                       "score": round(best_score, 4),
                                       "confidence": round(confidence, 3)})
        previous_faces = current_faces
    capture.release()
    ratio = faces / frames if frames else 0.0
    usable = frames >= 3 and ratio >= 0.25
    for track in tracks.values():
        track["mean_motion"] = round(track["motion_sum"] / max(1, track["samples"]), 4)
        del track["motion_sum"]
    stable_ids = {
        track_id for track_id, track in tracks.items()
        if track["samples"] >= 3 and track["last_seen"] - track["first_seen"] >= 0.5
    }
    active_windows = [item for item in active_windows if item["face_id"] in stable_ids]
    active_windows = _merge_active_windows(active_windows, max_gap=max(0.5, 2.0 / fps))
    stable_tracks = [track for track in tracks.values() if track["face_id"] in stable_ids]
    # Report only compact, JSON-friendly evidence.  The full face boxes are
    # intentionally omitted because they are not needed by the current UI.
    return {"enabled": True, "usable": usable, "sampled_frames": frames,
            "face_frames": faces, "face_presence_ratio": round(ratio, 3),
            "max_faces": max(face_counts or [0]),
            "fps": round(fps, 3), "duration": round(duration, 3),
            "tracks": sorted(stable_tracks, key=lambda item: item["face_id"]),
            "active_windows": active_windows,
            "stable_track_count": len(stable_tracks),
            "active_window_count": len(active_windows),
            "active_speaker_available": bool(active_windows),
            "reason": (f"检测到 {len(stable_tracks)} 条稳定人脸轨迹和 {len(active_windows)} 个嘴部运动候选，"
                       "可参与低置信说话人片段的视觉/声纹融合。" if usable and active_windows else
                       "检测到人脸，但嘴部运动证据不足；不参与说话人判断。" if usable else
                       "人脸证据不足，不参与说话人判断。")}
