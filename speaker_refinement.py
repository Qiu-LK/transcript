"""Independent acoustic speaker assignment over timestamped ASR words.

Does not use transcript semantics or sample-specific reference labels.
"""
from pathlib import Path
import re
import os
import threading
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'speaker-diarization'
EXTRACTORS = {}
LOCK = threading.Lock()


def words_from_payload(payload):
    words = []
    sentence_number = 0
    for transcript in payload.get('transcripts', []):
        for sentence in transcript.get('sentences', []):
            source = sentence.get('words') or []
            if not source:
                raise RuntimeError('云端没有返回词级时间戳，无法进行独立说话人校正。')
            sentence_words = []
            for item in source:
                text = str(item.get('text') or '')
                punctuation = str(item.get('punctuation') or '')
                if punctuation and not text.endswith(punctuation):
                    text += punctuation
                if text.strip():
                    sentence_words.append({'start': float(item['begin_time']) / 1000,
                                           'end': float(item['end_time']) / 1000,
                                           'text': text, 'sentence_number': sentence_number})
            if sentence_words:
                sentence_words[-1]['sentence_end'] = True
                words.extend(sentence_words)
            sentence_number += 1
    words.sort(key=lambda word: word['start'])
    for left, right in zip(words, words[1:]):
        if (left['text'].rstrip().endswith('.') and re.match(r'^[a-z]', right['text'].strip())
                and right['start'] - left['end'] <= 0.12):
            left['text'] = left['text'].rstrip().removesuffix('.').rstrip()
    return words


def language_for(words):
    text = ''.join(word['text'] for word in words)
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    latin = len(re.findall(r'[A-Za-z]', text))
    return 'zh' if chinese > latin / 2 else 'en'


def make_units(words):
    """Create bounded acoustic turns while preserving cloud sentence boundaries."""
    units = []
    current = []
    max_duration = 1.65
    min_words_before_split = 3
    for index, word in enumerate(words):
        if current:
            gap = word['start'] - words[current[-1]]['end']
            duration = words[current[-1]]['end'] - words[current[0]]['start']
            terminal = bool(re.search(r'[。！？!?;；.]$', words[current[-1]]['text'].rstrip()))
            # 云端 sentence 边界通常来自声学/语言模型的停顿判断，即使没有标点
            # 也应保留；否则快速接话会被拼进同一个声纹窗口。
            sentence_boundary = bool(words[current[-1]].get('sentence_end'))
            soft_window = duration >= max_duration and len(current) >= min_words_before_split
            if gap >= 0.22 or terminal or sentence_boundary or soft_window:
                units.append(current)
                current = []
        current.append(index)
    if current:
        units.append(current)
    return units


def make_sliding_units(words, window=1.2, step=0.42):
    """Build overlapping acoustic windows independent of cloud sentences.

    Windows overlap on purpose: a short reply at a sentence boundary can then
    borrow acoustic evidence from both sides instead of being represented by a
    single, often sub-second embedding.
    """
    if not words:
        return []
    first = float(words[0]['start'])
    last = float(words[-1]['end'])
    units = []
    anchor = first
    while anchor < last:
        end = anchor + window
        members = [index for index, word in enumerate(words)
                   if float(word['end']) > anchor and float(word['start']) < end]
        if members:
            start = float(words[members[0]]['start'])
            finish = float(words[members[-1]]['end'])
            if finish - start >= 0.42:
                units.append(members)
        anchor += step
    # Remove exact duplicates caused by sparse ASR timestamps.
    unique = []
    seen = set()
    for unit in units:
        key = tuple(unit)
        if key not in seen:
            seen.add(key)
            unique.append(unit)
    return unique


def vote_overlapping_labels(words, units, labels, durations):
    """Project window labels back to words using duration-weighted voting."""
    votes = [dict() for _ in words]
    for unit, label, duration in zip(units, labels, durations):
        if label is None:
            continue
        for word_index in unit:
            word = words[word_index]
            overlap = max(0.0, min(float(word['end']),
                                    float(words[unit[-1]]['end']))
                          - max(float(word['start']),
                                float(words[unit[0]]['start'])))
            if overlap <= 0:
                overlap = min(float(word['end']) - float(word['start']), 0.15)
            votes[word_index][label] = votes[word_index].get(label, 0.0) + overlap * min(duration, 1.8)
    projected = []
    for score in votes:
        if not score:
            projected.append(None)
            continue
        ranked = sorted(score.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * 1.12:
            projected.append(None)
        else:
            projected.append(ranked[0][0])
    return projected


def cluster_vectors(vectors, durations, num_speakers):
    """Duration-weighted average-link clustering; count is a hint, not invented speakers."""
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    similarities = vectors @ vectors.T
    clusters = [[i] for i in range(len(vectors))]
    target = num_speakers if num_speakers > 0 else 0
    while len(clusters) > target:
        best, pair = -2.0, None
        for i in range(len(clusters)):
            for j in range(i):
                left, right = clusters[i], clusters[j]
                weights = np.outer(np.minimum(np.asarray(durations)[left], 2.0),
                                   np.minimum(np.asarray(durations)[right], 2.0))
                score = float(np.average(similarities[np.ix_(left, right)], weights=weights))
                if score > best:
                    best, pair = score, (i, j)
        if pair is None or (num_speakers < 0 and best < 0.40):
            break
        i, j = pair
        clusters[j].extend(clusters[i])
        del clusters[i]
    labels = [None] * len(vectors)
    centers = []
    for label, members in enumerate(clusters):
        weights = np.minimum(np.asarray(durations)[members], 2.0)
        center = np.average(vectors[members], axis=0, weights=weights)
        center /= max(np.linalg.norm(center), 1e-8)
        centers.append(center)
        for index in members:
            labels[index] = label
    return labels, np.asarray(centers)


def _collapse_weak_auto_clusters(labels, centers, durations):
    """Merge tiny automatic clusters into the dominant voice."""
    labels = list(labels)
    durations = np.asarray(durations, dtype=np.float32)
    cluster_ids = sorted(set(labels))
    if len(cluster_ids) <= 1:
        return labels, centers
    mass = {cluster: float(sum(durations[i] for i, value in enumerate(labels) if value == cluster))
            for cluster in cluster_ids}
    counts = {cluster: sum(value == cluster for value in labels) for cluster in cluster_ids}
    dominant = max(cluster_ids, key=mass.get)
    total = float(durations.sum())
    weak_limit = min(0.95, max(0.55, total * 0.018))
    remap = {cluster: cluster for cluster in cluster_ids}
    for cluster in cluster_ids:
        if cluster == dominant or mass[cluster] >= weak_limit:
            continue
        similarity = float(centers[cluster] @ centers[dominant])
        if counts[cluster] == 1 or similarity >= 0.18:
            remap[cluster] = dominant
    labels = [remap[value] for value in labels]
    kept = sorted(set(labels))
    renumber = {value: index for index, value in enumerate(kept)}
    return [renumber[value] for value in labels], np.asarray([centers[value] for value in kept])


def reconcile_vectors(primary, durations, num_speakers, auxiliary=None):
    """Build centers from longer speech; short replies cannot move the centers.

    Absolute cosine values are model-specific diagnostics, not probabilities.
    A second embedding model can arbitrate a weak primary match against the
    same independently established clusters, without trusting cloud labels.
    """
    vectors = np.asarray(primary, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    durations = np.asarray(durations)
    # Include medium-length replies as anchor candidates, so a speaker who only
    # has short turns is not necessarily absent from the initial clusters.
    anchors = np.flatnonzero(durations >= 0.45)
    minimum = max(2, num_speakers) if num_speakers > 0 else 2
    if len(anchors) < minimum:
        anchors = np.arange(len(vectors))
    anchor_labels, centers = cluster_vectors(vectors[anchors], durations[anchors], num_speakers)
    if num_speakers < 0:
        anchor_labels, centers = _collapse_weak_auto_clusters(anchor_labels, centers, durations[anchors])
    auxiliary_centers = None
    if auxiliary is not None:
        auxiliary = np.asarray(auxiliary, dtype=np.float32)
        auxiliary /= np.maximum(np.linalg.norm(auxiliary, axis=1, keepdims=True), 1e-8)
        auxiliary_centers = []
        for label in range(len(centers)):
            members = anchors[np.asarray(anchor_labels) == label]
            center = np.average(auxiliary[members], axis=0, weights=durations[members])
            auxiliary_centers.append(center / max(np.linalg.norm(center), 1e-8))
        auxiliary_centers = np.asarray(auxiliary_centers)
    assignments = []
    score_rows = []
    auxiliary_wins = set()
    if len(centers):
        masses = [sum(durations[anchors][i] for i, value in enumerate(anchor_labels) if value == label)
                  for label in range(len(centers))]
        dominant = int(np.argmax(masses))
    else:
        dominant = 0
    for index, vector in enumerate(vectors):
        scores = vector @ centers.T
        score_rows.append(scores)
        order = np.argsort(scores)
        label = int(order[-1])
        margin = float(scores[order[-1]] - scores[order[-2]]) if len(order) > 1 else 1.0
        best = float(scores[label])
        if auxiliary_centers is not None and best < 0.65 and margin < 0.25:
            other = auxiliary[index] @ auxiliary_centers.T
            ranked = np.argsort(other)
            other_margin = float(other[ranked[-1]] - other[ranked[-2]]) if len(ranked) > 1 else 1.0
            if other[ranked[-1]] >= 0.78 and other_margin >= 0.07:
                label = int(ranked[-1])
                auxiliary_wins.add(index)
            elif best < 0.5 or margin < 0.15:
                label = None
        elif best < 0.45 or (best < 0.65 and margin < 0.08):
            label = None
        assignments.append(label)
    # 用高置信度的非锚点片段补充中心，再做一次匹配。这样短句角色只要
    # 有几段相互一致的语音，就不会永远被最近的主说话人吞掉。
    if len(centers) > 1:
        updated = []
        for label in range(len(centers)):
            members = [i for i, value in enumerate(assignments)
                       if value == label and (durations[i] >= 0.45 or
                                              float(vectors[i] @ centers[label]) >= 0.72)]
            if members:
                center = np.average(vectors[members], axis=0,
                                    weights=np.minimum(durations[members], 2.0))
                updated.append(center / max(np.linalg.norm(center), 1e-8))
            else:
                updated.append(centers[label])
        centers = np.asarray(updated)
        first_assignments = assignments
        assignments = []
        score_rows = []
        for index, vector in enumerate(vectors):
            if index in auxiliary_wins:
                assignments.append(first_assignments[index])
                continue
            scores = vector @ centers.T
            score_rows.append(scores)
            order = np.argsort(scores)
            label = int(order[-1])
            margin = float(scores[order[-1]] - scores[order[-2]]) if len(order) > 1 else 1.0
            # 只有极短且没有任何明显归属证据的片段才保留未判定。
            if (float(scores[label]) < 0.45 or
                    (float(scores[label]) < 0.65 and margin < 0.08)):
                label = None
            assignments.append(label)
    return smooth_short_assignments(assignments, durations, score_rows)


def smooth_short_assignments(assignments, durations, scores):
    """Use nearby long turns only when they agree on a short ambiguous turn."""
    labels = list(assignments)
    durations = np.asarray(durations, dtype=np.float32)
    for index, label in enumerate(labels):
        if durations[index] >= 0.72:
            continue
        previous = next((labels[j] for j in range(index - 1, max(-1, index - 3), -1)
                         if labels[j] is not None and durations[j] >= 0.72), None)
        following = next((labels[j] for j in range(index + 1, min(len(labels), index + 3))
                          if labels[j] is not None and durations[j] >= 0.72), None)
        if label is None and previous is not None and previous == following:
            labels[index] = previous
    return labels


def render_words(words, labels):
    """Join English word spacing, end rows on genuine speaker or sentence boundaries."""
    lines, names = [], {}
    for word, label in zip(words, labels):
        if label is None:
            speaker = '说话人未判定'
        else:
            if label not in names:
                names[label] = chr(65 + len(names))
            speaker = '说话人 ' + names[label]
        gap = word['start'] - lines[-1]['end'] if lines else 99
        if (lines and lines[-1]['speaker'] == speaker and gap <= 0.65
                and not re.search(r'[。！？!?;；.]$', lines[-1]['text'].rstrip())
                and word['end'] - lines[-1]['start'] < 20):
            previous = lines[-1]['text']
            text = word['text'].strip()
            space = ' ' if re.search(r'[A-Za-z0-9,]$', previous) and re.match(r'[A-Za-z0-9]', text) else ''
            lines[-1]['text'] += space + text
            lines[-1]['end'] = round(word['end'], 2)
        else:
            lines.append({'start': round(word['start'], 2), 'end': round(word['end'], 2),
                          'text': word['text'].strip(), 'speaker': speaker})
    return lines, len(names)


def fill_isolated_unknown(labels):
    """Fill only an unknown unit bracketed by the same confident speaker.

    This repairs a split word or very short acoustic window without turning
    every uncertain segment into the dominant speaker.
    """
    labels = list(labels)
    for index in range(1, len(labels) - 1):
        if labels[index] is None and labels[index - 1] is not None and labels[index - 1] == labels[index + 1]:
            labels[index] = labels[index - 1]
    return labels


def attach_tiny_unknown_suffixes(words, labels):
    """Keep an unvoiced final Han character with its immediately preceding turn.

    A 0.1-second suffix cannot yield a trustworthy speaker embedding.  Only
    attach it when ASR places it directly after a labelled Han word in the
    same cloud sentence. A terminal punctuation mark followed by a pause also
    permits a cloud sentence boundary inside a word; never cross a pause at
    the actual speaker boundary.
    """
    labels = list(labels)
    for index in range(1, len(words)):
        if labels[index] is not None or labels[index - 1] is None:
            continue
        previous, current = words[index - 1], words[index]
        text = current['text'].strip()
        same_sentence = current.get('sentence_number') == previous.get('sentence_number')
        terminal_with_pause = (text.endswith(('，', '。', '！', '？'))
                               and (index + 1 == len(words)
                                    or words[index + 1]['start'] - current['end'] >= 0.4))
        if ((same_sentence or terminal_with_pause)
                and re.fullmatch(r'[\u4e00-\u9fff][，、]?', text)
                and re.search(r'[\u4e00-\u9fff]$', previous['text'].strip())
                and current['start'] - previous['end'] <= 0.10
                and current['end'] - current['start'] <= 0.25):
            labels[index] = labels[index - 1]
    return labels


def refine(payload, samples, num_speakers, progress=None, model_override=None):
    import sherpa_onnx
    words = words_from_payload(payload)
    if not words:
        raise RuntimeError('没有可用于声纹校正的语音。')
    language = language_for(words)
    model = Path(model_override) if model_override else MODEL_DIR / (
        'wespeaker_zh_cnceleb_resnet34_LM.onnx' if language == 'zh' else 'wespeaker_en_voxceleb_resnet34_LM.onnx')
    if not model.is_file():
        raise RuntimeError('缺少独立说话人模型：' + model.name)
    progress = progress or (lambda *args: None)
    progress('diarizing', '正在按音频片段独立提取声纹并校正说话人边界。')
    # Keep the validated cloud-boundary path as the default.  The overlapping
    # window implementation is available behind a flag while it is evaluated
    # on more human-labelled clips; this avoids silently regressing the known
    # baseline when a new diarization strategy is experimental.
    if os.getenv('WX_SLIDING_SPEAKER', '0').lower() in {'1', 'true', 'yes', 'on'}:
        units = make_sliding_units(words, window=1.2, step=0.4)
        if len(units) < 2:
            units = make_units(words)
    else:
        units = make_units(words)
    vectors, durations, usable, auxiliary = [], [], [], []
    with LOCK:
        if str(model) not in EXTRACTORS:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=4, provider='cpu')
            EXTRACTORS[str(model)] = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        extractor = EXTRACTORS[str(model)]
        other_extractor = None
        if language == 'zh' and model_override is None:
            other_model = MODEL_DIR / 'wespeaker_en_voxceleb_resnet34_LM.onnx'
            if not other_model.is_file():
                raise RuntimeError('缺少交叉核对模型：' + other_model.name)
            if str(other_model) not in EXTRACTORS:
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(other_model), num_threads=4, provider='cpu')
                EXTRACTORS[str(other_model)] = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            other_extractor = EXTRACTORS[str(other_model)]
        for index, unit in enumerate(units):
            start, end = words[unit[0]]['start'], words[unit[-1]]['end']
            if end - start < 0.28:
                continue
            audio = samples[max(0, int(start * 16000)):min(len(samples), int(end * 16000))]
            stream = extractor.create_stream()
            stream.accept_waveform(16000, audio)
            stream.input_finished()
            if not extractor.is_ready(stream):
                continue
            vector = np.asarray(extractor.compute(stream), dtype=np.float32)
            if np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-8:
                if other_extractor is not None:
                    other_stream = other_extractor.create_stream()
                    other_stream.accept_waveform(16000, audio)
                    other_stream.input_finished()
                    if not other_extractor.is_ready(other_stream):
                        continue
                    other_vector = np.asarray(other_extractor.compute(other_stream), dtype=np.float32)
                    if not np.isfinite(other_vector).all() or np.linalg.norm(other_vector) < 1e-8:
                        continue
                    auxiliary.append(other_vector)
                vectors.append(vector)
                durations.append(end - start)
                usable.append(index)
    if not vectors:
        raise RuntimeError('音频过短，无法提取可靠声纹。')
    assignments = reconcile_vectors(vectors, durations, num_speakers, auxiliary if auxiliary else None)
    unit_labels = [None] * len(units)
    for index, label in zip(usable, assignments):
        unit_labels[index] = label
    unit_labels = fill_isolated_unknown(unit_labels)
    usable_units = [units[index] for index in usable]
    usable_labels = [unit_labels[index] for index in usable]
    usable_durations = [durations[pos] for pos in range(len(durations))]
    word_labels = vote_overlapping_labels(words, usable_units, usable_labels, usable_durations)
    word_labels = fill_isolated_unknown(word_labels)
    word_labels = attach_tiny_unknown_suffixes(words, word_labels)
    lines, count = render_words(words, word_labels)
    notice = '已使用独立 WeSpeaker 声纹与词级时间戳校正。'
    if num_speakers < 0:
        notice += '自动模式已合并低证据的孤立声纹类别。'
    if auxiliary:
        notice += '中文模糊短句增加第二声纹模型交叉核对。'
    notice += '短句和重叠发言仍可能误判，证据不足标为未判定。'
    if num_speakers > 0 and count != num_speakers:
        notice += f'指定 {num_speakers} 人，本次仅标注 {count} 人。'
    return {'segments': lines, 'speaker_count': count, 'speaker_notice': notice,
            'language': language, 'role_method': 'acoustic-refinement', 'speaker_model': model.name}
