import unittest

from visual_fusion import fuse


class VisualFusionTests(unittest.TestCase):
    def test_mouth_motion_proxy_cannot_relabel(self):
        rows = [{"start": 0, "end": 1, "speaker": "说话人未判定", "text": "嗯"}]
        evidence = {"active_speaker_available": True, "active_windows": [
            {"start": 0, "end": 1, "face_id": 1, "confidence": 1.0}]}
        result, info = fuse(rows, evidence)
        self.assertEqual(result, rows)
        self.assertFalse(info["applied"])

    def test_repairs_unknown_short_segment_from_face_mapping(self):
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "说话人 A", "text": "前句"},
            {"start": 1.0, "end": 1.15, "speaker": "说话人未判定", "text": "酒，"},
            {"start": 1.2, "end": 2.2, "speaker": "说话人 A", "text": "后句"},
        ]
        evidence = {"active_speaker_available": True, "method": "LR-ASD-pretrain-AVA", "active_windows": [
            {"start": 0.0, "end": 1.0, "face_id": 1, "confidence": 0.95},
            {"start": 1.0, "end": 1.2, "face_id": 1, "confidence": 0.95},
            {"start": 1.2, "end": 2.2, "face_id": 1, "confidence": 0.95},
        ]}
        result, info = fuse(segments, evidence)
        self.assertTrue(info["applied"])
        self.assertEqual("说话人 A", result[0]["speaker"])
        self.assertIn("酒，", result[0]["text"])

    def test_does_not_overwrite_known_audio_speaker(self):
        segments = [{"start": 0.0, "end": 1.0, "speaker": "说话人 B", "text": "已确定"}]
        evidence = {"active_speaker_available": True, "method": "LR-ASD-pretrain-AVA", "active_windows": [
            {"start": 0.0, "end": 1.0, "face_id": 1, "confidence": 0.95},
        ]}
        result, info = fuse(segments, evidence)
        self.assertEqual("说话人 B", result[0]["speaker"])
        self.assertFalse(info["applied"])

    def test_ambiguous_face_mapping_is_rejected(self):
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "说话人 A", "text": "甲"},
            {"start": 1.0, "end": 2.0, "speaker": "说话人 B", "text": "乙"},
            {"start": 2.0, "end": 2.1, "speaker": "说话人未判定", "text": "嗯"},
        ]
        evidence = {"active_speaker_available": True, "method": "LR-ASD-pretrain-AVA", "active_windows": [
            {"start": 0.0, "end": 1.0, "face_id": 1, "confidence": 0.8},
            {"start": 1.0, "end": 2.0, "face_id": 1, "confidence": 0.8},
            {"start": 2.0, "end": 2.1, "face_id": 1, "confidence": 0.8},
        ]}
        result, info = fuse(segments, evidence)
        self.assertEqual("说话人未判定", result[-1]["speaker"])
        self.assertFalse(info["applied"])


if __name__ == "__main__":
    unittest.main()
