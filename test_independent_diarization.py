import unittest
from independent_diarization import align_audio_labels


class IndependentDiarizationTests(unittest.TestCase):
    def test_maps_independent_id_from_long_anchor_and_repairs_unknown(self):
        lines = [
            {"start": 0.0, "end": 2.0, "text": "长句", "speaker": "说话人 B"},
            {"start": 2.1, "end": 2.5, "text": "嗯", "speaker": "说话人未判定"},
        ]
        evidence = {"available": True, "turns": [
            {"start": 0.0, "end": 2.0, "speaker_id": 4, "confidence": 0.9},
            {"start": 2.1, "end": 2.5, "speaker_id": 4, "confidence": 0.9},
        ]}
        result, info = align_audio_labels(lines, evidence)
        self.assertEqual(result[1]["speaker"], "说话人 B")
        self.assertEqual(info["changed_segments"], 1)

    def test_ambiguous_mapping_keeps_unknown(self):
        lines = [
            {"start": 0, "end": 1, "text": "甲", "speaker": "说话人 A"},
            {"start": 1, "end": 2, "text": "乙", "speaker": "说话人 B"},
            {"start": 2, "end": 2.2, "text": "嗯", "speaker": "说话人未判定"},
        ]
        evidence = {"available": True, "turns": [
            {"start": 0, "end": 2.2, "speaker_id": 1, "confidence": 0.8},
        ]}
        result, info = align_audio_labels(lines, evidence)
        self.assertEqual(result[-1]["speaker"], "说话人未判定")
        self.assertFalse(info["applied"])


if __name__ == "__main__":
    unittest.main()
