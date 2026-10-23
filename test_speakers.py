import io
import json
import unittest
from unittest.mock import patch

import server
import visual_evidence
from server import (
    align_speakers,
    build_transcription_prompt,
    group_role_segments,
    is_narration_fragment,
    is_standalone_address,
    parse_dialogue_candidates,
    parse_role_script,
)


class SpeakerAlignmentTests(unittest.TestCase):
    def test_visual_evidence_schema_has_candidate_fields(self):
        # The conservative unavailable path must still expose a stable shape
        # for the UI and future ASD fusion code.
        with patch.dict("sys.modules", {"cv2": None}):
            result = visual_evidence.inspect_video("missing.mp4")
        self.assertIn("enabled", result)
        self.assertIn("usable", result)

    def test_visual_evidence_is_conservative_when_opencv_is_unavailable(self):
        with patch.dict("sys.modules", {"cv2": None}):
            result = visual_evidence.inspect_video("missing.mp4")
        self.assertFalse(result["usable"])
        self.assertIn("未启用", result["reason"])
    def test_maps_role_script_back_to_untouched_asr_segments(self):
        segments = [
            {"text": "男主第一次见家长"},
            {"text": "显得十分紧张"},
            {"text": "叔叔好"},
            {"text": "这是我男朋友"},
        ]
        script = "A：男主第一次见家长，显得十分紧张。\nB：叔叔好。\nC：这是我男朋友。"

        self.assertEqual(parse_role_script(script, segments), ["A", "A", "B", "C"])

    def test_filters_action_narration_from_dialogue_candidates(self):
        raw = '{"dialogue":[2,"5",99,"bad"]}'

        self.assertEqual(parse_dialogue_candidates(raw, 10), {2, 5})
        self.assertTrue(is_narration_fragment("此时男主给旁边的岳父敬酒"))
        self.assertFalse(is_narration_fragment("叔叔好"))
        self.assertTrue(is_standalone_address("叔叔"))
        self.assertTrue(is_standalone_address("阿姨好！"))
        self.assertFalse(is_standalone_address("不是让你俩换鞋啊"))

    def test_groups_adjacent_role_segments_without_rewriting_text(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "这是你女儿啊"},
            {"start": 1.0, "end": 2.0, "text": "都长这么大了"},
            {"start": 4.0, "end": 5.0, "text": "旁白"},
        ]

        result = group_role_segments(segments, ["B", "B", "A"])

        self.assertEqual([line["speaker"] for line in result], ["说话人 B", "说话人 A"])
        self.assertEqual(result[0]["text"], "这是你女儿啊都长这么大了")

    def test_assigns_each_word_to_the_overlapping_speaker_turn(self):
        transcript = [{
            "start": 0.0,
            "end": 2.0,
            "text": "你好世界",
            "words": [
                {"start": 0.0, "end": 0.7, "text": "你好"},
                {"start": 1.0, "end": 2.0, "text": "世界"},
            ],
        }]
        turns = [
            {"speaker_id": 0, "start": 0.0, "end": 0.8},
            {"speaker_id": 1, "start": 0.9, "end": 2.1},
        ]

        result, speaker_count = align_speakers(transcript, turns)

        self.assertEqual(speaker_count, 2)
        self.assertEqual([line["speaker"] for line in result], ["说话人 A", "说话人 B"])
        self.assertEqual([line["text"] for line in result], ["你好", "世界"])

    def test_marks_speaker_unknown_when_no_turns_are_returned(self):
        transcript = [{"start": 0.0, "end": 1.0, "text": "测试", "words": []}]

        result, speaker_count = align_speakers(transcript, [])

        self.assertEqual(speaker_count, 0)
        self.assertEqual(result[0]["speaker"], "说话人未判定")

    def test_splits_sentences_for_the_same_speaker(self):
        transcript = [{
            "start": 0.0,
            "end": 2.0,
            "text": "你好。世界！",
            "words": [
                {"start": 0.0, "end": 0.5, "text": "你好。"},
                {"start": 0.6, "end": 1.0, "text": "世界！"},
            ],
        }]
        turns = [{"speaker_id": 0, "start": 0.0, "end": 2.0}]

        result, speaker_count = align_speakers(transcript, turns)

        self.assertEqual(speaker_count, 1)
        self.assertEqual([line["text"] for line in result], ["你好。", "世界！"])

    def test_limits_long_same_speaker_paragraphs(self):
        words = [{"start": i * 0.2, "end": i * 0.2 + 0.15, "text": "文字"} for i in range(80)]
        transcript = [{"start": 0.0, "end": 16.0, "text": "文字" * 80, "words": words}]
        turns = [{"speaker_id": 0, "start": 0.0, "end": 17.0}]

        result, _ = align_speakers(transcript, turns)

        self.assertGreater(len(result), 1)
        self.assertEqual("".join(line["text"] for line in result), "文字" * 80)

    def test_splits_english_sentences_on_periods(self):
        transcript = [{
            "start": 0.0,
            "end": 2.0,
            "text": "Hello. How are you?",
            "words": [
                {"start": 0.0, "end": 0.5, "text": "Hello."},
                {"start": 0.6, "end": 1.0, "text": " How are you?"},
            ],
        }]
        turns = [{"speaker_id": 0, "start": 0.0, "end": 2.0}]

        result, _ = align_speakers(transcript, turns)

        self.assertEqual([line["text"] for line in result], ["Hello.", "How are you?"])

    def test_transcription_prompt_does_not_force_chinese(self):
        prompt = build_transcription_prompt({"title": "Amy's interview", "description": "English video"})

        self.assertEqual(prompt, "Amy's interview English video")
        self.assertIsNone(build_transcription_prompt({}))

    def test_does_not_carry_previous_speaker_across_unmatched_gap(self):
        transcript = [{
            "start": 0.0,
            "end": 3.0,
            "text": "甲乙",
            "words": [
                {"start": 0.0, "end": 0.4, "text": "甲"},
                {"start": 2.5, "end": 2.9, "text": "乙"},
            ],
        }]
        turns = [
            {"speaker_id": 0, "start": 0.0, "end": 0.5},
            {"speaker_id": 1, "start": 2.4, "end": 3.0},
        ]

        result, speaker_count = align_speakers(transcript, turns)

        self.assertEqual(speaker_count, 2)
        self.assertEqual([line["speaker"] for line in result], ["说话人 A", "说话人 B"])

    def test_does_not_split_a_long_unpunctuated_sentence_early(self):
        text = "This is a long sentence that should remain together even when it passes the old forty two character limit."
        words = [{"start": i * 0.2, "end": i * 0.2 + 0.15, "text": word + " "} for i, word in enumerate(text.split())]
        transcript = [{"start": 0.0, "end": len(words) * 0.2, "text": text, "words": words}]
        turns = [{"speaker_id": 0, "start": 0.0, "end": 30.0}]

        result, _ = align_speakers(transcript, turns)

        self.assertEqual(len(result), 1)

    @patch("server.time.sleep")
    @patch("server.urlopen")
    def test_parse_retries_a_timeout_once(self, mock_urlopen, _mock_sleep):
        payload = {
            "code": 200,
            "data": {"url": "https://finder.video.qq.com/video.mp4", "title": "test"},
        }
        mock_urlopen.side_effect = [TimeoutError(), io.BytesIO(json.dumps(payload).encode())]
        with patch.object(server, "API_TEMPLATES", "https://parser.example/?url={url}"):
            media_url, metadata = server.parse_video("https://weixin.qq.com/sph/abc")

        self.assertEqual(media_url, payload["data"]["url"])
        self.assertEqual(metadata["title"], "test")
        self.assertEqual(mock_urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
