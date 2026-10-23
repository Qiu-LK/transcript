import unittest
from unittest.mock import patch

from text_refinement import _review_candidates, review_with_local


class TextRefinementTests(unittest.TestCase):
    def test_review_skips_short_lines_and_limits_cpu_work(self):
        lines = ([{"start": 0.0, "end": 0.5, "text": "只有短句"}]
                 + [{"start": float(i), "end": float(i) + 3.0,
                     "text": "这是一句足够长的中文对白内容"} for i in range(20)])
        with patch.dict("text_refinement.os.environ", {"WX_TEXT_REVIEW_MAX_CLIPS": "3"}):
            self.assertEqual(_review_candidates(lines, "zh"), [1, 2, 3])
            self.assertEqual(_review_candidates(lines, "en"), [])

    @patch("text_refinement._local_clip_texts")
    def test_repairs_short_opening_without_replacing_next_speaker(self, local):
        local.return_value = ({0: {"text": "怕，没事的，我都成了狗了让我别怕。",
                                  "prefix_text": "别怕，没事的", "avg_logprob": -0.3}}, None)
        source = [{"start": 18.74, "end": 22.01, "speaker": "说话人 A",
                   "text": "对，爸，没事的，我都成了狗了让我娘了。"}]
        result, info = review_with_local("unused.mp4", source, "zh")
        self.assertEqual(result[0]["text"], "别怕，没事的，我都成了狗了让我娘了。")
        self.assertEqual(info["changed_segments"], 1)

    @patch("text_refinement._local_clip_texts")
    def test_repairs_substantial_disagreement_without_changing_speaker(self, local):
        local.return_value = ({0: {"text": "别怕，没事的，我会陪着你。", "prefix_text": "别怕，没事的。",
                                  "avg_logprob": -0.3}}, None)
        source = [{"start": 18.0, "end": 20.0, "speaker": "说话人 B",
                   "text": "对，爸，没事的，我会陪着你。"}]
        result, info = review_with_local("unused.mp4", source, "zh")
        self.assertEqual(result[0]["text"], "别怕，没事的，我会陪着你。")
        self.assertEqual(result[0]["speaker"], "说话人 B")
        self.assertEqual(info["changed_segments"], 1)

    @patch("text_refinement._local_clip_texts")
    def test_does_not_replace_unanchored_short_reply(self, local):
        local.return_value = ({0: {"text": "文字", "avg_logprob": -0.3}}, None)
        source = [{"start": 39.0, "end": 39.5, "speaker": "说话人 A", "text": "啊？"}]
        result, info = review_with_local("unused.mp4", source, "zh")
        self.assertEqual(result, source)
        self.assertEqual(info["changed_segments"], 0)

    @patch("text_refinement._local_clip_texts")
    def test_does_not_change_a_similar_sounding_name(self, local):
        local.return_value = ({0: {"text": "丁车，你来的正好，来看看雷霆快。",
                                  "prefix_text": "丁车，你来的正好", "avg_logprob": -0.3}}, None)
        source = [{"start": 43.5, "end": 47.0, "speaker": "说话人 A",
                   "text": "订车，你来的正好，来看看雷霆快。"}]
        result, info = review_with_local("unused.mp4", source, "zh")
        self.assertEqual(result, source)
        self.assertEqual(info["changed_segments"], 0)

    @patch("text_refinement._local_clip_texts")
    def test_low_confidence_local_result_cannot_replace_cloud(self, local):
        local.return_value = ({}, None)
        source = [{"start": 0.0, "end": 1.0, "speaker": "说话人 A", "text": "原句"}]
        result, info = review_with_local("unused.mp4", source, "zh")
        self.assertEqual(result, source)
        self.assertEqual(info["changed_segments"], 0)

    @patch("text_refinement._local_clip_texts")
    def test_misaligned_local_segment_cannot_rewrite_another_speaker(self, local):
        local.return_value = ({}, None)
        source = [{"start": 18.0, "end": 22.0, "speaker": "说话人 A",
                   "text": "对，爸，没事的，我都成了狗了。"}]
        result, info = review_with_local("unused.mp4", source, "zh")
        self.assertEqual(result, source)
        self.assertEqual(info["changed_segments"], 0)


if __name__ == "__main__":
    unittest.main()
