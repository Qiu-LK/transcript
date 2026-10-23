import unittest
import numpy as np
from speaker_refinement import (words_from_payload, make_units, render_words,
                                reconcile_vectors, attach_tiny_unknown_suffixes,
                                make_sliding_units, vote_overlapping_labels)


class RefinementTests(unittest.TestCase):
    def test_sliding_units_overlap_cloud_sentence_boundary(self):
        words = [
            {'start': 0.0, 'end': 0.4, 'text': '甲'},
            {'start': 0.42, 'end': 0.8, 'text': '乙'},
            {'start': 0.82, 'end': 1.2, 'text': '丙'},
        ]
        units = make_sliding_units(words, window=0.8, step=0.3)
        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(any(0 in unit and 1 in unit for unit in units))

    def test_overlapping_votes_assign_each_word(self):
        words = [
            {'start': 0.0, 'end': 0.4, 'text': '甲'},
            {'start': 0.4, 'end': 0.8, 'text': '乙'},
        ]
        units = [[0], [0, 1], [1]]
        labels = [0, 0, 1]
        projected = vote_overlapping_labels(words, units, labels, [0.6, 1.0, 0.6])
        # The mixed window provides stronger evidence than the isolated short`r`n        # window, so the conservative vote keeps the second word with 0.`r`n        self.assertEqual(projected, [0, 0])

    def test_tiny_unknown_suffix_stays_with_previous_han_word(self):
        words = [
            {'start': 56.7, 'end': 57.06, 'text': '敬', 'sentence_number': 0},
            {'start': 57.07, 'end': 57.23, 'text': '酒，', 'sentence_number': 0},
        ]
        labels = attach_tiny_unknown_suffixes(words, [0, None])
        lines, _ = render_words(words, labels)
        self.assertEqual(labels, [0, 0])
        self.assertEqual([line['text'] for line in lines], ['敬酒，'])

    def test_tiny_unknown_suffix_never_crosses_a_real_pause(self):
        words = [
            {'start': 0, 'end': 0.3, 'text': '敬', 'sentence_number': 0},
            {'start': 0.31, 'end': 0.47, 'text': '酒', 'sentence_number': 1},
            {'start': 0.8, 'end': 1.1, 'text': '敬', 'sentence_number': 2},
            {'start': 1.3, 'end': 1.46, 'text': '酒，', 'sentence_number': 2},
        ]
        self.assertEqual(attach_tiny_unknown_suffixes(words, [0, None, 0, None]),
                         [0, None, 0, None])

    def test_repairs_lowercase_continuation_and_preserves_word_times(self):
        words = words_from_payload({'transcripts': [{'sentences': [
            {'words': [{'begin_time': 0, 'end_time': 700, 'text': 'Can I', 'punctuation': '.'}]},
            {'words': [{'begin_time': 700, 'end_time': 1400, 'text': 'have it', 'punctuation': '?'}]},
        ]}]})
        lines, _ = render_words(words, [0, 0])
        self.assertEqual(lines[0]['text'], 'Can I have it?')
        self.assertEqual(lines[0]['end'], 1.4)

    def test_real_speaker_change_is_not_merged_for_grammar(self):
        words = [{'start': 0, 'end': 1, 'text': 'I think'}, {'start': 1, 'end': 2, 'text': 'so.'}]
        lines, count = render_words(words, [0, 1])
        self.assertEqual(count, 2)
        self.assertEqual(len(lines), 2)

    def test_short_reply_cannot_contaminate_anchor_centers(self):
        labels = reconcile_vectors([[1, 0], [0.98, 0.02], [0, 1], [0.6, 0.8]], [2, 2, 2, 0.4], 2)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[2])

    def test_second_model_can_resolve_weak_ambiguous_match(self):
        primary = [[1, 0, 0], [0, 1, 0], [0.5, 0.45, 0.74]]
        auxiliary = [[1, 0], [0, 1], [0.1, 0.99]]
        labels = reconcile_vectors(primary, [2, 2, 0.4], 2, auxiliary)
        self.assertEqual(labels[2], labels[1])
        self.assertIsNone(reconcile_vectors(primary, [2, 2, 0.4], 2)[2])

    def test_pause_splits_mixed_cloud_sentence(self):
        words = [{'start': 0, 'end': 0.7, 'text': '旁白'}, {'start': 1, 'end': 1.4, 'text': '阿姨'}]
        self.assertEqual(make_units(words), [[0], [1]])

    def test_cloud_sentence_boundary_splits_fast_reply_without_pause(self):
        payload = {'transcripts': [{'sentences': [
            {'words': [{'begin_time': 0, 'end_time': 700, 'text': '你好吗'}]},
            {'words': [{'begin_time': 710, 'end_time': 1200, 'text': '我很好'}]},
        ]}]}
        words = words_from_payload(payload)
        self.assertEqual(make_units(words), [[0], [1]])

    def test_ambiguous_auto_reply_is_not_forced_to_dominant_speaker(self):
        primary = [[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0.4, 0.4, 0.82]]
        labels = reconcile_vectors(primary, [2, 2, 2, 0.4], -1)
        self.assertIsNone(labels[3])

    def test_missing_timestamps_does_not_silently_reuse_cloud_speaker(self):
        with self.assertRaises(RuntimeError):
            words_from_payload({'transcripts': [{'sentences': [{'text': 'No words', 'speaker_id': 0}]}]})

    def test_auto_mode_merges_tiny_island_into_dominant_voice(self):
        primary = [[1, 0], [0.99, 0.1], [0.1, 0.99], [0.95, 0.2]]
        labels = reconcile_vectors(primary, [3, 2, 1.5, 0.4], -1)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[3], labels[0])

    def test_auto_mode_keeps_repeated_short_speaker(self):
        primary = [[1, 0], [0.99, 0.1], [0.1, 0.99], [0.1, 0.98]]
        labels = reconcile_vectors(primary, [3, 2, 0.8, 0.8], -1)
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


if __name__ == '__main__':
    unittest.main()

