"""
Tests for the paired bootstrap comparison of corpus BLEU scores.
"""

import unittest

import pytest

from nltk.translate.bleu_score import (
    SmoothingFunction,
    corpus_bleu,
    paired_bootstrap_bleu,
)


def _two_document_corpus():
    """
    Two documents of very different lengths: a document of six short
    sentences and a document of two long sentences. System A is a
    near-perfect translation, system B is noticeably worse.
    """
    short_pairs = [
        ("the cat sat on the mat", "the cat sat on the mat", "a cat sat on mat"),
        ("she sells sea shells", "she sells sea shells", "he sells sea shells"),
        (
            "the quick brown fox jumps",
            "the quick brown fox jumps",
            "the quick red fox jumps",
        ),
        ("i like to read books", "i like to read books", "i like reading books"),
        ("they went to the park", "they went to the park", "they go to park"),
        ("we eat dinner at home", "we eat dinner at home", "we ate dinner home"),
    ]
    long_pairs = [
        (
            "the committee announced yesterday that it would postpone the final "
            "decision on the new budget until next monday morning",
            "the committee announced yesterday that it would postpone the final "
            "decision on the new budget until next monday morning",
            "the committee said it would postpone the decision on the budget "
            "until monday",
        ),
        (
            "despite the heavy rain the marathon runners continued along the "
            "river bank toward the finish line near the old bridge",
            "despite the heavy rain the marathon runners continued along the "
            "river bank toward the finish line near the old bridge",
            "despite the rain the runners continued along the river toward "
            "the finish line",
        ),
    ]
    references, hyp_a, hyp_b, document_ids = [], [], [], []
    for document_id, pairs in enumerate([short_pairs, long_pairs]):
        for ref, hyp1, hyp2 in pairs:
            references.append([ref.split()])
            hyp_a.append(hyp1.split())
            hyp_b.append(hyp2.split())
            document_ids.append(document_id)
    return references, hyp_a, hyp_b, document_ids


class TestPairedBootstrapBLEU(unittest.TestCase):
    def test_document_level_vs_sentence_level_sampling(self):
        """
        With two documents of very different lengths, sampling whole
        documents should give the same point estimates but a wider
        confidence interval than sampling individual sentences, because
        the long document dominates the corpus and is either present
        (possibly duplicated) or absent in each resample.
        """
        references, hyp_a, hyp_b, document_ids = _two_document_corpus()

        doc_level = paired_bootstrap_bleu(
            references,
            hyp_a,
            hyp_b,
            document_ids=document_ids,
            n_resamples=500,
            seed=7,
        )
        sent_level = paired_bootstrap_bleu(
            references, hyp_a, hyp_b, n_resamples=500, seed=7
        )

        # Point estimates do not depend on the sampling unit.
        self.assertAlmostEqual(doc_level.bleu_a, sent_level.bleu_a)
        self.assertAlmostEqual(doc_level.bleu_b, sent_level.bleu_b)
        self.assertAlmostEqual(doc_level.difference, sent_level.difference)

        # The intervals differ: with only two documents of very unequal
        # size, each document-level resample is dominated by which of the
        # two documents were drawn, so the bootstrap distribution of the
        # differences collapses to the few possible document combinations
        # (two draws with replacement from two documents), while the
        # sentence-level distribution varies much more smoothly.
        assert len(set(doc_level.differences)) <= 3
        assert len(set(sent_level.differences)) > 3
        assert (
            doc_level.ci_low != sent_level.ci_low
            or doc_level.ci_high != sent_level.ci_high
        )

        # Document-level resamples keep whole documents together.
        doc_sizes = {0: 6, 1: 2}
        for replicate in doc_level.sample_indices:
            n_sents = sum(doc_sizes[unit] for unit in replicate)
            assert n_sents == sum(6 if unit == 0 else 2 for unit in replicate)

    def test_document_level_replicates_match_corpus_bleu(self):
        """
        Every document-level resample score should equal a direct
        corpus_bleu call on the correspondingly resampled corpus.
        """
        references, hyp_a, hyp_b, document_ids = _two_document_corpus()
        sample_indices = [(0, 0), (0, 1), (1, 1), (1, 0)]
        result = paired_bootstrap_bleu(
            references,
            hyp_a,
            hyp_b,
            document_ids=document_ids,
            sample_indices=sample_indices,
        )
        units = [(0, 1, 2, 3, 4, 5), (6, 7)]
        assert result.n_resamples == result.n_valid == len(sample_indices)
        for replicate, score_a, score_b in zip(
            result.sample_indices, result.scores_a, result.scores_b
        ):
            sentence_ids = [i for unit in replicate for i in units[unit]]
            sub_refs = [references[i] for i in sentence_ids]
            expected_a = corpus_bleu(sub_refs, [hyp_a[i] for i in sentence_ids])
            expected_b = corpus_bleu(sub_refs, [hyp_b[i] for i in sentence_ids])
            self.assertAlmostEqual(score_a, expected_a)
            self.assertAlmostEqual(score_b, expected_b)

    def test_small_sample_replicates_match_corpus_bleu(self):
        """
        On a small corpus with multiple references, a short sentence, an
        empty hypothesis and a smoothing function, every resample score
        should equal a direct corpus_bleu call on the resampled corpus.
        """
        references = [
            [
                "the cat is on the mat".split(),
                "there is a cat on the mat".split(),
            ],
            ["he reads".split()],
            ["she sells sea shells by the shore".split()],
            ["we play football in the park every sunday".split()],
        ]
        hyp_a = [
            "the cat is on the mat".split(),
            "he reads".split(),
            "she sells sea shells by the shore".split(),
            "we play football in the park every sunday".split(),
        ]
        hyp_b = [
            "the cat is on mat".split(),
            "he reads".split(),
            [],  # empty hypothesis
            "we play football in park".split(),
        ]
        sample_indices = [(0, 1, 2, 3), (3, 3, 2, 0), (1, 1, 1, 1), (2, 0, 3, 1)]
        smoothing = SmoothingFunction().method1
        result = paired_bootstrap_bleu(
            references,
            hyp_a,
            hyp_b,
            smoothing_function=smoothing,
            sample_indices=sample_indices,
        )

        # Point estimates equal direct corpus_bleu calls on the full corpus.
        self.assertAlmostEqual(
            result.bleu_a, corpus_bleu(references, hyp_a, smoothing_function=smoothing)
        )
        self.assertAlmostEqual(
            result.bleu_b, corpus_bleu(references, hyp_b, smoothing_function=smoothing)
        )
        self.assertAlmostEqual(result.difference, result.bleu_a - result.bleu_b)

        assert result.n_resamples == result.n_valid == len(sample_indices)
        assert result.sample_indices == [tuple(r) for r in sample_indices]
        for replicate, score_a, score_b, difference in zip(
            result.sample_indices,
            result.scores_a,
            result.scores_b,
            result.differences,
        ):
            sub_refs = [references[i] for i in replicate]
            sub_a = [hyp_a[i] for i in replicate]
            sub_b = [hyp_b[i] for i in replicate]
            self.assertAlmostEqual(
                score_a, corpus_bleu(sub_refs, sub_a, smoothing_function=smoothing)
            )
            self.assertAlmostEqual(
                score_b, corpus_bleu(sub_refs, sub_b, smoothing_function=smoothing)
            )
            self.assertAlmostEqual(difference, score_a - score_b)

    def test_seed_reproducibility(self):
        references, hyp_a, hyp_b, _ = _two_document_corpus()
        first = paired_bootstrap_bleu(
            references, hyp_a, hyp_b, n_resamples=50, seed=123
        )
        second = paired_bootstrap_bleu(
            references, hyp_a, hyp_b, n_resamples=50, seed=123
        )
        assert first.sample_indices == second.sample_indices
        assert first.differences == second.differences
        assert first.ci_low == second.ci_low
        assert first.ci_high == second.ci_high

        third = paired_bootstrap_bleu(
            references, hyp_a, hyp_b, n_resamples=50, seed=456
        )
        assert first.sample_indices != third.sample_indices

    def test_both_systems_share_resamples(self):
        """
        Both systems are evaluated on identical resamples, so a replicate
        that samples only sentences where the systems tie contributes a
        zero difference.
        """
        references = [["a b c".split()], ["d e f".split()]]
        hyp_a = ["a b c".split(), "d e f".split()]
        hyp_b = ["a b c".split(), "x y z".split()]
        result = paired_bootstrap_bleu(
            references, hyp_a, hyp_b, sample_indices=[(0, 0), (1, 1)]
        )
        # Resample (0, 0): systems tie -> difference 0.
        # Resample (1, 1): A perfect, B disjoint -> positive difference.
        assert result.differences[0] == 0
        assert result.differences[1] > 0

    def test_invalid_inputs(self):
        references, hyp_a, hyp_b, document_ids = _two_document_corpus()
        with pytest.raises(ValueError):
            paired_bootstrap_bleu(references, hyp_a[:-1], hyp_b)
        with pytest.raises(ValueError):
            paired_bootstrap_bleu([], [], [])
        with pytest.raises(ValueError):
            paired_bootstrap_bleu(
                references, hyp_a, hyp_b, document_ids=document_ids[:-1]
            )
        with pytest.raises(ValueError):
            paired_bootstrap_bleu(
                references, hyp_a, hyp_b, weights=[(0.5, 0.5), (0.25,) * 4]
            )
        with pytest.raises(ValueError):
            paired_bootstrap_bleu(references, hyp_a, hyp_b, confidence=1.5)
        with pytest.raises(ValueError):
            paired_bootstrap_bleu(references, hyp_a, hyp_b, n_resamples=0)
        with pytest.raises(ValueError):
            paired_bootstrap_bleu(references, hyp_a, hyp_b, sample_indices=[(0, 100)])
