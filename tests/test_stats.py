"""Tests for src/stats.py.

Hand-worked cases always run. Where scikit-learn or scipy is installed, the
same functions are also checked against the reference implementations on
random data, including ties, which are where hand-rolled AP usually goes wrong.
"""
import math
import random
import unittest

from src.stats import (average_precision, bootstrap_ci, brier, ece, paired_summary,
                       sign_test_p)

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    average_precision_score = None

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None


class AveragePrecision(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertEqual(average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)

    def test_worst_ranking(self):
        # positives ranked last: precision is 1/3 at the first hit and 2/4 at the second
        ap = average_precision([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
        self.assertAlmostEqual(ap, 0.5 * (1 / 3) + 0.5 * (2 / 4))

    def test_hand_worked_interleaved(self):
        # ranking: 1 0 1 0 -> 0.5 * 1/1 + 0.5 * 2/3
        ap = average_precision([0.9, 0.7, 0.5, 0.3], [1, 0, 1, 0])
        self.assertAlmostEqual(ap, 0.5 + 0.5 * (2 / 3))

    def test_tied_scores_enter_the_curve_together(self):
        # all four tied: a single threshold with precision 2/4 and recall 1.0
        self.assertAlmostEqual(average_precision([0.5] * 4, [1, 0, 1, 0]), 0.5)

    def test_no_positives_is_undefined(self):
        self.assertIsNone(average_precision([0.9, 0.1], [0, 0]))

    @unittest.skipIf(average_precision_score is None, "scikit-learn not installed")
    def test_matches_sklearn_with_ties(self):
        rng = random.Random(7)
        for _ in range(200):
            n = rng.randint(2, 60)
            labels = [rng.randint(0, 1) for _ in range(n)]
            if sum(labels) == 0:
                labels[0] = 1
            scores = [rng.choice([0.1, 0.25, 0.5, 0.75, 0.9]) for _ in range(n)]  # heavy ties
            self.assertAlmostEqual(average_precision(scores, labels),
                                   average_precision_score(labels, scores), places=10)


class SignTest(unittest.TestCase):
    def test_no_informative_pairs(self):
        self.assertEqual(sign_test_p([0, 0, 0]), 1.0)

    def test_even_split_is_not_significant(self):
        self.assertEqual(sign_test_p([1, -1, 1, -1]), 1.0)

    def test_ten_of_ten(self):
        self.assertAlmostEqual(sign_test_p([0.1] * 10), 2 / 2 ** 10)

    def test_ties_are_dropped(self):
        self.assertEqual(sign_test_p([0.1] * 10 + [0] * 50), sign_test_p([0.1] * 10))

    @unittest.skipIf(binomtest is None, "scipy not installed")
    def test_matches_scipy(self):
        for wins in range(0, 21):
            diffs = [1] * wins + [-1] * (20 - wins)
            self.assertAlmostEqual(sign_test_p(diffs), binomtest(wins, 20, 0.5).pvalue, places=12)


class Bootstrap(unittest.TestCase):
    def test_interval_brackets_the_mean(self):
        est, lo, hi = bootstrap_ci([0.6, 0.7, 0.8, 0.9, 1.0], n_boot=2000)
        self.assertAlmostEqual(est, 0.8)
        self.assertLessEqual(lo, est)
        self.assertGreaterEqual(hi, est)

    def test_constant_data_has_zero_width(self):
        self.assertEqual(bootstrap_ci([0.5] * 10, n_boot=500), (0.5, 0.5, 0.5))

    def test_deterministic_for_a_seed(self):
        v = [random.Random(1).random() for _ in range(30)]
        self.assertEqual(bootstrap_ci(v, seed=3, n_boot=1000), bootstrap_ci(v, seed=3, n_boot=1000))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([])


class Paired(unittest.TestCase):
    def test_consistent_small_gap_is_detected(self):
        # large spread across networks, but model a is always 0.02 ahead
        rng = random.Random(0)
        b = [rng.uniform(0.3, 1.0) for _ in range(20)]
        a = [x + 0.02 for x in b]
        s = paired_summary(a, b, n_boot=2000)
        self.assertAlmostEqual(s["mean_diff"], 0.02)
        self.assertGreater(s["ci_low"], 0)
        self.assertEqual((s["wins"], s["losses"]), (20, 0))
        self.assertLess(s["sign_test_p"], 0.001)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_summary([1, 2], [1])


class Calibration(unittest.TestCase):
    def test_brier(self):
        self.assertAlmostEqual(brier([1.0, 0.0, 0.5], [1, 0, 1]), 0.25 / 3)

    def test_perfectly_calibrated_bin_has_zero_ece(self):
        # every prediction 0.25 and exactly a quarter are positive
        self.assertAlmostEqual(ece([0.25] * 8, [1, 1, 0, 0, 0, 0, 0, 0]), 0.0)

    def test_overconfident_model(self):
        # says 0.95 but only half are positive
        self.assertAlmostEqual(ece([0.95] * 4, [1, 1, 0, 0]), 0.45)

    def test_probability_of_one_lands_in_top_bin(self):
        self.assertTrue(math.isfinite(ece([1.0, 0.0], [1, 0])))


if __name__ == "__main__":
    unittest.main()
