import unittest

from tennis_ml.elo import DEFAULT_ELO, dynamic_k, expected_score, update_elo


class TestEloRatings(unittest.TestCase):
    def test_expected_score_equal_ratings(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5, places=6)

    def test_expected_score_symmetry(self):
        p_ab = expected_score(1700, 1500)
        p_ba = expected_score(1500, 1700)
        self.assertGreater(p_ab, 0.5)
        self.assertAlmostEqual(p_ab + p_ba, 1.0, places=9)

    def test_update_elo_zero_sum(self):
        new_a, new_b = update_elo(DEFAULT_ELO, DEFAULT_ELO, score_a=1.0, k=32.0)
        self.assertAlmostEqual(new_a, 1516.0, places=6)
        self.assertAlmostEqual(new_b, 1484.0, places=6)
        self.assertAlmostEqual(new_a + new_b, 2 * DEFAULT_ELO, places=9)

    def test_dynamic_k_decays_with_experience(self):
        k_rookie = dynamic_k(0)
        k_veteran = dynamic_k(200)
        self.assertGreater(k_rookie, k_veteran)
        self.assertGreater(k_veteran, 0.0)


if __name__ == "__main__":
    unittest.main()
