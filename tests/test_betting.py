import unittest

from tennis_ml.betting import closing_line_value, kelly_fraction, remove_vig


class TestBettingHelpers(unittest.TestCase):
    def test_remove_vig_normalizes_to_one(self):
        fair_p1, fair_p2, overround = remove_vig(1.90, 1.90)
        self.assertAlmostEqual(fair_p1, 0.5, places=6)
        self.assertAlmostEqual(fair_p2, 0.5, places=6)
        self.assertAlmostEqual(fair_p1 + fair_p2, 1.0, places=9)
        self.assertGreater(overround, 1.0)

    def test_remove_vig_asymmetric_odds(self):
        fair_p1, fair_p2, overround = remove_vig(1.50, 2.75)
        self.assertGreater(fair_p1, fair_p2)
        self.assertAlmostEqual(fair_p1 + fair_p2, 1.0, places=9)
        self.assertAlmostEqual(overround, 1 / 1.50 + 1 / 2.75, places=9)

    def test_remove_vig_rejects_invalid_odds(self):
        with self.assertRaises(ValueError):
            remove_vig(1.0, 2.0)

    def test_kelly_fraction_positive_and_negative_ev(self):
        # Even odds (2.0) with 60% win probability -> f = (0.6*1 - 0.4)/1 = 0.2
        self.assertAlmostEqual(kelly_fraction(0.60, 2.0), 0.20, places=6)
        # Negative EV -> clipped to 0.0
        self.assertEqual(kelly_fraction(0.40, 2.0), 0.0)

    def test_closing_line_value(self):
        # Took 2.20, closed at 2.00 -> fair prob rose from 1/2.20 to 1/2.00 (+10% CLV)
        clv = closing_line_value(2.20, 2.00)
        self.assertAlmostEqual(clv, 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
