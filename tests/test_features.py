import datetime as dt
import math
import unittest

from tennis_ml.elo import DEFAULT_ELO
from tennis_ml.features import DEFAULT_SRV, FEATURES, StateTracker, build_features


class TestFeatureGeneration(unittest.TestCase):
    def test_build_features_antisymmetry_and_keys(self):
        a = {
            "elo": 1650.0,
            "selo": 1700.0,
            "fatigue": 3,
            "rank": 10,
            "age": 24.5,
            "srv": dict(DEFAULT_SRV, ace_rate=0.10, first_won=0.78, second_won=0.55),
            "h2h_wins": 3,
            "h2h_total": 4,
        }
        b = {
            "elo": 1550.0,
            "selo": 1520.0,
            "fatigue": 1,
            "rank": 40,
            "age": 28.5,
            "srv": dict(DEFAULT_SRV),
            "h2h_wins": 1,
            "h2h_total": 4,
        }
        fa = build_features(a, b, best_of=3)
        fb = build_features(b, a, best_of=3)

        self.assertEqual(set(fa.keys()), set(FEATURES))
        self.assertAlmostEqual(fa["elo_diff"], -fb["elo_diff"], places=9)
        self.assertAlmostEqual(fa["surface_elo_diff"], -fb["surface_elo_diff"], places=9)
        self.assertAlmostEqual(fa["rank_diff"], math.log(40) - math.log(10), places=9)
        self.assertAlmostEqual(fa["rank_diff"], -fb["rank_diff"], places=9)
        self.assertAlmostEqual(fa["h2h_winrate"] + fb["h2h_winrate"], 1.0, places=9)
        self.assertEqual(fa["best_of"], 3)

    def test_state_tracker_updates_leak_free(self):
        st = StateTracker()
        d1 = dt.date(2026, 1, 10)
        d2 = dt.date(2026, 1, 15)

        pre_w = st.get_state("Player A", "Hard", d1, rank=20, age=24.0)
        self.assertEqual(pre_w["elo"], DEFAULT_ELO)
        self.assertEqual(pre_w["selo"], DEFAULT_ELO)
        self.assertEqual(pre_w["fatigue"], 0)

        st.update_after_match(
            "Player A",
            "Player B",
            "Hard",
            d1,
            {
                "w_svpt": 60,
                "w_ace": 6,
                "w_df": 2,
                "w_1stIn": 40,
                "w_1stWon": 32,
                "w_2ndWon": 12,
                "l_svpt": 60,
                "l_ace": 3,
                "l_df": 4,
                "l_1stIn": 36,
                "l_1stWon": 24,
                "l_2ndWon": 10,
            },
        )

        post_w = st.get_state("Player A", "Hard", d2, rank=18, age=24.0)
        post_l = st.get_state("Player B", "Hard", d2, rank=50, age=26.0)
        self.assertGreater(post_w["elo"], DEFAULT_ELO)
        self.assertLess(post_l["elo"], DEFAULT_ELO)
        self.assertEqual(post_w["fatigue"], 1)
        self.assertEqual(st.h2h_record("Player A", "Player B"), (1, 0, 1))
        self.assertAlmostEqual(post_w["srv"]["ace_rate"], 6 / 60, places=6)


if __name__ == "__main__":
    unittest.main()
