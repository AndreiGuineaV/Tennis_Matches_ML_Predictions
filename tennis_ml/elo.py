"""Elo rating calculations with dynamic K-factor."""

DEFAULT_ELO = 1500


def expected_score(rating_a, rating_b):
    """Return expected win probability for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a, rating_b, score_a, k):
    """Return updated (rating_a, rating_b) after a match with outcome score_a in {0, 1}."""
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - exp_a))
    return new_a, new_b


def dynamic_k(n_matches):
    """Higher K-factor for players with few matches; decays as history stabilizes."""
    return 250.0 / ((n_matches + 5.0) ** 0.4)
