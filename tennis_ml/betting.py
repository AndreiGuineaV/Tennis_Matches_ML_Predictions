"""Betting math helpers: vig removal, closing line value, and Kelly staking."""


def remove_vig(odds_p1, odds_p2):
    """Remove bookmaker overround and return (fair_p1, fair_p2, overround)."""
    if odds_p1 <= 1.0 or odds_p2 <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    raw_p1, raw_p2 = 1.0 / odds_p1, 1.0 / odds_p2
    overround = raw_p1 + raw_p2
    return raw_p1 / overround, raw_p2 / overround, overround


def closing_line_value(odds_taken, closing_odds):
    """Compute fractional Closing Line Value (CLV) relative to closing odds."""
    if odds_taken <= 1.0 or closing_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    fair_taken = 1.0 / odds_taken
    fair_close = 1.0 / closing_odds
    return (fair_close - fair_taken) / fair_taken


def kelly_fraction(prob, odds):
    """Return full-Kelly stake fraction clipped at 0 for non-positive EV bets."""
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, f)
