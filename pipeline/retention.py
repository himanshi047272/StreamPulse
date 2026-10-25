"""Retention recommendation engine (rules on top of the churn probability)."""

LOW, MED = 0.35, 0.55


def risk_level(p):
    return "Low" if p < LOW else ("Medium" if p < MED else "High")


def recommend(row, p):
    """row: dict-like with the RAW Netflix columns (missing keys are fine)."""
    g = lambda k, d=0: row.get(k, d) if row.get(k, d) == row.get(k, d) else d  # NaN-safe get
    if p < LOW:
        return "Low risk - no action needed"
    if g("last_login_days") >= 21:
        return "Inactive user - send re-engagement campaign"
    if str(g("subscription_type", "")).lower() == "basic" and p >= MED:
        return "Basic plan - offer Standard upgrade trial"
    if g("monthly_fee") >= 15 and p >= MED:
        return "High spend - offer loyalty discount (10-20%)"
    if g("watch_hours", 99) < 10:
        return "Low engagement - personalised content picks"
    if g("number_of_profiles", 9) <= 1:
        return "Single profile - promote family / profile sharing"
    return "General - personalised retention offer"
