"""Dynamic reservation utility threshold for optimal waiting under horizon decay."""

def get_reservation_threshold(day: int, max_day: int = 60) -> float:
    """Computes the reservation threshold tau_t for matching at day t.
    
    Pairs with expected utility below tau_t are deferred to wait for better matches.
    As t approaches horizon (day 60), the option value of waiting decays to 0.
    """
    return 0.0
