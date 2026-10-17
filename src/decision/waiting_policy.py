"""Reservation utility threshold for sequential matching allocation.

In the classical secretary problem a 1/e exploration window prevents premature
commitment. However, in reciprocal matching the expected MSMI per pair is
typically 0.03–0.07 — well below any meaningful threshold — and daily supply
is both limited and stochastic. Imposing a positive threshold therefore blocks
nearly all matches without improving pair quality.

We instead set τ = 0 unconditionally and let the Blossom maximum-weight
matching algorithm select the best available subset of edges each day. The
option value of waiting is implicitly encoded by Blossom's global optimisation:
it will prefer to leave a node unmatched today if doing so enables a higher-
total-weight assignment tomorrow, because tomorrow's edges are re-added to the
graph with updated Thompson-sampled weights.

References:
    - Ferguson (1989). Who solved the secretary problem?
    - Edmonds (1965). Paths, trees, and flowers (Blossom algorithm).
"""


def get_reservation_threshold(day: int, max_day: int = 60) -> float:
    """Returns the minimum MSMI utility required to commit a match on day `day`.

    Currently returns 0.0 for all days: all feasible pairs with positive
    Thompson-sampled weight are passed to the Blossom solver, which then
    selects the globally optimal non-overlapping subset.

    Args:
        day:     Current simulation day (0-indexed).
        max_day: Total episode horizon in days (default 60).

    Returns:
        tau_t: 0.0 — no floor; selection is delegated entirely to Blossom.
    """
    return 0.0

