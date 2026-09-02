"""
Mission-level sizing: solving for a free mission parameter such that
the mission satisfies a weight/fuel constraint, by repeatedly running
the FULL mission and root-finding on the result.

This is a different kind of iteration than anything else in the
codebase so far, and it's worth being explicit about why it lives in
its own module rather than as a MissionSegment:

  - atmosphere/aero/propulsion are point calculations -- no iteration.
  - solver.py root-finds a single number (flight-path angle) at a
    single point in space, given a fixed thrust setting and weight.
  - segments.py integrates ONE segment's own physics along its own
    independent variable (distance, time, or altitude), calling
    solver.py at each step. A segment only ever sees its own local
    starting weight; it has no idea what ran before it or after it.
  - mission.py sequences a list of segments and runs it ONCE, carrying
    weight forward -- still no iteration at the mission level.

"Maximum range on the fuel loaded" is a fundamentally different
problem: it requires running the ENTIRE mission repeatedly, adjusting
one segment's parameter (cruise range) each time, until the mission's
ending weight matches a target. A segment's contract is deliberately
narrow (start weight in, end weight out) specifically so segments stay
composable and independently testable -- reaching into that contract to
give one segment awareness of fuel spent before it and required after
it would mean either giving it a back-reference to the whole mission or
duplicating Mission's sequencing logic inside a segment. Keeping this
solver as a separate module preserves that boundary: segments stay
dumb and local, and "solve across the whole mission" logic lives here,
one level up from segments.py the same way segments.py sits one level
up from solver.py. Three nested iterations, three modules:

    solver.py          -- point-in-space force balance (innermost)
    segments.py         -- per-segment RK4 march over one segment's ODE
    mission_sizing.py   -- whole-mission closure (outermost)

A consequence worth relying on rather than working around: every
MissionSegment.run() is stateless with respect to its own instance (it
only reads self.* configuration and returns a fresh SegmentResult) --
none of them mutate self. That means the same climb/descent/reserve
segment objects can be safely reused across many mission re-runs here
without rebuilding them each iteration; only the segment whose
parameter is actually being searched over (cruise range) needs to be
freshly constructed per guess.
"""

from typing import Callable, List, Tuple

from scipy.optimize import brentq

from aircraft_build import Aircraft
from mission import Mission, MissionResult
from segments import MissionSegment, ConstantAltCruiseSegment


class MissionSizingError(RuntimeError):
    """
    Raised when no valid cruise range exists within the search bracket.
    """
    pass


class RangeResult:
    def __init__(self, cruise_range_nm: float, mission_result: MissionResult, residual_lb: float, iterations: int):
        self.cruise_range_nm = cruise_range_nm
        self.mission_result = mission_result
        self.residual_lb = residual_lb
        self.iterations = iterations

    def __repr__(self):
        return (
            f"RangeResult(cruise_range_nm={self.cruise_range_nm:.1f}, "
            f"total_distance_nm={self.mission_result.total_distance_nm:.1f}, "
            f"end_weight_lb={self.mission_result.end_weight_lb:.1f}, "
            f"residual_lb={self.residual_lb:.4f}, iterations={self.iterations})"
        )

def solve_cruise_range(
    aircraft: Aircraft,
    build_segments_fn: Callable[[float], List[MissionSegment]],
    start_weight_lb: float,
    zero_fuel_weight_lb: float,
    range_bracket_nm: Tuple[float, float] = (0, 6000),
    xtol_nm: float = 0.1,
    max_bracket_expansions: int = 10) -> RangeResult:
    """
    Solve for the cruise range at which flying the mission
    (built by 'build_segments_fn' at some range) ends at 'zero_fuel_weight_lb'.

    Parameters
    ----------
    build_segments_fn : callable(cruise_range_nm) -> list[MissionSegment]
        Builds the COMPLETE ordered segment list for a test cruise range, e.g.:

            def build(range_nm):
                return [climb, CruiseSegment(..., range_nm=range_nm), descent]

        Only one segment entry can scale with the range_nm argument passed in.
    start_weight_lb : float
        Fixed takeoff weight (OEW + payload + fuel load). This is NOT
        iterated.
    zero_fuel_weight_lb : float
        OEW + payload. The target ending weight.
    range_bracket_nm : tuple
        Initial (low, high) search bracket. Automatically widened (up to
        max_bracket_expansions doublings) if the upper bound doesn't locate
        a sign change for brentq.
    xtol_nm : float
        Convergence tolerance on range, passed to brentq.

    Returns
    -------
    RangeResult

    Raises
    ------
    MissionSizingError
        If the fixed portions of the mission alone already consume more
        fuel than is available (no cruise range, however short, is
        flyable), or if a sign change can't be bracketed within
        max_bracket_expansions doublings of the upper bound.
    """
    call_count = 0

    def residual(cruise_range_nm: float) -> float:
        nonlocal call_count
        call_count += 1
        segments = build_segments_fn(cruise_range_nm)
        mission = Mission(aircraft=aircraft, segments=segments)
        result = mission.run(start_weight_lb)
        # Positive: mission ended ABOVE zero-fuel weight (fuel left over,
        # could fly further). Negative: this range isn't achievable on
        # the fuel available (would need to burn more than is loaded).
        return result.end_weight_lb - zero_fuel_weight_lb

    lo, hi = range_bracket_nm
    f_lo = residual(lo)

    if f_lo < 0:
        raise MissionSizingError(
            f"Insufficient fuel to complete the fixed (non-cruise) portions of the "
            f"mission at all: even zero cruise range ends {abs(f_lo):.0f} kg below "
            f"zero-fuel weight. Check climb/descent/reserve fuel burn against the "
            f"fuel actually loaded (start_weight_kg - zero_fuel_weight_kg)."
        )

    f_hi = residual(hi)
    expansions = 0
    while f_hi > 0 and expansions < max_bracket_expansions:
        # Aircraft still has fuel to spare even at the upper bound --
        # widen the bracket rather than require the caller to guess an
        # exact-enough upper bound up front.
        hi *= 2.0
        f_hi = residual(hi)
        expansions += 1

    if f_hi > 0:
        raise MissionSizingError(
            f"Could not bracket a maximum range within {hi:.0f} nm after "
            f"{expansions} search bracket expansions. "
            f"Provide a larger range_bracket_nm, or check for errors in fuel flow "
        )

    converged_range_nm  = brentq(residual, lo, hi, xtol=xtol_nm)
    final_segments      = build_segments_fn(converged_range_nm)
    final_mission       = Mission(aircraft=aircraft, segments=final_segments)
    final_result        = final_mission.run(start_weight_lb)
    final_residual_lb   = final_result.end_weight_lb - zero_fuel_weight_lb

    return RangeResult(
        cruise_range_nm = converged_range_nm,
        mission_result  = final_result,
        residual_lb     = final_residual_lb,
        iterations      = call_count,
    )


#----------------------------- WRAPPER FUNC -----------------------------------
    """
    Wrapper that is quivalent to calling 'solve_cruise_range' with a 
    'build_segments_fn' that inserts a ConstantAltCruiseSegment().
    """
def solve_cruise_range_iterate(
    aircraft: Aircraft,
    MissionSegmentList: List[MissionSegment],
    IndexToPlaceIteration: int,
    cruise_altitude_ft: float,
    cruise_mach: float,
    start_weight_lb: float,
    zero_fuel_weight_lb: float,
    cruise_num_steps: int = 100,
    range_bracket_nm: Tuple[float, float] = (0, 6000),
    xtol_nm: float = 0.1,
    ) -> RangeResult:

    # This def is called by the iteration with various 'range_nm' values
    def build(range_nm: float) -> List[MissionSegment]:
        nonlocal MissionSegmentList
        nonlocal IndexToPlaceIteration
        IterateSegment = ConstantAltCruiseSegment(
            altitude_ft=cruise_altitude_ft,
            mach=cruise_mach,
            range_nm=range_nm,
            num_steps=cruise_num_steps,
        )
        FullMissionSegments = MissionSegmentList.copy()
        FullMissionSegments.insert(IndexToPlaceIteration, IterateSegment)
        return FullMissionSegments

    return solve_cruise_range(
        aircraft, build, start_weight_lb, zero_fuel_weight_lb,
        range_bracket_nm, xtol_nm,
    )