"""
This script solves for the maximum cruise range in a mission through brentq() 
to locate the range at which 0 residual fuel remains in the entire mission.

This tool iterates outside of the larger mission.py context rather than within 
a single segment. This architecture maintains the MissionSegment framework 
where individual segments do not interact with the mission beofre or after.

brentq is used to iterate on the cruise range, and uses standard scipy inputs 
to allocate the search bracket andsolution tolerance. Logic is in place to 
expand the search bracket if the root (max range & zero fuel) cannot be found. 
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


class MaxRangeIteratedResult:
    def __init__(self, cruise_range_nm: float, mission_result: MissionResult, residual_lb: float, iterations: int):
        self.cruise_range_nm = cruise_range_nm
        self.mission_result = mission_result
        self.residual_lb = residual_lb
        self.iterations = iterations

        print(
            f"Range converged in {self.iterations} iterations "
            f"(residual: {self.residual_lb:.4f} lb)\n"
            f"Iterated to Segment range = {self.cruise_range_nm:.1f} [nm]"
            )

def solve_cruise_range(
    aircraft: Aircraft,
    build_segments_fn: Callable[[float], List[MissionSegment]],
    saveDir: str,
    range_bracket_nm: Tuple[float, float] = (0, 6000),
    converge_tol: float = 0.1,
    max_bracket_expansions: int = 10) -> MaxRangeIteratedResult:
    """
    Solve for the cruise range at which flying the mission
    (built by 'build_segments_fn' at some range) ends at 'zero_fuel_weight_lb'.

    Parameters
    ----------
    build_segments_fn : callable(cruise_range_nm) -> list[MissionSegment]
        Builds the COMPLETE mission segment list for a test cruise range, e.g.:

            def build(range_nm):
                return [climb, CruiseSegment(..., range_nm=range_nm), descent]

        Only one segment entry can scale with the range_nm argument passed in.
    range_bracket_nm : tuple
        Initial (low, high) search bracket. Automatically widened (up to
        max_bracket_expansions doublings) if the upper bound doesn't locate
        a sign change for brentq.
    converge_tol : float
        Convergence tolerance on range, passed to brentq.

    Returns
    -------
    MaxRangeIteratedResult
        High level stats of iteration attempts requried to converge

    Raises
    ------
    MissionSizingError
        If the fixed portions of the mission already consume more
        fuel than is available (no cruise range is flyable), or if a sign 
        change can't be bracketed within max_bracket_expansions doublings 
        of the upper bound.
    """
    call_count = 0

    def residual(cruise_range_nm: float) -> float:
        nonlocal call_count
        call_count += 1
        segments = build_segments_fn(cruise_range_nm)
        mission = Mission(aircraft=aircraft, segments=segments,saveDir=None) # do not save or display intermediate runs
        result = mission.run()
        # Positive: mission ended ABOVE zero-fuel weight (fuel left over,
        # could fly further). Negative: this range isn't achievable on
        # the fuel available (would need to burn more than is loaded).
        return result.end_weight_lb - aircraft.zero_fuel_weight_lb

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

    converged_range_nm  = brentq(residual, lo, hi, xtol=converge_tol)
    final_segments      = build_segments_fn(converged_range_nm)
    final_mission       = Mission(aircraft=aircraft, segments=final_segments, saveDir=saveDir)
    final_result        = final_mission.run()
    final_residual_lb   = final_result.end_weight_lb - aircraft.zero_fuel_weight_lb

    return MaxRangeIteratedResult(
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
    saveDir:                str,
    aircraft:               Aircraft,
    MissionSegmentList:     List[MissionSegment],
    IndexToPlaceIteration:  int,
    cruise_altitude_ft:     float,
    cruise_mach:            float,
    cruise_num_steps:       int = 100,
    range_bracket_nm:       Tuple[float, float] = (0, 6000),
    converge_tol:           float = 0.1,
    ) -> MaxRangeIteratedResult:

    # This def is called by the iteration with various 'range_nm' values
    def build(range_nm: float) -> List[MissionSegment]:
        nonlocal MissionSegmentList
        nonlocal IndexToPlaceIteration
        IterateSegment = ConstantAltCruiseSegment(
            altitude_ft = cruise_altitude_ft,
            mach        = cruise_mach,
            range_nm    = range_nm,
            num_steps   = cruise_num_steps,
        )
        FullMissionSegments = MissionSegmentList.copy()
        FullMissionSegments.insert(IndexToPlaceIteration, IterateSegment)
        return FullMissionSegments
    return solve_cruise_range(aircraft,build,saveDir,range_bracket_nm,converge_tol)