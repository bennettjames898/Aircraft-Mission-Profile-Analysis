"""
This script solves for the minimum fuel load required to complete a mission of
a FIXED range through brentq() to locate the fuel weight at which 0 residual
fuel remains at the end of the mission (or exactly the required reserve).

This is the inverse of solver_mission_range.py: that solver holds fuel fixed
and iterates the cruise range, this one holds the range fixed and iterates the
fuel loaded. Both iterate outside of the larger mission.py context rather than
within a single segment, maintaining the MissionSegment framework where
individual segments do not interact with the mission before or after.

One structural difference worth noting against solver_mission_range.py: the
range solver's free variable lives in the SEGMENT LIST (a cruise segment is
rebuilt at each trial range), so it iterates by calling a build_segments_fn.
Here the free variable is the fuel load, which lives on the AIRCRAFT
(Aircraft derives gross_weight_lb from fuel_weight_lb at construction, and
Mission.run() reads that weight directly). So this solver holds the segment
list fixed and rebuilds a trial AIRCRAFT at each iteration instead. The
segments themselves never change and never need to know fuel is being solved
for.

brentq is used to iterate on the fuel weight, and uses standard scipy inputs
to allocate the search bracket and solution tolerance. Logic is in place to
expand the search bracket if the root (minimum fuel & zero residual) cannot be
found.

Note the residual sign convention here is OPPOSITE to solver_mission_range.py.
More fuel leaves more fuel remaining (residual increases with fuel), whereas
more range leaves less fuel remaining (residual decreases with range). The
bracket expansion logic is flipped to match.
"""

import copy
from typing import List, Tuple

from scipy.optimize import brentq

from aircraft_build import Aircraft
from mission import Mission, MissionResult
from segments import MissionSegment, ConstantAltCruiseSegment


class MissionFuelSizingError(RuntimeError):
    """
    Raised when no valid fuel load exists within the search bracket.
    """
    pass


class MinFuelIteratedResult:
    def __init__(self, fuel_weight_lb: float, mission_result: MissionResult, residual_lb: float, reserve_fuel_lb: float, iterations: int):
        self.fuel_weight_lb = fuel_weight_lb
        self.mission_result = mission_result
        self.residual_lb = residual_lb
        self.reserve_fuel_lb = reserve_fuel_lb
        self.iterations = iterations

        print(
            f"Fuel converged in {self.iterations} iterations "
            f"(residual: {self.residual_lb:.4f} lb)\n"
            f"Iterated to Fuel required = {self.fuel_weight_lb:.1f} [lb]"
            f" (Includes {self.reserve_fuel_lb:.1f} [lb] reserves)"
            )


def _update_ac_fuel(aircraft: Aircraft, fuel_weight_lb: float) -> Aircraft:
    """
    Return a copy of 'aircraft' carrying a different fuel load, with the
    derived gross weight updated to match.

    A copy is used rather than mutating the caller's aircraft so the object
    passed in is never modified as a side effect of solving (same reasoning as
    copying the segment list in solver_mission_range.py's build()). The copy is
    shallow on purpose: aero_model and propulsion_model hold no per-mission
    state, so every trial aircraft can safely share the same model instances.
    """
    trial = copy.copy(aircraft)
    trial.fuel_weight_lb = fuel_weight_lb
    trial.gross_weight_lb = trial.zero_fuel_weight_lb + fuel_weight_lb
    return trial


def solve_min_fuel(
    aircraft: Aircraft,
    MissionSegmentList: List[MissionSegment],
    saveDir: str,
    reserve_fuel_lb: float = 0,
    fuel_bracket_lb: Tuple[float, float] = (0, 100000),
    converge_tol: float = 0.1,
    max_bracket_expansions: int = 10) -> MinFuelIteratedResult:
    """
    Solve for the fuel load at which flying the fixed mission in
    'MissionSegmentList' ends at 'zero_fuel_weight_lb' + 'reserve_fuel_lb'.

    Parameters
    ----------
    MissionSegmentList : list[MissionSegment]
        The COMPLETE mission segment list, fully specified. Unlike
        solver_mission_range.py this list is fixed and is not rebuilt per
        iteration, since the free variable is the fuel on the aircraft.
    reserve_fuel_lb : float
        Fuel that must REMAIN at the end of the mission (holding/diversion
        reserve). The default of 0 solves for burning every usable pound.
        Note this is fuel held in reserve, not fuel burned in a reserve
        segment, if the mission already ends with a reserve loiter segment
        that burn is already counted in the mission itself.
    fuel_bracket_lb : tuple
        Initial (low, high) search bracket. Automatically widened (up to
        max_bracket_expansions doublings of the UPPER bound) if the bracket
        doesn't locate a sign change for brentq.
    converge_tol : float
        Convergence tolerance on fuel weight, passed to brentq.

    Returns
    -------
    MinFuelIteratedResult
        High level stats of iteration attempts required to converge

    Raises
    ------
    MissionFuelSizingError
        If the mission can be completed with no fuel at all (bad mission
        definition), if no fuel load within the expanded bracket is enough to
        complete the mission, or if a trial fuel load produces a flight
        condition the aircraft cannot sustain.
    """
    call_count = 0
    target_end_weight_lb = aircraft.zero_fuel_weight_lb + reserve_fuel_lb

    def residual(fuel_weight_lb: float) -> float:
        nonlocal call_count
        call_count += 1
        trial_aircraft = _update_ac_fuel(aircraft, fuel_weight_lb)
        mission = Mission(aircraft=trial_aircraft, segments=MissionSegmentList, saveDir=None) # do not save or display intermediate runs
        try:
            result = mission.run()
        except RuntimeError as err:
            # Two different causes produce this:
            #   - HIGH trial fuel: AC too heavy to hold the cruise
            #     condition.
            #   - LOW trial fuel: AC runs dry mid-mission and the
            #     segments keep integrating into negative-weight territory,
            #     where required CL becomes nonphysical.
            raise MissionFuelSizingError(
                f"Mission could not be flown at a fuel load of "
                f"{fuel_weight_lb:.0f} lb (gross wt. "
                f"{trial_aircraft.gross_weight_lb:.0f} lb).\n"
                f"If this fuel load is at the LOW end of the search bracket, the "
                f"mission is likely infeasible at any fuel load. If it is at the "
                f"HIGH end, the aircraft is too heavy to sustain the cruise "
                f"condition.\nUnderlying error: {err}"
            ) from err
        # Positive: fuel left over at the end (more than needed was loaded).
        # Negative: mission ran out before finishing (not enough was loaded).
        return result.end_weight_lb - target_end_weight_lb

    lo, hi = fuel_bracket_lb
    f_lo = residual(lo)

    if f_lo > 0:
        raise MissionFuelSizingError(
            f"Mission ends {f_lo:.0f} lb above the target end weight with "
            f"{lo:.0f} lb (minimum) of fuel, so there is no minimum fuel to solve for. "
            f"Increase mission range or reduce 'reserve_fuel_lb'."
        )

    f_hi = residual(hi)
    expansions = 0
    while f_hi < 0 and expansions < max_bracket_expansions:
        # running out of fuel at the upper bound = widen the bracket
        hi *= 2.0
        f_hi = residual(hi)
        expansions += 1

    if f_hi < 0:
        raise MissionFuelSizingError(
            f"Could not bracket a minimum fuel within {hi:.0f} lb after "
            f"{expansions} search bracket expansions. "
            f"The mission range may be greater than the aircraft's capability."
        )

    converged_fuel_lb   = brentq(residual, lo, hi, xtol=converge_tol)
    final_aircraft      = _update_ac_fuel(aircraft, converged_fuel_lb)
    final_mission       = Mission(aircraft=final_aircraft, segments=MissionSegmentList, saveDir=saveDir)
    final_result        = final_mission.run()
    final_residual_lb   = final_result.end_weight_lb - target_end_weight_lb

    return MinFuelIteratedResult(
        fuel_weight_lb  = converged_fuel_lb,
        mission_result  = final_result,
        residual_lb     = final_residual_lb,
        reserve_fuel_lb = reserve_fuel_lb,
        iterations      = call_count,
    )

#----------------------------- WRAPPER FUNC -----------------------------------
    """
    Wrapper that is equivalent to calling 'solve_min_fuel' with a segment list
    that already has a fixed-range ConstantAltCruiseSegment() inserted.
    """
def solve_min_fuel_iterate(
    saveDir:                str,
    aircraft:               Aircraft,
    MissionSegmentList:     List[MissionSegment],
    # IndexToPlaceIteration:  int,
    # cruise_range_nm:        float,
    # cruise_altitude_ft:     float,
    # cruise_mach:            float,
    # cruise_num_steps:       int = 100,
    reserve_fuel_lb:        float = 0,
    fuel_bracket_lb:        Tuple[float, float] = (0, 100000),
    converge_tol:           float = 0.1,
    ) -> MinFuelIteratedResult:

    # Unlike solver_mission_range.py this is built ONCE, not per iteration --
    # the cruise range is the fixed input here, the fuel is what is solved for.
    # CruiseSegment = ConstantAltCruiseSegment(
    #     altitude_ft = cruise_altitude_ft,
    #     mach        = cruise_mach,
    #     range_nm    = cruise_range_nm,
    #     num_steps   = cruise_num_steps,
    # )
    FullMissionSegments = MissionSegmentList.copy()
    # FullMissionSegments.insert(IndexToPlaceIteration, CruiseSegment)

    return solve_min_fuel(
        aircraft,
        FullMissionSegments,
        saveDir,
        reserve_fuel_lb,
        fuel_bracket_lb,
        converge_tol,
        )
