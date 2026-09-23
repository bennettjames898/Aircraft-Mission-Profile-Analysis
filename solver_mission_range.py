"""
This script solves for the maximum cruise range in a mission through brentq() 
to locate the range at which 0 residual fuel remains in the entire mission.

This tool iterates outside of the larger mission.py context rather than within 
a single segment. This architecture maintains the MissionSegment framework 
where individual segments do not interact with the mission before or after.

brentq is used to iterate on the cruise range, and uses standard scipy inputs 
to allocate the search bracket and solution tolerance. Logic is in place to 
expand the search bracket if the root (max range & zero fuel) cannot be found. 

A wrapper function is included at the bottom of this file and should be used 
to call this functionality into a mission analysis.

    solve_cruise_range_iterate(
        saveDir --------------- Save directory string (use 'None' to not save)
        aircraft -------------- Aircraft class containing Mass/Aero/Prop
        MissionSegmentList ---- Mission segments WITHOUT the cruise segment to be solved
        IndexToPlaceIteration - MissionSegmentList.insert() location to place the iterated cruise segment
        cruise_altitude_ft ---- Cruise altitude for the iterated segment
        cruise_mach ----------- Cruise Mach for the iterated segment
        cruise_num_steps ------ # of analysis steps over the iterated segment
        range_bracket_nm ------ Initial range solution bracket
        converge_tol ---------- Tolerance on range output to consider success
        ) -> MaxRangeIteratedResult:

See 'examples/max_range_iterate_mission.py' for quick reference on application.

===============================================================================
RADIUS MISSIONS
===============================================================================
The same nested-root-find system solves a radius mission, where the aircraft
flies out to a point and then returns to base, requiring both legs to cover the
same ground distance. Segments are tagged with the 'leg' input
("outbound"/"inbound", see MissionLeg in segments/base.py).

    solve_radius_iterate(
        saveDir ------------------------ Save directory string (use 'None' to not save)
        aircraft ----------------------- Aircraft class containing Mass/Aero/Prop
        MissionSegmentList ------------- Mission segments WITHOUT the two cruise
                                         segments to be solved, each already
                                         tagged leg="outbound"/"inbound"
        OutboundIndexToPlaceIteration -- insert() location for the outbound cruise
        InboundIndexToPlaceIteration --- insert() location for the inbound cruise
        cruise_altitude_ft ------------- Cruise altitude for the iterated segments
        cruise_mach -------------------- Cruise Mach for the iterated segments
        ...
        ) -> MaxRadiusIteratedResult

Two things are being solved at once, so there are two nested loops:

    OUTER (brentq on radius)   find the radius at which the mission ends at
                               exactly zero residual fuel -> the MAX radius.
    INNER (_balance_legs)      at a trial radius, find the outbound and inbound
                               cruise ranges that make both legs measure that
                               radius.

The inner loop is not a bracketed root find but a direct correction. Leg
distance is (cruise range) + (climb/accel/descent distance), and that second
term barely moves when the cruise range changes, so subtracting the measured
error from the cruise range converges in two or three passes:

    cruise_range <- cruise_range - (measured_leg_distance - target_radius)

See 'examples/radius_mission.py' for quick reference on application.
"""

import copy
from typing import Callable, List, Tuple

from scipy.optimize import brentq

from aircraft_build import Aircraft
from mission import Mission, MissionResult
from segments import MissionSegment, ConstantAltCruiseSegment, MissionLeg

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
        # Positive: mission ended ABOVE zero-fuel weight
        # Negative: range isn't achievable on the fuel available
        return result.end_weight_lb - aircraft.zero_fuel_weight_lb

    lo, hi = range_bracket_nm
    f_lo = residual(lo)

    if f_lo < 0:
        raise MissionSizingError(
            f"Insufficient fuel to complete the fixed portions of the "
            f"mission: zero cruise range ends {abs(f_lo):.0f} kg below "
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
    """
    Wrapper that is equivalent to calling 'solve_cruise_range' with a
    'build_segments_fn' that inserts a ConstantAltCruiseSegment().
    """
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

#==============================================================================
# RADIUS MISSIONS
#==============================================================================
class RadiusMissionError(MissionSizingError):
    """
    Raised when a radius mission cannot be balanced or sized.
      - the mission definition is not a valid radius mission (legs interleaved,
        or distance credited to an untagged segment),
      - the requested radius is geometrically impossible (the climb, accel and
        descent alone already cover more than the radius, leaving no room for
        cruise),
      - no radius in the search bracket ends the mission at zero fuel.
    """
    def __init__(self, message: str, min_radius_nm: float = None):
        super().__init__(message)
        # Set when the failure was geometric, so the outer solver can lift its
        # lower bracket to something flyable instead of giving up.
        self.min_radius_nm = min_radius_nm

class MaxRadiusIteratedResult:
    def __init__(self, radius_nm: float, outbound_cruise_nm: float, inbound_cruise_nm: float,
                 mission_result: MissionResult, residual_lb: float, iterations: int,
                 leg_imbalance_nm: float):
        self.radius_nm          = radius_nm
        self.outbound_cruise_nm = outbound_cruise_nm
        self.inbound_cruise_nm  = inbound_cruise_nm
        self.mission_result     = mission_result
        self.residual_lb        = residual_lb
        self.iterations         = iterations
        self.leg_imbalance_nm   = leg_imbalance_nm

        print(
            f"Radius converged in {self.iterations} mission evaluations "
            f"(fuel residual: {self.residual_lb:.4f} lb, "
            f"leg imbalance: {self.leg_imbalance_nm:.4f} nm)\n"
            f"Iterated to Mission radius = {self.radius_nm:.1f} [nm] "
            f"(outbound cruise = {self.outbound_cruise_nm:.1f} nm, "
            f"inbound cruise = {self.inbound_cruise_nm:.1f} nm)"
        )

def _check_leg_definition(segments: List[MissionSegment]) -> None:
    """
    Reject a segment list that is not a well-formed radius mission.
      1. Every outbound segment must come before every inbound segment.
      2. Both leg directions must exist.
    """
    legs = [getattr(seg, "leg", MissionLeg.UNASSIGNED) for seg in segments]
    if MissionLeg.OUTBOUND not in legs or MissionLeg.INBOUND not in legs:
        raise RadiusMissionError(
            f"A radius mission needs at least one segment tagged "
            f"leg=\"{MissionLeg.OUTBOUND}\" and one tagged "
            f"leg=\"{MissionLeg.INBOUND}\". Tagged legs found: "
            f"{sorted(set(legs))}."
        )
    last_outbound  = max(i for i, leg in enumerate(legs) if leg == MissionLeg.OUTBOUND)
    first_inbound  = min(i for i, leg in enumerate(legs) if leg == MissionLeg.INBOUND)
    if last_outbound > first_inbound:
        raise RadiusMissionError(
            f"Mission legs are interleaved: segment {last_outbound} "
            f"({segments[last_outbound].name}) is tagged "
            f"\"{MissionLeg.OUTBOUND}\" but flies after segment {first_inbound} "
            f"({segments[first_inbound].name}), which is tagged "
            f"\"{MissionLeg.INBOUND}\". All outbound segments must precede all "
            f"inbound segments."
        )

def _check_distance_accounted(result: MissionResult, tol_nm: float) -> None:
    """
    Reject a radius mission that flies distance on an UNASSIGNED segment.

    Use leg="neutral" to denote segmnets that should be ignored for radius 
    sizing. UNASSIGNED segments are only permitted in range-based missions.
    """
    if result.unassigned_distance_nm > tol_nm:
        offenders = [
            f"{seg.segment_name} ({seg.distance_nm:.1f} nm)"
            for seg in result.segment_results
            if seg.leg == MissionLeg.UNASSIGNED and seg.distance_nm > tol_nm
        ]
        raise RadiusMissionError(
            f"{result.unassigned_distance_nm:.1f} nm of the mission is flown by "
            f"segments that carry no leg tag: {', '.join(offenders)}. "
            f"Tag each segment with leg=\"{MissionLeg.OUTBOUND}\"/\"{MissionLeg.INBOUND}\", "
            f"or leg=\"{MissionLeg.NEUTRAL}\" to state that it deliberately "
            f"counts toward neither."
        )

def _balance_legs(
    aircraft: Aircraft,
    build_segments_fn: Callable[[float, float], List[MissionSegment]],
    radius_nm: float,
    leg_tol_nm: float,
    max_leg_iterations: int,
    extras_guess_nm: Tuple[float, float]) -> dict:
    """
    Find the two cruise ranges that make both legs measure 'radius_nm'.

    Each leg's distance is

        leg_distance = cruise_range + extra

    where 'extra' is everything else on that leg that covers ground (climb,
    accel/decel, descent). 'extra' is only weakly coupled to the cruise range
    (flying a longer cruise arrives at the descent a little lighter, which
    changes the descent distance a little), so this is solved by direct
    correction rather than a bracketed root find:

        cruise_range <- radius - extra_measured_last_pass

    which converges in two or three passes. 'extras_guess_nm' seeds the first
    pass with the extras measured at the previous radius, so the outer loop
    gets progressively cheaper.

    Returns a dict with the converged cruise ranges, the MissionResult, the
    measured extras (to seed the next call) and the evaluation count.
    """
    extra_out, extra_in = extras_guess_nm
    evaluations = 0

    for _ in range(max_leg_iterations):
        outbound_cruise_nm = radius_nm - extra_out
        inbound_cruise_nm  = radius_nm - extra_in

        if outbound_cruise_nm <= 0 or inbound_cruise_nm <= 0:
            # The climb/accel/descent alone already cover the whole radius.
            min_radius_nm = max(extra_out, extra_in)
            raise RadiusMissionError(
                f"A radius of {radius_nm:.1f} nm allows no cruise range. "
                f"The smallest radius this segment list can fly is approx. "
                f"{min_radius_nm:.1f} nm.",
                min_radius_nm = min_radius_nm,
            )

        segments = build_segments_fn(outbound_cruise_nm, inbound_cruise_nm)
        _check_leg_definition(segments)
        evaluations += 1
        try:
            result = Mission(aircraft=aircraft, segments=segments, saveDir=None).run()
        except RuntimeError as err:
            # Same two causes as solver_mission_fuel: too heavy to hold the
            # cruise condition, or the aircraft ran dry mid-mission and the
            # segments integrated into nonphysical negative-weight territory.
            raise RadiusMissionError(
                f"Mission could not be flown at a radius of {radius_nm:.1f} nm "
                f"(outbound cruise {outbound_cruise_nm:.1f} nm, inbound cruise "
                f"{inbound_cruise_nm:.1f} nm).\nIf this radius is at the LOW end "
                f"of the search bracket the mission is likely infeasible at any "
                f"radius. If it is at the HIGH end, the aircraft cannot "
                f"fly this mission on the fuel loaded.\nUnderlying error: {err}"
            ) from err
        _check_distance_accounted(result, leg_tol_nm)

        error_out = result.outbound_distance_nm - radius_nm
        error_in  = result.inbound_distance_nm  - radius_nm

        # Measured non-cruise distance on each leg, used both to correct this
        # pass and to seed the next radius.
        extra_out = result.outbound_distance_nm - outbound_cruise_nm
        extra_in  = result.inbound_distance_nm  - inbound_cruise_nm

        if max(abs(error_out), abs(error_in)) <= leg_tol_nm:
            return {
                "outbound_cruise_nm": outbound_cruise_nm,
                "inbound_cruise_nm":  inbound_cruise_nm,
                "mission_result":     result,
                "extras_nm":          (extra_out, extra_in),
                "evaluations":        evaluations,
            }

    raise RadiusMissionError(
        f"Leg balance DID NOT converge within {max_leg_iterations} passes at a "
        f"radius of {radius_nm:.1f} nm (outbound leg off by {error_out:+.3f} nm, "
        f"inbound leg off by {error_in:+.3f} nm, tolerance {leg_tol_nm} nm). "
        f"Loosen 'leg_tol_nm' or raise 'max_leg_iterations'."
    )

def solve_radius(
    aircraft: Aircraft,
    build_segments_fn: Callable[[float, float], List[MissionSegment]],
    saveDir: str,
    radius_bracket_nm: Tuple[float, float] = (0, 3000),
    converge_tol: float = 0.1,
    leg_tol_nm: float = 0.05,
    max_leg_iterations: int = 25,
    max_bracket_expansions: int = 10) -> MaxRadiusIteratedResult:
    """
    Solve for the maximum radius a mission can fly on the fuel loaded with 
    both legs covering the same ground distance.

    Parameters
    ----------
    build_segments_fn : callable(outbound_cruise_nm, inbound_cruise_nm) -> list[MissionSegment]
        Builds the COMPLETE mission segment list for a trial pair of cruise
        ranges, with every segment already tagged leg="outbound"/"inbound"
        (or "neutral"), e.g.:

            def build(outbound_nm, inbound_nm):
                return [
                    ClimbSegment(..., leg="outbound"),
                    ConstantAltCruiseSegment(range_nm=outbound_nm, ..., leg="outbound"),
                    DescentSegment(..., leg="outbound"),
                    LoiterSegment(...),                     # on station, 0 nm
                    ClimbSegment(..., leg="inbound"),
                    ConstantAltCruiseSegment(range_nm=inbound_nm, ..., leg="inbound"),
                    DescentSegment(..., leg="inbound"),
                ]

    radius_bracket_nm : tuple
        Initial (low, high) search bracket for the mission radius. The low end 
        is increased automatically if it is too small to fit a cruise between 
        the climb and descent (so a lower bound of 0 is fine). The high end is
        widened (up to max_bracket_expansions doublings) if fuel is still left
        over.
    converge_tol : float
        Convergence tolerance on radius (nm), passed to brentq.
    leg_tol_nm : float
        How closely the two legs must match the radius before the inner balance
        loop is considered converged.

    Returns
    -------
    MaxRadiusIteratedResult
        Converged radius, the cruise range flown on each leg, and the final
        mission.

    Raises
    ------
    RadiusMissionError
        If the segment list is not a well-formed radius mission, if the fixed
        portions of the mission already consume more fuel than is available, or
        if a sign change can't be bracketed within max_bracket_expansions
        doublings of the upper bound.
    """
    call_count = 0
    # Carried between radii: the non-cruise (climb/accel/descent) distance on
    # each leg Seeded at 0 "assume cruise covers the whole leg" on the first pass.
    extras_nm = (0.0, 0.0)

    def residual(radius_nm: float) -> float:
        nonlocal call_count, extras_nm
        out = _balance_legs(
            aircraft, build_segments_fn, radius_nm,
            leg_tol_nm, max_leg_iterations, extras_nm)
        call_count += out["evaluations"]
        extras_nm = out["extras_nm"]
        # Positive: mission ended ABOVE zero-fuel weight (fuel left over). 
        # Negative: radius isn't achievable on the fuel available.
        return out["mission_result"].end_weight_lb - aircraft.zero_fuel_weight_lb

    lo, hi = radius_bracket_nm

    # ---- lower bound: lift it until a cruise actually fits on each leg ------
    for _ in range(5):
        try:
            f_lo = residual(lo)
            break
        except RadiusMissionError as err:
            if err.min_radius_nm is None:
                raise
            # The failure reported the smallest radius possible. Step just 
            # above it and try again.
            lo = err.min_radius_nm * 1.02 + 1.0
            if lo >= hi:
                raise RadiusMissionError(
                    f"No flyable radius inside radius_bracket_nm = "
                    f"({radius_bracket_nm[0]:.1f}, {radius_bracket_nm[1]:.1f}) nm: "
                    f"the constant segments alone need at least "
                    f"{err.min_radius_nm:.1f} nm per leg, which is above the "
                    f"top of the bracket. Raise the upper bound.",
                    min_radius_nm = err.min_radius_nm,
                ) from err
    else:
        raise RadiusMissionError(
            f"Could not find a lower bound for the radius search. The "
            f"constant segments cover so much ground that no cruise "
            f"fits on either leg below {lo:.1f} nm."
        )

    if f_lo < 0:
        raise MissionSizingError(
            f"Insufficient fuel to fly any radius mission. The smallest "
            f"flyable radius ({lo:.1f} nm) ends {abs(f_lo):.0f} lb below ZFW."
        )

    # ---- upper bound: widen until the aircraft runs out of fuel -------------
    f_hi = residual(hi)
    expansions = 0
    while f_hi > 0 and expansions < max_bracket_expansions:
        hi *= 2.0
        f_hi = residual(hi)
        expansions += 1

    if f_hi > 0:
        raise RadiusMissionError(
            f"Could not bracket a maximum radius within {hi:.0f} nm after "
            f"{expansions} search bracket expansions. "
            f"Provide a larger radius_bracket_nm, or check for errors in fuel flow "
        )

    converged_radius_nm = brentq(residual, lo, hi, xtol=converge_tol)

    # Re-balance at the converged radius and fly with saving/printing on.
    final = _balance_legs(aircraft, build_segments_fn, converged_radius_nm,
        leg_tol_nm, max_leg_iterations, extras_nm)
    call_count += final["evaluations"]
    final_segments = build_segments_fn(final["outbound_cruise_nm"], final["inbound_cruise_nm"])
    final_result   = Mission(aircraft=aircraft, segments=final_segments, saveDir=saveDir).run()
    final_residual_lb = final_result.end_weight_lb - aircraft.zero_fuel_weight_lb

    return MaxRadiusIteratedResult(
        radius_nm          = converged_radius_nm,
        outbound_cruise_nm = final["outbound_cruise_nm"],
        inbound_cruise_nm  = final["inbound_cruise_nm"],
        mission_result     = final_result,
        residual_lb        = final_residual_lb,
        iterations         = call_count,
        leg_imbalance_nm   = final_result.leg_imbalance_nm,
    )

#----------------------------- WRAPPER FUNC -----------------------------------
def solve_radius_iterate(
    saveDir:                        str,
    aircraft:                       Aircraft,
    MissionSegmentList:             List[MissionSegment],
    OutboundIndexToPlaceIteration:  int,
    InboundIndexToPlaceIteration:   int,
    outbound_cruise_altitude_ft:    float,
    outbound_cruise_mach:           float,
    inbound_cruise_altitude_ft:     float,
    inbound_cruise_mach:            float,
    cruise_num_steps:               int = 100,
    radius_bracket_nm:              Tuple[float, float] = (0, 3000),
    converge_tol:                   float = 0.1,
    leg_tol_nm:                     float = 0.05,
    ) -> MaxRadiusIteratedResult:
    """
    Wrapper that is equivalent to calling 'solve_radius' with a
    'build_segments_fn' that inserts one ConstantAltCruiseSegment into each leg.

    'MissionSegmentList' holds everything EXCEPT those two cruise segments, with
    each of its entries already tagged leg="outbound"/"inbound" (segments that
    cover no ground, such as GroundOps and an on-station LoiterSegment, can be
    left untagged). The two indices are insert() positions in that list, so
    InboundIndexToPlaceIteration must be the larger of the two.

    The inbound cruise altitude/Mach default to the outbound values. Pass them
    explicitly to cruise home higher or faster than the way out.

    The list is copied on every trial, so the segment objects the caller
    built are never re-run (see the note in solve_radius about '-1' altitudes
    latching on first use).
    """
    if InboundIndexToPlaceIteration <= OutboundIndexToPlaceIteration:
        raise RadiusMissionError(
            f"InboundIndexToPlaceIteration ({InboundIndexToPlaceIteration}) must be "
            f"greater than OutboundIndexToPlaceIteration ({OutboundIndexToPlaceIteration})"
        )

    # This def is called by the iteration with various cruise range pairs
    def build(outbound_cruise_nm: float, inbound_cruise_nm: float) -> List[MissionSegment]:
        OutboundSegment = ConstantAltCruiseSegment(
            altitude_ft = outbound_cruise_altitude_ft,
            mach        = outbound_cruise_mach,
            range_nm    = outbound_cruise_nm,
            num_steps   = cruise_num_steps,
            leg         = MissionLeg.OUTBOUND,
        )
        InboundSegment = ConstantAltCruiseSegment(
            altitude_ft = inbound_cruise_altitude_ft,
            mach        = inbound_cruise_mach,
            range_nm    = inbound_cruise_nm,
            num_steps   = cruise_num_steps,
            leg         = MissionLeg.INBOUND,
        )
        # Deep copy so each trial flies fresh segment objects.
        FullMissionSegments = copy.deepcopy(MissionSegmentList)
        # Insert the later index first, so the earlier index stays valid.
        FullMissionSegments.insert(InboundIndexToPlaceIteration, InboundSegment)
        FullMissionSegments.insert(OutboundIndexToPlaceIteration, OutboundSegment)
        return FullMissionSegments

    return solve_radius(
        aircraft, build, saveDir,
        radius_bracket_nm   = radius_bracket_nm,
        converge_tol        = converge_tol,
        leg_tol_nm          = leg_tol_nm,
    )