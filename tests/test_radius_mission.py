"""
Validation tests for radius (out-and-back) missions.

Three things are checked, in order of how much they matter:

  1. The DEFINITION. A segment's 'leg' input is normalized and validated at
     construction, and Mission totals each leg's distance.
  2. The SOLVE. solve_radius() has to satisfy two conditions at once - equal
     leg distances and zero residual fuel - so both are verified on the
     converged answer, and the leg distances are re-summed here from the
     segment results rather than read back from the solver's own bookkeeping.
  3. The GUARDS. A mission that isn't a well-formed radius mission (interleaved
     legs, distance on an untagged segment, a radius too small to fit a cruise)
     must fail loudly rather than quietly returning a wrong turn point.

The headline check is test_legs_are_equal_but_cruises_are_not: the two legs
must match to within tolerance while the two CRUISE ranges differ. Equal
cruise ranges would mean the climb and descent distance was ignored, which is
the whole failure mode this feature exists to avoid.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_conversions as convert
from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from mission import Mission
from segments import (
    GroundOps,
    ClimbSegment,
    ConstantAltCruiseSegment,
    DescentSegment,
    LoiterSegment,
    MissionLeg,
)
from solver_mission_range import (
    solve_radius,
    solve_radius_iterate,
    RadiusMissionError,
)

# Coarse step counts: these tests check bookkeeping and convergence, not
# integration accuracy, and solve_radius runs the mission ~20 times.
STEPS       = 25
CRUISE_MACH = 0.80
LEG_TOL_NM  = 0.05

def build_test_aircraft(fuel_weight_lb: float = 140000) -> Aircraft:
    return Aircraft(
        name                        = "Test Radius Aircraft",
        wing_area_ft2               = 3501,
        operating_empty_weight_lb   = 239200,
        payload_weight_lb           = 47040,
        fuel_weight_lb              = fuel_weight_lb,
        aero_model=SimpleDragPolar(
            cd0=0.02, aspect_ratio=10.58, oswald_efficiency=0.9, mach_crit=0.86),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_lbf=50066, tsfc_lb_per_lbfhr=0.5279,
            num_engines=2, lapse_exponent=0.6, idle_thrust_fraction=0.05),
        DISAF=0,
    )

def build_radius_mission(outbound_cruise_nm: float, inbound_cruise_nm: float):
    return [
        GroundOps(duration_min=10, throttle_set_pct=0),
        ClimbSegment(1500, 33000, schedule=CRUISE_MACH, num_steps=STEPS,leg=MissionLeg.OUTBOUND),
        ConstantAltCruiseSegment(mach=CRUISE_MACH, range_nm=outbound_cruise_nm,altitude_ft=-1, num_steps=STEPS,leg=MissionLeg.OUTBOUND),
        DescentSegment(-1, 25000, schedule=CRUISE_MACH, num_steps=STEPS, leg=MissionLeg.OUTBOUND),
        LoiterSegment(mach=0.6, duration_min=20, altitude_ft=33000, num_steps=STEPS),
        ClimbSegment(-1, 37000, schedule=CRUISE_MACH, num_steps=STEPS, leg=MissionLeg.INBOUND),
        ConstantAltCruiseSegment(mach=CRUISE_MACH, range_nm=inbound_cruise_nm, altitude_ft=-1, num_steps=STEPS,  leg=MissionLeg.INBOUND),
        DescentSegment(-1, 1500, schedule=CRUISE_MACH, num_steps=STEPS, leg=MissionLeg.INBOUND),
    ]

def leg_distance_from_results(mission_result, leg: str) -> float:
    """Re-sum a leg's distance from the segment results, independently of
    MissionResult's own leg bookkeeping."""
    return sum(seg.distance_nm for seg in mission_result.segment_results if seg.leg == leg)

#--------------------------- 1. THE DEFINITION --------------------------------
def test_leg_input_is_normalized():
    """Case and whitespace are forgiven; the stored value is canonical."""
    assert ClimbSegment(1500, 20000, schedule=0.7, leg="outbound").leg == MissionLeg.OUTBOUND
    assert ClimbSegment(1500, 20000, schedule=0.7, leg=" Inbound ").leg == MissionLeg.INBOUND
    assert ClimbSegment(1500, 20000, schedule=0.7, leg="NEUTRAL").leg == MissionLeg.NEUTRAL
    # Omitting 'leg' entirely is the one-way default.
    assert ClimbSegment(1500, 20000, schedule=0.7).leg == MissionLeg.UNASSIGNED
    # Every segment type accepts it.
    assert GroundOps(10, 0, leg="outbound").leg == MissionLeg.OUTBOUND
    assert ConstantAltCruiseSegment(0.8, 100, 35000, leg="inbound").leg == MissionLeg.INBOUND
    assert LoiterSegment(0.5, 10, 20000, leg="neutral").leg == MissionLeg.NEUTRAL
    assert DescentSegment(35000, 1500, schedule=0.7, leg="inbound").leg == MissionLeg.INBOUND

def test_bad_leg_input_raises():
    """A near-miss like "out" must be rejected, not silently ignored."""
    for bad in ["out", "return", "", 1, "outbound "*2]:
        with_error = False
        try:
            ClimbSegment(1500, 20000, schedule=0.7, leg=bad)
        except ValueError:
            with_error = True
        assert with_error, print(f"Expected ValueError for leg={bad!r}.")

def test_mission_totals_each_leg():
    """MissionResult's leg totals must match the segment results."""
    ac = build_test_aircraft()
    result = Mission(ac, build_radius_mission(500, 500), None).run()

    assert abs(result.outbound_distance_nm
               - leg_distance_from_results(result, MissionLeg.OUTBOUND)) < 1e-9
    assert abs(result.inbound_distance_nm
               - leg_distance_from_results(result, MissionLeg.INBOUND)) < 1e-9
    # Nothing untagged covers ground here (GroundOps and the loiter fly 0 nm).
    assert result.unassigned_distance_nm < 1e-9
    # Every leg's distance exceeds its cruise range, because climb and descent
    # cover ground too.
    assert result.outbound_distance_nm > 500
    assert result.inbound_distance_nm > 500
    # Equal cruise ranges do NOT give equal legs. This is the error the solver
    # exists to remove, so it had better be present before solving.
    assert result.leg_imbalance_nm > 1.0

def test_untagged_mission_reports_no_legs():
    """A normal one-way mission must be completely unaffected."""
    ac = build_test_aircraft()
    segments = [
        ClimbSegment(1500, 33000, schedule=CRUISE_MACH, num_steps=STEPS),
        ConstantAltCruiseSegment(mach=CRUISE_MACH, range_nm=500, altitude_ft=-1,
                                 num_steps=STEPS),
        DescentSegment(-1, 1500, schedule=CRUISE_MACH, num_steps=STEPS),
    ]
    result = Mission(ac, segments, None).run()
    assert result.outbound_distance_nm == 0.0
    assert result.inbound_distance_nm == 0.0
    assert result.leg_imbalance_nm == 0.0
    assert abs(result.unassigned_distance_nm - result.total_distance_nm) < 1e-9

#------------------------------ 2. THE SOLVE ----------------------------------
def test_legs_are_equal_but_cruises_are_not():
    """
    The headline check. Both legs must measure the converged radius, while the
    two cruise ranges differ, because the lighter inbound leg climbs and
    descends over a different distance.
    """
    ac  = build_test_aircraft()
    out = solve_radius(ac, build_radius_mission, saveDir=None,
                       radius_bracket_nm=(200, 2000), leg_tol_nm=LEG_TOL_NM)
    result = out.mission_result

    outbound_nm = leg_distance_from_results(result, MissionLeg.OUTBOUND)
    inbound_nm  = leg_distance_from_results(result, MissionLeg.INBOUND)

    assert abs(outbound_nm - inbound_nm) <= LEG_TOL_NM, print(
        f"Legs differ by {abs(outbound_nm - inbound_nm):.4f} nm.")
    assert abs(outbound_nm - out.radius_nm) <= LEG_TOL_NM
    assert abs(inbound_nm - out.radius_nm) <= LEG_TOL_NM
    # Different cruise ranges are the proof that climb/descent distance counted.
    assert abs(out.outbound_cruise_nm - out.inbound_cruise_nm) > LEG_TOL_NM
    # And both cruises are shorter than the radius by the climb/descent distance.
    assert out.outbound_cruise_nm < out.radius_nm
    assert out.inbound_cruise_nm < out.radius_nm

def test_max_radius_ends_at_zero_fuel():
    """The converged radius must land with no usable fuel left."""
    ac  = build_test_aircraft()
    out = solve_radius(ac, build_radius_mission, saveDir=None, radius_bracket_nm=(200, 2000), converge_tol=0.1, leg_tol_nm=LEG_TOL_NM)
    # converge_tol is on radius (nm). At roughly 25-30 lb/nm of fuel burn for
    # this aircraft, 0.1 nm of radius is a few lb of fuel.
    assert abs(out.residual_lb) < 25.0, print(
        f"Residual fuel {out.residual_lb:.1f} lb is too large to call converged.")
    assert abs(out.mission_result.end_weight_lb - ac.zero_fuel_weight_lb) < 25.0

def test_larger_radius_runs_out_of_fuel():
    """
    Independent monotonicity check on the converged answer: flying 5% further
    out must end BELOW zero-fuel weight, and 5% shorter must end above it.
    Without this, a solver that converged on the wrong root would still pass
    the residual check above.
    """
    ac  = build_test_aircraft()
    out = solve_radius(ac, build_radius_mission, saveDir=None, radius_bracket_nm=(200, 2000), leg_tol_nm=LEG_TOL_NM)

    scale = 1.05
    too_far = Mission(ac, build_radius_mission(out.outbound_cruise_nm * scale, out.inbound_cruise_nm * scale), None).run()
    too_near = Mission(ac, build_radius_mission(out.outbound_cruise_nm / scale, out.inbound_cruise_nm / scale), None).run()
    assert too_far.end_weight_lb  < ac.zero_fuel_weight_lb
    assert too_near.end_weight_lb > ac.zero_fuel_weight_lb

def test_more_time_on_station_shrinks_the_radius():
    """
    Physical sanity: fuel burned holding at the turn point is fuel not
    available for the legs, so a longer hold must reduce the max radius.
    """
    ac = build_test_aircraft()

    def build_long_hold(outbound_cruise_nm, inbound_cruise_nm):
        segments = build_radius_mission(outbound_cruise_nm, inbound_cruise_nm)
        # Replace the 20 min hold with a 90 min hold, same everything else.
        segments[4] = LoiterSegment(mach=0.6, duration_min=90, altitude_ft=-1, num_steps=STEPS)
        return segments

    baseline  = solve_radius(ac, build_radius_mission, saveDir=None,
                             radius_bracket_nm=(200, 2000), leg_tol_nm=LEG_TOL_NM)
    long_hold = solve_radius(ac, build_long_hold, saveDir=None, radius_bracket_nm=(200, 2000), leg_tol_nm=LEG_TOL_NM)
    assert long_hold.radius_nm < baseline.radius_nm, print(
        f"A 90 min hold gave radius {long_hold.radius_nm:.1f} nm, no smaller than "
        f"the 20 min hold's {baseline.radius_nm:.1f} nm.")

def test_wrapper_matches_explicit_build():
    """
    solve_radius_iterate() inserting the iterated cruise segments must reproduce
    solve_radius() driving a hand-built segment list.
    """
    ac = build_test_aircraft()

    explicit = solve_radius(ac, build_radius_mission, saveDir=None, radius_bracket_nm=(200, 2000), leg_tol_nm=LEG_TOL_NM)

    # Same mission, minus the two cruise segments the wrapper will insert.
    skeleton = [
        GroundOps(duration_min=10, throttle_set_pct=0),
        ClimbSegment(1500, 33000, schedule=CRUISE_MACH, num_steps=STEPS,leg=MissionLeg.OUTBOUND),
        # insert outbound ConstantAltCruiseSegment() at index 2,
        DescentSegment(-1, 25000, schedule=CRUISE_MACH, num_steps=STEPS, leg=MissionLeg.OUTBOUND),
        LoiterSegment(mach=0.6, duration_min=20, altitude_ft=33000, num_steps=STEPS),
        ClimbSegment(-1, 37000, schedule=CRUISE_MACH, num_steps=STEPS, leg=MissionLeg.INBOUND),
        # insert inbound ConstantAltCruiseSegment() at index 5,
        DescentSegment(-1, 1500, schedule=CRUISE_MACH, num_steps=STEPS, leg=MissionLeg.INBOUND),
    ]
    wrapped = solve_radius_iterate(
        saveDir                         = None,
        aircraft                        = ac,
        MissionSegmentList              = skeleton,
        OutboundIndexToPlaceIteration   = 2,    # after the outbound climb
        InboundIndexToPlaceIteration    = 5,    # after the inbound climb
        outbound_cruise_altitude_ft     = -1,   # inherit each climb's end altitude
        outbound_cruise_mach            = CRUISE_MACH,
        inbound_cruise_altitude_ft      = -1,   # inherit each climb's end altitude
        inbound_cruise_mach             = CRUISE_MACH,
        cruise_num_steps                = STEPS,
        radius_bracket_nm               = (200, 2000),
        leg_tol_nm                      = LEG_TOL_NM,
    )
    assert abs(wrapped.radius_nm - explicit.radius_nm) < 1.0, print(
        f"Wrapper radius {wrapped.radius_nm:.2f} nm vs explicit "
        f"{explicit.radius_nm:.2f} nm.")

#------------------------------ 3. THE GUARDS ---------------------------------
def test_interleaved_legs_raise():
    """An outbound segment flown after an inbound one is a mis-ordered list."""
    ac = build_test_aircraft()

    def build_interleaved(outbound_cruise_nm, inbound_cruise_nm):
        segments = build_radius_mission(outbound_cruise_nm, inbound_cruise_nm)
        segments[-1].leg = MissionLeg.OUTBOUND   # last descent mis-tagged
        return segments

    with_error = False
    try:
        solve_radius(ac, build_interleaved, saveDir=None, radius_bracket_nm=(200, 2000))
    except RadiusMissionError:
        with_error = True
    assert with_error, print("Expected RadiusMissionError for interleaved legs.")

def test_untagged_distance_raises():
    """Distance on an untagged segment leaves the radius undefined."""
    ac = build_test_aircraft()

    def build_untagged(outbound_cruise_nm, inbound_cruise_nm):
        segments = build_radius_mission(outbound_cruise_nm, inbound_cruise_nm)
        # Forget to tag the outbound cruise.
        segments[2] = ConstantAltCruiseSegment(
            mach=CRUISE_MACH, range_nm=outbound_cruise_nm, altitude_ft=-1,
            num_steps=STEPS)
        return segments

    with_error = False
    try:
        solve_radius(ac, build_untagged, saveDir=None, radius_bracket_nm=(200, 2000))
    except RadiusMissionError:
        with_error = True
    assert with_error, print("Expected RadiusMissionError for untagged distance.")

def test_neutral_distance_is_allowed():
    """
    leg="neutral" is the deliberate way to fly distance that belongs to neither
    leg. It must not trip the untagged-distance guard, and it must stay out of totals.
    """
    ac = build_test_aircraft()
    segments = build_radius_mission(400, 400)
    segments.append(ConstantAltCruiseSegment(
        mach=0.7, range_nm=150, altitude_ft=-1, num_steps=STEPS,
        leg=MissionLeg.NEUTRAL))

    result = Mission(ac, segments, None).run()
    assert abs(result.neutral_distance_nm - 150) < 1.0
    assert result.unassigned_distance_nm < 1e-9
    assert abs(result.total_distance_nm
               - (result.outbound_distance_nm + result.inbound_distance_nm
                  + result.neutral_distance_nm)) < 1e-9

def test_radius_too_small_for_a_cruise_raises():
    """
    Below some radius the climb and descent alone cover the whole leg and no
    cruise fits.
    """
    ac = build_test_aircraft()
    with_error = False
    try:
        # 20 nm radius against a climb that covers ~90 nm.
        solve_radius(ac, build_radius_mission, saveDir=None,
                     radius_bracket_nm=(20, 25), leg_tol_nm=LEG_TOL_NM)
    except RadiusMissionError as err:
        with_error = True
        # The error has to carry the actionable number: how much radius the
        # climb and descent alone need.
        assert err.min_radius_nm is not None, print(
            f"Error did not report a minimum flyable radius: {err}")
        assert err.min_radius_nm > 25, print(
            f"Reported minimum radius {err.min_radius_nm:.1f} nm should exceed "
            f"the top of the bracket that failed.")
    assert with_error, print(
        "Expected RadiusMissionError when the bracket cannot fit a cruise.")

def test_missing_leg_raises():
    """A radius mission with only one leg tagged is not a radius mission."""
    ac = build_test_aircraft()

    def build_one_leg(outbound_cruise_nm, inbound_cruise_nm):
        segments = build_radius_mission(outbound_cruise_nm, inbound_cruise_nm)
        for seg in segments:
            if seg.leg == MissionLeg.INBOUND:
                seg.leg = MissionLeg.NEUTRAL
        return segments

    with_error = False
    try:
        solve_radius(ac, build_one_leg, saveDir=None, radius_bracket_nm=(200, 2000))
    except RadiusMissionError:
        with_error = True
    assert with_error, print("Expected RadiusMissionError when a leg is missing.")

###############################################################################
if __name__ == "__main__":
    tests = [
        test_leg_input_is_normalized,
        test_bad_leg_input_raises,
        test_mission_totals_each_leg,
        test_untagged_mission_reports_no_legs,
        test_legs_are_equal_but_cruises_are_not,
        test_max_radius_ends_at_zero_fuel,
        test_larger_radius_runs_out_of_fuel,
        test_more_time_on_station_shrinks_the_radius,
        test_wrapper_matches_explicit_build,
        test_interleaved_legs_raise,
        test_untagged_distance_raises,
        test_neutral_distance_is_allowed,
        test_radius_too_small_for_a_cruise_raises,
        test_missing_leg_raises,
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll radius mission validation checks passed.")
