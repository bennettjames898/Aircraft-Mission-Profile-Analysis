"""
Validation tests for max_cruise_solver.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import ClimbSegment, DescentSegment, LoiterSegment
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from max_cruise_solver import solve_cruise_range_iterate, MissionSizingError

def build_test_aircraft() -> Aircraft:
    return Aircraft(
        name                        = "Test Aircraft",
        wing_area_ft2               = 1320,
        operating_empty_weight_lb   = 92500,
        aero_model=SimpleDragPolar(
            cd0                 = 0.020, 
            aspect_ratio        = 9.5, 
            oswald_efficiency   = 0.80),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_n  = 120000, 
            tsfc_kg_per_n_per_s = 1.75e-5, 
            num_engines         = 2
        ),
    )

def build_test_AC_weights(aircraft):
    payload_lb  = 33000
    fuel_lb     = 40000
    zero_fuel_weight_lb = aircraft.operating_empty_weight_lb + payload_lb
    start_weight_lb = aircraft.operating_empty_weight_lb + payload_lb + fuel_lb
    return payload_lb, fuel_lb, zero_fuel_weight_lb, start_weight_lb

def build_test_mission(aircraft):
    ### Climb and Descent Schedule Definition
    # follow 280 KCAS until M0.78, then follow M0.78.
    KCAS = 280
    MACH = 0.78
    climb_sched   = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    descent_sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    
    ### Mision Segments
    MissionSegments = [
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=40),
        # iterated ConstantAltCruiseSegment goes here (IndexToPlaceIteration = 1)
        DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=descent_sched, num_steps=40),
    ]
    return MissionSegments

### Regression tests start here
def test_solved_mission_lands_at_zero_fuel_weight():
    """The whole point of the solver: converged mission should end at
    (within a tight tolerance of) the specified zero-fuel weight."""
    aircraft = build_test_aircraft()
    MissionSegments = build_test_mission(aircraft)
    payload_lb, fuel_lb, zero_fuel_weight_lb, start_weight_lb = build_test_AC_weights(aircraft)
    
    result = solve_cruise_range_iterate(
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments,
        IndexToPlaceIteration   = 1,
        cruise_altitude_ft      = 35000,
        cruise_mach             = 0.78,
        start_weight_lb         = start_weight_lb,
        zero_fuel_weight_lb     = zero_fuel_weight_lb,
        cruise_num_steps        = 100,
        range_bracket_nm        = (0.0, 6000.0),
        xtol_nm                 = 0.1,
    )

    assert abs(result.mission_result.end_weight_lb - zero_fuel_weight_lb) < 1.0, (
        f"Expected end weight within 1 lb of ZFW={zero_fuel_weight_lb:.1f}, "
        f"got {result.mission_result.end_weight_lb:.1f}"
    )
    assert result.cruise_range_nm > 0, (
        f"Expected crtuise range > 0 nm, "
        f"got {result.cruise_range_nm:.1f}"
    )

def test_total_fuel_burned_matches_loaded_fuel():
    """Total fuel burned across all segments should equal the fuel
    actually loaded (start_weight - zero_fuel_weight), since the
    aircraft lands exactly at ZFW."""
    aircraft = build_test_aircraft()
    MissionSegments = build_test_mission(aircraft)
    payload_lb, fuel_lb, zero_fuel_weight_lb, start_weight_lb = build_test_AC_weights(aircraft)
    
    result = solve_cruise_range_iterate(
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments,
        IndexToPlaceIteration   = 1,
        cruise_altitude_ft      = 35000,
        cruise_mach             = 0.78,
        start_weight_lb         = start_weight_lb,
        zero_fuel_weight_lb     = zero_fuel_weight_lb,
        cruise_num_steps        = 100,
        range_bracket_nm        = (0.0, 6000.0),
        xtol_nm                 = 0.1,
    )

    assert abs(result.mission_result.total_fuel_burned_lb - fuel_lb) < 1.0

def test_more_fuel_gives_more_range():
    """Basic physical sanity: loading more fuel (same payload, same
    climb/descent) should solve for a longer cruise range."""
    aircraft = build_test_aircraft()
    MissionSegments = build_test_mission(aircraft)
    payload_lb = 15000

    ranges = []
    for fuel_lb in [20000, 30000]:
        climb, descent = build_test_mission(aircraft)  # fresh objects each time
        start_weight_lb = aircraft.operating_empty_weight_lb + payload_lb + fuel_lb
        zero_fuel_weight_lb = aircraft.operating_empty_weight_lb + payload_lb
        
        result = solve_cruise_range_iterate(
            aircraft                = aircraft,
            MissionSegmentList      = MissionSegments,
            IndexToPlaceIteration   = 1,
            cruise_altitude_ft      = 35000,
            cruise_mach             = 0.78,
            start_weight_lb         = start_weight_lb,
            zero_fuel_weight_lb     = zero_fuel_weight_lb,
            cruise_num_steps        = 100,
            range_bracket_nm        = (0.0, 6000.0),
            xtol_nm                 = 0.1,
        )
        
        ranges.append(result.cruise_range_nm)

    assert ranges[1] > ranges[0], f"Expected more fuel to give more range: {ranges}"

def test_insufficient_fuel_raises():
    """If fuel loaded can't cover climb + descent, the solver
    should fail rather than return a nonsensical negative range."""
    aircraft = build_test_aircraft()
    MissionSegments = build_test_mission(aircraft)
    
    fuel_lb = 100 # tiny fuel to test errors
    payload_lb = 15000
    start_weight_lb = aircraft.operating_empty_weight_lb + payload_lb + fuel_lb
    zero_fuel_weight_lb = aircraft.operating_empty_weight_lb + payload_lb

    raised = False
    try:
        solve_cruise_range_iterate(
            aircraft                = aircraft,
            MissionSegmentList      = MissionSegments,
            IndexToPlaceIteration   = 1,
            cruise_altitude_ft      = 35000,
            cruise_mach             = 0.78,
            start_weight_lb         = start_weight_lb,
            zero_fuel_weight_lb     = zero_fuel_weight_lb,
            cruise_num_steps        = 100,
            range_bracket_nm        = (0.0, 6000.0),
            xtol_nm                 = 0.1,
        )
    except MissionSizingError:
        raised = True
    assert raised, "MissionSizingError did NOT trigger."

def test_bracket_auto_expands_from_tiny_initial_guess():
    """Even a deliberately too-small initial bracket should still
    converge, since the solver doubles the upper bound until it finds
    a sign change."""
    aircraft = build_test_aircraft()
    MissionSegments = build_test_mission(aircraft)
    payload_lb, fuel_lb, zero_fuel_weight_lb, start_weight_lb = build_test_AC_weights(aircraft)

    result = solve_cruise_range_iterate(
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments,
        IndexToPlaceIteration   = 1,
        cruise_altitude_ft      = 35000,
        cruise_mach             = 0.78,
        start_weight_lb         = start_weight_lb,
        zero_fuel_weight_lb     = zero_fuel_weight_lb,
        cruise_num_steps        = 100,
        range_bracket_nm        = (0.0, 50), # Check that resizing works
        xtol_nm                 = 0.1,
    )

    assert abs(result.mission_result.end_weight_lb - zero_fuel_weight_lb) < 1.0
    assert result.cruise_range_nm > 1000  # sanity check to see a real answer is found

def test_reserve_segment_reduces_max_range():
    """Adding a fixed reserve loiter should reduce the solved cruise
    range relative to the same mission without a reserve, since some of
    the fixed fuel budget is now spent holding rather than cruising."""
    aircraft = build_test_aircraft()
    payload_lb, fuel_lb, zero_fuel_weight_lb, start_weight_lb = build_test_AC_weights(aircraft)

    ### Climb and Descent Schedule Definition
    KCAS = 280
    MACH = 0.78
    climb_sched   = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    descent_sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    
    ### Mision Segments with no reserve
    MissionSegments1 = [
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=40),
        DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=descent_sched, num_steps=40),
    ]    
    no_reserve = solve_cruise_range_iterate(
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments1,
        IndexToPlaceIteration   = 1,
        cruise_altitude_ft      = 35000,
        cruise_mach             = 0.78,
        start_weight_lb         = start_weight_lb,
        zero_fuel_weight_lb     = zero_fuel_weight_lb,
        cruise_num_steps        = 100,
        range_bracket_nm        = (0.0, 50), # Check that resizing works
        xtol_nm                 = 0.1,
    )

    ### Mision Segments with reserve
    MissionSegments2 = [
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=40),
        DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=descent_sched, num_steps=40),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=30.0, num_steps=30), # Add loiter "reserve"
    ]
    with_reserve = solve_cruise_range_iterate(
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments2,
        IndexToPlaceIteration   = 1,
        cruise_altitude_ft      = 35000,
        cruise_mach             = 0.78,
        start_weight_lb         = start_weight_lb,
        zero_fuel_weight_lb     = zero_fuel_weight_lb,
        cruise_num_steps        = 100,
        range_bracket_nm        = (0.0, 50), # Check that resizing works
        xtol_nm                 = 0.1,
    )
    assert with_reserve.cruise_range_nm < no_reserve.cruise_range_nm

if __name__ == "__main__":
    tests = [
        test_solved_mission_lands_at_zero_fuel_weight,
        test_total_fuel_burned_matches_loaded_fuel,
        test_more_fuel_gives_more_range,
        test_insufficient_fuel_raises,
        test_bracket_auto_expands_from_tiny_initial_guess,
        test_reserve_segment_reduces_max_range,
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll mission sizing validation checks passed.")