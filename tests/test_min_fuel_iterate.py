"""
Validation tests for solver_mission_fuel.py.

The headline check here is a round trip against solver_mission_range.py: the
two solvers are inverses of each other, so solving for the fuel needed to fly
a fixed range, then feeding that fuel into the range solver, must return the
original range. That is an independent check in the same spirit as validating
cruise against Breguet, neither solver can quietly agree with itself.

Note these tests take no arguments, so pytest can collect and run them
directly ('python -m pytest tests/ -v', the same command CI runs).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import GroundOps, ClimbSegment, DescentSegment, LoiterSegment, ConstantAltCruiseSegment
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from solver_mission_fuel import solve_min_fuel_iterate, MissionFuelSizingError
from solver_mission_range import solve_cruise_range_iterate


MACH = 0.825
CRUISE_ALT_FT = 37000
INSERT_INDEX = 2


def build_test_aircraft(fuel_weight_lb: float = 189760) -> Aircraft:
    return Aircraft(
        name                        = "Test Aircraft",
        wing_area_ft2               = 3501,
        operating_empty_weight_lb   = 239200,
        payload_weight_lb           = 47040,
        fuel_weight_lb              = fuel_weight_lb,
        aero_model=SimpleDragPolar(
            cd0                 = 0.02,
            aspect_ratio        = 10.58,
            oswald_efficiency   = 0.9,
            mach_crit           = 0.86,
        ),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_lbf    = 50066,
            tsfc_lb_per_lbfhr       = 0.5279,
            num_engines             = 2,
            lapse_exponent          = 0.8,
            idle_thrust_fraction    = 0.05,
        ),
        DISAF = 0,
    )


def build_segments():
    climb_sched1   = CASMachSchedule(cas_m_s=convert.kt_to_ms(250), mach=MACH)
    descent_sched1 = CASMachSchedule(cas_m_s=convert.kt_to_ms(277), mach=MACH)
    
    MissionSegs= [
        GroundOps(duration_min=15, throttle_set_pct=0.0),
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=CRUISE_ALT_FT, schedule=climb_sched1, num_steps=50),
        # fixed range cruise segment gets inserted here (INSERT_INDEX = 2)
        DescentSegment(start_altitude_ft=CRUISE_ALT_FT, end_altitude_ft=50, schedule=descent_sched1, num_steps=50),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=20, num_steps=50),
    ]
    return MissionSegs

def solve_fuel_for(range_nm: float, reserve_fuel_lb: float = 0, aircraft: Aircraft = None):
    MissionSegmentList = build_segments()
    cruise = ConstantAltCruiseSegment(altitude_ft=CRUISE_ALT_FT, mach=MACH, range_nm=range_nm, num_steps=300)
    MissionSegmentList.insert(INSERT_INDEX, cruise)
    return solve_min_fuel_iterate(
        saveDir                 = None,
        aircraft                = aircraft if aircraft is not None else build_test_aircraft(),
        MissionSegmentList      = MissionSegmentList,
        reserve_fuel_lb         = reserve_fuel_lb,
    )

def test_solved_mission_lands_at_zero_fuel_weight():
    """The converged mission should end at the zero fuel weight, every
    usable pound loaded is burned."""
    aircraft = build_test_aircraft()
    result = solve_fuel_for(2500)
    assert abs(result.mission_result.end_weight_lb - aircraft.zero_fuel_weight_lb) < 1.0
    assert result.fuel_weight_lb > 0

def test_fuel_loaded_matches_fuel_burned():
    result = solve_fuel_for(2500)
    assert abs(result.fuel_weight_lb - result.mission_result.total_fuel_burned_lb) < 1.0

def test_caller_aircraft_is_not_mutated():
    """The solver builds trial aircraft internally, it must not modify the
    aircraft handed to it (same reasoning as copying the segment list)."""
    aircraft = build_test_aircraft(fuel_weight_lb=189760)
    solve_fuel_for(2000, aircraft=aircraft)
    assert aircraft.fuel_weight_lb == 189760
    assert aircraft.gross_weight_lb == aircraft.zero_fuel_weight_lb + 189760

def test_longer_range_requires_more_fuel():
    short = solve_fuel_for(1500)
    long = solve_fuel_for(3500)
    assert long.fuel_weight_lb > short.fuel_weight_lb

def test_reserve_fuel_increases_required_fuel():
    """Requiring reserve fuel on landing should raise the fuel loaded by MORE
    than the reserve itself, the extra fuel has to be carried the whole way
    (fuel to carry fuel)."""
    no_reserve = solve_fuel_for(2000, reserve_fuel_lb=0)
    with_reserve = solve_fuel_for(2000, reserve_fuel_lb=10000)
    assert with_reserve.fuel_weight_lb > no_reserve.fuel_weight_lb + 10000

def test_reserve_fuel_remains_at_end_of_mission():
    aircraft = build_test_aircraft()
    reserve_lb = 8000
    result = solve_fuel_for(2000, reserve_fuel_lb=reserve_lb)

    fuel_remaining_lb = result.mission_result.end_weight_lb - aircraft.zero_fuel_weight_lb
    assert abs(fuel_remaining_lb - reserve_lb) < 1.0

def test_round_trip_against_range_solver():
    """Solving fuel for a fixed range, then solving range at that fuel, must
    return the original range. The two solvers are inverses."""
    target_range_nm = 3000.0

    fuel_result = solve_fuel_for(target_range_nm)

    range_result = solve_cruise_range_iterate(
        saveDir                 = None,
        aircraft                = build_test_aircraft(fuel_weight_lb=fuel_result.fuel_weight_lb),
        MissionSegmentList      = build_segments(),
        IndexToPlaceIteration   = INSERT_INDEX,
        cruise_altitude_ft      = CRUISE_ALT_FT,
        cruise_mach             = MACH,
        cruise_num_steps        = 150,
    )

    assert abs(range_result.cruise_range_nm - target_range_nm) < 1.0, (
        f"Round trip mismatch: solved {fuel_result.fuel_weight_lb:.1f} lb for "
        f"{target_range_nm:.1f} nm, but that fuel gives "
        f"{range_result.cruise_range_nm:.1f} nm"
    )


def test_infeasible_range_raises():
    """A range far beyond the aircraft's capability should fail loudly rather
    than return a nonsense fuel load."""
    raised = False
    try:
        solve_fuel_for(50000)
    except MissionFuelSizingError:
        raised = True
    assert raised

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_solved_mission_lands_at_zero_fuel_weight,
        test_fuel_loaded_matches_fuel_burned,
        test_caller_aircraft_is_not_mutated,
        test_longer_range_requires_more_fuel,
        test_reserve_fuel_increases_required_fuel,
        test_round_trip_against_range_solver,
        test_round_trip_against_range_solver,
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll Min Fuel Solver validation checks passed.")