"""
Validation tests for the Ps-ceiling climb termination in CommonGammaSegment.

    end_altitude_ft =  35000   climb to 35,000 ft
    end_altitude_ft =   -300   climb to the 300 ft/min Ps ceiling

The headline check is self consistency: the Ps reported in the LAST history
row, computed from the converged state, must equal the commanded ceiling. The
history is not fed the target, so a broken solve shows up immediately.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_conversions as convert
from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import ClimbSegment
from speed_schedule import CASMachSchedule
from solver_climb_descent import TrimSolverError


PS_FPM =  [100, 300, 500, 1000]
END_ALT_FT = [20000, 35000, 41000]


def build_test_aircraft(fuel_weight_lb: float = 140000) -> Aircraft:
    return Aircraft(
        name                        = "Test Aircraft",
        wing_area_ft2               = 3501,
        operating_empty_weight_lb   = 239200,
        payload_weight_lb           = 47040,
        fuel_weight_lb              = fuel_weight_lb,
        aero_model=SimpleDragPolar(
            cd0=0.02, aspect_ratio=10.58, oswald_efficiency=0.9, mach_crit=0.86),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_lbf=50066, tsfc_lb_per_lbfhr=0.5279,
            num_engines=2, lapse_exponent=0.8, idle_thrust_fraction=0.05),
        DISAF=0,
    )


def build_schedule():
    return CASMachSchedule(cas_m_s=convert.kt_to_ms(280), mach=0.80)


def start_weight(ac: Aircraft) -> float:
    return convert.lb_to_kg(ac.gross_weight_lb)

# --- The commanded ceiling is actually reached ------------------------------
def test_climb_terminates_at_commanded_ps():
    ac = build_test_aircraft()
    for ps in PS_FPM:
        seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-ps,schedule=build_schedule(), num_steps=60)
        result = seg.run(ac, start_weight(ac))

        assert abs(result.history[-1]["Ps_theor_fpm"] - ps) < 1.0, (
            f"climb ended with Ps = {result.history[-1]['Ps_theor_fpm']:.2f} ft/min, "
            f"commanded {PS_FPM}")

def test_solved_end_altitude_is_exposed_and_matches_result():
    ac = build_test_aircraft()
    seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300,schedule=build_schedule(), num_steps=60)
    assert seg.solved_end_altitude_ft is None      # unknown before the run
    result = seg.run(ac, start_weight(ac))
    assert seg.solved_end_altitude_ft is not None
    assert abs(seg.solved_end_altitude_ft - result.end_altitude_ft) < 1e-6

def test_lower_ps_target_gives_higher_ceiling():
    """Demanding less residual climb capability lets the aircraft go higher."""
    ac = build_test_aircraft()
    high_demand = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-1000,schedule=build_schedule(), num_steps=40).run(ac, start_weight(ac))
    low_demand = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-100,schedule=build_schedule(), num_steps=40).run(ac, start_weight(ac))
    assert low_demand.end_altitude_ft > high_demand.end_altitude_ft
    assert low_demand.fuel_burned_kg > high_demand.fuel_burned_kg

def test_heavier_aircraft_gets_a_lower_ceiling():
    light = build_test_aircraft(fuel_weight_lb=60000)
    heavy = build_test_aircraft(fuel_weight_lb=200000)

    r_light = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300, schedule=build_schedule(), num_steps=40).run(light, start_weight(light))
    r_heavy = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300, schedule=build_schedule(), num_steps=40).run(heavy, start_weight(heavy))
    assert r_heavy.end_altitude_ft < r_light.end_altitude_ft

def test_ceiling_exceeds_instantaneous_ceiling_at_start_weight():
    """Fuel burns on the way up, so the aircraft arrives lighter and can climb
    past the ceiling computed at its starting weight. The solved answer must
    therefore be strictly higher than that instantaneous value."""
    ac = build_test_aircraft()
    W0 = start_weight(ac)
    seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300, schedule=build_schedule(), num_steps=60)
    result = seg.run(ac, W0)

    ps_target_ms = convert.fts_to_ms(300 / 60.0)
    instantaneous_ft = convert.m_to_ft(seg._instantaneous_ceiling_m(ac, W0, ps_target_ms))
    assert result.end_altitude_ft > instantaneous_ft

# --- Backward compatibility -------------------------------------------------
def test_positive_end_altitude_unchanged():
    """A normal altitude target must behave exactly as before."""
    ac = build_test_aircraft()
    for alt in END_ALT_FT:
        seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=alt, schedule=build_schedule(), num_steps=60)
        result = seg.run(ac, start_weight(ac))
        assert abs(result.end_altitude_ft - alt) < 1.0
        assert seg.ps_ceiling_fpm is None
        assert seg.solved_end_altitude_ft == alt

def test_sea_level_end_altitude_is_not_treated_as_a_flag():
    """Zero must still mean sea level, only strictly negative is the flag."""
    seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=0,
                       schedule=build_schedule(), num_steps=10)
    assert seg.ps_ceiling_fpm is None
    assert seg.end_altitude_m == 0.0

# --- Physical sanity of the climb itself ------------------------------------
def test_rate_of_climb_and_ps_agree_above_the_tropopause():
    """
    Above the tropopause on a constant Mach schedule, dTAS/dh is zero, so the
    acceleration factor ka is exactly 1 and rate of climb must equal Ps. Two
    independently computed quantities that theory says must coincide, which
    also catches unit errors in either one.
    """
    ac = build_test_aircraft()
    seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300, schedule=build_schedule(), num_steps=60)
    result = seg.run(ac, start_weight(ac))
    top = result.history[-1]
    assert top["altitude_ft"] > 36089, "expected the ceiling above the tropopause here"
    assert abs(top["ka"] - 1.0) < 1e-6
    assert abs(top["rate_of_climb_fpm"] - top["Ps_theor_fpm"]) < 1.0

def test_rate_of_climb_is_physically_sized():
    """Guards the ft/s vs ft/min conversion: a transport climbing from low
    altitude is in the thousands of ft/min, not single digits."""
    ac = build_test_aircraft()
    result = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300, schedule=build_schedule(), num_steps=60).run(ac, start_weight(ac))
    assert 1000 < result.history[0]["rate_of_climb_fpm"] < 10000

def test_altitude_and_weight_monotonic():
    ac = build_test_aircraft()
    result = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-300, schedule=build_schedule(), num_steps=60).run(ac, start_weight(ac))
    altitudes = [p["altitude_ft"] for p in result.history]
    weights = [p["weight_lb"] for p in result.history]
    assert altitudes == sorted(altitudes)
    assert weights == sorted(weights, reverse=True)

# --- Guards -----------------------------------------------------------------
def test_ps_target_already_unreachable_raises():
    """If the aircraft cannot make the requested Ps even at the start
    altitude, there is no climb to fly and it must say so."""
    ac = build_test_aircraft()
    seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-20000, schedule=build_schedule(), num_steps=20)
    raised = False
    try:
        seg.run(ac, start_weight(ac))
    except TrimSolverError:
        raised = True
    assert raised

def test_search_ceiling_too_low_raises():
    ac = build_test_aircraft()
    seg = ClimbSegment(start_altitude_ft=1500, end_altitude_ft=-100, schedule=build_schedule(), num_steps=20, ps_search_ceiling_ft=15000)
    raised = False
    try:
        seg.run(ac, start_weight(ac))
    except TrimSolverError:
        raised = True
    assert raised
        
###############################################################################
if __name__ == "__main__":      
    tests = [
        test_climb_terminates_at_commanded_ps, 
        test_solved_end_altitude_is_exposed_and_matches_result, 
        test_heavier_aircraft_gets_a_lower_ceiling, 
        test_ceiling_exceeds_instantaneous_ceiling_at_start_weight, 
        test_positive_end_altitude_unchanged,
        test_sea_level_end_altitude_is_not_treated_as_a_flag,
        test_rate_of_climb_and_ps_agree_above_the_tropopause,
        test_rate_of_climb_is_physically_sized,
        test_altitude_and_weight_monotonic,
        test_ps_target_already_unreachable_raises,
        test_search_ceiling_too_low_raises
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll Ps Climb Ceiling validation checks passed.")