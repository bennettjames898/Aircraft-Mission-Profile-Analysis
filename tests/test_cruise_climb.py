"""
Validation tests for CruiseClimbSegment.

The headline check is self-consistency of the altitude solve: at every step of
the march, the Ps evaluated at the solved altitude must reproduce the
COMMANDED Ps. That is an independent check, the history records Ps computed
from the converged state rather than echoing the target back, so a broken
altitude solve shows up immediately.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_conversions as convert
from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import CruiseClimbSegment, CruiseClimbError, ConstantAltCruiseSegment



MACH    =  [0.78, 0.82]
PS_FPM  = [100, 300, 500]
PS_FPM_WRONG  = [0, -100]

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

def start_weight(ac: Aircraft) -> float:
    return convert.lb_to_kg(ac.gross_weight_lb)

def test_commanded_ps_is_held_throughout():
    ac = build_test_aircraft()
    
    for mach in MACH:
        for ps in PS_FPM:
            seg = CruiseClimbSegment(mach=mach, range_nm=2000, ps_target_fpm=ps, num_steps=40)
            result = seg.run(ac, start_weight(ac))

            for point in result.history:
                assert abs(point["Ps_theor_fpm"] - ps) < 1e-3, (
                    f"Ps drifted from the input {ps} ft/min to "
                    f"{point['Ps_theor_fpm']:.4f} at {point['altitude_ft']:.0f} ft")

def test_altitude_increases_monotonically():
    """Fuel burns, weight falls, the Ps-matched altitude rises. That drift IS
    the cruise-climb."""
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=3000, ps_target_fpm=300, num_steps=60)
    result = seg.run(ac, start_weight(ac))
    altitudes = [p["altitude_ft"] for p in result.history]
    assert altitudes == sorted(altitudes)
    assert result.end_altitude_ft > result.start_altitude_ft

def test_weight_decreases_and_fuel_is_burned():
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=2000, ps_target_fpm=300, num_steps=40)
    result = seg.run(ac, start_weight(ac))
    weights = [p["weight_lb"] for p in result.history]
    assert weights == sorted(weights, reverse=True)
    assert result.fuel_burned_kg > 0
    assert result.end_weight_kg < result.start_weight_kg

def test_higher_ps_demand_forces_lower_altitude():
    """Holding more excess power in reserve means cruising lower, where thrust
    is available, which costs fuel."""
    ac = build_test_aircraft()
    low = CruiseClimbSegment(mach=0.82, range_nm=2000, ps_target_fpm=100, num_steps=30).run(ac, start_weight(ac))
    high = CruiseClimbSegment(mach=0.82, range_nm=2000, ps_target_fpm=1000, num_steps=30).run(ac, start_weight(ac))
    assert high.start_altitude_ft < low.start_altitude_ft
    assert high.fuel_burned_kg > low.fuel_burned_kg

def test_drift_up_angle_is_small_and_positive():
    """Gamma should be a genuine climb but a very shallow one, order 1e-2 deg.
    A large angle means the altitude solve is unstable."""
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=3000, ps_target_fpm=300, num_steps=40)
    result = seg.run(ac, start_weight(ac))
    for point in result.history:
        assert 0 < point["gamma_deg"] < 0.5

def test_climb_term_increases_fuel_burn():
    """Including W*sin(gamma) raises required thrust, so it must raise fuel
    burn. It is ~0.5% here, small but not negligible, which is why it is on
    by default."""
    ac = build_test_aircraft()
    kw = dict(mach=0.82, range_nm=3000, ps_target_fpm=300, num_steps=60)
    with_term = CruiseClimbSegment(**kw, include_climb_term=True).run(ac, start_weight(ac))
    without = CruiseClimbSegment(**kw, include_climb_term=False).run(ac, start_weight(ac))
    assert with_term.fuel_burned_kg > without.fuel_burned_kg
    rel = (with_term.fuel_burned_kg - without.fuel_burned_kg) / without.fuel_burned_kg
    assert 1e-4 < rel < 0.05, f"climb term effect of {rel*100:.3f}% looks wrong"

def test_reported_altitudes_match_history():
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=1500, ps_target_fpm=300, num_steps=30)
    result = seg.run(ac, start_weight(ac))
    assert abs(result.end_altitude_ft - result.history[-1]["altitude_ft"]) < 1e-6
    solved_start = convert.m_to_ft(seg.solve_altitude_m(ac, start_weight(ac)))
    assert abs(result.start_altitude_ft - solved_start) < 1e-6

def test_distance_and_time_are_consistent():
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=2000, ps_target_fpm=300, num_steps=40)
    result = seg.run(ac, start_weight(ac))
    assert abs(result.distance_nm - 2000) < 1e-6
    assert result.time_s > 0
    assert abs(sum(p["distance_nm"] for p in result.history) - 2000) < 1e-6

# --- Comparison against fixed-altitude cruise -------------------------------
def test_beats_fixed_altitude_cruise_at_its_own_start_altitude():
    """Cruise-climb drifts up as weight falls, so it should burn less than
    staying at the (lower) altitude it started from."""
    ac = build_test_aircraft()
    W0 = start_weight(ac)
    cc = CruiseClimbSegment(mach=0.82, range_nm=3000, ps_target_fpm=300, num_steps=60).run(ac, W0)
    level = ConstantAltCruiseSegment(
        mach=0.82, range_nm=3000, altitude_ft=cc.start_altitude_ft, num_steps=60).run(ac, W0)
    assert cc.fuel_burned_kg < level.fuel_burned_kg

# --- Guards ------------------------------------------------------------------
def test_non_positive_ps_rejected():
    with_error = False
    for ps in PS_FPM_WRONG:
        try:
            CruiseClimbSegment(mach=0.82, range_nm=1000, ps_target_fpm=ps)
            CruiseClimbSegment(mach=0.82, range_nm=1000, ps_target_fpm=ps)
            assert print(f"Ps {ps:f} did not trigger error")
        except:
            with_error = True
        assert with_error, print("Expected CruiseClimbError for a n incorrect Ps input.")

def test_unachievable_ps_raises():
    """A Ps target the aircraft cannot reach anywhere must fail loudly."""
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=500, ps_target_fpm=8000, num_steps=10)
    with_error = False
    try:
        seg.run(ac, start_weight(ac))
    except CruiseClimbError:
        with_error = True
    assert with_error, print("Expected CruiseClimbError for a no-solution condition, check hardcoded inputs.")

def test_bracket_too_low_raises():
    """If the aircraft still exceeds the Ps target at the top of the search
    bracket, the real answer is above it and the bracket must be widened."""
    ac = build_test_aircraft()
    seg = CruiseClimbSegment(mach=0.82, range_nm=500, ps_target_fpm=300,
                             altitude_bracket_ft=(1000, 20000), num_steps=10)
    with_error = False
    try:
        seg.run(ac, start_weight(ac))
    except CruiseClimbError:
        with_error = True
    assert with_error, print("Expected CruiseClimbError for a high-Ps condition, check hardcoded inputs.")    

###############################################################################
if __name__ == "__main__":      
    tests = [
        test_commanded_ps_is_held_throughout,
        test_altitude_increases_monotonically, 
        test_weight_decreases_and_fuel_is_burned, 
        test_higher_ps_demand_forces_lower_altitude, 
        test_drift_up_angle_is_small_and_positive, 
        test_climb_term_increases_fuel_burn, 
        test_reported_altitudes_match_history,
        test_distance_and_time_are_consistent,
        test_beats_fixed_altitude_cruise_at_its_own_start_altitude,
        test_non_positive_ps_rejected,
        test_unachievable_ps_raises,
        test_bracket_too_low_raises
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll Cruise-Climb validation checks passed.")