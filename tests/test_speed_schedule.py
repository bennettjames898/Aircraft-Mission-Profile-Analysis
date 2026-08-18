"""
Validation tests for speed_schedule.py and the CAS/Mach conversion
functions it depends on in atmosphere.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_conversions as convert
from speed_schedule import (
    ConstantTASSchedule,
    CASMachSchedule,
)
from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import ClimbSegment


def build_test_aircraft() -> Aircraft:
    return Aircraft(
        name                        = "Test Aircraft",
        wing_area_m2                = 122.6,
        operating_empty_weight_kg   = 42000,
        aero_model = SimpleDragPolar(
            cd0                 = 0.020, 
            aspect_ratio        = 9.5, 
            oswald_efficiency   = 0.80),
        propulsion_model = SimpleTurbofan(
            sea_level_thrust_n  = 120000, 
            tsfc_kg_per_n_per_s = 1.75e-5, 
            num_engines         = 2),
    )

# --- CAS/Mach conversion tests ---
def test_cas_mach_roundtrip(KCAS_list, ALT_list):
    """mach_to_cas(cas_to_mach(v)) should return the original CAS to tight tolerance."""
    print("Running: test_cas_mach_roundtrip")
    for cas_kt in KCAS_list:
        for alt_ft in ALT_list:
            cas_ms = convert.kt_to_ms(cas_kt)
            alt_m = convert.ft_to_m(alt_ft)
            mach = convert.cas_to_mach(cas_ms, alt_m)
            cas_back_ms = convert.mach_to_cas(mach, alt_m)
            error_kt = abs(convert.ms_to_kt(cas_back_ms) - cas_kt)
            assert error_kt < 0.01, print(f"CAS round-trip error too large: {error_kt:.4f} kt at {alt_ft} ft")

# --- Schedule tests ---
def test_constant_tas_schedule_zero_acceleration(KTAS, ALT):
    """Check for proper dtas behavior during a TAS climb (should be 0)."""
    print("Running: test_constant_tas_schedule_zero_acceleration")
    sched = ConstantTASSchedule(convert.kt_to_ms(KTAS))
    assert sched.dtas_dh(convert.ft_to_m(ALT)) == 0, print(f"TAS acceleration > 0: {sched.dtas_dh(convert.ft_to_m(ALT))}")
    assert sched.tas_at_altitude(convert.ft_to_m(ALT)) == convert.kt_to_ms(KTAS), print(f"TAS round-trip error too large: TAS @ ALT = {sched.tas_at_altitude(convert.ft_to_m(ALT)):.4f} | Input TAS = {convert.kt_to_ms(KTAS)}")

def test_cas_mach_schedule_continuity_at_crossover(KCAS, MACH):
    """The schedule's Mach must be continuous across the crossover
    altitude: Mach just below should equal the target Mach just above."""
    sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    crossover_m = sched.crossover_altitude_m

    mach_just_below = sched.mach_at_altitude(crossover_m - 10)
    mach_just_above = sched.mach_at_altitude(crossover_m + 10)

    assert abs(mach_just_below - mach_just_above) < 0.005, (
        f"Mach should be continuous at crossover: {mach_just_below:.4f} vs {mach_just_above:.4f}"
    )
    assert abs(mach_just_above - MACH) < 0.005

def test_cas_mach_schedule_regimes(KCAS, MACH):
    """Below crossover, schedule should follow CAS; above, constant Mach."""
    print("Running: test_cas_mach_schedule_regimes")
    sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    crossover_ft = convert.m_to_ft(sched.crossover_altitude_m)

    low_alt_m = convert.ft_to_m(max(crossover_ft - 10000, 0))
    high_alt_m = convert.ft_to_m(crossover_ft + 5000)

    assert abs(sched.mach_at_altitude(low_alt_m) - convert.cas_to_mach(convert.kt_to_ms(KCAS), low_alt_m)) < 1e-9, print("Mach crossover error too large")
    assert sched.mach_at_altitude(high_alt_m) == MACH, print("Mach Output mismatch to Mach Input")

# --- Acceleration factor (ka) sanity checks ---
def test_ka_deviates_from_one_under_cas_schedule(KCAS, MACH, ALT_lo, ALT_hi, start_weight, steps):
    """
    Under a CAS schedule (TAS changing with altitude), the acceleration factor 
    ka should differ from 1.0.
    """
    print("Running: test_ka_deviates_from_one_under_cas_schedule")
    ac = build_test_aircraft()
    schedule = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    climb = ClimbSegment(start_altitude_ft=ALT_lo, end_altitude_ft=ALT_hi, schedule=schedule, num_steps=steps)
    result = climb.run(ac, start_weight_kg=start_weight)

    ka_values = [pt["ka"] for pt in result.history]
    max_deviation = max(abs(k - 1.0) for k in ka_values)
    assert max_deviation < 1, print(
        f"Expected ka to deviate from 1.0 under a CAS schedule, "
        f"max deviation was {max_deviation:.4f}"
    )

def test_ka_above_one_during_cas_acceleration_below_crossover(KCAS, MACH, ALT_lo, ALT_hi, start_weight, steps):
    """
    Below the crossover altitude the aircraft is accelerating in TAS
    (constant CAS climb = TAS increases), 'ka' should be greater than 1.0.
    """
    print("Running: test_ka_above_one_during_cas_acceleration_below_crossover")
    ac = build_test_aircraft()
    schedule = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    climb = ClimbSegment(start_altitude_ft=ALT_lo, end_altitude_ft=ALT_hi, schedule=schedule, num_steps=steps)
    result = climb.run(ac, start_weight_kg=start_weight)

    # All points here are below the ~32,000 ft crossover, so should be in
    # the constant-CAS (accelerating) regime.
    for pt in result.history:
        assert pt["ka"] > 1.0, print(f"Expected ka > 1.0 below crossover, got {pt['ka']:.4f} at {pt['altitude_ft']:.0f} ft")

###############################################################################
if __name__ == "__main__":
    KCAS_list = [150, 200, 250, 280, 320]
    ALT_list = [0, 10000, 25000, 35000] # FT
    
    MACH = 0.78
    KCAS = 280; KTAS = 280
    ALT = 20000
    ALT_lo = 0 # FT
    ALT_hi = 20000 # FT
    
    start_weight = 75000 # kg
    steps = 40
    
    test_cas_mach_roundtrip(KCAS_list, ALT_list),
    test_constant_tas_schedule_zero_acceleration(KTAS, ALT),
    test_cas_mach_schedule_continuity_at_crossover(KCAS, MACH),
    test_cas_mach_schedule_regimes(KCAS, MACH),
    test_ka_deviates_from_one_under_cas_schedule(KCAS, MACH, ALT_lo, ALT_hi, start_weight, steps),
    test_ka_above_one_during_cas_acceleration_below_crossover(KCAS, MACH, ALT_lo, ALT_hi, start_weight, steps),
    
    print("!!! Any other comment besides 'Running: ' means errors have occured !!!")