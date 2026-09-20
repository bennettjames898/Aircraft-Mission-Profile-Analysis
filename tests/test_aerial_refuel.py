"""
Validation tests for AerialRefuelSegment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_conversions as convert
from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from speed_schedule import CASMachSchedule
from mission import Mission
from segments import (AerialRefuelSegment, AerialRefuelError, LoiterSegment,
                      ClimbSegment)

MACH = 0.72
ALT_FT = 25000
QTY_LB = [10000, 30000]
BAD_QTY = [-5000, 0]

def build_test_aircraft(fuel_weight_lb: float = 60000) -> Aircraft:
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

# --- Transfer and burn superpose -------------------------------------------
def test_receiving_weight_change_equals_transfer_minus_burn():
    ac = build_test_aircraft()
    for qty in QTY_LB:
        seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                              fuel_transferred_lb=qty, num_steps=100)
        result = seg.run(ac, start_weight(ac))
        gained_lb = convert.kg_to_lb(result.end_weight_kg - result.start_weight_kg)
        assert abs(gained_lb - (qty - convert.kg_to_lb(result.AR_AC_flight_burn_kg))) < 1.0

def test_donating_weight_change_equals_transfer_plus_burn():
    ac = build_test_aircraft()
    for qty in QTY_LB:
        seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=-2000,
                              fuel_transferred_lb=qty, num_steps=100)
        result = seg.run(ac, start_weight(ac))
        lost_lb = convert.kg_to_lb(result.start_weight_kg - result.end_weight_kg)
        assert abs(lost_lb - (qty + convert.kg_to_lb(result.AR_AC_flight_burn_kg))) < 1.0

def test_receiving_gains_weight_and_donating_loses_it():
    ac = build_test_aircraft()
    rx = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                             fuel_transferred_lb=20000, num_steps=60).run(ac, start_weight(ac))
    tx = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=-2000,
                             fuel_transferred_lb=20000, num_steps=60).run(ac, start_weight(ac))
    assert rx.end_weight_kg > rx.start_weight_kg
    assert tx.end_weight_kg < tx.start_weight_kg

def test_burn_ordering_brackets_a_plain_loiter():
    """
    Over the same duration and condition, the donating aircraft is getting
    lighter so burns less than a plain loiter, and the receiving aircraft is
    getting heavier so burns more. An independent check that the weight/drag
    coupling is actually being integrated.
    """
    ac = build_test_aircraft()
    W0 = start_weight(ac)
    kw = dict(mach=MACH, altitude_ft=ALT_FT, duration_min=15.0, num_steps=100)
    rx = AerialRefuelSegment(transfer_rate_lb_min=2000, **kw); rx.run(ac, W0)
    res_rx = rx.run(ac, W0)
    tx = AerialRefuelSegment(transfer_rate_lb_min=-2000, **kw); tx.run(ac, W0)
    res_tx = tx.run(ac, W0)
    loiter = LoiterSegment(mach=MACH, duration_min=15.0, altitude_ft=ALT_FT, num_steps=100)
    loiter_burn = loiter.run(ac, W0).fuel_burned_kg

    assert res_tx.AR_AC_flight_burn_kg < loiter_burn < res_rx.AR_AC_flight_burn_kg

# --- Specification modes ----------------------------------------------------
def test_quantity_mode_sets_duration():
    seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                              fuel_transferred_lb=30000)
    assert abs(seg.duration_s / 60.0 - 15.0) < 1e-9

def test_duration_mode_transfers_rate_times_time():
    ac = build_test_aircraft()
    seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                              duration_min=12, num_steps=60)
    result = seg.run(ac, start_weight(ac))
    transferred = sum(p["Fuel_transfer_step"] for p in result.history)
    assert abs(transferred - 24000) < 1.0

def test_sign_of_transfer_column_follows_the_rate():
    ac = build_test_aircraft()
    for rate, expect_positive in ((2000, True), (-2000, False)):
        seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=rate,
                                  duration_min=5, num_steps=20)
        result = seg.run(ac, start_weight(ac))
        for p in result.history:
            assert (p["Fuel_transfer_step"] > 0) is expect_positive
            # the genuine burn column is always positive, either way
            assert p["Fuel_burn_lb"] > 0

def test_reported_fuel_burned_kg_is_net_and_negative_when_receiving():
    """Documented gotcha: SegmentResult.fuel_burned_kg is a NET weight change
    for consistency with every other segment, so an onload reports negative."""
    ac = build_test_aircraft()
    seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                              fuel_transferred_lb=20000, num_steps=60)
    result = seg.run(ac, start_weight(ac))
    assert result.fuel_burned_kg < 0
    assert result.AR_AC_flight_burn_kg > 0 # the true burn is still positive
    assert abs(result.fuel_burned_kg - (result.start_weight_kg - result.end_weight_kg)) < 1e-9

def test_altitude_is_held_constant():
    ac = build_test_aircraft()
    seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                              duration_min=10, num_steps=40)
    result = seg.run(ac, start_weight(ac))
    assert abs(result.start_altitude_ft - ALT_FT) < 1e-6
    assert abs(result.end_altitude_ft - ALT_FT) < 1e-6
    for p in result.history:
        assert abs(p["altitude_ft"] - ALT_FT) < 1e-6

# --- Integration with the rest of the tool ---------------------------------
def test_participates_in_altitude_inheritance():
    ac = build_test_aircraft()
    segments = [
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=ALT_FT,
                     schedule=CASMachSchedule(cas_m_s=convert.kt_to_ms(280), mach=MACH),
                     num_steps=40),
        AerialRefuelSegment(mach=MACH, altitude_ft=-1,
                            transfer_rate_lb_min=2000, duration_min=10, num_steps=40),
    ]
    result = Mission(aircraft=ac, segments=segments, saveDir=None).run()
    assert abs(result.segment_results[1].start_altitude_ft - ALT_FT) < 1.0

def test_onload_extends_a_mission():
    """An onload mid-mission must leave the aircraft heavier than the same
    mission without it."""
    ac = build_test_aircraft()
    sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(280), mach=MACH)

    def build(with_ar):
        segs = [ClimbSegment(start_altitude_ft=1500, end_altitude_ft=ALT_FT,
                             schedule=sched, num_steps=30)]
        if with_ar:
            segs.append(AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT,
                                            transfer_rate_lb_min=2000,
                                            fuel_transferred_lb=20000, num_steps=40))
        return Mission(aircraft=ac, segments=segs, saveDir=None).run()
    assert build(True).end_weight_lb > build(False).end_weight_lb

# --- Guards -----------------------------------------------------------------
def test_zero_transfer_rate_rejected():
    with_error = False
    try:
        AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=0,
                            duration_min=10)
    except ValueError:
        with_error = True
    assert with_error, print('Expected ValueError for bad rtransfer rate')

def test_must_specify_exactly_one_of_quantity_or_duration():
    with_error = False
    try:
        AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000,
                            fuel_transferred_lb=10000, duration_min=10)
    except ValueError:
        with_error = True
    assert with_error, print('Expected ValueError for double transfer')
        
    try:
        AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=2000)
    except ValueError:
        with_error = True
    assert with_error, print('Expected ValueError for 0 transfer terms')

def test_quantity_must_be_a_positive_magnitude():
    """Direction is carried by the rate, so the quantity is a magnitude. A
    signed quantity would let the two disagree."""
    with_error = False
    for qty in BAD_QTY:
        try:
            AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=-2000,
                                fuel_transferred_lb=qty)
        except ValueError:
            with_error = True
        assert with_error, print('Expected ValueError for negative fuel qty.')

def test_donation_below_zero_fuel_weight_raises():
    ac = build_test_aircraft(fuel_weight_lb=60000)
    seg = AerialRefuelSegment(mach=MACH, altitude_ft=ALT_FT, transfer_rate_lb_min=-3000,
                              fuel_transferred_lb=80000, num_steps=60)
    with_error = False
    try:
        seg.run(ac, start_weight(ac))
    except AerialRefuelError:
        with_error = True
    assert with_error, print('Expected AerialRefuelError for ZFW')
        
def test_unresolved_inherited_altitude_raises():
    ac = build_test_aircraft()
    seg = AerialRefuelSegment(mach=MACH, altitude_ft=-1,
                              transfer_rate_lb_min=2000, duration_min=10)
    with_error = False
    try:
        seg.run(ac, start_weight(ac))
    except ValueError:
        with_error = True
    assert with_error, print('Expected ValueError for bad altitude')
        
###############################################################################
if __name__ == "__main__":      
    tests = [
        test_receiving_weight_change_equals_transfer_minus_burn, 
        test_donating_weight_change_equals_transfer_plus_burn, 
        test_receiving_gains_weight_and_donating_loses_it, 
        test_burn_ordering_brackets_a_plain_loiter, 
        test_quantity_mode_sets_duration, 
        test_duration_mode_transfers_rate_times_time,
        test_sign_of_transfer_column_follows_the_rate, 
        test_reported_fuel_burned_kg_is_net_and_negative_when_receiving, 
        test_altitude_is_held_constant, 
        test_altitude_is_held_constant, 
        test_participates_in_altitude_inheritance, 
        test_onload_extends_a_mission,
        test_zero_transfer_rate_rejected,
        test_must_specify_exactly_one_of_quantity_or_duration,
        test_quantity_must_be_a_positive_magnitude,
        test_donation_below_zero_fuel_weight_raises,
        test_unresolved_inherited_altitude_raises
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll Air Refueling validation checks passed.")