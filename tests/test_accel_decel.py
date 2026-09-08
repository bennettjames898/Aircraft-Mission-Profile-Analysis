"""
Validation tests for AccelerationSegment/DecelerationSegment
(CommonAccelSegment in segments.py).
 
Unlike climb/descent, there's no implicit trim solve here -- gamma is
pinned at zero and the along-flight-path force balance is fully
explicit (T - D = m*dV/dt). These tests check basic physical sanity
(Mach moves the right direction, fuel burns, weight decreases), the
asymmetry between max-thrust acceleration and idle-thrust deceleration,
and that AccelDecelError fires in both directions when the requested
speed change genuinely isn't achievable at the commanded thrust.
"""
 
import sys
import os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
import unit_conversions as convert
from segments import AccelDecelSegment
 
def build_test_aircraft(idle_thrust_fraction: float = 0.05) -> Aircraft:
    return Aircraft(
        name="Test Aircraft",
        wing_area_ft2=1320,
        operating_empty_weight_lb=92500,
        payload_weight_lb=33000,
        fuel_weight_lb=40000,
        aero_model=SimpleDragPolar(cd0=0.020, aspect_ratio=9.5, oswald_efficiency=0.80),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_lbf=27000, tsfc_lb_per_lbfhr=0.62,
            num_engines=2, idle_thrust_fraction=idle_thrust_fraction,
        ),
        DISAF=0,
    )
 
def test_acceleration_increases_mach_and_burns_fuel():
    aircraft = build_test_aircraft()
    start_weight_kg = convert.lb_to_kg(aircraft.gross_weight_lb)
    seg = AccelDecelSegment(altitude_ft=35000, start_mach=0.5, end_mach=0.78, num_steps=50)
    result = seg.run(aircraft, start_weight_kg)
 
    assert result.history[0]["mach"] < result.history[-1]["mach"]
    assert abs(result.history[-1]["mach"] - 0.78) < 1e-6
    assert result.fuel_burned_kg > 0
    assert result.end_weight_kg < start_weight_kg
    assert result.time_s > 0
    assert result.distance_nm > 0
 
def test_deceleration_decreases_mach_and_burns_fuel():
    aircraft = build_test_aircraft()
    start_weight_kg = convert.lb_to_kg(aircraft.gross_weight_lb)
    seg = AccelDecelSegment(altitude_ft=35000, start_mach=0.78, end_mach=0.5, num_steps=50)
    result = seg.run(aircraft, start_weight_kg)
 
    assert result.history[0]["mach"] > result.history[-1]["mach"]
    assert abs(result.history[-1]["mach"] - 0.5) < 1e-6
    assert result.fuel_burned_kg > 0  # idle still burns SOME fuel
    assert result.time_s > 0
    assert result.distance_nm > 0
 
def test_deceleration_burns_much_less_fuel_than_acceleration():
    """Idle-thrust deceleration should burn a small fraction of what
    max-thrust acceleration over the same Mach range burns."""
    aircraft = build_test_aircraft()
    start_weight_kg = convert.lb_to_kg(aircraft.gross_weight_lb)
 
    accel = AccelDecelSegment(altitude_ft=35000, start_mach=0.5, end_mach=0.78, num_steps=50)
    decel = AccelDecelSegment(altitude_ft=35000, start_mach=0.78, end_mach=0.5, num_steps=50)
 
    accel_result = accel.run(aircraft, start_weight_kg)
    decel_result = decel.run(aircraft, start_weight_kg)
 
    assert decel_result.fuel_burned_kg < 0.25 * accel_result.fuel_burned_kg, (
        f"Expected idle deceleration fuel burn to be much less than max-thrust "
        f"acceleration: accel={accel_result.fuel_burned_kg:.1f} kg, "
        f"decel={decel_result.fuel_burned_kg:.1f} kg"
    )
 
def test_altitude_held_constant_throughout():
    aircraft = build_test_aircraft()
    start_weight_kg = convert.lb_to_kg(aircraft.gross_weight_lb)
    seg = AccelDecelSegment(altitude_ft=35000, start_mach=0.5, end_mach=0.78, num_steps=50)
    result = seg.run(aircraft, start_weight_kg)
 
    assert result.start_altitude_ft == 35000.0
    assert result.end_altitude_ft == 35000.0
    for pt in result.history:
        assert abs(pt["altitude_ft"] - 35000.0) < 1e-6
 
def test_zero_length_speed_change_raises():
    raised = False
    try:
        AccelDecelSegment(altitude_ft=35000, start_mach=0.7, end_mach=0.7, num_steps=10)
    except ValueError:
        raised = True
    assert raised
 
def test_acceleration_beyond_thrust_ceiling_raises():
    """At a condition where max thrust can't beat drag, acceleration
    should fail loudly rather than return a nonsensical result."""
    aircraft = build_test_aircraft()
    raised = False
    try:
        seg = AccelDecelSegment(altitude_ft=41000, start_mach=0.85, end_mach=0.95, num_steps=30)
        seg.run(aircraft, convert.lb_to_kg(aircraft.gross_weight_lb + 200000))
    except ValueError:
        raised = True
    assert raised
 
 
def test_deceleration_with_excessive_idle_thrust_raises():
    """If idle thrust is high enough to still exceed drag, deceleration
    is physically impossible at idle alone and should fail loudly."""
    aircraft = build_test_aircraft(idle_thrust_fraction=0.5)  # deliberately unrealistic
    raised = False
    try:
        seg = AccelDecelSegment(altitude_ft=1500, start_mach=0.4, end_mach=0.2, num_steps=30)
        seg.run(aircraft, convert.lb_to_kg(aircraft.gross_weight_lb))
    except ValueError:
        raised = True
    assert raised
 
if __name__ == "__main__":
    tests = [
        test_acceleration_increases_mach_and_burns_fuel,
        test_deceleration_decreases_mach_and_burns_fuel,
        test_deceleration_burns_much_less_fuel_than_acceleration,
        test_altitude_held_constant_throughout,
        test_zero_length_speed_change_raises,
        test_acceleration_beyond_thrust_ceiling_raises,
        test_deceleration_with_excessive_idle_thrust_raises,
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print("\nAll accel/decel validation checks passed.")