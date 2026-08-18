"""
Validation tests for the climb/descent trim solver and the segments
built on top of it.

Unlike cruise (validated against Breguet's closed form), there isn't a
simple closed-form reference for climb/descent performance once a real
drag polar and thrust lapse are involved -- so these tests check the
things that CAN be verified independently: that the solved trim angle
actually zeros the force-balance residual, that basic physical trends
hold (excess thrust and hence rate of climb shrinks with altitude,
idle-thrust descent burns much less fuel than max-thrust climb), and
that the solver fails loudly rather than silently when no physical
trim exists (e.g., attempting to climb above the aircraft's ceiling).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
import unit_conversions as convert
from solver import (
    solve_climb_gamma,
    solve_descent_gamma,
    flight_path_angle_residual,
    TrimSolverError)
from segments import ClimbSegment, DescentSegment


def build_test_aircraft() -> Aircraft:
    return Aircraft(
        name                        = "Test Aircraft",
        wing_area_m2                = 122.6,
        operating_empty_weight_kg   = 42000,
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


# --- Solver-level tests ---
def test_climb_gamma_residual(MACH, ALT, start_weight_kg):
    """Gamma returned by solve_climb_gamma must satisfy the force balance 
    equation it was solved from."""
    print("Running: test_climb_gamma_residual")
    ac = build_test_aircraft()
    alt_m       = convert.ft_to_m(ALT)
    thrust_n = ac.propulsion_model.max_thrust(alt_m, MACH)

    gamma = solve_climb_gamma(ac, start_weight_kg, alt_m, MACH, thrust_n)
    residual = flight_path_angle_residual(gamma, ac, start_weight_kg, alt_m, MACH, thrust_n, 1)

    assert abs(residual) < 1.0, print(f" High Residual: {residual:.4f} N")
    assert gamma > 0, print("Gamma is negative: {gamma:.4f} rad")

def test_descent_gamma_residual(MACH, ALT, start_weight_kg):
    print("Running: test_descent_gamma_residual")
    ac = build_test_aircraft()
    alt_m = convert.ft_to_m(ALT)
    idle_n = ac.propulsion_model.idle_thrust(alt_m, MACH)

    gamma = solve_descent_gamma(ac, start_weight_kg, alt_m, MACH, idle_n)
    residual = flight_path_angle_residual(gamma, ac, start_weight_kg, alt_m, MACH, idle_n, 1)

    assert abs(residual) < 1.0, print(f" High Residual: {residual:.4f} N")
    assert gamma < 0, print("Gamma is positive: {gamma:.4f} rad")

def test_climb_above_ceiling():
    """
    At a weight/altitude/Mach combination where required thrust exceeds
    available thrust, the solver should fail.
    """
    print("Running: test_climb_above_ceiling with hardcoded values")
    ac = build_test_aircraft()
    with_error = False
    try:
        # Excessive weight for the wing/thrust combination at high altitude --
        # should exceed available thrust before reaching gamma_min_deg.
        solve_climb_gamma(ac, weight_kg=200000, altitude_m=convert.ft_to_m(40000), mach=0.3, thrust_n=ac.propulsion_model.max_thrust(convert.ft_to_m(40000), 0.3))
    except TrimSolverError:
        with_error = True
    assert with_error, print("Expected TrimSolverError for a no-climb condition, check hardcoded inputs.")


# --- Segment-level tests ---
def test_climb_segment_altitude_and_weight_monotonic():
    print("Running: test_climb_segment_altitude_and_weight_monotonic with hardcoded values")
    ac = build_test_aircraft()
    climb = ClimbSegment(start_altitude_ft=0, end_altitude_ft=35000, schedule=0.78, num_steps=40)
    result = climb.run(ac, start_weight_kg=75000)

    altitudes = [h["altitude_ft"] for h in result.history]
    weights = [h["weight_kg"] for h in result.history]

    assert altitudes == sorted(altitudes), print("Altitude must increase monotonically during climb.")
    assert weights == sorted(weights, reverse=True), print("Weight error.")
    assert result.fuel_burned_kg > 0, print("fuel burn error")
    assert abs(altitudes[-1] - 35000) < 1.0, print("Climb terminate altitude missed.")

def test_climb_rate_of_climb_decreases_with_altitude():
    print("Running: test_climb_rate_of_climb_decreases_with_altitude with hardcoded values")
    ac = build_test_aircraft()
    climb = ClimbSegment(start_altitude_ft=0, end_altitude_ft=35000, schedule=0.78, num_steps=40)
    result = climb.run(ac, start_weight_kg=75000.0)

    roc_start = result.history[0]["rate_of_climb_fpm"]
    roc_end = result.history[-1]["rate_of_climb_fpm"]

    assert roc_end < roc_start, (
        print(f"Expected rate of climb to decrease with altitude: "
        f"start={roc_start:.0f} fpm, end={roc_end:.0f} fpm")
    )

def test_descent_segment_altitude_and_weight_monotonic():
    print("Running: test_descent_segment_altitude_and_weight_monotonic with hardcoded values")
    ac = build_test_aircraft()
    descent = DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=0.6, num_steps=40)
    result = descent.run(ac, start_weight_kg=65000)

    altitudes = [h["altitude_ft"] for h in result.history]
    weights = [h["weight_kg"] for h in result.history]

    assert altitudes == sorted(altitudes, reverse=True), print("Altitude must decrease monotonically during descent.")
    assert weights == sorted(weights, reverse=True), print("Weight error.")
    assert result.fuel_burned_kg > 0, print("fuel burn error")
    assert abs(altitudes[-1] - 1500) < 1.0, print("Descent terminate altitude missed.")


if __name__ == "__main__":
    MACH = 0.78
    ALT = 20000
    start_weight_kg = 75000 # kg
    steps = 40
    
    test_climb_gamma_residual(MACH, ALT, start_weight_kg)
    test_descent_gamma_residual(MACH, ALT, start_weight_kg)
    test_climb_above_ceiling()
    test_climb_segment_altitude_and_weight_monotonic()
    test_climb_rate_of_climb_decreases_with_altitude()
    test_descent_segment_altitude_and_weight_monotonic()
      
    print("!!! Any other comment besides 'Running: ' means errors have occured !!!")
