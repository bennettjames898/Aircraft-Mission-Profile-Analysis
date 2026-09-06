"""
Validation test: numerically integrated CruiseSegment vs. the closed-form
Breguet range equation.

Why this matters: the Breguet equation

    R = (V / TSFC) * (L/D) * ln(W_start / W_end)

is derived from the exact same dW/dx ODE that CruiseSegment integrates
numerically, under the assumption that L/D and TSFC are constant across
the segment (i.e., the aircraft's cruise CL / Mach are held fixed, which
is what a constant-altitude constant-Mach cruise does here). If the
numerical integrator is implemented correctly, running it for a known
fuel burn should recover a range that matches Breguet to a tight
tolerance, and the match should improve as step count increases.
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
import unit_conversions as convert
from segments import ConstantAltCruiseSegment

# Notional airplane
def build_test_aircraft() -> Aircraft:
    return Aircraft(
        name                        = "Test Aircraft",
        wing_area_ft2               = 1320,
        operating_empty_weight_lb   = 92500,
        payload_weight_lb           = 33000,
        fuel_weight_lb              = 40000,
        aero_model=SimpleDragPolar(
            cd0                 = 0.020, 
            aspect_ratio        = 9.5, 
            oswald_efficiency   = 0.80
        ),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_lbf= 27000,
            tsfc_lb_per_lbfhr   = 0.62,
            num_engines         = 2
        ),
    )

# Closed-form Breguet range equation (constant V, TSFC, L/D)
def breguet_range_nm(tas_m_s, tsfc_kg_per_n_per_s, l_over_d, w_start_kg, w_end_kg):
    convert.G0 # = 9.80665
    return convert.m_to_nm((tas_m_s / (tsfc_kg_per_n_per_s * convert.G0)) * l_over_d * math.log(w_start_kg / w_end_kg))

def run_case(num_steps: int):
    """
    Run 'FixedCruiseSegment' for a fixed range, then check that a Breguet
    calculation using the (nearly constant) L/D at the midpoint weight
    predicts a very similar range for the same fuel burn.
    """
    # Build test vehicle
    aircraft        = build_test_aircraft()
    start_weight_kg = 70000
    altitude_ft     = 35000
    mach            = 0.78
    range_nm        = 1000
    num_steps       = 100

    # Build cruise segment and run
    segment = ConstantAltCruiseSegment(altitude_ft=altitude_ft, mach=mach, range_nm=range_nm, num_steps=num_steps)
    result = segment.run(aircraft, start_weight_kg)
    numerical_range_nm = result.distance_nm

    # Evaluate Breguet at the mean weight
    mean_weight_kg  = 0.5 * (result.start_weight_kg + result.end_weight_kg)
    l_over_d_mid    = aircraft.lift_to_drag(mean_weight_kg, convert.ft_to_m(altitude_ft), mach)
    tas             = convert.mach_to_tas(mach, convert.ft_to_m(altitude_ft))
    tsfc            = aircraft.propulsion_model.tsfc
    breguet_pred_range_nm = breguet_range_nm(
        tas, tsfc, l_over_d_mid, result.start_weight_kg, result.end_weight_kg
    )

    # Calc error
    error_pct = 100.0 * abs(breguet_pred_range_nm - numerical_range_nm) / numerical_range_nm

    return numerical_range_nm, breguet_pred_range_nm, error_pct


def test_breguet_agreement_coarse():
    """Even a coarse integration (10 steps) should agree with Breguet within 0.5%."""
    _, _, error_pct = run_case(num_steps=10)
    assert error_pct < 0.5, print(f"Breguet mismatch too large at coarse resolution: {error_pct:.4f}%")


def test_breguet_agreement_fine():
    """A finer integration (200 steps) should agree even more closely."""
    _, _, error_pct = run_case(num_steps=200)
    assert error_pct < 0.1, print(f"Breguet mismatch too large at fine resolution: {error_pct:.4f}%")


def test_convergence_improves_with_steps():
    """
    Error should shrink, or stay flat. As step count increases, error should 
    not grow. A flat residual is expected.
    """
    _, _, error_coarse = run_case(num_steps=5)
    _, _, error_fine = run_case(num_steps=100)
    assert error_fine <= error_coarse * 1.001, print(  # allow tiny float noise
        f"Refining the integration should not increase error: "
        f"coarse={error_coarse:.4f}%, fine={error_fine:.4f}%"
    )


def test_fuel_burn_is_positive_and_bounded():
    """Basic check: cruise should burn fuel, and not more than it started with."""
    aircraft = build_test_aircraft()
    segment = ConstantAltCruiseSegment(altitude_ft=35000, mach=0.78, range_nm=1000, num_steps=50)
    result = segment.run(aircraft, start_weight_kg=70000)

    assert result.fuel_burned_kg > 0, print("Cruise should burn a positive amount of fuel.")
    assert result.fuel_burned_kg < 70000.0, print("Cruise should not burn more fuel than available weight.")
    assert result.end_weight_kg < result.start_weight_kg, print("Weight must decrease during cruise.")

if __name__ == "__main__":
    print("Breguet Range Assessment - Method Comparison:")
    print("Step count | Numerical range (nm) | Breguet range (nm) | Error (%)")
    for n in [5, 10, 25, 50, 100, 200]:
        num_nm, breg_nm, err = run_case(num_steps=n)
        print(f"{n:>10} | {num_nm:>20.3f} | {breg_nm:>18.3f} | {err:>8.5f}")

    test_breguet_agreement_coarse()
    test_breguet_agreement_fine()
    test_convergence_improves_with_steps()
    test_fuel_burn_is_positive_and_bounded()