"""
Trim solver for climbing and descending flight.

Gamma is solved for numerically with scipy.optimize.brentq.
Both solvers below return gamma in radians.
"""

import math
from scipy.optimize import brentq
import unit_conversions as convert
from atmosphere import isa_conditions

class TrimSolverError(RuntimeError):
    """Raised when no valid gamma exists for the given thrust setting 
    (not enough thrust to climb, idle thrust exceeds drag)
    """
    pass


def flight_path_angle_residual(gamma_rad, aircraft, weight_kg, altitude_m, mach, thrust_n, ka):
    """
    Residual of the stability axis force balance:

        f(gamma) = T - D(gamma) - W * sin(gamma) * ka

    f(gamma) = 0 is trimmed. 
    f > 0 is excess thrust at this gamma.
    f < 0 is insufficient thrust.
    
    `ka` is the climb acceleration factor, ka = 1 + (V/g)(dV/dh) = 1.0 for
    quasi-steady climb/descent (constant TAS maneuvers). When the speed 
    schedule calls for TAS to change with altitude (e.g. constant CAS climbs), 
    some of the available excess thrust goes into accelerating the aircraft 
    rather than climbing, setting ka > 1. See 'speed_schedule.py' dtas_dh and
    'segments.py' CommonGammaSegment for where ka is computed.
    """
    
    # Atmosphere conditions
    tas         = convert.mach_to_tas(mach, altitude_m)
    rho         = isa_conditions(altitude_m)["density_kg_m3"]
    q           = 0.5 * rho * tas ** 2

    # Aero data
    weight_n        = weight_kg * convert.G0
    lift_required_n = weight_n * math.cos(gamma_rad)
    cl              = aircraft.aero_model.cl_for_lift(lift_required_n, q, aircraft.wing_area_m2)
    cd              = aircraft.aero_model.get_cd(cl, mach)
    drag_n          = cd * q * aircraft.wing_area_m2
    return thrust_n - drag_n - weight_n * math.sin(gamma_rad) * ka

def solve_climb_gamma(aircraft, weight_kg, altitude_m, mach, thrust_n,
    gamma_min_deg=0.05, gamma_max_deg=25.0, ka=1.0):
    """
    Solve for the climb gamma at which the aircraft is in equilibrium given full thrust.
    """
    args = (aircraft, weight_kg, altitude_m, mach, thrust_n, ka)
    lo = math.radians(gamma_min_deg) # Shallowest climb
    hi = math.radians(gamma_max_deg) # Steepest climb
    
    # Test limits to determine if possible
    f_lo = flight_path_angle_residual(lo, *args)
    f_hi = flight_path_angle_residual(hi, *args)
    if f_lo <= 0: # gamma_min_deg has insufficient thrust
        raise TrimSolverError(
            f"No positive climb angle achievable: thrust does not exceed drag "
            f"near level flight (weight={weight_kg:.0f} kg, altitude={altitude_m:.0f} m, "
            f"mach={mach:.2f}). Residual at gamma~0 is {f_lo:.0f} N."
        )
    if f_hi > 0: # gamma_max_deg has excess thrust 
        # Aircraft could climb even steeper than the bracket allows.
        return hi
    return brentq(flight_path_angle_residual, lo, hi, args=args)


def solve_descent_gamma(aircraft, weight_kg, altitude_m, mach, thrust_n,
    gamma_min_deg=0.05, gamma_max_deg=15.0, ka=1.0):
    """
    Solve for the descent flight-path angle at which the aircraft is in 
    equilibrium given an idle thrust setting. Same residual equation as climb.
    """
    args = (aircraft, weight_kg, altitude_m, mach, thrust_n, ka)
    lo = -math.radians(gamma_max_deg)   # steepest descent
    hi = -math.radians(gamma_min_deg)   # shallowest descent

    f_lo = flight_path_angle_residual(lo, *args)
    f_hi = flight_path_angle_residual(hi, *args)

    if f_hi >= 0:
        raise TrimSolverError(
            f"No descending trim found: thrust exceeds drag at {gamma_min_deg:.0f} "
            f"deg (weight={weight_kg:.0f} kg, "
            f"altitude={altitude_m:.0f} m, mach={mach:.2f})."
        )
    if f_lo < 0:
        # Drag exceeds idle thrust at the steepest angle.
        # Aircraft needs a steeper descent than gamma_max_deg allows.
        return lo
    return brentq(flight_path_angle_residual, lo, hi, args=args)
