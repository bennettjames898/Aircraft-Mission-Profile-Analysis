"""
DescentSegment: idle-thrust descent along a speed schedule.

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

from aircraft_build import Aircraft
import solver_climb_descent

from .common_gamma import CommonGammaSegment

class DescentSegment(CommonGammaSegment):
    """
    Descent at idle thrust from a start to an end altitude following a speed 
    schedule exactly as ClimbSegment does.

    Idle thrust comes from the 'propulsion_model.py' idle_thrust() method,
    which is currently a simple fraction of max thrust.
    """
    name = "descent"
    def _thrust_n(self, aircraft: Aircraft, altitude_m: float, mach: float) -> float:
        return aircraft.propulsion_model.idle_thrust(altitude_m, mach, aircraft.DISAC)

    def _solve_gamma(self, aircraft: Aircraft, weight_kg: float, altitude_m: float, mach: float, thrust_n: float, ka: float) -> float:
        return solver_climb_descent.solve_descent_gamma(
            aircraft, 
            weight_kg, 
            altitude_m, 
            mach, 
            thrust_n,
            gamma_min_deg=self.gamma_min_deg, 
            gamma_max_deg=self.gamma_max_deg, 
            ka=ka
        )