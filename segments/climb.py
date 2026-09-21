"""
ClimbSegment: max-thrust climb along a speed schedule.

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

from aircraft_build import Aircraft
import solver_climb_descent

from .common_gamma import CommonGammaSegment

class ClimbSegment(CommonGammaSegment):
    """
    Climb at full thrust from a start to an end altitude, following a speed 
    schedule (see speed_schedule.py). 
    `schedule` accepts either a SpeedScheduleBase instance or a plain float, 
    which is treated as a constant Mach for convenience:

        ClimbSegment(0, 35000, schedule=0.78, ...) # constant Mach
        ClimbSegment(0, 35000, schedule=CASMachSchedule(kt_to_ms(280), 0.78), ...)  # realistic

    Flight-path angle is solved via solve_climb_gamma at every RK4
    evaluation point, including the acceleration correction (ka) when
    the schedule's TAS varies with altitude.
    """
    name = "climb"
    def _thrust_n(self, aircraft: Aircraft, altitude_m: float, mach: float) -> float:
        return aircraft.propulsion_model.max_thrust(altitude_m, mach, aircraft.DISAC)

    def _solve_gamma(self, aircraft: Aircraft, weight_kg: float, altitude_m: float, mach: float, thrust_n: float, ka: float) -> float:
        return solver_climb_descent.solve_climb_gamma(
            aircraft, 
            weight_kg, 
            altitude_m, 
            mach, 
            thrust_n,
            gamma_min_deg=self.gamma_min_deg, 
            gamma_max_deg=self.gamma_max_deg, 
            ka=ka
        )