"""
Mission segment classes.

Each segment's `run()` method returns a SegmentResult and the ending
aircraft weight, so segments can be chained by Mission (see mission.py).
"""

from dataclasses import dataclass, field
from typing import List

import unit_conversions as convert
from aircraft_build import Aircraft
import solver
import speed_schedule

@dataclass
class SegmentResult:
    segment_name:       str
    start_weight_kg:    float
    end_weight_kg:      float
    fuel_burned_kg:     float
    distance_m:         float
    time_s:             float
    # Fine-grained trace for plotting: one entry per integration step
    history: List[dict] = field(default_factory=list)

# Base class. Subclasses implement run().
class MissionSegment:
    name = "generic_segment"
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        raise NotImplementedError

class FixedCruiseSegment(MissionSegment):
    """
    Constant altitude & Mach cruise for a specified range.
    'FixedCruiseSegment' uses stepped numerical integration (RK4) on the
    weight-vs-distance ODE:

        dW/dx = -g * TSFC_effective / V   (Breguet's differential form)
    """
    name = "cruise"

    def __init__(self, altitude_ft: float, mach: float, range_nm: float, num_steps: int = 100):
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self.mach       = mach
        self.range_m    = convert.nm_to_m(range_nm)
        self.num_steps  = num_steps

    # dW/dx (kg fuel per meter of range) at an instantaneous weight.
    def _dW_dx(self, aircraft: Aircraft, weight_kg: float) -> float:
        tas = convert.mach_to_tas(self.mach, self.altitude_m)
        fuel_flow_kg_s = aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)
        # dW/dt = -fuel_flow ; dt/dx = 1/V  =>  dW/dx = -fuel_flow / V
        return -fuel_flow_kg_s / tas

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        dx          = self.range_m / self.num_steps
        weight_kg   = start_weight_kg
        distance_m  = 0.0
        time_s      = 0.0
        tas         = convert.mach_to_tas(self.mach, self.altitude_m)

        history = [{
            "distance_nm":  0.0,
            "weight_kg":    weight_kg,
            "altitude_ft":  convert.m_to_ft(self.altitude_m),
            "mach":         self.mach,
            "l_over_d":     aircraft.lift_to_drag(weight_kg, self.altitude_m, self.mach),
        }]

        for _ in range(self.num_steps):
            # RK4 steps on dW/dx
            k1 = self._dW_dx(aircraft, weight_kg)
            k2 = self._dW_dx(aircraft, weight_kg + 0.5 * dx * k1)
            k3 = self._dW_dx(aircraft, weight_kg + 0.5 * dx * k2)
            k4 = self._dW_dx(aircraft, weight_kg + dx * k3)

            weight_kg   += (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            distance_m  += dx
            time_s      += dx / tas

            history.append({
                "distance_nm":  convert.m_to_nm(distance_m),
                "weight_kg":    weight_kg,
                "altitude_ft":  convert.m_to_ft(self.altitude_m),
                "mach":         self.mach,
                "l_over_d":     aircraft.lift_to_drag(weight_kg, self.altitude_m, self.mach),
            })

        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_m      = distance_m,
            time_s          = time_s,
            history         = history,
        )

class LoiterSegment(MissionSegment):
    """
    Constant altitude & Mach loiter for a specified duration. 
    Same ODE as cruise but integrated over time.
    """
    name = "loiter"

    def __init__(self, altitude_ft: float, mach: float, duration_min: float, num_steps: int = 50):
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self.mach       = mach
        self.duration_s = duration_min * 60.0
        self.num_steps  = num_steps

    # dW/dt (kg fuel per second) at an instantaneous weight.
    def _dW_dt(self, aircraft: Aircraft, weight_kg: float) -> float:
        return -aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        dt          = self.duration_s / self.num_steps
        weight_kg   = start_weight_kg
        time_s      = 0.0

        history = [{"time_min": 0.0, "weight_kg": weight_kg}]

        for _ in range(self.num_steps):
            # RK4 steps over dW/dt
            k1 = self._dW_dt(aircraft, weight_kg)
            k2 = self._dW_dt(aircraft, weight_kg + 0.5 * dt * k1)
            k3 = self._dW_dt(aircraft, weight_kg + 0.5 * dt * k2)
            k4 = self._dW_dt(aircraft, weight_kg + dt * k3)

            weight_kg   += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            time_s      += dt
            history.append({"time_min": time_s / 60.0, "weight_kg": weight_kg})

        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_m      = 0.0,
            time_s=time_s,
            history=history,
        )
    
class CommonGammaSegment(MissionSegment):
    """
    Common implementation for ClimbSegment and DescentSegment. Both 
    integrate over altitude.

    Speed is governed by a SpeedSchedule (speed_schedule.py). Because
    TAS changes with altitude, this segment class applies the climb 
    acceleration correction 'ka', captured by the factor
    1 + (V/g)(dV/dh) in the force balance solved by solver.py.

    At every altitude step, this solves for the trimmed gamma
    given the prescribed thrust setting and 'ka' acceleration, then uses 
    that gamma to compute dt/dh, dx/dh, and dW/dh. All three are integrated 
    together with RK4.
    """

    def __init__(
        self,
        start_altitude_ft:  float,
        end_altitude_ft:    float,
        schedule,
        num_steps:          int     = 50,
        gamma_min_deg:      float   = 0.05,
        gamma_max_deg:      float   = 25,
    ):
        self.start_altitude_m = convert.ft_to_m(start_altitude_ft)
        self.end_altitude_m = convert.ft_to_m(end_altitude_ft)
        self.schedule = speed_schedule.as_schedule(schedule)
        self.num_steps = num_steps
        self.gamma_min_deg = gamma_min_deg
        self.gamma_max_deg = gamma_max_deg

    # Thrust calculation placeholder
    def _thrust_n(self, aircraft: Aircraft, altitude_m: float, mach: float) -> float:
        raise NotImplementedError
    # gamma solver placeholder
    def _solve_gamma(self, aircraft: Aircraft, weight_kg: float, altitude_m: float, mach: float, thrust_n: float, ka: float) -> float:
        raise NotImplementedError

    # Calculates dt/dh, dx/dh, dW/dh, and gamma at a given altitude & weight
    def _derivatives(self, aircraft: Aircraft, altitude_m: float, weight_kg: float) -> dict:
        import math

        mach    = self.schedule.mach_at_altitude(altitude_m)
        tas     = self.schedule.tas_at_altitude(altitude_m)
        dtas_dh = self.schedule.dtas_dh(altitude_m)
        ka      = 1.0 + (tas / convert.G0) * dtas_dh

        thrust_n = self._thrust_n(aircraft, altitude_m, mach)
        gamma_rad = self._solve_gamma(aircraft, weight_kg, altitude_m, mach, thrust_n, ka)

        rate_of_climb = tas * math.sin(gamma_rad)  # dh/dt; negative during descent

        if abs(rate_of_climb) < 1e-6:
            raise solver.TrimSolverError(
                f"Rate of climb/descent numerically ~0 at altitude={altitude_m:.0f} m "
                f"(gamma={math.degrees(gamma_rad):.4f} deg)"
                f"Check gamma_min_deg bracket."
            )

        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, altitude_m, mach)

        return {
            "dt_dh": 1.0 / rate_of_climb,
            "dx_dh": (tas * math.cos(gamma_rad)) / rate_of_climb,
            "dW_dh": -fuel_flow_kg_s / rate_of_climb,
            "gamma_rad": gamma_rad,
            "tas": tas,
            "mach": mach,
            "ka": ka,
        }
    
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        import math

        # Collect initial conditions
        h0 = self.start_altitude_m
        h1 = self.end_altitude_m
        dh = (h1 - h0) / self.num_steps
        altitude_m  = h0
        weight_kg   = start_weight_kg
        time_s      = 0.0
        distance_m  = 0.0
        d0          = self._derivatives(aircraft, altitude_m, weight_kg)
        history     = [{
            "altitude_ft":          convert.m_to_ft(altitude_m),
            "distance_nm":          0.0,
            "weight_kg":            weight_kg,
            "time_min":             0.0,
            "mach":                 d0["mach"],
            "tas_kt":               d0["tas"] / 0.514444,
            "gamma_deg":            math.degrees(d0["gamma_rad"]),
            "rate_of_climb_fpm":    d0["tas"] * math.sin(d0["gamma_rad"]) * 196.850394,
            "ka":                   d0["ka"],
        }]

        # Calculate weight time and distance across the segment
        for _ in range(self.num_steps):
            k1 = self._derivatives(aircraft, altitude_m, weight_kg)
            k2 = self._derivatives(aircraft, altitude_m + 0.5 * dh, weight_kg + 0.5 * dh * k1["dW_dh"])
            k3 = self._derivatives(aircraft, altitude_m + 0.5 * dh, weight_kg + 0.5 * dh * k2["dW_dh"])
            k4 = self._derivatives(aircraft, altitude_m + dh, weight_kg + dh * k3["dW_dh"])

            weight_kg   += (dh / 6.0) * (k1["dW_dh"] + 2 * k2["dW_dh"] + 2 * k3["dW_dh"] + k4["dW_dh"])
            time_s      += (dh / 6.0) * (k1["dt_dh"] + 2 * k2["dt_dh"] + 2 * k3["dt_dh"] + k4["dt_dh"])
            distance_m  += (dh / 6.0) * (k1["dx_dh"] + 2 * k2["dx_dh"] + 2 * k3["dx_dh"] + k4["dx_dh"])
            altitude_m  += dh

            d_end = self._derivatives(aircraft, altitude_m, weight_kg)
            history.append({
                "altitude_ft":          convert.m_to_ft(altitude_m),
                "distance_nm":          convert.m_to_ft(distance_m),
                "weight_kg":            weight_kg,
                "time_min":             time_s / 60.0,
                "mach":                 d_end["mach"],
                "tas_kt":               convert.ms_to_kt(d_end["tas"]),
                "gamma_deg":            math.degrees(d_end["gamma_rad"]),
                "rate_of_climb_fpm":    d_end["tas"] * math.sin(d_end["gamma_rad"]) * 196.850394,
                "ka":                   d_end["ka"],
            })

        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_m      = distance_m,
            time_s          = time_s,
            history         = history,
        )
    
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
        return aircraft.propulsion_model.max_thrust(altitude_m, mach)

    def _solve_gamma(self, aircraft: Aircraft, weight_kg: float, altitude_m: float, mach: float, thrust_n: float, ka: float) -> float:
        return solver.solve_climb_gamma(aircraft, weight_kg, altitude_m, mach, thrust_n,
            gamma_min_deg=self.gamma_min_deg, gamma_max_deg=self.gamma_max_deg, ka=ka)
    
class DescentSegment(CommonGammaSegment):
    """
    Descent at idle thrust from a start to an end altitude following a speed 
    schedule exactly as ClimbSegment does.

    Idle thrust comes from the propulsion model's idle_thrust() method,
    which is a simple fraction of max thrust in propulsion_model.py.
    """
    name = "descent"

    def _thrust_n(self, aircraft: Aircraft, altitude_m: float, mach: float) -> float:
        return aircraft.propulsion_model.idle_thrust(altitude_m, mach)

    def _solve_gamma(self, aircraft: Aircraft, weight_kg: float, altitude_m: float, mach: float, thrust_n: float, ka: float) -> float:
        return solver.solve_descent_gamma(aircraft, weight_kg, altitude_m, mach, thrust_n,
            gamma_min_deg=self.gamma_min_deg, gamma_max_deg=self.gamma_max_deg, ka=ka)