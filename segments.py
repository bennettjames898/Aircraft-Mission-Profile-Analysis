"""
Mission segment classes.

Each segment's `run()` method returns a SegmentResult and the ending
aircraft weight, so segments can be chained by Mission (see mission.py).

FixedCruiseSegment uses fixed-step numerical integration (RK4) on the
weight-vs-distance ODE:

    dW/dx = -g * TSFC_effective / V   (Breguet's differential form)

which is validated against the closed-form Breguet range equation as a
unit test (see tests/test_breguet_sanity_check.py).
"""

from dataclasses import dataclass, field
from typing import List

import unit_conversions as convert
from aircraft_build import Aircraft

@dataclass
class SegmentResult:
    segment_name: str
    start_weight_kg: float
    end_weight_kg: float
    fuel_burned_kg: float
    distance_m: float
    time_s: float
    # Fine-grained trace for plotting: one entry per integration step
    history: List[dict] = field(default_factory=list)

# Base class. Subclasses implement run().
class MissionSegment:
    name = "generic_segment"
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        raise NotImplementedError

class FixedCruiseSegment(MissionSegment):
    """
    Constant-altitude, constant-Mach cruise for a specified range.
    Integrates the weight/distance ODE using 4th-order Runge-Kutta
    (RK4) for accuracy at coarse step counts.
    """
    name = "cruise"

    def __init__(self, altitude_ft: float, mach: float, range_nm: float, num_steps: int = 100):
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self.mach       = mach
        self.range_m    = convert.nm_to_m(range_nm)
        self.num_steps  = num_steps

    def _dW_dx(self, aircraft: Aircraft, weight_kg: float) -> float:
        """
        dW/dx (kg fuel per meter of range) at a given instantaneous weight.
        """
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
    Constant-altitude, constant-Mach loiter for a
    specified duration. Same ODE as cruise but integrated over time.
    """
    name = "loiter"

    def __init__(self, altitude_ft: float, mach: float, duration_min: float, num_steps: int = 50):
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self.mach       = mach
        self.duration_s = duration_min * 60.0
        self.num_steps  = num_steps

    def _dW_dt(self, aircraft: Aircraft, weight_kg: float) -> float:
        return -aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        dt          = self.duration_s / self.num_steps
        weight_kg   = start_weight_kg
        time_s      = 0.0

        history = [{"time_min": 0.0, "weight_kg": weight_kg}]

        for _ in range(self.num_steps):
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