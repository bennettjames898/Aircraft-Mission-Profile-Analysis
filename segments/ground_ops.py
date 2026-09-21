"""
GroundOps: fixed-duration fuel burn on the ground (taxi, APU, run-up).

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

import unit_conversions as convert
from aircraft_build import Aircraft

from .base import MissionSegment, SegmentResult, _parse_leg_input

class GroundOps(MissionSegment):
    """
    Simple mission segment to input fuel burned on the ground for a specified 
    time. 
    """
    name = "groundOps"
    def __init__(self, duration_min: float, throttle_set_pct: float, leg: str = None):
        self.leg = _parse_leg_input(leg, type(self).__name__)
        self.duration_s = duration_min * 60
        self.throttle_set_pct = throttle_set_pct
        if self.throttle_set_pct < 0.0 or self.throttle_set_pct > 1.0:
            raise ValueError("groundOps: throttle_set_pct must be between 0 and 1")
        
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        weight_kg   = start_weight_kg
        
        maxFn = aircraft.propulsion_model.max_thrust(0, 0, aircraft.DISAC)
        minFn = aircraft.propulsion_model.idle_thrust(0, 0, aircraft.DISAC)
        thrust_n = ((maxFn-minFn)*self.throttle_set_pct)+minFn
        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, 0, 0, aircraft.DISAC)
        
        weight_kg = weight_kg - fuel_flow_kg_s*self.duration_s
        
        history=[]
        history.append({
            "time_min":     self.duration_s/60,
            "distance_nm":  0,
            "weight_lb":    convert.kg_to_lb(weight_kg),
            "altitude_ft":  0,
            "mach":         0,
            "tas_kt":       0,
            "thrust_lb":    convert.kg_to_lb(thrust_n/convert.G0),
            "fuel_flow_lbphr": convert.kg_to_lb(fuel_flow_kg_s)*3600,
            "Fuel_burn_lb": convert.kg_to_lb(start_weight_kg - weight_kg),
            # "l_over_d":     aircraft.lift_to_drag(weight_kg, self.altitude_m, self.mach),
            "drag_lb":      0,
            "Ps_theor_fpm": 0,
            "gamma_deg":    0,
        })
        
        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_nm     = 0,
            time_s          = self.duration_s,
            history         = history,
            start_altitude_ft   = 0,
            end_altitude_ft     = 0,
            leg             = self.leg,
        )