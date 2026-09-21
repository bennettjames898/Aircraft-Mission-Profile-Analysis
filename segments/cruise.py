"""
ConstantAltCruiseSegment: constant-altitude, constant-Mach cruise over a
fixed distance (Breguet range ODE, RK4 over range).

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

import unit_conversions as convert
from aircraft_build import Aircraft

from .base import MissionSegment, SegmentResult, _parse_altitude_input, _parse_leg_input

class ConstantAltCruiseSegment(MissionSegment):
    """
    Constant altitude & Mach cruise for a specified range.
    'ConstantAltCruiseSegment' uses stepped numerical integration (RK4) on the
    weight-vs-distance ODE:

        dW/dx = -g * TSFC_effective / V   (Breguet's differential form)
                                           
   `altitude_ft` can be left as 'None' to defer the previous segment end alt 
   (see MissionSegment.needs_start_altitude).
    """
    name = "cruise"
    def __init__(self, mach: float, range_nm: float, altitude_ft: float, num_steps: int = 100,
                 leg: str = None):
        self.leg = _parse_leg_input(leg, type(self).__name__)
        self.altitude_m, self._inherit_altitude = _parse_altitude_input(altitude_ft, type(self).__name__)
        self.mach       = mach
        self.range_m    = convert.nm_to_m(range_nm)
        self.num_steps  = num_steps
        
    # --- altitude inheritance (altitude is both start and end) ---
    def needs_start_altitude(self) -> bool:
        return self._inherit_altitude
    def resolve_start_altitude(self, altitude_ft: float) -> None:
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self._inherit_altitude = False
    def declared_start_altitude_ft(self):
        return None if self.altitude_m is None else convert.m_to_ft(self.altitude_m)
    def _require_altitude(self) -> None:
        if self.altitude_m is None:
            raise ValueError(
                f"{type(self).__name__}.altitude was set to inherit the previous "
                f"segment's ending altitude but was never resolved.")

    # dW/dx (kg fuel per meter of range) at an instantaneous weight.
    def _dW_dx(self, aircraft: Aircraft, weight_kg: float) -> float:
        tas = convert.mach_to_tas(self.mach, self.altitude_m)
        fuel_flow_kg_s = aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)
        # dW/dt = -fuel_flow ; dt/dx = 1/V  =>  dW/dx = -fuel_flow / V
        return -fuel_flow_kg_s / tas

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        dx          = self.range_m / self.num_steps
        weight_kg   = start_weight_kg
        distance_m  = 0
        time_s      = 0
        tas         = convert.mach_to_tas(self.mach, self.altitude_m, aircraft.DISAC)
        
        history=[]
        for _ in range(self.num_steps):
            tprev = time_s
            dprev = distance_m
            wprev = weight_kg
            # RK4 steps on dW/dx
            k1 = self._dW_dx(aircraft, weight_kg)
            k2 = self._dW_dx(aircraft, weight_kg + 0.5 * dx * k1)
            k3 = self._dW_dx(aircraft, weight_kg + 0.5 * dx * k2)
            k4 = self._dW_dx(aircraft, weight_kg + dx * k3)
            weight_kg   += (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            distance_m  += dx
            time_s      += dx / tas
            
            # Thrust & Fuel
            fuel_flow_kg_s = -k1*tas
            fn_reqd = aircraft.thrust_required_n(weight_kg, self.altitude_m, self.mach)
            fuel_weight_kg = wprev-weight_kg
            
            # Theoretical Ps calculation
            fn_max = aircraft.propulsion_model.max_thrust(self.altitude_m, self.mach, aircraft.DISAC)
            drag_n = aircraft.drag_n(weight_kg, self.altitude_m, self.mach)
            Ps = (fn_max - drag_n)/(weight_kg*convert.G0)*tas

            history.append({
                "time_min":     (time_s-tprev)/60,
                "distance_nm":  convert.m_to_nm(distance_m-dprev),
                "weight_lb":    convert.kg_to_lb(weight_kg),
                "altitude_ft":  convert.m_to_ft(self.altitude_m),
                "mach":         self.mach,
                "tas_kt":       convert.ms_to_kt(tas),
                "thrust_lb":    convert.kg_to_lb(fn_reqd/convert.G0),
                "fuel_flow_lbphr": convert.kg_to_lb(fuel_flow_kg_s)*3600,
                "Fuel_burn_lb":  convert.kg_to_lb(fuel_weight_kg),
                # "l_over_d":     aircraft.lift_to_drag(weight_kg, self.altitude_m, self.mach),
                "drag_lb":      convert.kg_to_lb(aircraft.drag_n(weight_kg, self.altitude_m, self.mach)/convert.G0),
                "Ps_theor_fpm": convert.ms_to_fts(Ps)*60,
                "gamma_deg": 0,
            })

        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_nm     = convert.m_to_nm(distance_m),
            time_s          = time_s,
            history         = history,
            start_altitude_ft=convert.m_to_ft(self.altitude_m),
            end_altitude_ft=convert.m_to_ft(self.altitude_m),
            leg             = self.leg,
        )