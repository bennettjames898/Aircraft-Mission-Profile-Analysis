"""
AccelDecelSegment: level-flight speed change at constant altitude,
integrated over Mach with RK4.

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

import unit_conversions as convert
from aircraft_build import Aircraft

from .base import MissionSegment, SegmentResult, _parse_altitude_input, _parse_leg_input

class AccelDecelSegment(MissionSegment):
    """
    Common implementation for Accel and Decel segments:
    level flight (gamma = 0) speed change at constant altitude,
    integrated over MACH as the independent variable.
 
        L = W 
        
    The along-flight-path balance is fully explicit:
 
        T - D = m * dV/dt   =>   dV/dt = (T - D) / weight_kg
 
    Transforming to Mach as the independent variable (constant altitude, so 
    speed of sound is constant). V = mach * a_local:
 
        dt/dmach = a_local / (dV/dt)
        dx/dmach = V * dt/dmach
        dW/dmach = -fuel_flow * dt/dmach
 
    which is integrated with RK4 exactly like CommonGammaSegment's
    dt/dh, dx/dh, dW/dh.
    """
    name = "accel/decel"
 
    def __init__(self, altitude_ft: float, start_mach: float, end_mach: float, num_steps: int = 100,
                 leg: str = None):
        self.leg = _parse_leg_input(leg, type(self).__name__)
        if start_mach == end_mach:
            raise ValueError("accel_decel: 'start_mach' and 'end_mach' must differ.")
        self.altitude_m, self._inherit_altitude = _parse_altitude_input(altitude_ft, type(self).__name__)
        self.start_mach = start_mach
        self.end_mach = end_mach
        self.num_steps = num_steps
        self.accelerating = end_mach > start_mach
        
    # --- altitude inheritance (altitude is the start and the end) ---
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
 
    # Calculates dt/dmach, dx/dmach, dW/dmach at a given Mach & weight.
    def _derivatives(self, aircraft: Aircraft, mach: float, weight_kg: float) -> dict:
        a_local = convert.mach_to_tas(1, self.altitude_m, aircraft.DISAC)
        tas = mach * a_local
        
        # Find thrust based on maneuver type (accel or decel)
        if self.start_mach < self.end_mach:
            thrust_n = aircraft.propulsion_model.max_thrust(self.altitude_m, mach, aircraft.DISAC)
        if self.start_mach > self.end_mach:
            thrust_n = aircraft.propulsion_model.idle_thrust(self.altitude_m, mach, aircraft.DISAC)
        drag_n = aircraft.drag_n(weight_kg, self.altitude_m, mach)
        net_force_n = thrust_n - drag_n
 
        # Check for sufficient thrust & low enough idle
        if self.accelerating and net_force_n <= 0:
            raise ValueError(
                f"Cannot accelerate: Insufficient thrust at mach={mach:.3f}, "
                f"altitude={convert.m_to_ft(self.altitude_m):.0f} ft, "
                f"weight={convert.kg_to_lb(weight_kg):.0f} lb "
                f"(net force = {net_force_n:.0f} N)."
            )
        if not self.accelerating and net_force_n >= 0:
            raise ValueError(
                f"Cannot decelerate: Too high idle thrust at mach={mach:.3f}, "
                f"altitude={convert.m_to_ft(self.altitude_m):.0f} ft, "
                f"weight={convert.kg_to_lb(weight_kg):.0f} lb "
                f"(net force = {net_force_n:.0f} N)."
            )
 
        accel_ms2 = net_force_n / weight_kg
        dt_dmach = a_local / accel_ms2
        dx_dmach = tas * dt_dmach
 
        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, self.altitude_m, mach, aircraft.DISAC)
        dW_dmach = -fuel_flow_kg_s * dt_dmach
 
        # Theoretical Ps
        fn_max = aircraft.propulsion_model.max_thrust(self.altitude_m, mach, aircraft.DISAC)
        Ps_ms = (fn_max - drag_n) / (weight_kg * convert.G0) * tas
 
        return {
            "dt_dmach":         dt_dmach,
            "dx_dmach":         dx_dmach,
            "dW_dmach":         dW_dmach,
            "tas":              tas,
            "thrust_n":         thrust_n,
            "drag_n":           drag_n,
            "accel_ms2":        accel_ms2,
            "fuel_flow_kg_s":   fuel_flow_kg_s,
            "Ps_theor_ms":      Ps_ms,
        }
 
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        m0 = self.start_mach
        m1 = self.end_mach
        dmach = (m1 - m0) / self.num_steps
 
        mach        = m0
        weight_kg   = start_weight_kg
        time_s      = 0
        distance_m  = 0
        
        history=[]
        for _ in range(self.num_steps):
            tprev = time_s
            dprev = distance_m
            wprev = weight_kg
 
            # RK4 over Mach
            k1 = self._derivatives(aircraft, mach, weight_kg)
            k2 = self._derivatives(aircraft, mach + 0.5 * dmach, weight_kg + 0.5 * dmach * k1["dW_dmach"])
            k3 = self._derivatives(aircraft, mach + 0.5 * dmach, weight_kg + 0.5 * dmach * k2["dW_dmach"])
            k4 = self._derivatives(aircraft, mach + dmach, weight_kg + dmach * k3["dW_dmach"])
 
            weight_kg   += (dmach / 6.0) * (k1["dW_dmach"] + 2 * k2["dW_dmach"] + 2 * k3["dW_dmach"] + k4["dW_dmach"])
            time_s      += (dmach / 6.0) * (k1["dt_dmach"] + 2 * k2["dt_dmach"] + 2 * k3["dt_dmach"] + k4["dt_dmach"])
            distance_m  += (dmach / 6.0) * (k1["dx_dmach"] + 2 * k2["dx_dmach"] + 2 * k3["dx_dmach"] + k4["dx_dmach"])
            mach        += dmach
 
            fuel_weight_kg = wprev - weight_kg
            d_end = self._derivatives(aircraft, mach, weight_kg)
 
            history.append({
                "time_min":         (time_s - tprev) / 60,
                "distance_nm":      convert.m_to_nm(distance_m - dprev),
                "weight_lb":        convert.kg_to_lb(weight_kg),
                "altitude_ft":      convert.m_to_ft(self.altitude_m),
                "mach":             mach,
                "tas_kt":           convert.ms_to_kt(d_end["tas"]),
                "thrust_lb":        convert.kg_to_lb(d_end["thrust_n"] / convert.G0),
                "fuel_flow_lbphr":  convert.kg_to_lb(d_end["fuel_flow_kg_s"]) * 3600,
                "Fuel_burn_lb":     convert.kg_to_lb(fuel_weight_kg),
                "drag_lb":          convert.kg_to_lb(d_end["drag_n"] / convert.G0),
                "gamma_deg":        0,
                "accel_kt_s":       convert.ms_to_kt(d_end["accel_ms2"]),
                "Ps_theor_fpm":     convert.ms_to_fts(d_end["Ps_theor_ms"]) * 60,
            })
 
        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_nm     = convert.m_to_nm(distance_m),
            time_s          = time_s,
            history         = history,
            start_altitude_ft = convert.m_to_ft(self.altitude_m),
            end_altitude_ft   = convert.m_to_ft(self.altitude_m),
            leg             = self.leg,
        )