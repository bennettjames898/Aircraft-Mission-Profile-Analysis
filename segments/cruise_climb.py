"""
CruiseClimbSegment: cruise at a commanded specific excess power, solving
the altitude that holds that Ps at every RK4 stage (MIL-STD-3013 convention).

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

import unit_conversions as convert
from aircraft_build import Aircraft

from .base import MissionSegment, SegmentResult, _parse_leg_input


class CruiseClimbError(RuntimeError):
    """
    Raised when no altitude in the search bracket delivers the requested
    specific excess power, i.e. the aircraft cannot hold the commanded Ps at
    this weight and Mach anywhere in the bracket.
    """
    pass

class CruiseClimbSegment(MissionSegment):
    """
    Cruise-climb at a specified Excess Power (Ps), integrated over range.

    At every point this segment solves for the altitude at which

        Ps = (Fn_max - D) * V / W  =  'Ps_target'    (e.g. 300 ft/min)

    Cruise-Climb is the standard convention for MIL-STD-3013 ground rules. 
    Cruise altitude is capped by a required residual climb capability at max 
    continuous thrust rather than flown at the aerodynamic optimum, so the 
    aircraft always retains some maneuver and climb margin.

    WHY THE RK4 STRUCTURE IS UNCHANGED
    -------------------------------------------------------------------------
    Altitude is a function of weight alone (given Mach and Ps_target),
    h = h(W). So the governing ODE

        dW/dx = -fuel_flow(W, h(W), M) / V(h(W))

    still depends only on W, exactly like ConstantAltCruiseSegment. It remains
    autonomous, and the same RK4 march applies without modification. Each right 
    hand side evaluation runs a bracketed altitude solve (brentq) before it 
    can evaluate drag and fuel flow. That is the same pattern as in the climb 
    solver, a root find inside an integrator.

    THE DRIFT-UP TERM - !!! USE WITH CAUTION !!!
    -------------------------------------------------------------------------
    Because altitude is changing, the aircraft is climbing, so strictly

        T = D + W*sin(gamma)

    rather than T = D. The drift is slow (a few thousand feet climb over
    a few thousand miles, resulting in gamma's on the oder of 1e-5 rad), which 
    is why classical cruise-climb treatments drop the term. It is included here 
    for completeness, via one corrector pass. 'include_climb_term=False' turns 
    this functionality off. The reported gamma_deg in the history is the actual 
    drift-up angle calculated in this segment, regardless of setting.

    Ps_target must be strictly positive. Ps = 0 is the absolute ceiling, where
    thrust required equals thrust available and numerical analysis stability 
    is challenged due to extremely small climb rate fallout..
    """
    name = "cruise_climb"

    def __init__(self, mach: float, range_nm: float, ps_target_fpm: float = 300,
                 altitude_bracket_ft: tuple = (1000, 55000),
                 num_steps: int = 100,
                 max_continuous_fraction: float = 1.0,
                 include_climb_term: bool = True,
                 leg: str = None):
        """
        ps_target_fpm --------- Commanded specific excess power (ft/min).
        altitude_bracket_ft --- Search bracket for the altitude solve.
        max_continuous_fraction Fraction of propulsion max_thrust treated as
                                MAX CONTINUOUS for the Ps calculation. 1.0 
                                reproduces the convention used for Ps_theor_fpm
                                elsewhere in this file.
        include_climb_term ---- Include W*sin(gamma) in required thrust.
        """
        self.leg = _parse_leg_input(leg, type(self).__name__)
        if ps_target_fpm <= 0:
            raise CruiseClimbError("ps_target_fpm must be > 0.")
        self.mach                    = mach
        self.range_m                 = convert.nm_to_m(range_nm)
        self.ps_target_ms            = convert.fts_to_ms(ps_target_fpm / 60.0)
        self.ps_target_fpm           = ps_target_fpm
        self.altitude_bracket_ft     = altitude_bracket_ft
        self.num_steps               = num_steps
        self.max_continuous_fraction = max_continuous_fraction
        self.include_climb_term      = include_climb_term

    #---------------------------- ALTITUDE SOLVE ------------------------------
    def _ps_at(self, aircraft: Aircraft, weight_kg: float, altitude_m: float) -> float:
        """
        Specific excess power (m/s) at a condition, same convention as
        'Ps_theor_fpm' output reported by the other segments.
        """
        tas    = convert.mach_to_tas(self.mach, altitude_m, aircraft.DISAC)
        fn_max = aircraft.propulsion_model.max_thrust(altitude_m, self.mach, aircraft.DISAC) * self.max_continuous_fraction
        drag_n = aircraft.drag_n(weight_kg, altitude_m, self.mach)
        return (fn_max - drag_n) / (weight_kg * convert.G0) * tas

    def solve_altitude_m(self, aircraft: Aircraft, weight_kg: float) -> float:
        """
        Altitude at which Ps equals the commanded value for this weight.

        Ps falls with altitude (thrust lapses faster than drag), so the
        residual is monotonically decreasing and brentq is well suited.
        """
        from scipy.optimize import brentq

        lo_m = convert.ft_to_m(self.altitude_bracket_ft[0])
        hi_m = convert.ft_to_m(self.altitude_bracket_ft[1])

        residual = lambda h: self._ps_at(aircraft, weight_kg, h) - self.ps_target_ms
        f_lo, f_hi = residual(lo_m), residual(hi_m)

        if f_lo < 0:
            raise CruiseClimbError(
                f"Cannot achieve Ps = {self.ps_target_fpm:.0f} ft/min at "
                f"{self.altitude_bracket_ft[0]:.0f} ft (weight = "
                f"{convert.kg_to_lb(weight_kg):.0f} lb, Mach = {self.mach:.3f}). "
                f"Available Ps = "
                f"{convert.ms_to_fts(self._ps_at(aircraft, weight_kg, lo_m))*60:.0f} ft/min."
            )
        if f_hi > 0:
            raise CruiseClimbError(
                f"Exceeding Ps = {self.ps_target_fpm:.0f} ft/min at the top of "
                f"the search bracket ({self.altitude_bracket_ft[1]:.0f} ft). "
                f"Increase 'altitude_bracket_ft'."
            )
        return brentq(residual, lo_m, hi_m, xtol=1e-4)

    #------------------------------ DERIVATIVES -------------------------------
    def _state_at(self, aircraft: Aircraft, weight_kg: float) -> dict:
        """
        Everything needed at one weight: the Ps-matched altitude, the drift-up
        angle, thrust, fuel flow and dW/dx.
        """
        import math

        altitude_m = self.solve_altitude_m(aircraft, weight_kg)
        tas        = convert.mach_to_tas(self.mach, altitude_m, aircraft.DISAC)
        drag_n     = aircraft.drag_n(weight_kg, altitude_m, self.mach)

        # First pass, level flight: T = D.
        ff  = aircraft.propulsion_model.fuel_flow(drag_n, altitude_m, self.mach, aircraft.DISAC)
        dW_dx = -ff / tas
        thrust_n = drag_n
        gamma_rad = 0

        ## This block aims to capture the tiny climb impact to range over the 
        # climb-cruise. 
        if self.include_climb_term:
            # dh/dx = (dh/dW)(dW/dx). 
            # dh/dW comes from re-solving the altitude at a slightly different 
            # weight, requiring an extra brentq call.
            dW = -max(weight_kg * 1e-6, 1e-3)          # lighter -> higher
            h2 = self.solve_altitude_m(aircraft, weight_kg + dW)
            dh_dW = (h2 - altitude_m) / dW
            dh_dx = dh_dW * dW_dx
            gamma_rad = math.atan(dh_dx)
            
            # One corrector pass. The term is ~1e-3 of drag, so a second pass
            # changes nothing measurable.
            thrust_n = drag_n + weight_kg * convert.G0 * math.sin(gamma_rad)
            ff       = aircraft.propulsion_model.fuel_flow(thrust_n, altitude_m, self.mach, aircraft.DISAC)
            dW_dx    = -ff / tas
        return {
            "altitude_m":   altitude_m,
            "tas":          tas,
            "drag_n":       drag_n,
            "thrust_n":     thrust_n,
            "fuel_flow":    ff,
            "dW_dx":        dW_dx,
            "gamma_rad":    gamma_rad,
        }

    def _dW_dx(self, aircraft: Aircraft, weight_kg: float) -> float:
        return self._state_at(aircraft, weight_kg)["dW_dx"]

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        import math

        dx          = self.range_m / self.num_steps
        weight_kg   = start_weight_kg
        distance_m  = 0
        time_s      = 0

        start_altitude_m = self.solve_altitude_m(aircraft, start_weight_kg)

        history = []
        for _ in range(self.num_steps):
            tprev, dprev, wprev = time_s, distance_m, weight_kg

            # RK4 on dW/dx, identical in form to ConstantAltCruiseSegment. Each
            # stage solves its own altitude.
            k1 = self._dW_dx(aircraft, weight_kg)
            k2 = self._dW_dx(aircraft, weight_kg + 0.5 * dx * k1)
            k3 = self._dW_dx(aircraft, weight_kg + 0.5 * dx * k2)
            k4 = self._dW_dx(aircraft, weight_kg + dx * k3)
            weight_kg += (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            state = self._state_at(aircraft, weight_kg)
            distance_m += dx
            time_s     += dx / state["tas"]
            fuel_weight_kg = wprev - weight_kg

            # Ps at the solved altitude, which should reproduce the commanded
            # value at every step.
            Ps = self._ps_at(aircraft, weight_kg, state["altitude_m"])

            history.append({
                "time_min":         (time_s - tprev) / 60,
                "distance_nm":      convert.m_to_nm(distance_m - dprev),
                "weight_lb":        convert.kg_to_lb(weight_kg),
                "altitude_ft":      convert.m_to_ft(state["altitude_m"]),
                "mach":             self.mach,
                "tas_kt":           convert.ms_to_kt(state["tas"]),
                "thrust_lb":        convert.kg_to_lb(state["thrust_n"] / convert.G0),
                "fuel_flow_lbphr":  convert.kg_to_lb(state["fuel_flow"]) * 3600,
                "Fuel_burn_lb":     convert.kg_to_lb(fuel_weight_kg),
                "drag_lb":          convert.kg_to_lb(state["drag_n"] / convert.G0),
                "Ps_theor_fpm":     convert.ms_to_fts(Ps) * 60,
                "gamma_deg":        math.degrees(state["gamma_rad"]),
            })
        return SegmentResult(
            segment_name     = self.name,
            start_weight_kg  = start_weight_kg,
            end_weight_kg    = weight_kg,
            fuel_burned_kg   = start_weight_kg - weight_kg,
            distance_nm      = convert.m_to_nm(distance_m),
            time_s           = time_s,
            history          = history,
            start_altitude_ft= convert.m_to_ft(start_altitude_m),
            end_altitude_ft  = history[-1]["altitude_ft"],
            leg             = self.leg,
        )
