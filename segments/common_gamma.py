"""
CommonGammaSegment: shared base for ClimbSegment and DescentSegment.
Solves the trimmed flight-path angle at every altitude step and marches
dt/dh, dx/dh and dW/dh with RK4.

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

import unit_conversions as convert
from aircraft_build import Aircraft
import solver_climb_descent
import speed_schedule

from .base import MissionSegment, SegmentResult, _parse_altitude_input, _parse_leg_input

class CommonGammaSegment(MissionSegment):
    """
    Common implementation for ClimbSegment and DescentSegment. Both 
    integrate over altitude.

    Speed is governed by a SpeedSchedule (speed_schedule.py). Because
    TAS changes with altitude, this segment class applies the climb
    acceleration correction 'ka', captured by the factor
    1 + (V/g)(dV/dh) in the force balance solved by solver_climb_descent.py.

    At every altitude step, this solves for the trimmed gamma
    given the prescribed thrust setting and 'ka' acceleration, then uses 
    that gamma to compute dt/dh, dx/dh, and dW/dh. All three are integrated 
    together with RK4.
    """

    def __init__(
            self, 
            start_altitude_ft: float, 
            end_altitude_ft: float, 
            schedule, 
            num_steps: int = 100, 
            gamma_min_deg: float = 0.05, 
            gamma_max_deg: float = 25,
            ps_search_steps: int = None,
            ps_search_ceiling_ft: float = 60000,
            leg: str = None
        ):
        """
        end_altitude_ft accepts either a target altitude or a Ps ceiling:

            end_altitude_ft =  35000   climb to 35,000 ft
            end_altitude_ft =   -300   climb until specific excess power falls
                                       to 300 ft/min, i.e. climb to the 300
                                       ft/min ceiling

        A negative value is unambiguous as a flag because the atmosphere model
        does not support altitudes below sea level, so no real end altitude can
        be negative. Zero still means sea level.

        THE CEILING IS NOT KNOWN UP FRONT
        ---------------------------------------------------------------------
        The altitude at which Ps falls to the target depends on the weight the
        aircraft has instantaneously, and that weight depends on how far it
        climbed:

            find h_end such that Ps(W_end(h_end), h_end) = Ps_target

        where 'W_end' comes from actually flying the climb to 'h_end'. That is an
        outer root find wrapped around the RK4 march, the same nesting used by
        solver_mission_range.py around a whole mission. See
        _solve_ps_ceiling_altitude_m().

        Because fuel burns on the way up, the answer is always higher than the
        instantaneous ceiling evaluated at the starting weight. That
        instantaneous value is used as the lower bracket for the search.

        ps_search_steps      Step count for the trial climbs inside the ceiling
                             search. Defaults to num_steps, so the search flies
                             the same discretisation as the final reported climb
                             and the converged Ps therefore reads back exactly.
                             Lower it to trade a small inconsistency for speed.
        ps_search_ceiling_ft Hard upper limit for the search.
        """
        self.leg = _parse_leg_input(leg, type(self).__name__)
        self.start_altitude_m, self._inherit_altitude = _parse_altitude_input(start_altitude_ft, type(self).__name__)
        if end_altitude_ft < 0:
            self.ps_ceiling_fpm = -float(end_altitude_ft)
            self.end_altitude_m = None          # resolved in run()
        else:
            self.ps_ceiling_fpm = None
            self.end_altitude_m = convert.ft_to_m(end_altitude_ft)
        self.schedule = speed_schedule.as_schedule(schedule)
        self.num_steps = num_steps
        self.gamma_min_deg = gamma_min_deg
        self.gamma_max_deg = gamma_max_deg
        self.ps_search_steps = ps_search_steps if ps_search_steps is not None else num_steps
        self.ps_search_ceiling_ft = ps_search_ceiling_ft

        # Populated by run() when a Ps ceiling was requested, so the caller can
        # see what altitude the climb terminated at.
        self.solved_end_altitude_ft = None if self.ps_ceiling_fpm else end_altitude_ft
        
    # --- altitude inheritance (only START altitude is inheritable) ---
    def needs_start_altitude(self) -> bool:
        return self._inherit_altitude
    def resolve_start_altitude(self, altitude_ft: float) -> None:
        self.start_altitude_m = convert.ft_to_m(altitude_ft)
        self._inherit_altitude = False
    def declared_start_altitude_ft(self):
        return None if self.start_altitude_m is None else convert.m_to_ft(self.start_altitude_m)
    def _require_altitude(self) -> None:
        if self.start_altitude_m is None:
            raise ValueError(
                f"{type(self).__name__}.start_altitude_ft was set to inherit the previous "
                f"segment's ending altitude but was never resolved.")

    # Thrust calculation placeholder
    def _thrust_n(self, aircraft: Aircraft, altitude_m: float, mach: float) -> float:
        raise NotImplementedError
    # gamma solver placeholder
    def _solve_gamma(self, aircraft: Aircraft, weight_kg: float, altitude_m: float, mach: float, thrust_n: float, ka: float) -> float:
        raise NotImplementedError

    # Calculates dt/dh, dx/dh, dW/dh, and gamma at a given altitude & weight
    def _derivatives(self, aircraft: Aircraft, altitude_m: float, weight_kg: float) -> dict:
        import math
        mach    = self.schedule.mach_at_altitude(altitude_m, aircraft.DISAC)
        tas     = self.schedule.tas_at_altitude(altitude_m, aircraft.DISAC)
        dtas_dh = self.schedule.dtas_dh(altitude_m)
        ka      = 1.0 + (tas / convert.G0) * dtas_dh
        
        thrust_n = self._thrust_n(aircraft, altitude_m, mach)
        gamma_rad = self._solve_gamma(aircraft, weight_kg, altitude_m, mach, thrust_n, ka)

        rate_of_climb = tas * math.sin(gamma_rad)  # dh/dt; negative during descent

        if abs(rate_of_climb) < 1e-6:
            raise solver_climb_descent.TrimSolverError(
                f"Rate of climb/descent numerically ~0 at altitude={altitude_m:.0f} m "
                f"(gamma={math.degrees(gamma_rad):.4f} deg)"
                f"Check gamma_min_deg bracket."
            )

        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, altitude_m, mach, aircraft.DISAC)
        
        fn_max = aircraft.propulsion_model.max_thrust(altitude_m, mach, aircraft.DISAC)
        drag_n = aircraft.drag_n(weight_kg, altitude_m, mach)
        Ps = (fn_max - drag_n)/(weight_kg*convert.G0)*tas

        return {
            "dt_dh":        1 / rate_of_climb,
            "dx_dh":        (tas * math.cos(gamma_rad)) / rate_of_climb,
            "dW_dh":        -fuel_flow_kg_s / rate_of_climb,
            "gamma_rad":    gamma_rad,
            "tas":          tas,
            "mach":         mach,
            "ka":           ka,
            "roc_ms":       rate_of_climb,
            "Ps_theor_ms": Ps,
            "fuel_flow_kg_s": fuel_flow_kg_s,
            "thrust_n":     thrust_n,
        }
    
    # Ps using MAX thrust, matching Ps_theor_ms convention reported by
    # _derivatives().
    def _ps_ms(self, aircraft: Aircraft, weight_kg: float, altitude_m: float) -> float:
        mach   = self.schedule.mach_at_altitude(altitude_m)
        tas    = self.schedule.tas_at_altitude(altitude_m)
        fn_max = aircraft.propulsion_model.max_thrust(altitude_m, mach, aircraft.DISAC)
        drag_n = aircraft.drag_n(weight_kg, altitude_m, mach)
        return (fn_max - drag_n) / (weight_kg * convert.G0) * tas

    def _instantaneous_ceiling_m(self, aircraft: Aircraft, weight_kg: float, ps_target_ms: float) -> float:
        """
        Altitude where Ps equals the target. Ps falls with.
        """
        from scipy.optimize import brentq

        lo = self.start_altitude_m
        hi = convert.ft_to_m(self.ps_search_ceiling_ft)
        residual = lambda h: self._ps_ms(aircraft, weight_kg, h) - ps_target_ms

        f_lo = residual(lo)
        if f_lo <= 0:
            raise solver_climb_descent.TrimSolverError(
                f"Aircraft is at or below the {convert.ms_to_fts(ps_target_ms)*60:.0f} "
                f"ft/min ceiling at the start altitude "
                f"({convert.m_to_ft(self.start_altitude_m):.0f} ft, weight "
                f"{convert.kg_to_lb(weight_kg):.0f} lb). Available Ps is "
                f"{convert.ms_to_fts(self._ps_ms(aircraft, weight_kg, lo))*60:.0f} ft/min")
        if residual(hi) > 0:
            raise solver_climb_descent.TrimSolverError(
                f"Aircraft exceeds the {convert.ms_to_fts(ps_target_ms)*60:.0f} ft/min "
                f"ceiling at {self.ps_search_ceiling_ft:.0f} ft. Increase "
                f"ps_search_ceiling_ft.")
        return brentq(residual, lo, hi, xtol=1e-3)

    def _solve_ps_ceiling_altitude_m(self, aircraft: Aircraft, start_weight_kg: float) -> float:
        """
        Solve for the altitude at which the climb should stop, accounting for
        the fuel burned.

        residual(h) = Ps(W_end(h), h) - Ps_target, where W_end(h) comes from
        actually flying the climb to h. Monotonically decreasing: climbing
        higher costs altitude (Ps down) faster than it saves weight (Ps up).

        If a trial altitude is unreachable the march raises TrimSolverError,
        which is treated as "Ps has already run out below here" rather than
        propagated, so the search converges on the true ceiling instead of
        failing at a probe point above it.
        """
        from scipy.optimize import brentq

        ps_target_ms = convert.fts_to_ms(self.ps_ceiling_fpm / 60.0)

        def residual(h1_m: float) -> float:
            if h1_m <= self.start_altitude_m:
                return self._ps_ms(aircraft, start_weight_kg, self.start_altitude_m) - ps_target_ms
            try:
                end_weight = self._march(aircraft, start_weight_kg, h1_m, self.ps_search_steps)["weight_kg"]
            except solver_climb_descent.TrimSolverError:
                # Unreachable, so effective Ps here is zero.
                return -ps_target_ms
            return self._ps_ms(aircraft, end_weight, h1_m) - ps_target_ms

        # Lower bracket: the ceiling ignoring fuel burn. Guaranteed reachable.
        lo = self._instantaneous_ceiling_m(aircraft, start_weight_kg, ps_target_ms)
        f_lo = residual(lo)
        if f_lo <= 0:
            # instantaneous answer is already the answer.
            return lo

        # Expand upward until Ps falls through the target or the climb stops
        # being flyable.
        hi = lo
        ceiling_m = convert.ft_to_m(self.ps_search_ceiling_ft)
        for _ in range(20):
            hi = min(hi + convert.ft_to_m(2000), ceiling_m)
            if residual(hi) <= 0:
                break
            if hi >= ceiling_m:
                raise solver_climb_descent.TrimSolverError(
                    f"Could not bracket the {self.ps_ceiling_fpm:.0f} ft/min ceiling below "
                    f"{self.ps_search_ceiling_ft:.0f} ft. Raise ps_search_ceiling_ft.")
        return brentq(residual, lo, hi, xtol=1e-2)

    # Shared RK4 march over altitude. run() uses it with history collection,
    # the Ps ceiling search uses it without, so both fly identical physics.
    def _march(self, aircraft: Aircraft, start_weight_kg: float, h1: float, num_steps: int) -> dict:
        import math

        h0 = self.start_altitude_m
        dh = (h1 - h0) / num_steps
        altitude_m  = h0
        weight_kg   = start_weight_kg
        time_s      = 0
        distance_m  = 0
        history=[]

        # Calculate weight time and distance across the segment
        for _ in range(num_steps):
            tprev = time_s
            dprev = distance_m
            wprev = weight_kg
            
            # RK4 over altitude
            k1 = self._derivatives(aircraft, altitude_m, weight_kg)
            k2 = self._derivatives(aircraft, altitude_m + 0.5 * dh, weight_kg + 0.5 * dh * k1["dW_dh"])
            k3 = self._derivatives(aircraft, altitude_m + 0.5 * dh, weight_kg + 0.5 * dh * k2["dW_dh"])
            k4 = self._derivatives(aircraft, altitude_m + dh, weight_kg + dh * k3["dW_dh"])

            weight_kg   += (dh / 6.0) * (k1["dW_dh"] + 2 * k2["dW_dh"] + 2 * k3["dW_dh"] + k4["dW_dh"])
            time_s      += (dh / 6.0) * (k1["dt_dh"] + 2 * k2["dt_dh"] + 2 * k3["dt_dh"] + k4["dt_dh"])
            distance_m  += (dh / 6.0) * (k1["dx_dh"] + 2 * k2["dx_dh"] + 2 * k3["dx_dh"] + k4["dx_dh"])
            altitude_m  += dh

            fuel_weight_kg = wprev-weight_kg
            d_end = self._derivatives(aircraft, altitude_m, weight_kg)
            
            history.append({
                "time_min":             (time_s-tprev) / 60,
                "distance_nm":          convert.m_to_nm(distance_m-dprev),
                "weight_lb":            convert.kg_to_lb(weight_kg),
                "altitude_ft":          convert.m_to_ft(altitude_m),
                "mach":                 d_end["mach"],
                "tas_kt":               convert.ms_to_kt(d_end["tas"]),
                "thrust_lb":            convert.kg_to_lb(d_end["thrust_n"]/convert.G0),
                "fuel_flow_lbphr":      convert.kg_to_lb(d_end["fuel_flow_kg_s"])*3600,
                "Fuel_burn_lb":         convert.kg_to_lb(fuel_weight_kg),
                "gamma_deg":            math.degrees(d_end["gamma_rad"]),
                "rate_of_climb_fpm":    convert.ms_to_fts(d_end["tas"] * math.sin(d_end["gamma_rad"]))*60,
                "Ps_theor_fpm":         convert.ms_to_fts(d_end["Ps_theor_ms"])*60,
                "ka":                   d_end["ka"],
                "drag_lb":              convert.kg_to_lb(aircraft.drag_n(weight_kg, altitude_m, d_end["mach"])/convert.G0)
            })
        return {
            "weight_kg":    weight_kg,
            "time_s":       time_s,
            "distance_m":   distance_m,
            "altitude_m":   altitude_m,
            "history":      history,
        }
    
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:

        # Collect initial conditions
        h0 = self.start_altitude_m
        if self.ps_ceiling_fpm is not None:
            h1 = self._solve_ps_ceiling_altitude_m(aircraft, start_weight_kg)
            self.solved_end_altitude_ft = convert.m_to_ft(h1)
        else:
            h1 = self.end_altitude_m
        out = self._march(aircraft, start_weight_kg, h1, self.num_steps)
        weight_kg   = out["weight_kg"]
        time_s      = out["time_s"]
        distance_m  = out["distance_m"]
        altitude_m  = out["altitude_m"]
        history     = out["history"]

        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg,
            distance_nm     = convert.m_to_nm(distance_m),
            time_s          = time_s,
            history         = history,
            start_altitude_ft = convert.m_to_ft(h0),
            end_altitude_ft = convert.m_to_ft(altitude_m),
            leg             = self.leg,
        )