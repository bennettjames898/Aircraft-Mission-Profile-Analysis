"""
Mission segment classes.

Each segment's `run()` method returns a SegmentResult and the ending
aircraft weight, so segments can be chained by Mission (see mission.py).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import unit_conversions as convert
from aircraft_build import Aircraft
import solver_climb_descent
import speed_schedule

def _parse_altitude_input(altitude_ft: float, class_name: str):
    """
    Interpret an altitude argument that may carry the inherit '-1' value.
 
    Returns (altitude_m, inherits). altitude_m is None when the segment must
    wait for the previous segment's ending altitude.
    """
    if altitude_ft == -1:
        return None, True
    if altitude_ft < 0:
        raise ValueError(
            f"{class_name} altitude = {altitude_ft} is not a valid altitude. "
            f"Use -1 to inherit the previous segment's ending altitude, or "
            f"input a real altitude in feet. Negative values other than -1 are rejected.")
    return convert.ft_to_m(altitude_ft), False

@dataclass
class SegmentResult:
    segment_name:       str
    start_weight_kg:    float
    end_weight_kg:      float
    fuel_burned_kg:     float
    distance_nm:        float
    time_s:             float
    # Time-history data for plotting: one entry per integration step
    history: List[dict] = field(default_factory=list)
    # Altitude the segment started/ended at.
    start_altitude_ft: Optional[float] = None
    end_altitude_ft: Optional[float] = None

# Base class. Subclasses implement run().
class MissionSegment:
    """
    Base class. Subclasses implement 'run()' specific for the flight segment.
    """
    name = "generic_segment"
    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        raise NotImplementedError
        
    # ------------------------ ALTITUDE INHERITANCE ------------------------
    # Mission.run() calls them: if needs_start_altitude() is True it passes the
    # previous segment's end_altitude_ft to resolve_start_altitude().
    def needs_start_altitude(self) -> bool:
        """True if this segment was built with 'start_altitude_ft = -1'."""
        return False
    def resolve_start_altitude(self, altitude_ft: float) -> None:
        """Supply the altitude this segment should start from."""
        raise NotImplementedError(
            f"{type(self).__name__} does not take an inheritable start altitude.")
    def declared_start_altitude_ft(self) -> Optional[float]:
        """This segment's start altitude if known, else None."""
        return None
        
class GroundOps(MissionSegment):
    """
    Simple mission segment to input fuel burned on the ground for a specified 
    time. 
    """
    name = "groundOps"
    def __init__(self, duration_min: float, throttle_set_pct: float):
        self.duration_s = duration_min * 60.0
        self.throttle_set_pct = throttle_set_pct
        
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
            start_altitude_ft=0,
            end_altitude_ft=0,
        )
    
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
 
    def __init__(self, altitude_ft: float, start_mach: float, end_mach: float, num_steps: int = 100):
        if start_mach == end_mach:
            raise ValueError("'start_mach' and 'end_mach' must differ.")
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
        )

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
    def __init__(self, mach: float, range_nm: float, altitude_ft: float, num_steps: int = 100):
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
        )

class LoiterSegment(MissionSegment):
    """
    Constant altitude & Mach loiter for a specified duration. 
    Same ODE as cruise but integrated over time.
    """
    name = "loiter"
    def __init__(self, mach: float, duration_min: float, altitude_ft: float, num_steps: int = 100):
        self.altitude_m, self._inherit_altitude = _parse_altitude_input(altitude_ft, type(self).__name__)
        self.mach       = mach
        self.duration_s = duration_min * 60
        self.num_steps  = num_steps
        
    # --- altitude inheritance (altitude is start and end) ---
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

    # dW/dt (kg fuel per second) at an instantaneous weight.
    def _dW_dt(self, aircraft: Aircraft, weight_kg: float) -> float:
        return -aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:            
        dt          = self.duration_s / self.num_steps
        weight_kg   = start_weight_kg
        time_s      = 0.0
        tas         = convert.mach_to_tas(self.mach, self.altitude_m, aircraft.DISAC)

        history=[]
        for _ in range(self.num_steps):
            wprev = weight_kg
            # RK4 steps over dW/dt
            k1 = self._dW_dt(aircraft, weight_kg)
            k2 = self._dW_dt(aircraft, weight_kg + 0.5 * dt * k1)
            k3 = self._dW_dt(aircraft, weight_kg + 0.5 * dt * k2)
            k4 = self._dW_dt(aircraft, weight_kg + dt * k3)
            weight_kg   += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            time_s      += dt
            
            # Thrust & Fuel
            fn_reqd = aircraft.thrust_required_n(weight_kg, self.altitude_m, self.mach)
            fuel_weight_kg = wprev-weight_kg
            
            # Theoretical Ps calculation
            fn_max = aircraft.propulsion_model.max_thrust(self.altitude_m, self.mach, aircraft.DISAC)
            drag_n = aircraft.drag_n(weight_kg, self.altitude_m, self.mach)
            Ps = (fn_max - drag_n)/(weight_kg*convert.G0)*tas
            
            history.append({
                "time_min":     (dt)/60, 
                "distance_nm":  0,
                "weight_lb":    convert.kg_to_lb(weight_kg),
                "altitude_ft":  convert.m_to_ft(self.altitude_m),
                "mach":         self.mach,
                "tas_kt":       convert.ms_to_kt(tas),
                "thrust_lb":    convert.kg_to_lb(fn_reqd/convert.G0),
                "fuel_flow_lbphr": convert.kg_to_lb(-k1)*3600,
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
            distance_nm     = 0.0,
            time_s          = time_s,
            history         = history,
            start_altitude_ft=convert.m_to_ft(self.altitude_m),
            end_altitude_ft=convert.m_to_ft(self.altitude_m),
        )
    
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
            ps_search_ceiling_ft: float = 60000
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
        aircraft has WHEN IT GETS THERE, and that weight depends on how far it
        climbed. So the terminal altitude cannot simply be looked up before
        integrating, it has to be solved:

            find h_end such that Ps(W_end(h_end), h_end) = Ps_target

        where W_end comes from actually flying the climb to h_end. That is an
        outer root find wrapped around the RK4 march, the same nesting used by
        solver_mission_range.py around a whole mission. See
        _solve_ps_ceiling_altitude_m().

        Because fuel burns on the way up, the answer is always HIGHER than the
        instantaneous ceiling evaluated at the starting weight. That
        instantaneous value is used as the lower bracket for the search.

        ps_search_steps      Step count for the trial climbs inside the ceiling
                             search. Defaults to num_steps, so the search flies
                             the same discretisation as the final reported climb
                             and the converged Ps therefore reads back exactly.
                             Lower it to trade a small inconsistency for speed.
        ps_search_ceiling_ft Hard upper limit for the search.
        """
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
        fn_max = aircraft.propulsion_model.max_thrust(altitude_m, mach)
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
                 include_climb_term: bool = True):
        """
        ps_target_fpm --------- Commanded specific excess power (ft/min).
        altitude_bracket_ft --- Search bracket for the altitude solve.
        max_continuous_fraction Fraction of propulsion max_thrust treated as
                                MAX CONTINUOUS for the Ps calculation. 1.0 
                                reproduces the convention used for Ps_theor_fpm
                                elsewhere in this file.
        include_climb_term ---- Include W*sin(gamma) in required thrust.
        """
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
        )
