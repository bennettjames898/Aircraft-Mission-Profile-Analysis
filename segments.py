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

@dataclass
class SegmentResult:
    segment_name:       str
    start_weight_kg:    float
    end_weight_kg:      float
    fuel_burned_kg:     float
    distance_nm:        float
    time_s:             float
    # Fine-grained trace for plotting: one entry per integration step
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
        
        maxFn = aircraft.propulsion_model.max_thrust(0, 0)
        minFn = aircraft.propulsion_model.idle_thrust(0, 0)
        thrust_n = ((maxFn-minFn)*self.throttle_set_pct)+minFn
        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, 0, 0)
        
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
    integrated over MACH as the independent variable, since Mach is the 
    start/end condition.
 
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
            raise ValueError("start_mach and end_mach must differ.")
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self.start_mach = start_mach
        self.end_mach = end_mach
        self.num_steps = num_steps
        self.a_local = convert.mach_to_tas(1.0, self.altitude_m)
        self.accelerating = end_mach > start_mach
 
    # Calculates dt/dmach, dx/dmach, dW/dmach at a given Mach & weight.
    def _derivatives(self, aircraft: Aircraft, mach: float, weight_kg: float) -> dict:
        tas = mach * self.a_local
        
        # Find thrust based on maneuver type (accel or decel)
        if self.start_mach < self.end_mach:
            thrust_n = aircraft.propulsion_model.max_thrust(self.altitude_m, mach)
        if self.start_mach > self.end_mach:
            thrust_n = aircraft.propulsion_model.idle_thrust(self.altitude_m, mach)
        drag_n = aircraft.drag_n(weight_kg, self.altitude_m, mach)
        net_force_n = thrust_n - drag_n
 
        # Check for sufficient thrust & low enough idle
        if self.accelerating and net_force_n <= 0:
            raise ValueError(
                f"Cannot accelerate: Insufficient thrust at mach={mach:.3f}, "
                f"altitude={self.altitude_m:.0f} m, weight={weight_kg:.0f} kg "
                f"(net force = {net_force_n:.0f} N)."
            )
        if not self.accelerating and net_force_n >= 0:
            raise ValueError(
                f"Cannot decelerate: Too high idle thrust at "
                f"mach={mach:.3f}, altitude={self.altitude_m:.0f} m, weight={weight_kg:.0f} kg "
                f"(net force = {net_force_n:.0f} N)."
            )
 
        accel_ms2 = net_force_n / weight_kg
        dt_dmach = self.a_local / accel_ms2
        dx_dmach = tas * dt_dmach
 
        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, self.altitude_m, mach)
        dW_dmach = -fuel_flow_kg_s * dt_dmach
 
        # Theoretical Ps
        fn_max = aircraft.propulsion_model.max_thrust(self.altitude_m, mach)
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
        distance_m  = 0
        time_s      = 0
        tas         = convert.mach_to_tas(self.mach, self.altitude_m)
        
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
            fn_max = aircraft.propulsion_model.max_thrust(self.altitude_m, self.mach)
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
        self.altitude_m = convert.ft_to_m(altitude_ft)
        self.mach       = mach
        self.duration_s = duration_min * 60
        self.num_steps  = num_steps

    # dW/dt (kg fuel per second) at an instantaneous weight.
    def _dW_dt(self, aircraft: Aircraft, weight_kg: float) -> float:
        return -aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:            
        dt          = self.duration_s / self.num_steps
        weight_kg   = start_weight_kg
        time_s      = 0.0
        tas         = convert.mach_to_tas(self.mach, self.altitude_m)

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
            fn_max = aircraft.propulsion_model.max_thrust(self.altitude_m, self.mach)
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
    1 + (V/g)(dV/dh) in the force balance solved by solver.py.

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
            gamma_max_deg: float = 25
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
            raise solver_climb_descent.TrimSolverError(
                f"Rate of climb/descent numerically ~0 at altitude={altitude_m:.0f} m "
                f"(gamma={math.degrees(gamma_rad):.4f} deg)"
                f"Check gamma_min_deg bracket."
            )

        fuel_flow_kg_s = aircraft.propulsion_model.fuel_flow(thrust_n, altitude_m, mach)
        
        fn_max = aircraft.propulsion_model.max_thrust(altitude_m, mach)
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
        history=[]
        
        # Calculate weight time and distance across the segment
        for _ in range(self.num_steps):
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
                "rate_of_climb_fpm":    convert.ms_to_fts(d_end["tas"] * math.sin(d_end["gamma_rad"]))/60,
                "Ps_theor_fpm":         convert.ms_to_fts(d_end["Ps_theor_ms"])*60,
                "ka":                   d_end["ka"],
                "drag_lb":              convert.kg_to_lb(aircraft.drag_n(weight_kg, altitude_m, d_end["mach"])/convert.G0)
            })

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
        return aircraft.propulsion_model.max_thrust(altitude_m, mach)

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

    Idle thrust comes from the propulsion model's idle_thrust() method,
    which is a simple fraction of max thrust in propulsion_model.py.
    """
    name = "descent"
    def _thrust_n(self, aircraft: Aircraft, altitude_m: float, mach: float) -> float:
        return aircraft.propulsion_model.idle_thrust(altitude_m, mach)

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