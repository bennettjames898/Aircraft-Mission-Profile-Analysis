"""
AerialRefuelSegment: constant-altitude, constant-Mach fuel transfer
(onload or offload) integrated over time, coupled with the receiving/
donating aircraft's own fuel burn.

Argument meanings, accepted values and the error messages this segment can
raise are documented in the catalog at the top of segments/base.py.
"""

import unit_conversions as convert
from aircraft_build import Aircraft

from .base import MissionSegment, SegmentResult, _parse_altitude_input, _parse_leg_input

class AerialRefuelError(RuntimeError):
    """
    Raised when a fuel transfer is not possible, e.g. a transfer that would 
    take the aircraft below ZFW.
    """
    pass

class AerialRefuelSegment(MissionSegment):
    """
    Aerial refuelling at constant altitude and Mach, integrated over time.

    TRANSFER SIGN CONVENTION
    -------------------------------------------------------------------------
        transfer_rate_lb_min > 0    RECEIVING  (onload fuel)
        transfer_rate_lb_min < 0    DONATING   (offload fuel)

    'fuel_transferred_lb' is a magnitude and should be positive for all cases.

    DERIVATION
    -------------------------------------------------------------------------
    Transfer and fuel burn happen simultaneously, so the governing ODE is

        dW/dt = transfer_rate - fuel_flow(W, h, M)

    A receiving aircraft gains weight at the net rate, and a donating aircraft 
    loses weight faster than the offload alone. Because drag and fuel flow 
    depend on the changing weight, this is integrated with RK4 over time in
    the same way as LoiterSegment.

    FUEL TRANSFER SPECIFICATIONS
    -------------------------------------------------------------------------
        fuel_transferred_lb --- how much to move (magnitude, > 0)
                            OR
        duration_min ---------- how long to stay on the boom/drogue

    The other follows from the rate. Transfer rate is constant, so
    duration = fuel_transferred / |rate|.

    FUEL ACCOUNTING
    -------------------------------------------------------------------------
    SegmentResult.fuel_burned_kg is start weight minus end weight, which is
    what every segment reports and what Mission uses for the running total. 
    For onloading fuel 'fuel_burned_kg' is NEGATIVE, because the aircraft ends 
    heavier than it started.

    The consequence is that MissionResult.total_fuel_burned_lb on a mission
    containing an onload is a NET WEIGHT CHANGE, not the fuel actually
    consumed. The true burn is available two ways: the 'Fuel_burn_lb' column
    in this segment's history is always the genuine burn, positive, and
    'SegmentResult.AR_AC_flight_burn_kg' holds the segment total (in kg) after
    run(). Both exclude transferred fuel.

    ASSUMPTIONS AND SIMPLIFICATIONS
    -------------------------------------------------------------------------
    A donation that would drive the aircraft below its zero fuel weight raises
    AerialRefuelError. There is no upper limit check on an onloading scenario 
    because Aircraft carries no tank capacity attribute.

    Drogue Contact is assumed instantaneous and the transfer rate is constant. 
    Increased drag from the refueling formation (boom/drogue effects, receiver 
    fling in the tanker downwash) is not modeled.
    """
    name = "air_refuel"

    def __init__(self, mach: float, altitude_ft: float,
                 transfer_rate_lb_min: float,
                 fuel_transferred_lb: float = None,
                 duration_min: float = None,
                 num_steps: int = 100,
                 leg: str = None):
        if transfer_rate_lb_min == 0:
            raise ValueError(
                "transfer_rate_lb_min must be non-zero. Positive recieves fuel, "
                "negative offloads fuel.")
        if (fuel_transferred_lb is None) == (duration_min is None):
            raise ValueError(
                f"Specify either 'fuel_transferred_lb' OR 'duration_min'. Got "
                f"fuel_transferred_lb={fuel_transferred_lb}, duration_min={duration_min}.")

        if fuel_transferred_lb is not None:
            if fuel_transferred_lb <= 0:
                raise ValueError(
                    f"'fuel_transferred_lb' must be > 0, got {fuel_transferred_lb}.")
            self.duration_s = fuel_transferred_lb / abs(transfer_rate_lb_min) * 60
        else:
            if duration_min <= 0:
                raise ValueError(f"'duration_min' must be > 0, got {duration_min}.")
            self.duration_s = duration_min * 60.0

        self.altitude_m, self._inherit_altitude = _parse_altitude_input(altitude_ft, type(self).__name__)
        self.mach                   = mach
        self.transfer_rate_lb_min   = transfer_rate_lb_min
        self.transfer_rate_kg_s     = convert.lb_to_kg(transfer_rate_lb_min)/60
        self.receiving              = transfer_rate_lb_min >0
        self.num_steps              = num_steps
        self.leg                    = _parse_leg_input(leg, type(self).__name__)

    # --- altitude inheritance ---
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
                f"{type(self).__name__}.altitude_ft was set to inherit the previous "
                f"segment's ending altitude but was never resolved.")

    # Net dW/dt: transfer rate minus the aircrfat native fuel burn.
    def _dW_dt(self, aircraft: Aircraft, weight_kg: float) -> float:
        burn = aircraft.fuel_flow_kg_s(weight_kg, self.altitude_m, self.mach)
        return self.transfer_rate_kg_s - burn

    def run(self, aircraft: Aircraft, start_weight_kg: float) -> SegmentResult:
        self._require_altitude()

        zfw_kg = convert.lb_to_kg(aircraft.zero_fuel_weight_lb)
        if start_weight_kg < zfw_kg - 1e-6:
            raise AerialRefuelError(
                f"Aircraft is below zero fuel weight "
                f"({convert.kg_to_lb(start_weight_kg):.0f} lb vs "
                f"{aircraft.zero_fuel_weight_lb:.0f} lb) entering the AR segment.")

        dt          = self.duration_s / self.num_steps
        weight_kg   = start_weight_kg
        time_s      = 0
        distance_m  = 0
        tas         = convert.mach_to_tas(self.mach, self.altitude_m, aircraft.DISAC)
        burned_kg   = 0
        transfer_kg = 0

        history=[]
        for _ in range(self.num_steps):
            wprev = weight_kg

            # RK4 over the NET weight change rate. The burn term inside _dW_dt 
            # is what makes this an ODE rather than a straight line in time.
            k1 = self._dW_dt(aircraft, weight_kg)
            k2 = self._dW_dt(aircraft, weight_kg + 0.5 * dt * k1)
            k3 = self._dW_dt(aircraft, weight_kg + 0.5 * dt * k2)
            k4 = self._dW_dt(aircraft, weight_kg + dt * k3)
            weight_kg += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            time_s    += dt

            if weight_kg < zfw_kg - 1e-6:
                raise AerialRefuelError(
                    f"Offloading {abs(self.transfer_rate_lb_min):.0f} lb/min for "
                    f"{self.duration_s/60:.1f} min drives the aircraft below its zero "
                    f"fuel weight ({convert.kg_to_lb(weight_kg):.0f} lb vs "
                    f"{aircraft.zero_fuel_weight_lb:.0f} lb).")

            # Split the weight change back into its two parts for reporting. 
            transferred_kg  = self.transfer_rate_kg_s * dt # AR transfer ONLY
            step_burn_kg    = transferred_kg - (weight_kg - wprev) # AC burn ONLY
            burned_kg      += step_burn_kg # AC burn tally
            transfer_kg    += transferred_kg # transfer tally

            # Ps calculation for reference
            fn_reqd = aircraft.thrust_required_n(weight_kg, self.altitude_m, self.mach)
            fn_max  = aircraft.propulsion_model.max_thrust(self.altitude_m, self.mach, aircraft.DISAC)
            drag_n  = aircraft.drag_n(weight_kg, self.altitude_m, self.mach)
            Ps      = (fn_max - drag_n) / (weight_kg * convert.G0) * tas

            history.append({
                "time_min":             dt / 60,
                "distance_nm":          0,
                "weight_lb":            convert.kg_to_lb(weight_kg),
                "altitude_ft":          convert.m_to_ft(self.altitude_m),
                "mach":                 self.mach,
                "tas_kt":               convert.ms_to_kt(tas),
                "thrust_lb":            convert.kg_to_lb(fn_reqd / convert.G0),
                "fuel_flow_lbphr":      convert.kg_to_lb(step_burn_kg / dt) * 3600, # Aircraft burn excludes transferred fuel.
                "Fuel_burn_lb":         convert.kg_to_lb(step_burn_kg), # positive onload, negative offload.
                "Fuel_transfer_step":   convert.kg_to_lb(transferred_kg),
                "Fuel_transfer_total":  convert.kg_to_lb(transfer_kg),
                "drag_lb":              convert.kg_to_lb(drag_n / convert.G0),
                "Ps_theor_fpm":         convert.ms_to_fts(Ps) * 60,
                "gamma_deg":            0,
            })

        return SegmentResult(
            segment_name    = self.name,
            start_weight_kg = start_weight_kg,
            end_weight_kg   = weight_kg,
            fuel_burned_kg  = start_weight_kg - weight_kg, # Net weight change. NEGATIVE = onload. See the docstring on fuel accounting.
            distance_nm     = convert.m_to_nm(distance_m),
            time_s          = time_s,
            history         = history,
            start_altitude_ft = convert.m_to_ft(self.altitude_m),
            end_altitude_ft   = convert.m_to_ft(self.altitude_m),
            AR_AC_flight_burn_kg = burned_kg,
            leg             = self.leg,
        )