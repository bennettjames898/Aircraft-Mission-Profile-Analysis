"""
Mission segment classes.

Each segment's `run()` method returns a SegmentResult and the ending
aircraft weight. Each segment below is built, then handed to 
Mission(aircraft, segments, saveDir) in flight order. Mission.run() calls 
each segment's run(aircraft, weight_kg) feeds the ending weight into the 
next.

All constructor arguments are in imperial units (ft, lb, kt, min). Segments 
compute internally in SI. 

===============================================================================
ALTITUDE INHERITANCE (user input '-1' for a starting altitude)
===============================================================================
Every segment that takes an altitude (AccelDecelSegment, ConstantAltCruiseSegment, 
LoiterSegment, ClimbSegment, DescentSegment) will accept -1 in place of a real 
altitude to mean "start at the altitude the previous segment ended at."

    ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, ...),
    AccelDecelSegment(altitude_ft=-1, start_mach=0.78, end_mach=0.85, ...),
    ConstantAltCruiseSegment(altitude_ft=-1, mach=0.85, range_nm=1200, ...),

SOURCES OF ERROR
  - The FIRST segment in a mission can not use -1 (there is no previous
    segment to inherit from). Mission.run() raises ValueError:
        "Segment 0 (<name>) must have an explicit altitude stated."
  - Any altitude input other than -1 that is negative is rejected outright
    (ValueError). The one exception is CommonGammaSegment, where ANY 
    negative value (not just -1) is a valid, different input (see 
    ClimbSegment/DescentSegment below).
  - CruiseClimbSegment has no altitude input at all (it solves altitude from
    a commanded Ps every step) and does not support -1. A later segment can
    still inherit ITS ending altitude.

===============================================================================
LEG ASSIGNMENT (the 'leg' input, for radius missions)
===============================================================================
Every segment accepts an optional keyword argument 'leg' that names which half
of a radius (out-and-back) mission it belongs to:

    leg = "outbound"    flown on the way out to the turn point
    leg = "inbound"     flown on the way home
    leg = "neutral"     deliberately counts toward NEITHER leg's distance
    leg omitted         unassigned (the default, and the choice for a one-way 
                        mission)

A radius mission is a mission whose outbound and inbound legs cover the same
ground distance. That common distance is the reported radius:

    radius_nm = sum(distance of every "outbound" segment)
              = sum(distance of every "inbound" segment)

Tagging costs nothing in a normal one-way mission: Mission.run() only records
the tag, and nothing in the physics reads it. The radius solver in
solver_mission_range.py is what acts on it (see solve_radius()).

SOURCES OF ERROR
  - ValueError at construction for any value other than the four above
    (matching is case-insensitive and ignores surrounding whitespace, so
    "Outbound" and " inbound " are accepted).
  - The radius solver, not the segment, rejects a mission whose legs are
    interleaved (an "outbound" segment after an "inbound" one), or that
    credits distance to a segment left unassigned. Use leg="neutral" to state
    on purpose that a distance-crediting segment belongs to neither leg
    (a diversion to an alternate, for example).

===============================================================================
GroundOps(duration_min, throttle_set_pct)
===============================================================================
Ground fuel burn (e.x. taxi, APU, run-up) at a fixed altitude/Mach = 0 for a 
fixed duration. No distance credited.

  duration_min ----- Minutes on the ground.
  throttle_set_pct - 0-1, interpolated linearly between idle and max static 
                     thrust to pick the fuel flow used.

SOURCES OF ERROR
  - throttle_set_pct outside [0, 1] will raise a ValueError if improper 
    percentages are used.
  - A propulsion model that isn't well-behaved at V=0 (e.g. a turboprop) will 
    surface here first.

===============================================================================
AccelDecelSegment(altitude_ft, start_mach, end_mach, num_steps=100)
===============================================================================
Level flight speed change at constant altitude, integrated over Mach via RK4. 
Full thrust if accelerating, idle thrust if decelerating.

  altitude_ft - altitude for the segment. Accepts -1 (see ALTITUDE INHERITANCE).
  start_mach, end_mach
                Must differ. Direction (start_mach vs end_mach) is what
                selects accelerating vs decelerating.
  num_steps --- RK4 step count over the Mach range. More steps = smoother
                history / slightly better accuracy, at the cost of runtime.

SOURCES OF ERROR
  - ValueError at construction if start_mach == end_mach.
  - ValueError during run() ("Cannot accelerate: Insufficient thrust...")
    if max thrust does not exceed drag anywhere along the Mach sweep. Usually
    means the target end_mach is unreachable at the altitude/weight.
  - ValueError during run() ("Cannot decelerate: Too high idle thrust...")
    if idle thrust exceeds drag.

===============================================================================
ConstantAltCruiseSegment(mach, range_nm, altitude_ft, num_steps=100)
===============================================================================
Constant altitude, constant Mach cruise for a fixed distance, integrated over
range via RK4 on dW/dx (Breguet's Range Eq. differential form).

  mach -------- Cruise Mach number (constant for the whole segment).
  range_nm ---- Distance to fly.
  altitude_ft - Cruise altitude. Accepts -1 (see ALTITUDE INHERITANCE).
  num_steps --- RK4 step count over the range.

SOURCES OF ERROR
  - RuntimeError from Aircraft.fuel_flow_kg_s() ("Thrust required (...) >
    max thrust (...). Aircraft cannot sustain this flight condition.") if
    the aircraft cannot hold this Mach/altitude at the current weight.

===============================================================================
LoiterSegment(mach, duration_min, altitude_ft, num_steps=100)
===============================================================================
Constant altitude, constant Mach for a fixed time (same governing ODE as 
cruise, integrated over time instead of range). No distance is credited to the 
mission, so this segment is intended for holds/reserves.

  mach ---------- Loiter Mach number.
  duration_min -- Minutes to hold.
  altitude_ft --- Loiter altitude. Accepts -1 (see ALTITUDE INHERITANCE).
  num_steps ----- RK4 step count over the duration.

SOURCES OF ERROR
  - Same RuntimeError as ConstantAltCruiseSegment ("...Aircraft cannot
    sustain this flight condition.") if thrust required exceeds max thrust
    at this altitude/Mach/weight.

===============================================================================
ClimbSegment(start_altitude_ft, end_altitude_ft, schedule, num_steps=100,
             gamma_min_deg=0.05, gamma_max_deg=25, ps_search_steps=None,
             ps_search_ceiling_ft=60000)
===============================================================================
Max thrust climb between altitudes, integrated over altitude via RK4. At every 
altitude step, the flight-path angle (gamma) that trims 
T - D = W sin(gamma) * ka is solved with brentq (solver_climb_descent.py),
where `ka` corrects for TAS changing with altitude on non-constant-Mach schedules.

  start_altitude_ft - Accepts -1 (see ALTITUDE INHERITANCE).
  end_altitude_ft --- EITHER a target altitude (>= 0) OR a Ps-ceiling flag:
                          end_altitude_ft =  35000   climb to 35,000 ft
                          end_altitude_ft =   -300   climb until specific
                                                      excess power decays to
                                                      300 ft/min
                        A negative value is read as -Ps_target_fpm. Because 
                        the eventual ceiling altitude depends on how much 
                        fuel is burned getting there, this runs a brentq 
                        search (see ps_search_steps/ps_search_ceiling_ft) 
                        around the climb itself.
  schedule ---------- A SpeedScheduleBase instance (speed_schedule.py) or a
                      plain float, treated as constant Mach:
                          ClimbSegment(..., schedule=0.78)                  # constant Mach
                          ClimbSegment(..., schedule=CASMachSchedule(...))  # CAS->Mach
  num_steps --------- RK4 step count over the altitude range (for a
                      Ps-ceiling climb, the step count used by the trial 
                      climbs too).
  gamma_min_deg/gamma_max_deg
                      Search bracket (deg) for the trimmed climb angle.
                      gamma_min_deg near 0 avoids a divide-by-zero in
                      dt/dh = 1/(TAS*sin(gamma)). Increase it a 
                      "Rate of climb ~0" error appears. gamma_max_deg caps
                      unrealistically steep solutions. High thrust scenarios
                      will be capped at this gamma.
  ps_search_steps --- Step count used by the trial climbs inside a Ps-ceiling
                      search (end_altitude_ft < 0 only). Defaults to
                      num_steps so the search matches the final reported
                      climb's discretization.
  ps_search_ceiling_ft
                      Upper altitude limit for a Ps-ceiling search.
                      Raise this if a "could not bracket" error appears
                      for an aircraft with a high ceiling.

SOURCES OF ERROR
  - TrimSolverError ("No positive climb angle achievable...") max thrust
    does not exceed drag at gamma_min_deg.
  - ValueError ("Rate of climb/descent numerically ~0...") the solved
    gamma is so close to zero that dt/dh blows up. Usually means
    gamma_min_deg is set too low.
  - Ps-ceiling search only (end_altitude_ft < 0), all TrimSolverError:
      - "...already at or below the ceiling at the start altitude" the
        aircraft can't climb at the start condition.
      - "...exceeds the ceiling at ps_search_ceiling_ft. Increase
        ps_search_ceiling_ft.".
      - "Could not bracket the ... ceiling below ps_search_ceiling_ft...".

===============================================================================
DescentSegment(start_altitude_ft, end_altitude_ft, schedule, num_steps=100,
               gamma_min_deg=0.05, gamma_max_deg=15, ps_search_steps=None,
               ps_search_ceiling_ft=60000)
===============================================================================
Idle thrust descent, identical in structure to ClimbSegment.

SOURCES OF ERROR
  - TrimSolverError ("No descending trim found: thrust exceeds drag...")
    idle thrust exceeds drag at the shallowest allowed angle (gamma_min_deg).
  - Same "Rate of climb/descent numerically ~0..." ValueError as ClimbSegment
    if the solved (negative) gamma is too close to zero.

===============================================================================
CruiseClimbSegment(mach, range_nm, ps_target_fpm=300,
                    altitude_bracket_ft=(1000, 55000), num_steps=100,
                    max_continuous_fraction=1.0, include_climb_term=True)
===============================================================================
Cruise-climb integrated over range like ConstantAltCruiseSegment, but instead
of a fixed altitude, every RK4 stage solves (via brentq) for the altitude
that holds an input specific excess power (following the MIL-STD-3013 
cruise-climb convention), so the aircraft drifts upward as weight drops rather 
than flying level. There is no altitude input, this segment does NOT support -1
altitude inheritance. A later segment can inherit its ending altitude.

  mach ------------------ Cruise-climb Mach.
  range_nm -------------- Cruise distance.
  ps_target_fpm --------- Target specific excess power. Must be > 0.
  altitude_bracket_ft --- (lo, hi) search bracket brentq solves the altitude
                          within at every step. Must bracket the true
                          altitude at every weight flown in the segment.
  num_steps ------------- RK4 step count over the range.
  max_continuous_fraction Fraction of propulsion_model.max_thrust() treated
                          as MAX CONTINUOUS thrust for the Ps calculation.
                          1.0 (default) reproduces the Ps_theor_fpm
                          convention reported by the other segments.
  include_climb_term ---- If True (default), adds the small W*sin(gamma)
                          drift-up term to the required-thrust balance (an
                          extra brentq altitude solve per RK4 stage, to get
                          dh/dW). If False, treats the climb as level at each
                          instant (T = D), which is the classical textbook
                          simplification and slightly faster to run.

SOURCES OF ERROR
  - CruiseClimbError at construction if ps_target_fpm <= 0.
  - CruiseClimbError during run() ("Cannot achieve Ps = ... at <bracket
    lo> ft...") the aircraft can't hold the commanded Ps at the bottom
    of altitude_bracket_ft.
  - CruiseClimbError during run() ("Exceeding Ps = ... at the top of the
    search bracket... Increase altitude_bracket_ft.") the aircraft has
    more Ps than commanded at the top of the bracket.

===============================================================================
AerialRefuelSegment(mach, altitude_ft, transfer_rate_lb_min,
                     fuel_transferred_lb=None, duration_min=None,
                     num_steps=100)
===============================================================================
Constant altitude, constant Mach fuel transfer (onload or offload), integrated
over time via RK4 on the net weight-change ODE dW/dt = transfer_rate -
fuel_flow(W, h, M). See the class docstring in segments/air_refuel.py for the 
full derivation and the fuel accounting convention.

  mach ------------------ Constant Mach flown during the transfer.
  altitude_ft ------------ Constant altitude. Accepts -1 (see ALTITUDE
                          INHERITANCE).
  transfer_rate_lb_min --- Signed rate: > 0 receiving (onload), < 0 donating
                          (offload). Must be non-zero.
  fuel_transferred_lb ---- How much fuel to move (a positive magnitude,
                          regardless of direction). Specify this OR
                          duration_min, not both.
  duration_min ----------- How long to stay connected. Specify this OR
                          fuel_transferred_lb, not both; the other is derived
                          from the constant rate (duration = quantity / |rate|).
  num_steps -------------- RK4 step count over the transfer duration.

SOURCES OF ERROR
  - ValueError at construction if transfer_rate_lb_min == 0.
  - ValueError at construction if neither or both of fuel_transferred_lb /
    duration_min are given, or if the one given is <= 0.
  - AerialRefuelError at the start of run() if the aircraft enters the segment
    already below its zero fuel weight.
  - AerialRefuelError during run() if a donation (transfer_rate_lb_min < 0)
    would drive the aircraft below its zero fuel weight before the transfer
    completes. There is no equivalent upper-bound check on an onload, because
    Aircraft carries no tank-capacity attribute.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import unit_conversions as convert
from aircraft_build import Aircraft

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

class MissionLeg:
    """
    The accepted values of a segment's 'leg' input (see LEG ASSIGNMENT above).

    A plain namespace of strings rather than an enum.Enum, so that a segment
    can be tagged with either the constant or the bare string and the two
    compare equal:

        ClimbSegment(..., leg=MissionLeg.OUTBOUND)
        ClimbSegment(..., leg="outbound")        # identical
    """
    OUTBOUND   = "outbound"     # flown on the way out to the turn point
    INBOUND    = "inbound"      # flown on the way home
    NEUTRAL    = "neutral"      # deliberately counts toward neither leg
    UNASSIGNED = "unassigned"   # the default: not part of a radius mission

    VALID = (OUTBOUND, INBOUND, NEUTRAL, UNASSIGNED)

def _parse_leg_input(leg, class_name: str) -> str:
    """
    Normalize a segment's 'leg' argument.

    None (the default, meaning the caller never mentioned legs) becomes
    MissionLeg.UNASSIGNED. Anything else must be one of MissionLeg.VALID,
    matched case-insensitively and ignoring surrounding whitespace.
    """
    if leg is None:
        return MissionLeg.UNASSIGNED
    if not isinstance(leg, str):
        raise ValueError(
            f"{class_name} leg = {leg!r} is not a valid leg assignment. "
            f"Use one of {MissionLeg.VALID}, or omit 'leg' entirely for a "
            f"one-way mission.")
    normalized = leg.strip().lower()
    if normalized not in MissionLeg.VALID:
        raise ValueError(
            f"{class_name} leg = {leg!r} is not a valid leg assignment. "
            f"Use one of {MissionLeg.VALID}, or omit 'leg' entirely for a "
            f"one-way mission.")
    return normalized

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
    # Air Refuel special case: fuel burned by the aircraft natively (no transfer tracked here)
    AR_AC_flight_burn_kg: Optional[float] = None
    # Which half of a radius mission this segment belongs to. Stamped by
    # Mission.run() from the segment's 'leg' input, so a segment's run()
    # does not have to carry it. See MissionLeg.
    leg: str = MissionLeg.UNASSIGNED

# Base class. Subclasses implement run().
class MissionSegment:
    """
    Base class. Subclasses implement 'run()' specific for the flight segment.
    """
    name = "generic_segment"
    # Default for any segment whose __init__ does not set one (including
    # user-written segments), so Mission.run() can always read segment.leg.
    leg = MissionLeg.UNASSIGNED

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