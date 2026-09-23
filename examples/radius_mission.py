"""
Demonstrates solver_mission_range.py:solve_radius_iterate. Solves for the
maximum radius this aircraft can fly on a given fuel load. Both the outbound 
and inbound legs cover the same ground distance. The mission iterates cruise 
segments on each leg to complete the mission at zero-fuel weight.

===============================================================================
QUICK START 4 of 4 -- SOLVE MAXIMUM RADIUS (fixed fuel)
===============================================================================
Answers "how far out and back can this airplane fly on the fuel it is
carrying, if it has to return to where it started?"

    Run with:   python examples/radius_mission.py

THE FOUR WAYS TO RUN AN ANALYSIS
-------------------------------------------------------------------------------
  1. examples/full_mission_profile.py
        Fixed mission, no iteration.        Fuel and range are both inputs.
  2. examples/max_range_iterate_mission.py
        Fixed fuel   -> solves cruise range. "How far can it go on this fuel?"
        (solver_mission_range.py:solve_cruise_range_iterate)
  3. examples/min_fuel_iterate_mission.py
        Fixed range  -> solves minimum fuel. "How much fuel does this mission require?"
        (solver_mission_fuel.py)
  4. examples/radius_mission.py             THIS FILE
        Fixed fuel   -> solves a radius, flying out and back to base. 
        (solver_mission_range.py:solve_radius_iterate)

Mode 4 is a variant of mode 2: instead of one unknown cruise range, there are
two (outbound and inbound), solved together so both legs end on the same
radius. See full_mission_profile.py for the full description of STEPS 1-3
(defining the Aircraft and the speed schedules). Only the segment list (STEP
3) and the solver call (STEP 4) differ, documented below.

ONE DIFFERENCE IN STEP 3 (segment list, and the 'leg' tag)
-------------------------------------------------------------------------------
Every segment that covers ground on the flight out is tagged leg="outbound", 
and every segment on the way back is tagged leg="inbound" (import MissionLeg 
from segments, or just pass the plain string). Segments that
do not assess mission distance (i.e. GroundOps, LoiterSegment) can be left 
untagged.

    ClimbSegment(1500, 35000, schedule=climb_sched, leg="outbound", ...)
    DescentSegment(-1, 1500, schedule=descent_sched, leg="inbound", ...)

Outbound segments must all come before inbound segments (no interleaving),
and at least one segment of each leg must be present, or solve_radius_iterate
raises RadiusMissionError. See segments/base.py (MissionLeg) and
solver_mission_range.py for the full set of guards.

Like mode 2, this segment list is incomplete: it holds everything except the
two cruise segments the solver builds itself at every iteration.

STEP 4: call solve_radius_iterate()
-------------------------------------------------------------------------------
This is a wrapper function, similar to `solve_cruise_range_iterate` but solving 
two cruise segments (outbound and inbound) at once so both legs measure the 
same radius.

    result = solve_radius_iterate(
        saveDir                         = saveDir,
        aircraft                        = aircraft,
        MissionSegmentList              = MissionSegments,
        OutboundIndexToPlaceIteration   = 2,
        InboundIndexToPlaceIteration    = 5,
        outbound_cruise_altitude_ft     = 41000,
        outbound_cruise_mach            = 0.85,
        inbound_cruise_altitude_ft      = 41000,
        inbound_cruise_mach             = 0.85,
        cruise_num_steps                = 300,
        radius_bracket_nm               = (200, 4000),
        converge_tol                    = 0.1,
        leg_tol_nm                      = 0.05,
    )

INPUTS:
  OutboundIndexToPlaceIteration / InboundIndexToPlaceIteration [0-base array position]
      insert() locations (in the ORIGINAL, uninserted MissionSegmentList) for
      the outbound and inbound cruise segments. InboundIndexToPlaceIteration
      must be the larger of the two. As with mode 2, do not put a cruise
      segment there manually.

  outbound_cruise_altitude_ft / outbound_cruise_mach
  inbound_cruise_altitude_ft / inbound_cruise_mach
      Conditions flown on each leg's iterated cruise segment.

  radius_bracket_nm [nm]
      Initial search bracket on the entire radius (not a single cruise range). 
      The lower bound is raised automatically if it is too small to fit a 
      cruise within one of the out/inbound legs. The upper bound is doubled 
      (up to 10 times) if fuel is left over.

  converge_tol [nm]
      Convergence tolerance on the radius, passed to the outer brentq.

  leg_tol_nm [nm]
      How closely the two legs must match the radius before the inner
      outbound/inbound balance is considered converged (see the module
      docstring in solver_mission_range.py for how the inner loop works).

READING THE OUTPUT
-------------------------------------------------------------------------------
  Returns MaxRadiusIteratedResult:
      .radius_nm            solved out-and-back radius (both legs measure this)
      .outbound_cruise_nm   solved length of the outbound cruise segment
      .inbound_cruise_nm    solved length of the inbound cruise segment
      .mission_result       the MissionResult for the converged mission
      .residual_lb          fuel left at the end, should be ~0
      .iterations           how many full missions were flown to converge
      .leg_imbalance_nm     |outbound leg - inbound leg| at convergence

  NOTE `.outbound_cruise_nm` / `.inbound_cruise_nm` are shorter than 
  `.radius_nm`. These values denote the cruise segment iterated on the achieve 
  the misison radius.

  saveDir behaves the same as in mode 1. The mission summary also prints a
  RADIUS block (outbound/inbound leg distance and imbalance) whenever either
  leg carries distance -- see mission.py:MissionResult.buildSummary().
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import GroundOps, ClimbSegment, DescentSegment, LoiterSegment, MissionLeg
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from solver_mission_range import solve_radius_iterate


def main():
    ### Save Directory
    saveDir = ".//"

    ### Aircraft Definition
    aircraft = Aircraft(
        name                        = "MockB787-Radius",
        wing_area_ft2               = 3501,
        operating_empty_weight_lb   = 239200,
        payload_weight_lb           = 47040,
        fuel_weight_lb              = 189760,
        aero_model=SimpleDragPolar(
            cd0                 = 0.02,
            aspect_ratio        = 10.58,
            oswald_efficiency   = 0.9,
            mach_crit           = 0.86,
        ),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_lbf    = 50066,
            tsfc_lb_per_lbfhr       = 0.5279,
            num_engines             = 2,
            lapse_exponent          = 0.8,
            idle_thrust_fraction    = 0.05,
        ),
        DISAF = 0,
    )

    ### Climb and Descent Schedule Definition
    # follow 280 KCAS until M0.78, then follow M0.78, both directions.
    MACH = 0.78
    climb_sched   = CASMachSchedule(cas_m_s=convert.kt_to_ms(280), mach=MACH)
    descent_sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(280), mach=MACH)

    ### Mission Segments (INCOMPLETE -- the two cruise segments are inserted
    ### by solve_radius_iterate at OutboundIndexToPlaceIteration / InboundIndexToPlaceIteration)
    MissionSegments = [
        GroundOps(duration_min=60, throttle_set_pct=0.0),                                                                               # 0
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=100, leg=MissionLeg.OUTBOUND),      # 1
        # iterated outbound ConstantAltCruiseSegment goes here (OutboundIndexToPlaceIteration = 2)
        DescentSegment(start_altitude_ft=-1, end_altitude_ft=25000, schedule=descent_sched, num_steps=100, leg=MissionLeg.OUTBOUND),    # 2
        LoiterSegment(altitude_ft=-1, mach=0.5, duration_min=20.0, num_steps=100),                                                      # 3, on station, untagged
        ClimbSegment(start_altitude_ft=-1, end_altitude_ft=37000, schedule=climb_sched, num_steps=100, leg=MissionLeg.INBOUND),         # 4
        # iterated inbound ConstantAltCruiseSegment goes here (InboundIndexToPlaceIteration = 5)
        DescentSegment(start_altitude_ft=-1, end_altitude_ft=1500, schedule=descent_sched, num_steps=100, leg=MissionLeg.INBOUND),      # 5
        GroundOps(duration_min=60, throttle_set_pct=0.0),                                                                               # 6
    ]

    ## Mission solver with iteration on two cruise segments (outbound + inbound) to maximize radius
    result = solve_radius_iterate(
        saveDir                         = saveDir,
        aircraft                        = aircraft,
        MissionSegmentList              = MissionSegments,
        OutboundIndexToPlaceIteration   = 2,
        InboundIndexToPlaceIteration    = 5,
        outbound_cruise_altitude_ft     = 41000,
        outbound_cruise_mach            = 0.85,
        inbound_cruise_altitude_ft      = 41000,
        inbound_cruise_mach             = 0.85,
        cruise_num_steps                = 300,
        radius_bracket_nm               = (200, 4000),
        converge_tol                    = 0.1,   # nm
        leg_tol_nm                      = 0.05,  # nm
    )

    if saveDir is None:
        print(result.mission_result.summary)

if __name__ == "__main__":
    main()
