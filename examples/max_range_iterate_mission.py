"""
Demonstrates mission_sizing.py: solves for the maximum range this
aircraft can fly on a given fuel load, with a fixed climb, descent, and
diversion reserve, converging until the mission lands at exactly
zero-fuel weight.

===============================================================================
QUICK START 2 of 3 -- SOLVE MAXIMUM RANGE (fixed fuel -> solves cruise range)
===============================================================================
Answers "how far can this airplane fly on the fuel it is carrying?"
 
Fuel is FIXED (from the Aircraft() definition). The length of ONE cruise
segment is the unknown. solver_mission_range.py runs the mission with different 
cruise ranges, using brentq to find the range at which the aircraft lands at 
exactly its zero fuel weight.
 
    Run with:   python examples/max_range_iterate_mission.py
 
 
THE THREE WAYS TO RUN AN ANALYSIS
-------------------------------------------------------------------------------
  1. examples/full_mission_profile.py
        Fixed mission, no iteration.        Fuel and range are both inputs.
  2. examples/max_range_iterate_mission.py  THIS FILE
        Fixed fuel   -> solves cruise range. "How far can it go on this fuel?"
        (solver_mission_range.py)
  3. examples/min_fuel_iterate_mission.py
        Fixed range  -> solves minimum fuel. "How much fuel does this mission require?"
        (solver_mission_fuel.py)
 
Modes 2 and 3 are inverses and will round trip. Feed this solver's answer for
fuel into mode 3 and you get this mission's range back.
 
See full_mission_profile.py for the full description of STEPS 1-3 (defining
the Aircraft, the speed schedules, and the segment list). They are identical
here. Only STEP 4 differs, and is documented below.
 
 
STEP 4 (THIS MODE): call solve_cruise_range_iterate()
-------------------------------------------------------------------------------
This is a wrapper function to utilize the range root-find logic found in 
solver_mission_range.py. The wrapper places a cruise segment into the provided 
mission segment list and iterates that segment's range to locate the zero-fuel 
condition.
 
    result = solve_cruise_range_iterate(
        saveDir               = saveDir,
        aircraft              = aircraft,
        MissionSegmentList    = MissionSegments,
        IndexToPlaceIteration = 9,
        cruise_altitude_ft    = 41000,
        cruise_mach           = 0.85,
        cruise_num_steps      = 300,
        range_bracket_nm      = (0, 6000),
        converge_tol          = 0.1,
    )
 
INPUTS: 
  IndexToPlaceIteration [0-base array position]
      The list index where the solved cruise segment gets inserted. It is a
      plain list 'insert()', so the segment currently at that index and onwards
      shifts back one. DO NOT put a cruise segment there manually,
      the solver builds it fresh at every iteration. Mark the spot with a
      comment in the segment list (as done below) so the index stays obvious
      if segments are added or removed later. Double-check the summary output 
      to ensure the iterated segment is placed in the correct order.
 
  cruise_altitude_ft [ft] / cruise_mach [nd]
      The conditions the iterated cruise leg flies at.
 
  range_bracket_nm [nm]
      Initial search bracket. The upper bound is doubled automatically (up to
      10 times) if the aircraft still has fuel at that range. If it cannot 
      bracket an answer it raises MissionSizingError rather than returning 
      meaningless results.
 
  converge_tol [nm]
      Convergence tolerance on the cruise range.
 
 
READING THE OUTPUT
-------------------------------------------------------------------------------
  Returns a MaxRangeIteratedResult:
      .cruise_range_nm    solved length of the iterated cruise segment
      .mission_result     the MissionResult for the converged mission
      .residual_lb        fuel left at the end, should be ~0
      .iterations         how many full missions were flown to converge
 
  NOTE .cruise_range_nm is the ITERATED SEGMENT distance. Total mission distance
  (including climb, descent, and all cruise legs) is recorded in
  .mission_result.total_distance_nm.
 
  saveDir behaves the same as in mode 1. Intermediate iterations are not 
  written to an output file, only the final converged mission.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import GroundOps, ConstantAltCruiseSegment, ClimbSegment, DescentSegment, LoiterSegment, AccelDecelSegment
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from solver_mission_range import solve_cruise_range_iterate


def main():
    ### Save Directory
    saveDir = ".//"
    
    ### Aircraft Definition
    aircraft = Aircraft(
        name                        = "MockB787-IterRange",
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
    # follow 280 KCAS until M0.78, then follow M0.78.
    MACH = 0.825
    climb_sched1   = CASMachSchedule(cas_m_s=convert.kt_to_ms(250), mach=MACH)
    climb_sched2   = CASMachSchedule(cas_m_s=convert.kt_to_ms(277), mach=MACH)
    descent_sched1 = CASMachSchedule(cas_m_s=convert.kt_to_ms(277), mach=MACH)
    descent_sched2 = CASMachSchedule(cas_m_s=convert.kt_to_ms(250), mach=MACH)

    ### Mision Segments
    MissionSegments = [
        GroundOps(duration_min=60, throttle_set_pct=0.0),
        GroundOps(duration_min=1, throttle_set_pct=1.0),
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=10000, schedule=climb_sched1, num_steps=100),
        ClimbSegment(start_altitude_ft=10000, end_altitude_ft=37000, schedule=climb_sched2, num_steps=100),
        AccelDecelSegment(altitude_ft=37000, start_mach=0.825, end_mach=0.85, num_steps=20),
        ConstantAltCruiseSegment(altitude_ft=37000, mach=0.85, range_nm=2750, num_steps=300),
        AccelDecelSegment(altitude_ft=35000, start_mach=0.85, end_mach=0.825, num_steps=20),
        ClimbSegment(start_altitude_ft=37000, end_altitude_ft=41000, schedule=climb_sched2, num_steps=100),
        AccelDecelSegment(altitude_ft=41000, start_mach=0.825, end_mach=0.85, num_steps=20),
        # iterated ConstantAltCruiseSegment goes here (IndexToPlaceIteration = 9)
        DescentSegment(start_altitude_ft=41000, end_altitude_ft=10000, schedule=descent_sched1, num_steps=100),
        DescentSegment(start_altitude_ft=10000, end_altitude_ft=50, schedule=descent_sched2, num_steps=100),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=20, num_steps=200),
        GroundOps(duration_min=60, throttle_set_pct=0),
    ]
    
    ## Mission solver with iteration on a cruise segment to maximize range
    result = solve_cruise_range_iterate(
        saveDir                 = saveDir,
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments,
        IndexToPlaceIteration   = 9,
        cruise_altitude_ft      = 41000,
        cruise_mach             = 0.85,
        cruise_num_steps        = 300,
        range_bracket_nm        = (0, 6000),
        converge_tol            = 0.1, # nm
    )

    if saveDir is None:
        print(result.mission_result.summary)

if __name__ == "__main__":
    main()