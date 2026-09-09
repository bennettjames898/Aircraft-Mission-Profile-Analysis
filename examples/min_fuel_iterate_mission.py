"""
Demonstrates sdolver_mission_fuel.py: solves for the minimum fuel required for
a saircraft to fly the provided mission and end at ZFW + reserves.

===============================================================================
QUICK START 3 of 3 -- SOLVE MINIMUM FUEL (fixed range -> solves fuel loaded)
===============================================================================
Answers "how much fuel does this mission require?"
 
The mission is FIXED, every segment length including the cruise range is an
input. The FUEL WEIGHT is the unknown. solver_mission_fuel.py runs the whole
mission with different fuel loads, using brentq to find the load at which the 
aircraft lands at exactly its zero fuel weight (plus some reserve).
 
    Run with:   python examples/min_fuel_iterate_mission.py
 
 
THE THREE WAYS TO RUN AN ANALYSIS
-------------------------------------------------------------------------------
  1. examples/full_mission_profile.py
        Fixed mission, no iteration.        Fuel and range are both inputs.
  2. examples/max_range_iterate_mission.py  
        Fixed fuel   -> solves cruise range. "How far can it go on this fuel?"
        (solver_mission_range.py)
  3. examples/min_fuel_iterate_mission.py   THIS FILE
        Fixed range  -> solves minimum fuel. "How much fuel does this mission require?"
        (solver_mission_fuel.py)
 
Modes 2 and 3 are inverses and will round trip. Feed this solver's answer for
fuel into mode 2 and you get this mission's range back.
 
See full_mission_profile.py for the full description of STEPS 1-3 (defining
the Aircraft, the speed schedules, and the segment list). They are identical
here. Only STEP 4 differs, and is documented below.
 
 
ONE DIFFERENCE IN STEP 1 (Aircraft definition)
-------------------------------------------------------------------------------
fuel_weight_lb on the Aircraft is only a STARTING POINT for the search. The
solver builds its own trial aircraft at each iteration and never modifies the
one passed in, so whatever is written there does not constrain the answer and
the aircraft object is unchanged afterward. Empty weight and payload DO matter,
and set the zero fuel weight the solver is converging to.
 
 
ONE DIFFERENCE IN STEP 3 (segment list)
-------------------------------------------------------------------------------
Unlike mode 2, the segment list here is COMPLETE. The cruise segment is written
in with its range already set.
 
 
STEP 4 (THIS MODE): call solve_min_fuel_iterate()
-------------------------------------------------------------------------------
This is a wrapper function to utilize the fuel root-find logic found in 
solver_mission_fuel.py.

    result = solve_min_fuel_iterate(
        saveDir            = saveDir,
        aircraft           = aircraft,
        MissionSegmentList = MissionSegments,
        reserve_fuel_lb    = 0,             # <-- see below
        fuel_bracket_lb    = (0, 100000),
        converge_tol       = 0.1,           # lb
    )
 
INPUTS:
  reserve_fuel_lb [lb]
      Fuel that must still be in the tanks at the end of the mission. The
      default of 0 solves for burning every pound. This is fuel held
      in RESERVE, not fuel burned by a reserve segment. If the mission
      already ends with a LoiterSegment (for example), that burn is counted in 
      the mission itself. Take care not to double count resever fuel 
      requirements.
 
  fuel_bracket_lb (tuple) [lb]
      Initial search bracket. The upper bound is doubled automatically (up to
      10 times) if the mission fails.
 
  converge_tol [lb]
      Convergence tolerance on the fuel weight.
 
  This function will raise a MissionFuelSizingError if no fuel load can 
  complete the mission.
 
 
READING THE RESULT
-------------------------------------------------------------------------------
  Returns a MinFuelIteratedResult:
      .fuel_weight_lb     solved fuel load required
      .reserve_fuel_lb    the input end-of-mission fuel load
      .mission_result     the MissionResult for the converged mission
      .residual_lb        fuel remaining beyond target, should be ~0
      .iterations         how many full missions were flown to converge
 
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
from solver_mission_fuel import solve_min_fuel_iterate


def main():
    ### Save Directory
    saveDir = ".//"
    
    ### Aircraft Definition
    aircraft = Aircraft(
        name                        = "MockB787-IterFuel",
        wing_area_ft2               = 3501,
        operating_empty_weight_lb   = 239200,        
        payload_weight_lb           = 47040,
        fuel_weight_lb              = 189760, # This value will be ignored by solve_min_fuel_iterate()
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
    climb_sched2   = CASMachSchedule(cas_m_s=convert.kt_to_ms(277), mach=MACH)
    descent_sched2 = CASMachSchedule(cas_m_s=convert.kt_to_ms(250), mach=MACH)

    ### Mision Segments
    MissionSegments = [
        GroundOps(duration_min=60, throttle_set_pct=0.0),
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=37000, schedule=climb_sched2, num_steps=100),
        AccelDecelSegment(altitude_ft=37000, start_mach=MACH, end_mach=0.85, num_steps=20),
        ConstantAltCruiseSegment(altitude_ft=37000, mach=0.85, range_nm=3000, num_steps=300),
        AccelDecelSegment(altitude_ft=37000, start_mach=0.85, end_mach=MACH, num_steps=20),
        DescentSegment(start_altitude_ft=37000, end_altitude_ft=1500, schedule=descent_sched2, num_steps=100),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=20, num_steps=200),
        GroundOps(duration_min=60, throttle_set_pct=0),
    ]
    
    ## Mission solver with iteration on a cruise segment to maximize range
    result = solve_min_fuel_iterate(
        saveDir                 = saveDir,
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments,
        reserve_fuel_lb         = 0,
        fuel_bracket_lb         = (0, 200000),
        converge_tol            = 0.1, # nm
    )

    if saveDir is None:
        print(result.mission_result.summary)

if __name__ == "__main__":
    main()