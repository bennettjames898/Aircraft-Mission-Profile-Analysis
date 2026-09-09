"""
Full mission example: climb, cruise, descent, and a loiter segment. 
This exercises every segment type currently implemented and
is the profile to run when checking that the whole mission chain behaves properly.

===============================================================================
QUICK START 1 of 3 -- RUN A FIXED MISSION (no mission-level iteration)
===============================================================================
Flies a fully specified mission exactly as written and reports how much fuel 
remains at conclusion. Every segment length is an input, nothing is iteratively 
solved for at the mission level.
 
Use this mode when you already know the mission and want the fuel burn, time,
and distance. This method exercises every segment type currently implemented, 
so it is also the profile to run when checking that the whole mission chain 
behaves as expected.
 
    Run with:   python examples/full_mission_profile.py
 
THE THREE WAYS TO RUN AN ANALYSIS
-------------------------------------------------------------------------------
  1. examples/full_mission_profile.py       THIS FILE
        Fixed mission, no iteration.        Fuel and range are both inputs.
  2. examples/max_range_iterate_mission.py
        Fixed fuel   -> solves cruise range. "How far can it go on this fuel?"
        (solver_mission_range.py)
  3. examples/min_fuel_iterate_mission.py
        Fixed range  -> solves fuel loaded.  "What fuel does this trip need?"
        (solver_mission_fuel.py)
 
All three share the same initial three setup steps below and differ only in the
final step. Modes 2 and 3 are inverses of each other and will round trip: the
fuel that mode 3 solves for a given range is the fuel that makes mode 2 return
that same range.
 
 
THE WORKFLOW (steps 1-3 are identical in all three modes)
-------------------------------------------------------------------------------
  STEP 1  Define the Aircraft.
          An Aircraft is one SPECIFIC LOADED AIRPLANE, not an aircraft type.
          Empty weight, payload, and fuel are all fixed at construction, and
          gross_weight_lb / zero_fuel_weight_lb are derived from them. To
          analyze a different loading, build another Aircraft. Mission.run()
          takes no weight argument, it reads aircraft.gross_weight_lb.
            - Weights and areas are POUNDS and FEET on the public API.
            - DISAF is the non-standard day temperature offset in deg F
              (DISAF = 0 is a standard day, DISAF = 45 is ISA+25C).
            - aero_model and propulsion_model are swappable depending on what 
              models are coded into those containing files.
 
  STEP 2  Define the climb/descent speed schedules.
          CASMachSchedule(cas_m_s=..., mach=...) locates the MACH/CAS crossover 
          speed internally. Pass a bare float instead of a schedule object for 
          a simple constant-Mach climb.
 
  STEP 3  Build the mission segment list, in flight order.
          Available segments:
            GroundOps(duration_min, throttle_set_pct)
                Taxi / ground burn. throttle_set_pct interpolates between
                idle (0.0) and max (1.0) thrust. No distance covered.
            ClimbSegment(start_altitude_ft, end_altitude_ft, schedule, num_steps)
                Max thrust climb. Integrates over ALTITUDE, solving the
                flight path angle at every step.
            DescentSegment(start_altitude_ft, end_altitude_ft, schedule, num_steps)
                Idle thrust descent. Same methods as climb.
            ConstantAltCruiseSegment(mach, range_nm, altitude_ft, num_steps)
                Level cruise for a set distance. Integrates over RANGE.
            LoiterSegment(mach, duration_min, altitude_ft, num_steps)
                Level cruise for a set time. Integrates over TIME. no distance 
                credit (intended for resevres).
            AccelDecelSegment(altitude_ft, start_mach, end_mach, num_steps)
                Level flight speed change at constant altitude. Max thrust if
                end_mach > start_mach, idle thrust if it is lower.
 
          NOTE: segments do not know about each other's initial/final 
          conditions. Nothing checks that one segment's end altitude matches 
          the next one's start altitude, so a typo will "teleport" the aircraft 
          between conditions.
 
  STEP 4  Run it. THIS is the step that differs between the three modes.
          Here: build a Mission and call .run() directly.
 
              mission = Mission(aircraft, MissionSegments, saveDir)
              result  = mission.run()
 
 
OUTPUT / saveDir
-------------------------------------------------------------------------------
  saveDir = None      Nothing is written. Use `print(result.summary)` to get
                      the summary table on the console.
  saveDir = ".//"     Writes the summary table and the full time history to
                      files in that directory (current folder here).
 
  result is a MissionResult carrying total_fuel_burned_lb, end_weight_lb,
  total time and distance, and per-segment results with a time history of each 
  segment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import GroundOps, ClimbSegment, ConstantAltCruiseSegment, DescentSegment, LoiterSegment, AccelDecelSegment
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from mission import Mission

def main():
    ### Save Directory
    saveDir = ".//"    
    
    ### Aircraft Definition
    aircraft = Aircraft(
        name                        = "MockB787-Simple",
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
    KCAS = 250
    MACH = 0.75
    climb_sched   = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    descent_sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    
    ### Mision Segments
    MissionSegments = [
        GroundOps(duration_min=60, throttle_set_pct=0),
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=100),
        AccelDecelSegment(altitude_ft=35000, start_mach=0.75, end_mach=0.85, num_steps=100),
        ConstantAltCruiseSegment(altitude_ft=35000, mach=0.85, range_nm=1200, num_steps=200),
        AccelDecelSegment(altitude_ft=35000, start_mach=0.85, end_mach=0.75, num_steps=100),
        DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=descent_sched, num_steps=100),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=20.0, num_steps=100),
    ]

    ### BUILD & RUN a noniterative mission @ aircraft.gross_weight_lb
    # No changes to segment ranges will occur, excess fuel may remain at end.
    mission = Mission(
        aircraft=aircraft, 
        segments=MissionSegments, 
        saveDir=saveDir
    )
    result = mission.run() # runs the mission
    if saveDir is None:
        print(result.summary) # print summary to cmd line (if saveDir=None)

if __name__ == "__main__":
    main()