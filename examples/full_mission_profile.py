"""
Full mission example: climb, cruise, descent, and a loiter segment. 
This exercises every segment type currently implemented and
is the profile to run when checking that the whole mission chain
(not just individual segments) behaves properly.

Run with:  python3 examples/full_mission_profile.py
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