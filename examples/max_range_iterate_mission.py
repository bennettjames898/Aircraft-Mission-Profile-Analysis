"""
Demonstrates mission_sizing.py: solves for the maximum range this
aircraft can fly on a given fuel load, with a fixed climb, descent, and
diversion reserve, converging until the mission lands at exactly
zero-fuel weight.

Run with:  python3 examples/max_range_mission.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import GroundOps, ClimbSegment, DescentSegment, LoiterSegment
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from solver_mission_range import solve_cruise_range_iterate


def main():
    ### Save Directory
    saveDir = ".//"
    
    ### Aircraft Definition
    aircraft = Aircraft(
        name                        = "Generic Narrowbody Twin",
        wing_area_ft2               = 1320,
        operating_empty_weight_lb   = 92500,
        payload_weight_lb           = 33000,
        fuel_weight_lb              = 40000,
        aero_model=SimpleDragPolar(
            cd0                 = 0.020,
            aspect_ratio        = 9.5, 
            oswald_efficiency   = 0.80
            ),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_n  = 120000,
            tsfc_kg_per_n_per_s = 1.75e-5,
            num_engines         = 2,
        ),
    )

    ### Climb and Descent Schedule Definition
    # follow 280 KCAS until M0.78, then follow M0.78.
    KCAS = 280
    MACH = 0.78
    climb_sched   = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    descent_sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)

    ### Mision Segments
    MissionSegments = [
        GroundOps(duration_min=60, throttle_set_pct=0.0),
        GroundOps(duration_min=1, throttle_set_pct=1.0),
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=100),
        # iterated ConstantAltCruiseSegment goes here (IndexToPlaceIteration = 1)
        DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=descent_sched, num_steps=100),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=20, num_steps=200),
        GroundOps(duration_min=60, throttle_set_pct=0),
    ]
    
    ## Mission solver with iteration on a cruise segment to maximize range
    result = solve_cruise_range_iterate(
        saveDir                 = saveDir,
        aircraft                = aircraft,
        MissionSegmentList      = MissionSegments,
        IndexToPlaceIteration   = 2,
        cruise_altitude_ft      = 35000,
        cruise_mach             = 0.78,
        cruise_num_steps        = 100,
        range_bracket_nm        = (0, 6000),
        converge_tol            = 0.1, # nm
    )

    # print(f"Payload: {aircraft.payload_weight_lb:.0f} lb, Fuel: {aircraft.fuel_weight_lb:.0f} lb, "
    #       f"Takeoff weight: {aircraft.gross_weight_lb:.0f} lb")
    # print(f"Converged in {result.iterations} iterations "
    #       f"(residual: {result.residual_lb:.4f} lb)\n")
    # print(f"Total mission distance: {result.mission_result.total_distance_nm:.1f} nm\n")
    # print(result.mission_result.summary())

if __name__ == "__main__":
    main()