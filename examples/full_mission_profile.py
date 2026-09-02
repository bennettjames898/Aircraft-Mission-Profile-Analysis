"""
Full mission example: climb, cruise, descent, and a loiter segment. 
This exercises every segment type currently implemented and
is the profile to run when checking that the whole mission chain
(not just individual segments) behaves sensibly.

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
from segments import ClimbSegment, ConstantAltCruiseSegment, DescentSegment, LoiterSegment
from speed_schedule import CASMachSchedule
import unit_conversions as convert
from mission import Mission


def main():
    ### Aircraft Definition
    aircraft = Aircraft(
        name                        = "Generic Narrowbody Twin",
        wing_area_ft2               = 1320,
        operating_empty_weight_lb   = 92500,
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

    # Fuel and Stores
    payload_lb  = 33000
    fuel_lb     = 40000
    zero_fuel_weight_lb = aircraft.operating_empty_weight_lb + payload_lb
    start_weight_lb = aircraft.operating_empty_weight_lb + payload_lb + fuel_lb

    ### Climb and Descent Schedule Definition
    # follow 280 KCAS until M0.78, then follow M0.78.
    KCAS = 280
    MACH = 0.78
    climb_sched   = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    descent_sched = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=MACH)
    
    ### Mision Segments
    MissionSegments = [
        ClimbSegment(start_altitude_ft=1500, end_altitude_ft=35000, schedule=climb_sched, num_steps=100),
        ConstantAltCruiseSegment(altitude_ft=35000, mach=0.78, range_nm=1200, num_steps=200),
        DescentSegment(start_altitude_ft=35000, end_altitude_ft=1500, schedule=descent_sched, num_steps=100),
        LoiterSegment(altitude_ft=1500, mach=0.3, duration_min=20.0, num_steps=200),
    ]

    ### RUN MissionSegments @ start_weight_lb
    mission = Mission(aircraft=aircraft, segments=MissionSegments)
    result = mission.run(start_weight_lb)
    print(result.summary())
    if result.end_weight_lb < zero_fuel_weight_lb:
        print(
            f"\nWARNING: mission ends below zero-fuel weight "
            f"({result.end_weight_lb:.0f} lb < {zero_fuel_weight_lb:.0f} lb). "
            f"Not flyable with the fuel loaded."
        )
    else:
        print(f"\nFuel remaining at end of mission: {result.end_weight_lb - zero_fuel_weight_lb:.0f} lb")

    # --- Build a single altitude-vs-distance profile across all segments ---
    cumulative_distance_nm = 0.0
    profile_distance = []
    profile_altitude = []
    profile_weight = []

    for seg_result in result.segment_results:
        for point in seg_result.history:
            if "altitude_ft" in point:
                profile_distance.append(cumulative_distance_nm + point["distance_nm"])
                profile_altitude.append(point["altitude_ft"])
                profile_weight.append(convert.kg_to_lb(point["weight_kg"]))
            else:
                # Loiter has no distance axis 
                profile_distance.append(cumulative_distance_nm)
                profile_altitude.append(profile_altitude[-1] if profile_altitude else 0.0)
                profile_weight.append(convert.kg_to_lb(point["weight_kg"]))
        cumulative_distance_nm += seg_result.distance_nm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(profile_distance, profile_altitude, color="#1f4e79", linewidth=2)
    ax1.set_ylabel("Altitude (ft)")
    ax1.set_title(f"{aircraft.name} — Full Mission Profile")
    ax1.grid(alpha=0.3)

    ax2.plot(profile_distance, profile_weight, color="#c0504d", linewidth=2)
    ax2.set_xlabel("Distance (nm)")
    ax2.set_ylabel("Weight (lb)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "full_mission_profile.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    main()