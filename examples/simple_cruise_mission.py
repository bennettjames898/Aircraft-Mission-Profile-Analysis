"""
End-to-end example: a narrowbody-class aircraft flying a cruise segment
followed by a holding-pattern loiter (e.g. a diversion reserve), using
only the components built so far (aero, propulsion, aircraft, segments,
mission). Climb/descent segments are a natural next addition -- this
example only requires steady-level flight physics, which is what's
implemented today.

Run with:  python3 examples/simple_cruise_mission.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving PNGs
import matplotlib.pyplot as plt

from aero_model import SimpleDragPolar
from propulsion_model import SimpleTurbofan
from aircraft_build import Aircraft
from segments import FixedCruiseSegment, LoiterSegment
from mission import Mission

## Run a performance analysis
def Example_Mission_Scenario():
    
    ## Construct Aircraft Model (Aero & Prop data)
    aircraft = Aircraft(
        name                        = "Generic Narrowbody Twin",
        wing_area_m2                = 122.6,
        operating_empty_weight_kg   = 42000,
        # max_fuel_weight_kg          =20000,
        # max_payload_weight_kg       =20000,
        aero_model=SimpleDragPolar(
            cd0                 = 0.020, 
            aspect_ratio        = 9.5, 
            oswald_efficiency   = 0.80),
        propulsion_model=SimpleTurbofan(
            sea_level_thrust_n  = 120000,
            tsfc_kg_per_n_per_s = 1.75e-5,
            num_engines         = 2,
        ),
    )
    
    # Construct starting weight with fuel and payload
    payload_kg      = 15000
    fuel_kg         = 18000
    start_weight_kg = aircraft.operating_empty_weight_kg + payload_kg + fuel_kg

    ## Construct the mission profile by segments
    # (cruise at 35kft/M0.78 for 1,500nm, hold 30min at 10kft)
    MissionProfile = [FixedCruiseSegment(altitude_ft=35000, mach=0.78, range_nm=1500.0, num_steps=100),
    LoiterSegment(altitude_ft=10000, mach=0.35, duration_min=30.0, num_steps=10)]

    ####### RUN mission #######
    mission = Mission(
        aircraft = aircraft,
        segments = MissionProfile
        )
    result = mission.run(start_weight_kg)
    print(result.summary())

    # Sanity checks:
    zero_fuel_weight_kg = aircraft.operating_empty_weight_kg + payload_kg
    if result.end_weight_kg < zero_fuel_weight_kg:
        print(
            f"\nWARNING: mission ends below zero-fuel weight "
            f"({result.end_weight_kg:.0f} kg < {zero_fuel_weight_kg:.0f} kg)."
        )
    else:
        remaining_fuel_kg = result.end_weight_kg - zero_fuel_weight_kg
        print(f"\nFuel remaining: {remaining_fuel_kg:.0f} kg")

    # --- Plot: weight vs. cumulative distance for the cruise segment ---
    cruise_result = result.segment_results[0]
    distances_nm = [h["distance_nm"] for h in cruise_result.history]
    weights_kg = [h["weight_kg"] for h in cruise_result.history]
    l_over_d = [h["l_over_d"] for h in cruise_result.history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(distances_nm, weights_kg, color="#1f4e79", linewidth=2)
    ax1.set_ylabel("Aircraft Weight (kg)")
    ax1.set_title(f"{aircraft.name} — Cruise Fuel Burn (35,000 ft, M0.78)")
    ax1.grid(alpha=0.3)

    ax2.plot(distances_nm, l_over_d, color="#c0504d", linewidth=2)
    ax2.set_xlabel("Distance (nm)")
    ax2.set_ylabel("L/D")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "cruise_mission_output.png")
    # plt.savefig(output_path, dpi=150)
    # print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    Example_Mission_Scenario()
