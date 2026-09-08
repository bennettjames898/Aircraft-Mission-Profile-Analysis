# Aircraft-Mission-Profile-Analysis

[![Tests](https://github.com/bennettjames898/Aircraft-Mission-Profile-Analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/bennettjames898/Aircraft-Mission-Profile-Analysis/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/github/license/bennettjames898/Aircraft-Mission-Profile-Analysis)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A solver to analyze the performance of a conceptual aircraft along a 
user-defined mission profile. 

## Why this project
I transitioned into more general mod/sim roles after spending several years in 
the aeronautics industry as an aircraft performance engineer. With this time 
away from the field, I figured an example portfolio of my skillsets would help 
in the job search and provide recruiters confidence in my abilities. I also
hated the mission profile analysis tool I used at work, and never got funding 
to rewrite it.

## Tool theory of operation
Mission analysis is fundamentally an iterative problem: weight
decreases continuously as fuel burns, which changes the lift coefficient
required for level flight, which changes L/D and fuel flow, which
changes the burn rate going forward. You can't solve for total fuel burn
algebraically except under simplifying assumptions (Breguet). 
`ConstantAltCruiseSegment` solves the coupled problem by numerically
integrating the weight-vs-distance ODE, and validates that integration
against the closed-form Breguet range equation as a unit test
(`tests/test_breguet_range_check.py`).

## Architecture
```
aero_model.py           - Aero interface + simple parabolic drag polar implementation
aircraft_build.py 	    - Aircraft class: wraps geometry, weights, aero + propulsion models
atmosphere.py           - ISA atmosphere model (temp, pressure, density, speed of sound)
solver_mission_range.py - Outer-loop solver for max range given fixed fuel
solver_climb_descent.py - Holds the brentq trim solution used in Climb/Descent
mission.py              - Mission class: sequences segments, carries weight forward
propulsion_model.py     - Propulsion interface + simple constant-TSFC turbofan implementation
segments.py             - MissionSegment base class, ConstantAltCruiseSegment (RK4), LoiterSegment (RK4), ClimbSegment/DescentSegment (brentq)
speed_schedule.py       - Climb/descent speed schedules (constant Mach/TAS/CAS, CAS/Mach crossover)
unit_conversions.py     - Collection of unit conversions used across the project
examples/               - Runnable end-to-end mission scripts
tests/                  - Validation tests (Breguet convergence, climb/descent validation, speed schedule)
```

**Design principle:** `aero_model.py` and `propulsion_model.py` define
abstract interfaces (`AeroModelBase`, `PropulsionModelBase`). Everything
downstream — `Aircraft`, `segments.py`, `Mission` — only calls those
interface methods. This means a real aero deck (CFD-derived lookup
table, DATCOM build-up) or a real engine cycle deck can be substituted
by writing one new class, with no changes necessary for the solver code.

## Physics implemented
- **ISA atmosphere** (0–20 km), including an ISA+ΔT offset option.
- **Steady, level trim**: L = W, T = D solved at each point via the
  required-CL relationship.
- **Climbing/descending trim**: L = W cos(γ), T − D = W sin(γ), where D
  depends on CL which depends on γ. The implicit equation is solved 
  numerically via `scipy.optimize.brentq` in `solver_climb_descent.py`, at every point 
  along the climb/descent profile.
- **Climb acceleration correction**: Excess thrust required to accelerate 
  in TAS is accounted for in climbs and descents by the factor
  `ka = 1 + (V/g)(dV/dh)` in the force balance (`solver.py`), computed
  from the schedule's `dtas_dh` at every point. See the
  references and full derivation in `solver_climb_descent.py`'s module 
  docstring (Marchman, *Aerodynamics and Aircraft Performance*, Virginia Tech)
- **Coupled weight/fuel-burn integration**: 4th-order Runge-Kutta on
  `dW/dx = -fuel_flow / V` for cruise, `dW/dt = -fuel_flow` for loiter
- **Breguet range equation** as an independent closed-form check on the
  numerical integrator.
- **Mission-level max-range sizing** (`solver_mission_range.py`): given a
  fixed fuel weight, this tool solves for the maximum cruise range. This is 
  a distinct iteration loop wrapping the entire mission in an outer root-find 
  rather than iterating within a single segment. 

## Quick start
```bash
pip install -r requirements.txt
python3 examples/full_mission_profile.py
python3 examples/max_range_iterate_mission.py
python3 -m pytest tests/ -v
```
The examples showcase a simple and fully iterated mission for an example 
airliner based on the B787. Both will output two text files showing a summary 
of the mission as well as a time-history output of each segment.

Defining a climb/descent schedule:
```python
from speed_schedule import CASMachSchedule, ConstantMachSchedule

# 280 kt CAS to M0.78, then constant M0.78
schedule = CASMachSchedule(cas_m_s=kt_to_ms(280), mach=0.78)

# Or just pass a float for constant-Mach behavior
# ClimbSegment/DescentSegment accept either
climb = ClimbSegment(start_altitude_ft=0, end_altitude_ft=35000, schedule=schedule)
climb_simple = ClimbSegment(start_altitude_ft=0, end_altitude_ft=35000, schedule=0.78)
```

## Validation
`tests/test_breguet_range_check.py` checks that the numerically
integrated cruise segment agrees with the closed-form Breguet range
equation to within a tight tolerance.
`tests/test_speed_schedule.py` checks that the various behaviors required to 
construct different speed schedules behaves as intended. 
`tests/test_climb_descent.py` checks that the behaviors expected in climb and 
descent are appearing normal, the trimmed gamma solution is believable, 
and that the climb acceleration correction 'ka' is properly accounted.
`tests/test_max_cruise_iterate.py` checks that the max cruise range iteration 
is providing realistic outputs, and that its error catching is functioning.
`tests/test_accel_decel.py` checks the behaviors of the acceleration segment.

Individual file outputs are checked against a hand-computable or
independently-derivable reference in the corresponding test or
`__name__ == "__main__"` block of each `.py` file

## Simplifications & Assumptions
- Constant TSFC propulsion model (no altitude/Mach/throttle variation)
- Idle thrust is modeled as a fixed fraction of max thrust at the same
  altitude/Mach.
- Simple aero model is whole aircraft and assumes critical mach behavior 
  based on Anderson textbook methods.
- Missions are only ran in a single 'direction' (no radius missions, or 
  outbound/inbound legs).
- Mission segment continuity is ignored between segments. The aircraft can 
  'teleport' to a different flight condition between two named segments (i.e. 
  between a ClimbSegment and a CruiseSegment).

## Future Work
- [ ] Improved outputting & plot generation (currently ad hoc plotting)
- [ ] Functionality for Mission-level fuel sizing: 
		iteration that guesses initiqal fuel weight and converges when 
		reserves are met (similar to range root-find).
- [ ] Create an implementation of `AeroModelBase` / `PropulsionModelBase`, to
      read in table data from an outside source (i.e. DATCOM) to demonstrate 
	  knowledge of iterpolated data handling.

## References
- **ISA atmosphere model** — standard 1976 US Standard Atmosphere
  relations (`atmosphere.py`).
- **Compressible CAS↔Mach conversion** (`unit_conversions.py:cas_to_mach`,
  `mach_to_cas`) — compressible pitot-static relation from FAA,
  *Pilot's Handbook of Aeronautical Knowledge*
- **Climb acceleration factor / "kinetic correction factor"**
  (`ka` in `solver.py`) — derived from the flight path force
  balance and the chain rule for dV/dt:
  - Marchman, J.F., *Aerodynamics and Aircraft Performance*, 3rd ed.,
    Virginia Tech (open textbook), Ch. 5:
    https://eng.libretexts.org/Bookshelves/Aerospace_Engineering/Aerodynamics_and_Aircraft_Performance_3e_(Marchman)/05:_Altitude_Change-_Climb_and_Guide
    
## Disclaimer
This is a portfolio project, not a certified performance analysis tool. The 
goal of this project is to demonstrate my knowledge of performance analysis 
theory and numerical methods behind mission analysis. While I have attempted to 
ensure the outputs from this tool are calculated accurately, I give no warranty 
to the tool's accuracy. Users should do their own due dilligence before any 
commercial use.