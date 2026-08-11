# Aircraft-Mission-Profile-Analysis
A solver to analyze the performance of a conceptual aircraft along a 
user-defined mission profile. 

This is a portfolio project, not a certified performance tool. Aero and
propulsion models are simplified (parabolic drag polar, constant-TSFC
turbofan) for the purpose of example. The goal of this project is to
demonstrate my knowledge of performance analysis theory and numerical-methods 
behind mission analysis.

## Why this project
After spending several years in industry working as an aircraft performance engineer,
I found it increasingly difficult to find a new position. I figured an example portfolio 
of my background would help my job search and get me back into the field. I also hated
the performance analysis tool I used at work, and never got funding to rewrite it.

## Tool theory of operation
Mission analysis is fundamentally an iterative problem: weight
decreases continuously as fuel burns, which changes the lift coefficient
required for level flight, which changes L/D and fuel flow, which
changes the burn rate going forward. You can't solve for total fuel burn
algebraically except under simplifying assumptions (Breguet). 
`FixedCruiseSegment` solves the coupled problem by numerically
integrating the weight-vs-distance ODE, and validates that integration
against the closed-form Breguet range equation as a unit test
(`tests/test_breguet_sanity_check.py`).

## Architecture
```
atmosphere.py       - ISA atmosphere model (temp, pressure, density, speed of sound)
aero_model.py       - Aero interface + simple parabolic drag polar implementation
propulsion_model.py - Propulsion interface + simple constant-TSFC turbofan implementation
aircraft_build.py 	- Aircraft class: wraps geometry, weights, aero + propulsion models
segments.py         - MissionSegment base class, FixedCruiseSegment (RK4), LoiterSegment (RK4)
mission.py          - Mission class: sequences segments, carries weight forward
examples/           - Runnable end-to-end mission scripts
tests/              - Validation tests (Breguet convergence, physical sanity checks)
```

**Design principle:** `aero_model.py` and `propulsion_model.py` define
abstract interfaces (`AeroModelBase`, `PropulsionModelBase`). Everything
downstream — `Aircraft`, `segments.py`, `Mission` — only calls those
interface methods. This means a real aero deck (CFD-derived lookup
table, DATCOM build-up) or a real engine cycle deck can be substituted
by writing one new class, with zero changes to the mission-solving code.

## Physics implemented
- **ISA atmosphere** (0–20 km), including an ISA+ΔT offset option
- **Steady, level trim**: L = W, T = D solved at each point via the
  required-CL relationship — no separate root-find needed in this
  version, since level cruise is not over-constrained; the trim solver
  becomes necessary once climb/descent flight-path angle is added (see
  Roadmap)
- **Coupled weight/fuel-burn integration**: 4th-order Runge-Kutta on
  `dW/dx = -fuel_flow / V` for cruise, `dW/dt = -fuel_flow` for loiter
- **Breguet range equation** as an independent closed-form check on the
  numerical integrator

## Quick start
```bash
pip install -r requirements.txt
python3 examples/simple_cruise_mission.py
python3 -m pytest tests/ -v
```
The example runs a 1,500 nm cruise at 35,000 ft / M0.78 followed by a
30-minute diversion loiter, prints a segment-by-segment fuel/time/weight
summary, and saves a weight-and-L/D-vs-distance plot.

## Validation
`tests/test_breguet_sanity_check.py` checks that the numerically
integrated cruise segment agrees with the closed-form Breguet range
equation to within a tight tolerance. One result worth understanding
rather than skimming past: **the residual error (~0.05%) does not shrink
as integration step count increases.** That's expected, not a bug —
RK4 is 4th-order accurate, so it's already converged to the true ODE
solution by ~5 steps. The residual is coming from Breguet's own
approximation (constant L/D evaluated at mean segment weight) rather
than from the numerical integrator. The test docstring walks through
how to tell which side of a comparison like this is actually the
under-resolved one, since it's easy to misattribute a mismatch to the
wrong component.

## Known simplifications (by design, for this stage)
- Constant TSFC (no altitude/Mach/throttle variation)
- No climb/descent segments yet (steady-level flight only)

## Roadmap
- [ ] Mission segments for climb and descent
- [ ] Separate class definition for mass properties (currently held in `aero_model.py`)
- [ ] Imporved outputting & plot generation
- [ ] Functionality for radius profiles (outbound and inbound segments)
- [ ] Functionality for Mission-level fuel sizing: 
		iteration that guesses takeoff fuel weight and converges when 
		required reserves are met — layered on top of `Mission.run()` 
		without modifying it.
- [ ] Functionality for Mission-level range sizing:
		Iterate on a specified cruise leg to zero out fuel at the end of a mission 		
- [ ] Create an implementation of `AeroModelBase` / `PropulsionModelBase`, to
      read in table data from an outside source (DATCOM) to demonstrate a 
	  knowledge of iterpolated data handling

## Author's note
Built to demonstrate applied flight mechanics + numerical methods +
software structure. Individual file outputs are checked against a hand-computable or
independently-derivable reference in the corresponding test or
`__name__ == "__main__"` block of each `.py` file
