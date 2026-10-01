"""
Aerodynamic model interface.

The mission code only ever asks the aero model two questions:
    1. "What CL is needed to make this much lift?"  -> cl_for_lift()
    2. "What is CD at this CL and Mach?"            -> get_cd()
It does not care whether the answer comes from a textbook parabolic polar, a
DATCOM build-up, a CFD table, or a wind-tunnel database. Every aero model
subclasses AeroModelBase below and implements those two methods, so anything
downstream (Aircraft, the segments/ package, Mission) works unchanged when the
aero model is swapped.

===============================================================================
SimpleDragPolar(cd0, aspect_ratio, oswald_efficiency, mach_crit=0.78,
                mach_drag_rise_coeff=20.0)
===============================================================================
Textbook parabolic drag polar (Anderson), CD = CD0 + K*CL^2 with
K = 1 / (pi * AR * e), plus a placeholder cubic wave-drag rise above
mach_crit so cruise-Mach trends are qualitatively correct.

  cd0 -------------------- Zero-lift (parasite) drag coefficient.
  aspect_ratio ----------- Wing aspect ratio, b^2 / S.
  oswald_efficiency ------ Span efficiency factor e (typically 0.7-0.85).
  mach_crit -------------- Mach above which the wave-drag term turns on.
  mach_drag_rise_coeff --- Severity of the cubic wave-drag rise.

SOURCES OF ERROR
  - ValueError from cl_for_lift() if dynamic pressure or area <= 0.
  - No CL_max / stall check. A segment that demands an unrealistic 
    CL (very high altitude, very low speed) will not be flagged.

===============================================================================
ADDING A NEW AERO MODEL
===============================================================================
    1. Create aero_model/<new_model>.py. Subclass AeroModelBase from .base,
       give it a class-level `name`, and implement get_cd() and cl_for_lift().
    2. Set self.modelID = self.name, and build self.inputs, the dict of
       values Mission writes into the mission summary file's "Aerodynamic
       Model" block.
    3. Import it in aero_model/__init__.py, add it to __all__ and to
       AERO_MODEL_TYPES.
    4. Document it in the catalog above.
"""

from abc import ABC, abstractmethod

class AeroModelBase(ABC):
    """Abstract interface all aero models must implement."""
    name = "AeroModelBase"

    @abstractmethod
    def get_cd(self, cl: float, mach: float) -> float:
        """Return total drag coefficient for given lift coefficient and Mach."""
        raise NotImplementedError

    @abstractmethod
    def cl_for_lift(self, lift_n: float, dynamic_pressure_pa: float, area_m2: float) -> float:
        """Return CL required to produce a given lift force at given q and area."""
        raise NotImplementedError
