"""
Simple Aerodynamic model buildup. Future work planned to implement other 
aerodynamic model types (interpolated CFD table, DATCOM build-up, etc.).

NOTE: Any future implementation (interpolated CFD table, DATCOM build-up, etc.)
should subclass AeroModelBase and implement the same two methods.
"""

import math
from abc import ABC, abstractmethod


class AeroModelBase(ABC):
    """Abstract interface all aero models must implement."""

    @abstractmethod
    def get_cd(self, cl: float, mach: float) -> float:
        """Return total drag coefficient for given lift coefficient and Mach."""
        raise NotImplementedError

    @abstractmethod
    def cl_for_lift(self, lift_n: float, dynamic_pressure_pa: float, area_m2: float) -> float:
        """Return CL required to produce a given lift force at given q and area."""
        raise NotImplementedError


class SimpleDragPolar(AeroModelBase):
    """
    Simple parabolic drag polar: CD = CD0 + K * CL^2

    This is a textbook model (Anderson) for early conceptual-level mission 
    analysis. It intentionally ignores:
      - Mach-dependent CD0 rise (wave drag) above a critical Mach
      - CL_max / stall limits
      - Compressibility effects on K

    `mach_drag_rise` correction is included as a placeholder so the tool 
    produces qualitatively correct cruise-Mach behavior.
    """

    def __init__(
        self,
        cd0: float,
        aspect_ratio: float,
        oswald_efficiency: float,
        mach_crit: float = 0.78,
        mach_drag_rise_coeff: float = 20.0,
    ):
        self.cd0 = cd0
        self.aspect_ratio = aspect_ratio
        self.e = oswald_efficiency
        self.k = 1.0 / (math.pi * aspect_ratio * oswald_efficiency)
        self.mach_crit = mach_crit
        self.mach_drag_rise_coeff = mach_drag_rise_coeff

    def get_cd(self, cl: float, mach: float) -> float:
        cd0_effective = self.cd0
        if mach > self.mach_crit:
            # Simple empirical wave-drag effect above critical Mach.
            cd0_effective += self.mach_drag_rise_coeff * (mach - self.mach_crit) ** 3
        return cd0_effective + self.k * cl ** 2

    def cl_for_lift(self, lift_n: float, dynamic_pressure_pa: float, area_m2: float) -> float:
        if dynamic_pressure_pa <= 0 or area_m2 <= 0:
            raise ValueError("Dynamic pressure and area must be positive.")
        return lift_n / (dynamic_pressure_pa * area_m2)

    def l_over_d(self, cl: float, mach: float) -> float:
        cd = self.get_cd(cl, mach)
        return cl / cd

#------------------------------ DEBUGGING ------------------------------------- 
if __name__ == "__main__":
    # Sanity check: L/D should peak somewhere reasonable and drag should rise sharply past mach_crit.
    aero = SimpleDragPolar(cd0=0.020, aspect_ratio=9.5, oswald_efficiency=0.80)
    print("CL     CD       L/D")
    for cl in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        cd = aero.get_cd(cl, mach=0.78)
        print(f"{cl:.2f}   {cd:.4f}   {cl/cd:.2f}")

    print("\nMach   CD (CL=0.5)")
    for m in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]:
        print(f"{m:.2f}   {aero.get_cd(0.5, m):.4f}")
