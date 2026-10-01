"""
Propulsion model interface.

Same philosophy as 'aero_model.py'. The mission code asks for thrust
available and fuel flow at a flight condition + throttle setting, and
does not care whether that comes from a real engine deck, a scaled
manufacturer chart, or one of the conceptual-design approximations
implemented here. Every propulsion model subclasses PropulsionModelBase 
below and implements max_thrust(), idle_thrust() and fuel_flow().

All methods work in SI and return values for all engines combined.
Each model must also carry a `num_engines` attribute (Mission uses it to
report per-engine thrust in the time-history file).

===============================================================================
SimpleTurbofan(sea_level_thrust_lbf, tsfc_lb_per_lbfhr, num_engines=2,
               lapse_exponent=0.8, idle_thrust_fraction=0.05)
===============================================================================
Constant SFC, thrust producing engine. Fuel flow tracks thrust directly and
is Mach independent. Max thrust lapses with density ratio and a mild Mach
correction (Mattingly conceptual-design approximation):

    T_max(h, M) = T_SL * (rho/rho0)^lapse_exponent * (1 - 0.25*M)

  sea_level_thrust_lbf --- Static sea-level thrust, per engine.
  tsfc_lb_per_lbfhr ------ Thrust specific fuel consumption, lb/(lbf*hr).
  num_engines ------------ Engine count.
  lapse_exponent --------- 'm' in (rho/rho0)^m, typically ~0.7-1.0 for high-bypass turbofans.
  idle_thrust_fraction --- Idle thrust as a fraction of max thrust.

SOURCES OF ERROR
  - Constant TSFC ignores the real
    variation of TSFC with altitude, Mach and throttle, so partial power fuel
    burn is the least accurate output.

===============================================================================
SimpleTurboprop(sea_level_power_shp, psfc_lb_per_shphr, prop_diameter_ft,
                prop_rpm, num_blades=4, eta_penalty_per_blade=0.0,
                num_engines=2, lapse_exponent=0.8, idle_power_fraction=0.05,
                eta_prop_max=0.87, advance_ratio_peak=1.6,
                advance_ratio_width=1.8, tip_mach_crit=0.88,
                tip_mach_loss_coeff=25.0)
===============================================================================
Constant SFC, power producing engine driving a constant-speed propeller.
Thrust is the lesser of the momentum theory static limit and eta*P/V, where
eta combines an advance-ratio curve fit with a helical tip-Mach loss. See the
class docstring in simple_turboprop.py for the full derivation, sources and 
simplifications.

  sea_level_power_shp ---- Sea-level shaft power, per engine (shp).
  psfc_lb_per_shphr ------ Power specific fuel consumption, lb/(shp*hr).
  prop_diameter_ft ------- Propeller diameter.
  prop_rpm --------------- Propeller shaft RPM (constant-speed prop).
  num_blades ------------- Only matters if eta_penalty_per_blade != 0.
  eta_penalty_per_blade -- Opt-in linear efficiency penalty per blade relative
                           to a 4-blade reference. Defaults to 0.
  num_engines ------------ Engine count.
  lapse_exponent --------- Power lapse with density ratio.
  idle_power_fraction ---- Idle power as a fraction of max power.
  eta_prop_max ----------- Peak propeller efficiency.
  advance_ratio_peak ----- J at which efficiency peaks.
  advance_ratio_width ---- J offset from the peak at which eta falls to 0.
  tip_mach_crit ---------- Helical tip Mach where compressibility losses start.
  tip_mach_loss_coeff ---- Severity of the cubic tip-Mach loss.

SOURCES OF ERROR
  - Efficiency is minimum at 1e-3 to avoid divide-by-zero, causing badly mismatched 
    J or a transonic tip to show up as ~0 thrust.

===============================================================================
ADDING A NEW PROPULSION MODEL
===============================================================================
    1. Create propulsion_model/<new_model>.py. Subclass PropulsionModelBase
       from .base, give it a class-level `name`, and implement max_thrust(),
       idle_thrust() and fuel_flow() (all engines combined, SI units).
    2. Set self.modelID = self.name and self.num_engines, and build
       self.inputs, the dict of values Mission writes into the mission
       summary file's "Propulsion Model" block.
    3. Import it in propulsion_model/__init__.py, add it to __all__ and to
       PROPULSION_MODEL_TYPES.
    4. Document it in the catalog above.
"""

from abc import ABC, abstractmethod


class PropulsionModelBase(ABC):
    """
    Abstract interface all propulsion models must implement.
    """
    name = "PropulsionModelBase"
    @abstractmethod
    def max_thrust(self, altitude_m: float, mach: float, DISAC: float) -> float:
        """Maximum available thrust (N) at altitude/Mach, full throttle."""
        raise NotImplementedError

    @abstractmethod
    def idle_thrust(self, altitude_m: float, mach: float, DISAC: float) -> float:
        """Idle (flight-idle) thrust (N) at altitude/Mach, used for descent."""
        raise NotImplementedError

    @abstractmethod
    def fuel_flow(self, thrust_n: float, altitude_m: float, mach: float, DISAC: float) -> float:
        """Fuel mass flow rate (kg/s) for a given thrust setting."""
        raise NotImplementedError
