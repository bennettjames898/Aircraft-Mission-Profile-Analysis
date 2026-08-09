"""
Propulsion model interface.

Same philosophy as aero_model.py: the mission code asks for thrust
available and fuel flow at a flight condition + throttle setting, and
does not care whether that comes from a real engine deck, a scaled
manufacturer chart, or (as here) a simple TSFC-based approximation.
"""

from abc import ABC, abstractmethod

G0 = 9.80665  # m/s^2, for weight/thrust unit consistency


class PropulsionModelBase(ABC):
    @abstractmethod
    def max_thrust(self, altitude_m: float, mach: float) -> float:
        """Maximum available thrust (N) at altitude/Mach, full throttle."""
        raise NotImplementedError

    @abstractmethod
    def idle_thrust(self, altitude_m: float, mach: float) -> float:
        """Idle (flight-idle) thrust (N) at altitude/Mach, used for descent."""
        raise NotImplementedError

    @abstractmethod
    def fuel_flow(self, thrust_n: float, altitude_m: float, mach: float) -> float:
        """Fuel mass flow rate (kg/s) for a given thrust setting."""
        raise NotImplementedError


class SimpleTurbofan(PropulsionModelBase):
    """
    Simplified turbofan model using:
      - A altitude/Mach thrust lapse approximation (common conceptual-design form)
      - Constant TSFC (thrust-specific fuel consumption)

    Thrust lapse: T_max(h, M) = T_sea_level * (rho/rho0)^m * f(M)
    where m ~ 0.7-1.0 for high-bypass turbofans, and f(M) is a mild
    Mach correction. These are illustrative conceptual-design
    approximations (consistent with Mattingly).
    """

    def __init__(
        self,
        sea_level_thrust_n: float,
        tsfc_kg_per_n_per_s: float,
        num_engines: int = 2,
        lapse_exponent: float = 0.8,
        idle_thrust_fraction: float = 0.05,
    ):
        self.sea_level_thrust_n = sea_level_thrust_n
        self.tsfc = tsfc_kg_per_n_per_s
        self.num_engines = num_engines
        self.lapse_exponent = lapse_exponent
        self.idle_thrust_fraction = idle_thrust_fraction

    def max_thrust(self, altitude_m: float, mach: float) -> float:
        from atmosphere import isa_conditions, RHO0

        rho = isa_conditions(altitude_m)["density_kg_m3"]
        density_ratio = rho / RHO0

        # Mild Mach correction: thrust drops off slightly with increasing
        # Mach at constant altitude for a high-bypass turbofan.
        mach_factor = 1.0 - 0.25 * mach

        thrust_per_engine = (
            self.sea_level_thrust_n * (density_ratio ** self.lapse_exponent) * mach_factor
        )
        return self.num_engines * thrust_per_engine

    def idle_thrust(self, altitude_m: float, mach: float) -> float:
        return self.idle_thrust_fraction * self.max_thrust(altitude_m, mach)

    def fuel_flow(self, thrust_n: float, altitude_m: float, mach: float) -> float:
        # Simplification: Constant TSFC model: fuel flow scales linearly with thrust.
        # (Real engines show TSFC variation with altitude/Mach/throttle)
        return thrust_n * self.tsfc

#------------------------------ DEBUGGING ------------------------------------- 
if __name__ == "__main__":
    from atmosphere import ft_to_m

    engine = SimpleTurbofan(
        sea_level_thrust_n=120000.0,  # ~27,000 lbf per engine, x2
        tsfc_kg_per_n_per_s=1.75e-5,  # ~0.62 lb/lbf/hr, typical modern turbofan cruise TSFC
        num_engines=2,
    )

    print(f"{'Alt (ft)':>10} {'Mach':>6} {'Max Thrust (N)':>16} {'Fuel Flow (kg/s)':>18}")
    for alt_ft, mach in [(0, 0.3), (35000, 0.78), (39000, 0.78)]:
        alt_m = ft_to_m(alt_ft)
        t_max = engine.max_thrust(alt_m, mach)
        ff = engine.fuel_flow(t_max, alt_m, mach)
        print(f"{alt_ft:>10} {mach:>6.2f} {t_max:>16.1f} {ff:>18.4f}")
