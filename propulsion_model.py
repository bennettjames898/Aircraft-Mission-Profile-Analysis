"""
Propulsion model interface.

Same philosophy as 'aero_model.py'. The mission code asks for thrust
available and fuel flow at a flight condition + throttle setting, and
does not care whether that comes from a real engine deck, a scaled
manufacturer chart, or one of the conceptual-design approximations
implemented here.

Two implementations are provided:
  - SimpleTurbofan: constant-TSFC, thrust-producing engine. Fuel flow
    tracks thrust directly and is Mach independent.
  - SimpleTurboprop: constant-PSFC, power-producing engine driving a
    propeller. Thrust is derived from shaft power via a propeller
    efficiency model (advance ratio + helical tip-Mach compressibility).
    See its class docstring for the fullbderivation and known simplifications.
"""

import math
from abc import ABC, abstractmethod
import unit_conversions as convert


class PropulsionModelBase(ABC):
    name = "PropulsionModelBase"
    @abstractmethod
    def max_thrust(self, altitude_m: float, mach: float) -> float:
        """Maximum available thrust (N) at altitude/Mach, full throttle."""
        raise NotImplementedError

    @abstractmethod
    def idle_thrust(self, altitude_m: float, mach: float) -> float:
        """Idle (flight-idle) thrust (N) at altitude/Mach, used for descent."""
        raise NotImplementedError

    @abstractmethod
    def fuel_flow(self, thrust_n: float, altitude_m: float, mach: float, DISAC: float) -> float:
        """Fuel mass flow rate (kg/s) for a given thrust setting."""
        raise NotImplementedError


class SimpleTurbofan(PropulsionModelBase):
    """
    Simplified turbofan model using:
      - A altitude/Mach thrust lapse approximation (conceptual-design)
      - Constant TSFC

    Thrust lapse: T_max(h, M) = T_sea_level * (rho/rho0)^m * f(M)
    where m ~ 0.7-1.0 for high-bypass turbofans, and f(M) is a mild
    Mach correction. These are conceptual-design approximations (Mattingly).
    """
    name = "SimpleTurbofan"
    def __init__(
        self,
        sea_level_thrust_lbf:   float, # PER ENGINE
        tsfc_lb_per_lbfhr:      float, # FuelFlow/Thrust [lb/(hr*lbf)]
        num_engines:            int = 2,
        lapse_exponent:         float = 0.8,
        idle_thrust_fraction:   float = 0.05,
    ):
        self.modelID                = self.name
        self.sea_level_thrust_n     = convert.lb_to_kg(sea_level_thrust_lbf)*convert.G0
        self.tsfc                   = tsfc_lb_per_lbfhr/35310 # converts to kg/(N*s)
        self.num_engines            = num_engines
        self.lapse_exponent         = lapse_exponent
        self.idle_thrust_fraction   = idle_thrust_fraction # % of max thrust for idle approximation
        self.inputs                 = self.__dict__ # collect input terms for output files

    def max_thrust(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        from atmosphere import isa_conditions, RHO0

        rho = isa_conditions(altitude_m, DISAC)["density_kg_m3"]
        density_ratio = rho / RHO0

        # Mild Mach correction: thrust drops off slightly with increasing
        # Mach at constant altitude for a high-bypass turbofan.
        mach_factor = 1.0 - 0.25 * mach

        thrust_per_engine = (
            self.sea_level_thrust_n * (density_ratio**self.lapse_exponent) * mach_factor
        )
        return self.num_engines * thrust_per_engine

    def idle_thrust(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        return self.idle_thrust_fraction * self.max_thrust(altitude_m, mach, DISAC)

    def fuel_flow(self, thrust_n: float, altitude_m: float, mach: float, DISAC: float) -> float:
        # Simplification: Constant TSFC model: fuel flow scales linearly with thrust.
        # (Real engines show TSFC variation with altitude/Mach/throttle)
        return thrust_n * self.tsfc

class SimpleTurboprop(PropulsionModelBase):
    """
    Simplified turboprop model, requiring inputs for engine performance and 
    propeller geometry.
 
    Propeller-converted thrust from a shaft-power engine:
 
        T = eta_prop * P_shaft / V
 
    At constant power, thrust falls as a facotr of 1/V. This relationship is 
    why turboprops dominate at low speed and lose to turbofans in cruise, 
    and is the reason this class takes propeller geometry as an input.
 
    SPEED EFFECTS ON EFFICIENCY (Modeled in this class)
    -------------------------------------------------------------------------
    1. THRUST LAPSE WITH SPEED T = eta*P/V [N]
 
    2. ADVANCE RATIO J = V / (n*D)   [n = rev/s, D = diameter]
       The nondimensional ratio of forward distance per revolution to
       propeller diameter. This parameter directly impacts prop efficiency. 
       Efficiency peaks near a design J and falls away parabolicallyas modeled.
 
    3. HELICAL TIP MACH NUMBER
           M_tip = sqrt(V^2 + (pi*n*D)^2) / a
       The blade tip sees the vector sum of the aircraft's forward velocity
       and the tip's own rotational velocity, causing the tip to go transonic 
       long before the aircraft. Once M_tip passes roughly 0.85-0.90, shock
       losses cut efficiency sharply.
 
    STATIC & LOW SPEED THRUST
    -------------------------------------------------------------------------
    T = eta*P/V is singular at V = 0, so it cannot be used on the ground.
    Actuator disk (momentum) theory gives the static limit instead 
    (NASA Glenn, https://www.grc.nasa.gov/WWW/K-12/airplane/propth.html):
 
        T_static = (2 * rho * A_disk * P^2)^(1/3)
 
    where A_disk is the propeller swept area and FM is the figure of merit
    (a real propeller's fraction of the ideal momentum theory result, ~0.7-0.8).
    The model reports min(T_static, eta*P/V), causing the momentum limit to 
    drive at low speed and the power/efficiency relation drives in cruise.
 
    SIMPLIFICATIONS
    -------------------------------------------------------------------------
      - The eta(J) parabola is a tunable curve fit (using 'advance_ratio_peak' 
        and 'advance_ratio_width'). Its shape is qualitatively accurate to 
        empirical data (single peak & falls off either side).
      - This model assumes a constant speed propellers, 'prop_rpm' is constant 
        throughout the mission profile.
      - 'num_blades' has no effect on the model unless a blade number 
        efficiency penalty is specified in 'eta_penalty_per_blade'. This term 
        applies a linear efficiency penalty relative to a 4-blade reference, 
        and defaults to 0. This term is intednded as a calibration slot for 
        the simplified model.
      - Residual jet thrust from the core exhaust is NOT modeled.
      - Flat thrust rating is not modeled. Real turboprops hold constant shaft power
        up to some altitude/temperature, this model lapses immediately.
      - PSFC is constant, as TSFC is in SimpleTurbofan.
      - Static thrust is RPM independent, because momentum theory only sees
        disk area and power.
      - Static power-thrust conversion is assumed perfect.
 
    USING THIS MODEL FOR PROPELLER SIZING STUDIES
    -------------------------------------------------------------------------
    'advance_ratio_opt' represents the J a given propeller was designed
    around (its pitch/twist distribution). It is an independent input, so
    sweeping diameter or RPM on independently asks, "what if the SAME propeller 
    spun faster/slower?", moving J away from its design point. Sizing 
    studies should recalculate 'advance_ratio_opt' using the following:
        
        J_opt_new = J_opt_old * (n_old * D_old) / (n_new * D_new)
 
    to ensure each tested prop is "re-pitched" for its new operating point. 
    Sweep the geometry alone to study off design behavior of one propeller.
    Sweep the geometry and advance_ratio_opt together to compare separately
    optimized propeller designs.
    """
    name = "SimpleTurboprop"
    def __init__(
        self,
        sea_level_power_shp:    float,          # PER ENGINE, shaft horsepower
        psfc_lb_per_shphr:      float,          # FuelFlow/Power [lb/(hr*shp)]
        prop_diameter_ft:       float,          # propeller diameter
        prop_rpm:               float,          # propeller shaft RPM
        num_blades:             int = 4,
        eta_penalty_per_blade:  float = 0.0,    # approximated impact of increased blade count on overall prop efficiency
        num_engines:            int = 2,
        lapse_exponent:         float = 0.8,    # power lapse w/ density ratio
        idle_power_fraction:    float = 0.05,   # % of max power for idle approximation
        eta_prop_max:           float = 0.87,   # peak propeller efficiency
        advance_ratio_peak:     float = 1.6,    # J at which eta peaks
        advance_ratio_width:    float = 1.8,    # J offset at which eta falls to 0
        tip_mach_crit:          float = 0.88,   # M_tip where compressibility losses start
        tip_mach_loss_coeff:    float = 25.0,   # severity of the tip compressibility loss
    ):
        self.modelID                = self.name
        self.sea_level_power_w      = sea_level_power_shp * 745.699872        # shp -> W
        self.psfc                   = convert.lb_to_kg(psfc_lb_per_shphr) / (745.699872 * 3600) # [lb/(shp*hr)] -> [kg/(W*s)]
        self.prop_diameter_m        = convert.ft_to_m(prop_diameter_ft)
        self.prop_rps               = prop_rpm / 60 # rev/s
        self.prop_disk_area_m2      = math.pi * (self.prop_diameter_m/2)**2
        self.num_blades             = num_blades
        self.eta_penalty_per_blade  = eta_penalty_per_blade
        self.num_engines            = num_engines
        self.lapse_exponent         = lapse_exponent
        self.idle_power_fraction    = idle_power_fraction
        self.eta_prop_max           = eta_prop_max
        self.advance_ratio_peak     = advance_ratio_peak
        self.advance_ratio_width    = advance_ratio_width
        self.tip_mach_crit          = tip_mach_crit
        self.tip_mach_loss_coeff    = tip_mach_loss_coeff
        self.inputs                 = self.__dict__ # collect input terms for output files
 
    #--------------------------- PROPELLER TERMS ------------------------------
    def advance_ratio(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        """J = V / (n*D). Nondimensional forward travel per revolution."""
        tas = convert.mach_to_tas(mach, altitude_m, DISAC)
        return tas / (self.prop_rps * self.prop_diameter_m)
 
    def helical_tip_mach(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        """
        Mach number seen by the blade tip, the vector sum of forward flight
        speed and tip rotational speed. Always higher than aircraft Mach.
        """
        from atmosphere import isa_conditions
 
        tas         = convert.mach_to_tas(mach, altitude_m, DISAC)
        tip_speed   = math.pi * self.prop_rps * self.prop_diameter_m
        a_local     = isa_conditions(altitude_m, DISAC)["speed_of_sound_m_s"]
        return math.sqrt(tas ** 2 + tip_speed ** 2) / a_local
 
    def prop_efficiency(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        """
        Propeller efficiency, eta = (thrust power out) / (shaft power in).
 
        Combines the advance ratio term (curve fit) with the helical tip Mach
        compressibility term (physical) and the blade count handle.
        """
        # --- Advance ratio term: parabola peaking at advance_ratio_peak ---
        J           = self.advance_ratio(altitude_m, mach, DISAC)
        eta_J       = 1 - ((J - self.advance_ratio_peak) / self.advance_ratio_width) ** 2
        eta_J       = max(eta_J, 0)
 
        # --- Compressibility term: cubic loss above tip_mach_crit ---
        # Same functional form as SimpleDragPolar's wave drag rise, for
        # consistency of convention across the models.
        M_tip       = self.helical_tip_mach(altitude_m, mach, DISAC)
        eta_comp    = 1
        if M_tip > self.tip_mach_crit:
            eta_comp = 1-(self.tip_mach_loss_coeff*(M_tip-self.tip_mach_crit)**3)
            eta_comp = max(eta_comp, 0)
 
        # --- Blade count efficiency factor: OFF unless specified ---
        # eta_penalty_per_blade is a calibration input, defaulting to 0, 
        # so num_blades has no effect on efficiency unless a value is 
        # provided. See the class docstring for why a scalar cannot represent 
        # blade count properly.
        blade_factor = 1-(self.eta_penalty_per_blade*(self.num_blades-4))
 
        eta = self.eta_prop_max * eta_J * eta_comp * blade_factor
        return max(eta, 1e-3) # Floor prevents divide-by-zero downstream.
 
    #----------------------------- POWER TERMS --------------------------------
    def max_power_w(self, altitude_m: float, DISAC: float = 0) -> float:
        """Total shaft power available (W, all engines) at altitude."""
        from atmosphere import isa_conditions, RHO0
 
        rho = isa_conditions(altitude_m, DISAC)["density_kg_m3"]
        density_ratio = rho / RHO0
        return self.num_engines * self.sea_level_power_w * (density_ratio ** self.lapse_exponent)
 
    def _static_thrust_n(self, power_w: float, altitude_m: float, DISAC: float = 0) -> float:
        """
        Actuator disk (Rankine-Froude momentum theory) static thrust for a
        given shaft power:
 
            T = (2 * rho * A * P^2)^(1/3)
 
        DERIVATION
        ---------------------------------------------------------------------
        For a disk of area A in still air, with induced velocity vp at the
        prop disk and a far wake velocity of 2*vp:
 
            mass flow through the disk - mdot = rho * A * vp
            thrust --------------------- T = mdot * (2*vp) = 2 * rho * A * vp^2
            ideal power at the disk ---- P_ideal = T * vp
 
        Solving the thrust relation for vp = sqrt(T / (2*rho*A)) and
        substituting into the power relation gives P_ideal = T^1.5 /
        sqrt(2*rho*A), which rearranges to the expression above. The
        fuel_flow() method inverts exactly this relation to recover shaft
        power from a commanded thrust.
 
        SOURCES
        ---------------------------------------------------------------------
          - MIT OpenCourseWare 2.611, actuator disk notes
          - NASA Glenn, Propellor Analysis 
            (https://www.grc.nasa.gov/WWW/K-12/airplane/propanl.html)
        """
        from atmosphere import isa_conditions
 
        rho     = isa_conditions(altitude_m, DISAC)["density_kg_m3"]
        A_total = self.prop_disk_area_m2 * self.num_engines
        return (2 * rho * A_total * (power_w)**2)**(1/3)
 
    #---------------------------- BASE INTERFACE ------------------------------
    def _thrust_from_power(self, power_w: float, altitude_m: float, mach: float, DISAC: float) -> float:
        """
        Convert available shaft power to thrust, taking whichever of the two
        limits binds:
          - momentum theory static thrust (governs at low speed)
          - eta*P/V (governs in the cruise)
        """
        tas         = convert.mach_to_tas(mach, altitude_m, DISAC)
        T_static    = self._static_thrust_n(power_w, altitude_m, DISAC)
 
        if tas <= 1e-6:
            return T_static
 
        eta         = self.prop_efficiency(altitude_m, mach, DISAC)
        T_prop      = eta * power_w / tas
        return min(T_static, T_prop)
 
    def max_thrust(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        power_w = self.max_power_w(altitude_m, DISAC)
        return self._thrust_from_power(power_w, altitude_m, mach, DISAC)
 
    def idle_thrust(self, altitude_m: float, mach: float, DISAC: float = 0) -> float:
        power_w = self.idle_power_fraction * self.max_power_w(altitude_m, DISAC)
        return self._thrust_from_power(power_w, altitude_m, mach, DISAC)
 
    def fuel_flow(self, thrust_n: float, altitude_m: float, mach: float, DISAC: float) -> float:
        """
        Fuel flow (kg/s) for a commanded thrust.
 
        Fuel burn tracks power, not thrust, so the thrust model is
        inverted to recover the power the propeller must be absorbing. Because
        _thrust_from_power takes the MINIMUM of two increasing functions of
        power, the inverse takes the MAXIMUM of their two inverses.
 
        This is why a turboprop holding constant thrust burns more fuel as it
        speeds up (power = thrust x velocity), the opposite of the constant
        TSFC turbofan behavior in SimpleTurbofan, where fuel flow tracks
        thrust directly and is speed independent.
        """
        from atmosphere import isa_conditions
 
        if thrust_n <= 0:
            return 0.0
 
        # Invert the static (momentum theory) branch:  P = T^1.5 / sqrt(2*rho*A)
        rho             = isa_conditions(altitude_m, DISAC)["density_kg_m3"]
        A_total         = self.prop_disk_area_m2 * self.num_engines
        P_from_static   = thrust_n**1.5 / math.sqrt(2 * rho * A_total)
 
        # Invert the propeller branch:  P = T*V / eta
        tas = convert.mach_to_tas(mach, altitude_m, 0)
        if tas <= 1e-6:
            P_required = P_from_static
        else:
            eta         = self.prop_efficiency(altitude_m, mach, 0)
            P_from_prop = thrust_n * tas / eta
            P_required  = max(P_from_static, P_from_prop)
 
        return self.psfc * P_required
 
 
#------------------------------ DEBUGGING ------------------------------------- 
if __name__ == "__main__":
    DISAF = 0 # Delta standard conditions in degF
 
    engine = SimpleTurbofan(
        sea_level_thrust_lbf = 27000,  # ~27,000 lbf per engine, x2
        tsfc_lb_per_lbfhr    = 0.62,   # ~0.62 lb/lbf/hr, typical turbofan cruise TSFC
        num_engines          = 2,
    )
    
    print(f"{'Alt (ft)':>10} {'Mach':>6} {'Max Thrust (N)':>16} {'Fuel Flow (kg/s)':>18}")
    for alt_ft, mach in [(0, 0.3), (35000, 0.78), (39000, 0.78)]:
        alt_m   = convert.ft_to_m(alt_ft)
        t_max   = engine.max_thrust(alt_m, mach, convert.DISAF_to_C(DISAF)) # N
        WF      = engine.fuel_flow(t_max, alt_m, mach, convert.DISAF_to_C(DISAF)) # kg/s
        print(f"{alt_ft:>10} {mach:>6.2f} {t_max:>16.1f} {WF:>18.4f}")
 
    # --- Turboprop: ATR/Dash-8 class, ~2750 shp per engine ---
    # Checks to look for in the sweep below:
    #   - Thrust falls monotonically with Mach (T = eta*P/V), unlike the
    #     turbofan above.
    #   - eta peaks at advance_ratio_opt (J = 1.6 by default) and falls off
    #     either side.
    #   - M_tip is well above aircraft Mach at all times and goes transonic
    #     around M0.55-0.65, collapsing eta.
    prop = SimpleTurboprop(
        sea_level_power_shp = 2750,   # shp per engine
        psfc_lb_per_shphr   = 0.5,    # typical modern turboprop cruise PSFC
        prop_diameter_ft    = 13,
        prop_rpm            = 1200,
        num_blades          = 6,
        num_engines         = 2,
    )
 
    print(f"\n{prop.name}: static thrust at SL = "
          f"{convert.kg_to_lb(prop.max_thrust(0, 0)/convert.G0):.0f} lbf, "
          f"WF = {convert.kg_to_lb(prop.fuel_flow(prop.max_thrust(0,0), 0, 0, 0))*3600:.0f} lb/hr")
    print(f"{'Alt (ft)':>10} {'Mach':>6} {'J':>7} {'M_tip':>8} {'eta_prop':>10} "
          f"{'Max Thrust (N)':>16} {'Fuel Flow (kg/s)':>18}")
    for alt_ft, mach in [(0, 0.20), (10000, 0.35), (20000, 0.45), (20000, 0.55), (25000, 0.65)]:
        alt_m   = convert.ft_to_m(alt_ft)
        DISAC   = convert.DISAF_to_C(DISAF)
        J       = prop.advance_ratio(alt_m, mach, DISAC)
        M_tip   = prop.helical_tip_mach(alt_m, mach, DISAC)
        eta     = prop.prop_efficiency(alt_m, mach, DISAC)
        t_max   = prop.max_thrust(alt_m, mach, DISAC) # N
        WF      = prop.fuel_flow(t_max, alt_m, mach, DISAC)  # kg/s
        print(f"{alt_ft:>10} {mach:>6.2f} {J:>7.2f} {M_tip:>8.3f} {eta:>10.3f} "
              f"{t_max:>16.1f} {WF:>18.4f}")
