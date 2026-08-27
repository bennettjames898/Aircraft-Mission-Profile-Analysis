"""
Aircraft class: Combines weights and the aero/propulsion
models into a single object that mission segments query.

"""

from dataclasses import dataclass

from aero_model import AeroModelBase
from propulsion_model import PropulsionModelBase

from atmosphere import isa_conditions
import unit_conversions as convert

@dataclass
class Aircraft:    
    def __init__(self, name: str, wing_area_ft2: float, operating_empty_weight_lb: float, aero_model: AeroModelBase, propulsion_model: PropulsionModelBase):
        self.name = name
        self.wing_area_ft2 = wing_area_ft2
        self.operating_empty_weight_lb = operating_empty_weight_lb
        self.wing_area_m2 = wing_area_ft2 * convert.ft_to_m(1)**2
        self.operating_empty_weight_kg = convert.lb_to_kg(operating_empty_weight_lb)
        self.aero_model = aero_model
        self.propulsion_model = propulsion_model
        
    # Convert current mass (kg) to weight force (N).1
    def weight_n(self, current_weight_kg: float) -> float:
        return current_weight_kg * convert.G0

    # CL required for level, unaccelerated flight (L = W)
    def required_cl(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        tas = convert.mach_to_tas(mach, altitude_m)
        rho = isa_conditions(altitude_m)["density_kg_m3"]
        q = 0.5 * rho * tas ** 2
        weight = self.weight_n(weight_kg)
        return self.aero_model.cl_for_lift(weight, q, self.wing_area_m2)

    # Drag (N) for level flight at given weight/altitude/Mach
    def drag_n(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        cl = self.required_cl(weight_kg, altitude_m, mach)
        cd = self.aero_model.get_cd(cl, mach)
        tas = convert.mach_to_tas(mach, altitude_m)
        rho = isa_conditions(altitude_m)["density_kg_m3"]
        q = 0.5 * rho * tas ** 2
        return cd * q * self.wing_area_m2

    # Thrust required for steady, level, unaccelerated cruise (T = D)
    def thrust_required_n(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        return self.drag_n(weight_kg, altitude_m, mach)

    # Fuel flow (kg/s) to maintain steady level cruise
    def fuel_flow_kg_s(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        thrust_needed = self.thrust_required_n(weight_kg, altitude_m, mach)
        max_thrust = self.propulsion_model.max_thrust(altitude_m, mach)
        if thrust_needed > max_thrust:
            raise RuntimeError(
                f"Thrust required ({thrust_needed:.0f} N) > max thrust ({max_thrust:.0f}) "
                f"at ALT = {altitude_m:.0f}m, MACH = {mach:.2f}.\n"
                f"Aircraft cannot sustain this flight condition."
            )
        return self.propulsion_model.fuel_flow(thrust_needed, altitude_m, mach)

    def lift_to_drag(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        cl = self.required_cl(weight_kg, altitude_m, mach)
        cd = self.aero_model.get_cd(cl, mach)
        return cl / cd

#------------------------------ DEBUGGING ------------------------------------- 
if __name__ == "__main__":
    from aero_model import SimpleDragPolar
    from propulsion_model import SimpleTurbofan
   
    ac = Aircraft(
        name                        = "Generic Narrowbody",
        wing_area_ft2                = 1320,
        operating_empty_weight_lb    = 80000,
        aero_model=SimpleDragPolar(cd0=0.020, aspect_ratio=9.5, oswald_efficiency=0.80),
        propulsion_model=SimpleTurbofan(sea_level_thrust_n=120000, tsfc_kg_per_n_per_s=1.75e-5, num_engines=2),
    )
    
    total_weight_kg     = 100000
    alt_m               = convert.ft_to_m(25000)
    mach                = 0.78

    print(f"Aircraft: {ac.name}")
    print(f"Wing Area:     {ac.wing_area_m2:.0f} m2 | {ac.wing_area_ft2:.0f} ft2")
    print(f"Empty Weight:  {ac.operating_empty_weight_kg:.0f} kg | {ac.operating_empty_weight_lb:.0f} lb")
    print(f"Weight: {total_weight_kg:.0f} kg, Alt: {convert.m_to_ft(alt_m):.0f} ft, Mach: {mach}")
    print(f"Required CL:   {ac.required_cl(total_weight_kg, alt_m, mach):.4f}")
    print(f"Drag:          {ac.drag_n(total_weight_kg, alt_m, mach):.0f} N")
    print(f"L/D:           {ac.lift_to_drag(total_weight_kg, alt_m, mach):.2f}")
    print(f"Fuel flow:     {ac.fuel_flow_kg_s(total_weight_kg, alt_m, mach)*3600:.1f} kg/hr")
