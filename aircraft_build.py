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
    def __init__(
            self, 
            name: str, 
            wing_area_ft2: float, 
            operating_empty_weight_lb: float, 
            payload_weight_lb: float, 
            fuel_weight_lb: float, 
            aero_model: AeroModelBase, 
            propulsion_model: PropulsionModelBase,
            DISAF: float = 0,
        ):
        self.name = name
        self.wing_area_m2 = wing_area_ft2 * convert.ft_to_m(1)**2
        self.operating_empty_weight_lb = operating_empty_weight_lb
        self.payload_weight_lb = payload_weight_lb
        self.fuel_weight_lb = fuel_weight_lb
        self.aero_model = aero_model
        self.propulsion_model = propulsion_model
        self.DISAC = DISAF*5/9 # carry DISAC
        
        # mass props buildup
        self.zero_fuel_weight_lb = self.operating_empty_weight_lb + self.payload_weight_lb
        self.gross_weight_lb = self.zero_fuel_weight_lb + self.fuel_weight_lb
        
    # Convert current mass (kg) to weight force (N).
    def weight_n(self, weight_kg: float) -> float:
        return weight_kg * convert.G0

    # CL required for level, unaccelerated flight (L = W)
    def required_cl(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        tas = convert.mach_to_tas(mach, altitude_m)
        rho = isa_conditions(altitude_m, self.DISAC)["density_kg_m3"]
        q = 0.5 * rho * tas ** 2
        weight = self.weight_n(weight_kg)
        return self.aero_model.cl_for_lift(weight, q, self.wing_area_m2)

    # Drag (N) for level flight at given weight/altitude/Mach
    def drag_n(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        cl = self.required_cl(weight_kg, altitude_m, mach)
        cd = self.aero_model.get_cd(cl, mach)
        tas = convert.mach_to_tas(mach, altitude_m)
        rho = isa_conditions(altitude_m, self.DISAC)["density_kg_m3"]
        q = 0.5 * rho * tas ** 2
        return cd * q * self.wing_area_m2

    # Thrust required for steady, level, unaccelerated cruise (T = D)
    def thrust_required_n(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        return self.drag_n(weight_kg, altitude_m, mach)

    # Fuel flow (kg/s) to maintain steady level cruise
    def fuel_flow_kg_s(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        thrust_needed = self.thrust_required_n(weight_kg, altitude_m, mach)
        max_thrust = self.propulsion_model.max_thrust(altitude_m, mach, self.DISAC)
        if thrust_needed > max_thrust:
            raise RuntimeError(
                f"Thrust required ({thrust_needed:.0f} N) > max thrust ({max_thrust:.0f}) "
                f"at ALT = {altitude_m:.0f}m, MACH = {mach:.2f}.\n"
                f"Aircraft cannot sustain this flight condition."
            )
        return self.propulsion_model.fuel_flow(thrust_needed, altitude_m, mach, self.DISAC)

    def lift_to_drag(self, weight_kg: float, altitude_m: float, mach: float) -> float:
        cl = self.required_cl(weight_kg, altitude_m, mach)
        cd = self.aero_model.get_cd(cl, mach)
        return cl / cd

#------------------------------ DEBUGGING ------------------------------------- 
if __name__ == "__main__":
    from aero_model import SimpleDragPolar
    from propulsion_model import SimpleTurbofan
   
    ac = Aircraft(
        name                        = "Test Airplane",
        wing_area_ft2               = 1320,
        operating_empty_weight_lb   = 80000,
        payload_weight_lb           = 33000,
        fuel_weight_lb              = 40000,
        aero_model=SimpleDragPolar(cd0=0.020, aspect_ratio=9.5, oswald_efficiency=0.80),
        propulsion_model=SimpleTurbofan(sea_level_thrust_n=120000, tsfc_kg_per_n_per_s=1.75e-5, num_engines=2),
    )
    
    alt_m               = convert.ft_to_m(35000)
    mach                = 0.78

    print(f"Aircraft: {ac.name}")
    print(f"Wing Area:      {ac.wing_area_m2/convert.ft_to_m(1)**2:.0f} ft2")
    print(f"Empty Weight:   {ac.operating_empty_weight_lb:.0f} lb")
    print(f"Weight:         {ac.gross_weight_lb:.0f} lb") 
    print(f"Required CL:    {ac.required_cl(convert.lb_to_kg(ac.gross_weight_lb), alt_m, mach):.4f}")
    print(f"Drag:           {ac.drag_n(convert.lb_to_kg(ac.gross_weight_lb), alt_m, mach):.0f} N")
    print(f"L/D:            {ac.lift_to_drag(convert.lb_to_kg(ac.gross_weight_lb), alt_m, mach):.2f}")
    print(f"Fuel flow:      {ac.fuel_flow_kg_s(convert.lb_to_kg(ac.gross_weight_lb), alt_m, mach)*3600:.1f} kg/hr")
