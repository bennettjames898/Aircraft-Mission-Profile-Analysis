"""
Unit conversons used throughout the project.

"""
###############################################################################
# ---------------------------- Length Conversions -----------------------------
###############################################################################
# Convert FEET to METER
def ft_to_m(ft: float) -> float:
    return ft * 0.3048

# Convert METER to FEET
def m_to_ft(m: float) -> float:
    return m / 0.3048

# Convert NMi to METER
def nm_to_m(nm: float) -> float:
    return nm * 1852

# Convert METER to NMi
def m_to_nm(m: float) -> float:
    return m / 1852

###############################################################################
# ------------------------- Mass/Weight Conversions ---------------------------
###############################################################################
def kg_to_lb(kg: float) -> float:
    return kg * 2.20462

def lb_to_kg(lb: float) -> float:
    return lb / 2.20462

###############################################################################
# ---------------------------- Speed Conversions ------------------------------
###############################################################################
# Convert KNOT to METER PER SEC
def kt_to_ms(kt: float) -> float:
    return kt * 0.514444

# Convert METER PER SEC to KNOT
def ms_to_kt(ms: float) -> float:
    return ms / 0.514444

###############################################################################
# ------------------------ Airspeed Type Conversions --------------------------
###############################################################################
import atmosphere
# MACH to True airspeed (m/s)
def mach_to_tas(mach: float, altitude_m: float, delta_isa: float = 0.0) -> float:
    a = atmosphere.isa_conditions(altitude_m, delta_isa)["speed_of_sound_m_s"]
    return mach * a

# True airspeed (m/s) to MACH
def tas_to_mach(tas_m_s: float, altitude_m: float, delta_isa: float = 0.0) -> float:
    a = atmosphere.isa_conditions(altitude_m, delta_isa)["speed_of_sound_m_s"]
    return tas_m_s / a

# Dynamic pressure (Pa) lookup
def dynamic_pressure(tas_m_s: float, altitude_m: float, delta_isa: float = 0.0) -> float:
    rho = atmosphere.isa_conditions(altitude_m, delta_isa)["density_kg_m3"]
    return 0.5 * rho * tas_m_s ** 2