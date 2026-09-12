"""
Unit conversions used throughout the project.

"""
import math

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
# ------------------------- Temperature Conversions ---------------------------
###############################################################################
def DISAF_to_C(DISAF: float) -> float:
    """
    Convert Delta ISA conditions in degF to delta ISA condition in DegC/K
    """
    return DISAF * (5/9)

def DISAC_to_F(DISAC: float) -> float:
    """
    Convert Delta ISA conditions in degC/K to delta ISA condition in DegF
    """
    return DISAC * (9/5)

###############################################################################
# ------------------------- Mass/Weight Conversions ---------------------------
###############################################################################
G0 = 9.80665  # m/s^2

def kg_to_lb(kg: float) -> float:
    return kg * 2.20462

def lb_to_kg(lb: float) -> float:
    return lb / 2.20462

###############################################################################
# ---------------------------- Speed Conversions ------------------------------
###############################################################################
# Convert KNOT to METER/SEC
def kt_to_ms(kt: float) -> float:
    return kt * 0.514444

# Convert METER/SEC to KNOT
def ms_to_kt(ms: float) -> float:
    return ms / 0.514444

# Convert METER/SEC to FT/SEC
def ms_to_fts(ms: float) -> float:
    return ms * 3.28084

# Convert FT/SEC to METER/SEC
def fts_to_ms(fts: float) -> float:
    return fts / 3.28084

###############################################################################
# ------------------------ Airspeed Type Conversions --------------------------
###############################################################################
import atmosphere
# MACH to True airspeed (m/s)
def mach_to_tas(mach: float, altitude_m: float, DISAC: float = 0.0) -> float:
    a = atmosphere.isa_conditions(altitude_m, DISAC)["speed_of_sound_m_s"]
    return mach * a

# True airspeed (m/s) to MACH
def tas_to_mach(tas_m_s: float, altitude_m: float, DISAC: float = 0.0) -> float:
    a = atmosphere.isa_conditions(altitude_m, DISAC)["speed_of_sound_m_s"]
    return tas_m_s / a

# Calibrated Airspeed to MACH
def cas_to_mach(cas_m_s: float, altitude_m: float) -> float:
    """
    Reference: Anderson, Introduction to Flight
    """
    qc = atmosphere.P0 * ((1.0 + 0.2 * (cas_m_s / atmosphere.A0) ** 2) ** 3.5 - 1.0)
    p_local = atmosphere.isa_conditions(altitude_m, 0)["pressure_Pa"]
    return math.sqrt(5.0 * ((qc / p_local + 1.0) ** (2.0 / 7.0) - 1.0))

# MACH to Calibrated Airspeed
def mach_to_cas(mach: float, altitude_m: float) -> float:
    p_local = atmosphere.isa_conditions(altitude_m, 0)["pressure_Pa"]
    qc = p_local * ((1.0 + 0.2 * mach ** 2) ** 3.5 - 1.0)
    return atmosphere.A0 * math.sqrt(5.0 * ((qc / atmosphere.P0 + 1.0) ** (2.0 / 7.0) - 1.0))

# Dynamic pressure (Pa) lookup
def dynamic_pressure(tas_m_s: float, altitude_m: float, DISAC: float = 0.0) -> float:
    rho = atmosphere.isa_conditions(altitude_m, DISAC)["density_kg_m3"]
    return 0.5 * rho * tas_m_s ** 2