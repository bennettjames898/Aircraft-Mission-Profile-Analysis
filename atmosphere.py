"""
International Standard Atmosphere (ISA) model.

Implements the 1976 US Standard Atmosphere up through the stratosphere
(0 - 20 km / 0 - 65,617 ft), which covers essentially all commercial
and most military fixed-wing mission profiles.

All functions accept/return SI units

Reference: NASA TM-X-74335
"""

import math

# --- Sea level reference conditions ---
T0      = 288.15        # K
P0      = 101325.0      # Pa
RHO0    = 1.225         # kg/m^3
A0      = 340.294       # m/s, speed of sound at sea level
G0      = 9.80665       # m/s^2
R_AIR   = 287.05287     # J/(kg*K), specific gas constant for dry air
GAMMA   = 1.4           # ratio of specific heats for air

# Tropopause boundary (standard day)
TROPOPAUSE_ALT_M    = 11000.0
TROPOPAUSE_TEMP_K   = 216.65
LAPSE_RATE          = -0.0065  # K/m, valid 0-11 km

# Unit conversion helpers
def ft_to_m(ft: float) -> float:
    return ft * 0.3048
def m_to_ft(m: float) -> float:
    return m / 0.3048
def kt_to_ms(kt: float) -> float:
    return kt * 0.514444
def ms_to_kt(ms: float) -> float:
    return ms / 0.514444

def isa_conditions(altitude_m: float, delta_isa: float = 0.0) -> dict:
    """
    Compute atmospheric properties at a given geopotential altitude.

    Parameters
    ----------
    altitude_m : float
        Geopotential altitude in meters. Valid 0 to 20,000 m.
    delta_isa : float, optional
        Temperature offset from standard day, in Kelvin (e.g. ISA+10 -> 10.0).
        Applied as a constant offset to the temperature profile.

    Returns
    -------
    dict with keys: temperature_K, pressure_Pa, density_kg_m3, speed_of_sound_m_s
    """
    if altitude_m < 0:
        raise ValueError("Altitude below sea level not supported.")
    if altitude_m > 20000.0:
        raise ValueError(
            "Altitude above 20,000 m not supported by this simplified ISA model."
        )

    if altitude_m <= TROPOPAUSE_ALT_M:
        # Troposphere: linear lapse rate
        temperature_K = T0 + LAPSE_RATE * altitude_m + delta_isa
        temperature_std_K = T0 + LAPSE_RATE * altitude_m  # pressure uses standard T
        pressure_Pa = P0 * (temperature_std_K / T0) ** (-G0 / (LAPSE_RATE * R_AIR))
    else:
        # Lower stratosphere: isothermal
        temperature_K = TROPOPAUSE_TEMP_K + delta_isa
        temperature_std_K = TROPOPAUSE_TEMP_K
        pressure_tropopause = P0 * (TROPOPAUSE_TEMP_K / T0) ** (
            -G0 / (LAPSE_RATE * R_AIR)
        )
        pressure_Pa = pressure_tropopause * math.exp(
            -G0 * (altitude_m - TROPOPAUSE_ALT_M) / (R_AIR * TROPOPAUSE_TEMP_K)
        )

    # Density from equation of state using the (possibly offset) temperature.
    # Note: pressure profile is driven by the standard-day temperature (hydrostatic
    # balance assumption for off-standard days), density reflects the actual temp.
    density_kg_m3 = pressure_Pa / (R_AIR * temperature_K)
    speed_of_sound_m_s = math.sqrt(GAMMA * R_AIR * temperature_K)

    return {
        "temperature_K": temperature_K,
        "pressure_Pa": pressure_Pa,
        "density_kg_m3": density_kg_m3,
        "speed_of_sound_m_s": speed_of_sound_m_s,
    }


def mach_to_tas(mach: float, altitude_m: float, delta_isa: float = 0.0) -> float:
    """True airspeed (m/s) from Mach number at a given altitude."""
    a = isa_conditions(altitude_m, delta_isa)["speed_of_sound_m_s"]
    return mach * a


def tas_to_mach(tas_m_s: float, altitude_m: float, delta_isa: float = 0.0) -> float:
    """Mach number from true airspeed (m/s) at a given altitude."""
    a = isa_conditions(altitude_m, delta_isa)["speed_of_sound_m_s"]
    return tas_m_s / a


def dynamic_pressure(tas_m_s: float, altitude_m: float, delta_isa: float = 0.0) -> float:
    """Dynamic pressure q = 0.5 * rho * V^2, in Pa."""
    rho = isa_conditions(altitude_m, delta_isa)["density_kg_m3"]
    return 0.5 * rho * tas_m_s ** 2

#------------------------------ DEBUGGING ------------------------------------- 
if __name__ == "__main__":
    # check against known ISA table values
    test_altitudes_ft = [0, 10000, 36089, 40000]
    print(f"{'Alt (ft)':>10} {'Temp (K)':>10} {'Press (Pa)':>12} {'Rho (kg/m3)':>12} {'a (m/s)':>10}")
    for alt_ft in test_altitudes_ft:
        c = isa_conditions(ft_to_m(alt_ft))
        print(
            f"{alt_ft:>10} {c['temperature_K']:>10.2f} {c['pressure_Pa']:>12.1f} "
            f"{c['density_kg_m3']:>12.4f} {c['speed_of_sound_m_s']:>10.2f}"
        )
