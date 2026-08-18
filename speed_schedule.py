"""
Climb/descent speed schedules.

Every schedule below exposes the same three-method interface so
segments.py doesn't need to know which kind of schedule it's using:

    mach_at_altitude(altitude_m) -> Mach number to fly at that altitude
    tas_at_altitude(altitude_m)  -> true airspeed (m/s) at that altitude
    dtas_dh(altitude_m)          -> d(TAS)/d(altitude) (m/s per m)

If TAS changes with altitude during the climb, the aircraft is 
accelerating in true airspeed even during a "steady" climb, and that 
acceleration consumes some of the available excess power that would go 
into rate of climb.

solver.py uses dtas_dh to apply that correction (see "acceleration
factor" in solver.py).
"""

from abc import ABC, abstractmethod
import unit_conversions as convert

class SpeedScheduleBase(ABC):
    @abstractmethod
    def mach_at_altitude(self, altitude_m: float) -> float:
        raise NotImplementedError

    def tas_at_altitude(self, altitude_m: float) -> float:
        return convert.mach_to_tas(self.mach_at_altitude(altitude_m), altitude_m)

    def dtas_dh(self, altitude_m: float, eps_m: float = 10.0) -> float:
        """
        d(TAS)/d(altitude) via central finite difference.
        """
        h_lo = max(altitude_m - eps_m, 0.0)
        h_hi = altitude_m + eps_m
        return (self.tas_at_altitude(h_hi) - self.tas_at_altitude(h_lo)) / (h_hi - h_lo)


class ConstantMachSchedule(SpeedScheduleBase):
    """Fly a fixed Mach number at every altitude"""

    def __init__(self, mach: float):
        self.mach = mach

    def mach_at_altitude(self, altitude_m: float) -> float:
        return self.mach


class ConstantTASSchedule(SpeedScheduleBase):
    """Fly a fixed true airspeed at every altitude"""

    def __init__(self, tas_m_s: float):
        self.tas_m_s = tas_m_s

    def mach_at_altitude(self, altitude_m: float) -> float:
        return convert.tas_to_mach(self.tas_m_s, altitude_m)

    def tas_at_altitude(self, altitude_m: float) -> float:
        return self.tas_m_s

    def dtas_dh(self, altitude_m: float, eps_m: float = 10.0) -> float:
        return 0.0


class ConstantCASSchedule(SpeedScheduleBase):
    """
    Fly a fixed calibrated airspeed at every altitude"""

    def __init__(self, cas_m_s: float):
        self.cas_m_s = cas_m_s

    def mach_at_altitude(self, altitude_m: float) -> float:
        return convert.cas_to_mach(self.cas_m_s, altitude_m)


class CASMachSchedule(SpeedScheduleBase):
    """
    The realistic climb/descent schedule: constant CAS up to a crossover
    altitude, constant Mach above it. This is how essentially every jet
    transport climb/descent is actually flown (e.g. "280/.78").

    The crossover altitude is found by root-finding for the altitude at
    which the CAS schedule's instantaneous Mach equals the target cruise
    Mach -- CAS-implied Mach increases monotonically with altitude (see
    atmosphere.py's cas_to_mach sanity check), so this is well-posed for
    brentq.
    """

    def __init__(self, cas_m_s: float, mach: float, altitude_search_ceiling_m: float = 20000):
        self.cas_m_s = cas_m_s
        self.mach = mach
        self._crossover_altitude_m = self._find_crossover_altitude(altitude_search_ceiling_m)

    def _find_crossover_altitude(self, ceiling_m: float) -> float:
        from scipy.optimize import brentq

        def residual(altitude_m):
            return convert.cas_to_mach(self.cas_m_s, altitude_m) - self.mach

        f_lo = residual(0)
        f_hi = residual(ceiling_m)

        if f_lo >= 0:
            # CAS schedule's Mach already meets/exceeds target Mach at sea level.
            return 0.0
        if f_hi <= 0:
            raise ValueError(
                f"CAS={self.cas_m_s:.1f} m/s and Mach={self.mach:.3f} never cross within "
                f"{ceiling_m:.0f} m."
            )
        return brentq(residual, 0.0, ceiling_m)

    @property
    def crossover_altitude_m(self) -> float:
        return self._crossover_altitude_m

    def mach_at_altitude(self, altitude_m: float) -> float:
        if altitude_m <= self._crossover_altitude_m:
            return convert.cas_to_mach(self.cas_m_s, altitude_m)
        return self.mach


def as_schedule(value) -> SpeedScheduleBase:
    """
    Convenience feature: accept either a SpeedScheduleBase instance or
    a plain float (treated as a constant Mach). Used by ClimbSegment and
    DescentSegment so callers aren't forced to import speed_schedule.py
    for the simple constant-Mach case.
    """
    if isinstance(value, SpeedScheduleBase):
        return value
    return ConstantMachSchedule(float(value))

if __name__ == "__main__":
    
    KCAS = 280
    Mach = 0.78

    schedule = CASMachSchedule(cas_m_s=convert.kt_to_ms(KCAS), mach=Mach)
    print(f"{KCAS} kt / M{Mach} schedule -- crossover altitude: {convert.m_to_ft(schedule.crossover_altitude_m):.0f} ft\n")

    print(f"{'Alt (ft)':>10} {'Mach':>8} {'TAS (kt)':>10} {'dTAS/dh (m/s per 1000ft)':>26}")
    for alt_ft in [0, 5000, 10000, 20000, 28000, 31000, 32000, 35000]:
        alt_m = convert.ft_to_m(alt_ft)
        m = schedule.mach_at_altitude(alt_m)
        tas_kt = convert.ms_to_kt(schedule.tas_at_altitude(alt_m))
        dtdh = convert.m_to_ft(schedule.dtas_dh(alt_m))  # per 1000 ft
        print(f"{alt_ft:>10} {m:>8.4f} {tas_kt:>10.1f} {dtdh:>26.3f}")
