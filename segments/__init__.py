"""
Mission segment package.

Each segment's `run()` method returns a SegmentResult and the ending aircraft
weight. Segments are built, then handed to Mission(aircraft, segments, saveDir)
in flight order. Mission.run() calls each segment's run(aircraft, weight_kg) and
feeds the ending weight into the next.

All constructor arguments are in imperial units (ft, lb, kt, min). Segments
compute internally in SI.

FILE STRUCTURE
-------------------------------------------------------------------------------
    segments/base.py            The catalog: a full description of every segment
                                (arguments, accepted values, sources of error)
                                plus the shared pieces - SegmentResult, the
                                MissionSegment base class, the altitude
                                inheritance ('-1') rules, and MissionLeg (the
                                'leg' input used by radius missions).
                                READ THIS FIRST.
    segments/ground_ops.py      GroundOps
    segments/accel_decel.py     AccelDecelSegment
    segments/cruise.py          ConstantAltCruiseSegment
    segments/loiter.py          LoiterSegment
    segments/common_gamma.py    CommonGammaSegment (shared climb/descent march)
    segments/climb.py           ClimbSegment
    segments/descent.py         DescentSegment
    segments/cruise_climb.py    CruiseClimbSegment, CruiseClimbError
    segments/air_refuel.py      AerialRefuelSegment, AerialRefuelError

Importing is unchanged from when this was a single segments.py module:

    `from segments import ClimbSegment, ConstantAltCruiseSegment`

ADDING A NEW SEGMENT
-------------------------------------------------------------------------------
    1. Create segments/<new_segment>.py. Subclass MissionSegment from .base
       (or CommonGammaSegment from .common_gamma if it integrates over altitude),
       give it a class-level `name`, and implement run(aircraft, start_weight_kg)
       returning a SegmentResult.
    2. If it takes an altitude, run the input through _parse_altitude_input() and
       implement needs_start_altitude()/resolve_start_altitude()/
       declared_start_altitude_ft() so '-1' prior segment altitude works.
    2a. Accept a trailing 'leg: str = None' argument and store
       self.leg = _parse_leg_input(leg, type(self).__name__), so the segment can
       be used in a radius mission. Skipping this is safe - MissionSegment
       defaults self.leg to MissionLeg.UNASSIGNED - but the segment can then
       never be tagged outbound/inbound.
    3. Import it below, add it to __all__ and to SEGMENT_TYPES.
    4. Document it in the catalog at the top of segments/base.py.
"""

from .base import (
    MissionSegment,
    SegmentResult,
    MissionLeg,
    _parse_altitude_input,
    _parse_leg_input,
)
from .ground_ops import GroundOps
from .accel_decel import AccelDecelSegment
from .cruise import ConstantAltCruiseSegment
from .loiter import LoiterSegment
from .common_gamma import CommonGammaSegment
from .climb import ClimbSegment
from .descent import DescentSegment
from .cruise_climb import CruiseClimbSegment, CruiseClimbError
from .air_refuel  import AerialRefuelSegment, AerialRefuelError

__all__ = [
    # Shared plumbing
    "MissionSegment",
    "SegmentResult",
    "MissionLeg",
    "CommonGammaSegment",
    # Segments
    "GroundOps",
    "AccelDecelSegment",
    "ConstantAltCruiseSegment",
    "LoiterSegment",
    "ClimbSegment",
    "DescentSegment",
    "CruiseClimbSegment",
    "AerialRefuelSegment",
    # Errors
    "CruiseClimbError",
    "AerialRefuelError",
]

# Name -> class, keyed by each segment's class-level `name`. Useful for building
# a mission from a config file/dict rather than from Python constructor calls.
SEGMENT_TYPES = {
    GroundOps.name:                 GroundOps,
    AccelDecelSegment.name:         AccelDecelSegment,
    ConstantAltCruiseSegment.name:  ConstantAltCruiseSegment,
    LoiterSegment.name:             LoiterSegment,
    ClimbSegment.name:              ClimbSegment,
    DescentSegment.name:            DescentSegment,
    CruiseClimbSegment.name:        CruiseClimbSegment,
    AerialRefuelSegment.name:       AerialRefuelSegment,
}
