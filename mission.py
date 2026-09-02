"""
Mission class: sequences a list of MissionSegments end-to-end, carrying
weight forward from one segment to the next.

All physics lives in segments.py and aircraft.py.
Mission's only job is bookkeeping: run segment 1,
take the ending weight as segment 2's starting weight, and so on, while
accumulating totals and keeping every segment's history for plotting.
"""

from dataclasses import dataclass, field
from typing import List

from aircraft_build import Aircraft
from segments import MissionSegment, SegmentResult
import unit_conversions as convert


@dataclass
class MissionResult:
    aircraft_name: str
    start_weight_lb: float
    end_weight_lb: float
    total_fuel_burned_lb: float
    total_distance_nm: float
    total_time_s: float
    segment_results: List[SegmentResult] = field(default_factory=list)

    @property
    def total_time_hr(self) -> float:
        return self.total_time_s / 3600

    def summary(self) -> str:
        lines = [
            f"Mission summary: {self.aircraft_name}",
            f"{'Segment':<12}{'Time (min)':>12}{'Fuel (lb)':>12}{'Dist (nm)':>12}{'Weight (lb)':>14}",
        ]
        for seg in self.segment_results:
            lines.append(
                f"{seg.segment_name:<12}{seg.time_s/60.0:>12.1f}{convert.kg_to_lb(seg.fuel_burned_kg):>12.1f}"
                f"{seg.distance_nm:>12.1f}{convert.kg_to_lb(seg.end_weight_kg):>14.1f}"
            )
        lines.append("-" * 62)
        lines.append(
            f"{'TOTAL':<12}{self.total_time_hr*60:>12.1f}{self.total_fuel_burned_lb:>12.1f}"
            f"{self.total_distance_nm:>12.1f}{self.end_weight_lb:>14.1f}"
        )
        return "\n".join(lines)

class Mission:
    def __init__(self, aircraft: Aircraft, segments: List[MissionSegment]):
        self.aircraft = aircraft
        self.segments = segments

    # Evlauate the segment list beginning at some defined weight
    def run(self, start_weight_lb: float) -> MissionResult:
        start_weight_kg     = convert.lb_to_kg(start_weight_lb)
        weight_kg           = start_weight_kg
        segment_results     = []
        total_distance_nm   = 0.0
        total_time_s        = 0.0

        # Loop thru each mission segment, running segment-specific solver
        for segment in self.segments:
            result = segment.run(self.aircraft, weight_kg)
            segment_results.append(result)
            weight_kg           = result.end_weight_kg
            total_distance_nm   += result.distance_nm
            total_time_s        += result.time_s

        # Output whole mission summary data
        return MissionResult(
            aircraft_name        = self.aircraft.name,
            start_weight_lb      = start_weight_lb,
            end_weight_lb        = convert.kg_to_lb(weight_kg),
            total_fuel_burned_lb = convert.kg_to_lb(start_weight_kg - weight_kg),
            total_distance_nm    = total_distance_nm,
            total_time_s         = total_time_s,
            segment_results      = segment_results,
        )