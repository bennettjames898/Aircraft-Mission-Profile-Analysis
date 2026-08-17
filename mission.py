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
    start_weight_kg: float
    end_weight_kg: float
    total_fuel_burned_kg: float
    total_distance_m: float
    total_time_s: float
    segment_results: List[SegmentResult] = field(default_factory=list)

    @property
    def total_time_hr(self) -> float:
        return self.total_time_s / 3600

    def summary(self) -> str:
        lines = [
            f"Mission summary: {self.aircraft_name}",
            f"{'Segment':<12}{'Fuel (kg)':>12}{'Dist (nm)':>12}{'Time (min)':>12}{'End Wt (kg)':>14}",
        ]
        for seg in self.segment_results:
            lines.append(
                f"{seg.segment_name:<12}{seg.fuel_burned_kg:>12.1f}"
                f"{convert.m_to_nm(seg.distance_m):>12.1f}{seg.time_s/60.0:>12.1f}{seg.end_weight_kg:>14.1f}"
            )
        lines.append("-" * 62)
        lines.append(
            f"{'TOTAL':<12}{self.total_fuel_burned_kg:>12.1f}"
            f"{convert.m_to_nm(self.total_distance_m):>12.1f}{self.total_time_hr*60:>12.1f}{self.end_weight_kg:>14.1f}"
        )
        return "\n".join(lines)

class Mission:
    def __init__(self, aircraft: Aircraft, segments: List[MissionSegment]):
        self.aircraft = aircraft
        self.segments = segments

    # Evlauate the segment list beginning at some defined weight
    def run(self, start_weight_kg: float) -> MissionResult:
        weight_kg           = start_weight_kg
        segment_results     = []
        total_distance_m    = 0.0
        total_time_s        = 0.0

        # Loop thru each mission segment, running segment-specific solver
        for segment in self.segments:
            result = segment.run(self.aircraft, weight_kg)
            segment_results.append(result)
            weight_kg           = result.end_weight_kg
            total_distance_m    += result.distance_m
            total_time_s        += result.time_s

        # Output whole mission summary data
        return MissionResult(
            aircraft_name           = self.aircraft.name,
            start_weight_kg         = start_weight_kg,
            end_weight_kg           = weight_kg,
            total_fuel_burned_kg    = start_weight_kg - weight_kg,
            total_distance_m        = total_distance_m,
            total_time_s            = total_time_s,
            segment_results         = segment_results,
        )