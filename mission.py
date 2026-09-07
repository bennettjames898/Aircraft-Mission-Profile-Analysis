"""
Mission class: sequences a list of MissionSegments end-to-end, carrying
weight forward from one segment to the next. 
    - call 'result = mission.run(Aircraft(), List[MissionSegment], saveDir)' 
      to evaluate the 'Aircraft' model in the MissionSegment list.
          - 'saveDir=None' will not save any files (must use 
            'print(result.summary)' to see the output summary tabe).
          - Providing a filepath to save will also print the summary to the 
            command line.
    - 'Aircraft.gross_weight_lb' is the starting weight of the mission run. 
      Adjust the aircraft's mass properties to change the starting weight.

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
    """
    MissionResult creates results files for the whole mission (summary table 
    and a time history of each step through each segment). 
    """
    def __init__(
            self,     
            aircraft: Aircraft,
            end_weight_lb: float,
            total_fuel_burned_lb: float,
            total_distance_nm: float,
            total_time_s: float,
            saveDir: str,
            missionSuccess: str,
            segment_results: List[SegmentResult] = field(default_factory=list),
        ):
        self.aircraft = aircraft
        self.end_weight_lb = end_weight_lb
        self.total_fuel_burned_lb = total_fuel_burned_lb
        self.total_distance_nm = total_distance_nm
        self.total_time_s = total_time_s
        self.saveDir = saveDir
        self.missionSuccess = missionSuccess
        self.segment_results = segment_results
        
        # setup the summary file of the mission
        summary, fnameSum = self.buildSummary()
        self.summary = summary
        
        # build the time history file of the mission - TODO
        timehistory, fnameTH = self.buildTimeHistory()
        
        # display and print if desired 
        if self.saveDir is not None:
            print(summary)
            self.writeFile(fnameSum,summary)
            self.writeFile(fnameTH,timehistory)
            
    def buildSummary(self) -> str:
        # Collect aero input values
        aero_lines = ["\n----------------------------------Aerodynamic Model----------------------------------"]
        aeroIn = self.aircraft.aero_model.inputs
        for key, value in aeroIn.items():
            aero_lines.append(f"{key:<20} = {value}")
        aero_lines = aero_lines[:-1]
        sumOutAero = "\n".join(aero_lines)
            
        # Collect prop input values
        prop_lines = ["\n----------------------------------Propulsion Model-----------------------------------"]
        propIn = self.aircraft.propulsion_model.inputs
        for key, value in propIn.items():
            prop_lines.append(f"{key:<20} = {value}")
        prop_lines = prop_lines[:-1]
        sumOutProp = "\n".join(prop_lines)
        
        # Mass Props values
        lines = [
            "\n----------------------------------Aircraft Model-------------------------------------",
            f"Name:               {self.aircraft.name}",
            f"Wing Area [ft^2]:   {self.aircraft.wing_area_m2/(convert.ft_to_m(1)**2)}",
            f"OWE [lb]:           {self.aircraft.operating_empty_weight_lb}",
            f"Payload [lb]:       {self.aircraft.payload_weight_lb}",
            f"Internal Fuel [lb]: {self.aircraft.fuel_weight_lb}",
            f"Gross Weight [lb]:  {self.aircraft.gross_weight_lb}",
            ]
        lines.append(f"{'Mission summary':^112}")
        lines.append("." * 112)
        lines.append(
            f"{'Segment Name':<12}{'Time (min)':>16}{'Dist (nm)':>16}{'Fuel (lb)':>20}{'Gross Wt. (lb)':>16}"
            f"{'Altitude (ft)':>16}{'Mach Number':>16}"
        )
        lines.append(
            f"{'':^12}{'Seg':>8}{'Total':>8}{'Seg':>8}{'Total':>8}{'Seg':>10}{'Total':>10}{'':>16}"
            f"{'Start':>8}{'End':>8}{'Start':>8}{'End':>8}"
        )
        
        # Summary table of mission segments
        runTime = 0
        runDist = 0
        runFuel = 0
        for seg in self.segment_results:
            runTime = runTime + seg.time_s/60
            runDist = runDist + seg.distance_nm
            runFuel = runFuel + convert.kg_to_lb(seg.fuel_burned_kg)
            lines.append(
                f"{seg.segment_name:<12}" # segment name
                f"{seg.time_s/60:>8.1f}{runTime:>8.1f}" # Time
                f"{seg.distance_nm:>8.1f}{runDist:>8.1f}" # Distance
                f"{convert.kg_to_lb(seg.fuel_burned_kg):>10.1f}{runFuel:>10.1f}" # Fuel
                f"{convert.kg_to_lb(seg.end_weight_kg):>16.1f}" # Gross Weight
                f"{seg.start_altitude_ft:>8.1f}{seg.end_altitude_ft:>8.1f}" # Altitude
                f"{seg.history[0]['mach']:>8.3f}{seg.history[-1]['mach']:>8.3f}" # Mach
            )
        lines.append("-" * 112)
        lines.append(
            f"{'TOTAL':<12}{self.total_time_s/60:>16.1f}{self.total_distance_nm:>16.1f}"
            f"{self.total_fuel_burned_lb:>20.1f}{self.end_weight_lb:>16.1f}"
            f"{self.missionSuccess[0]}"
        )
        
        # build text file
        sumOutTab = "\n".join(lines)
        summaryOut = sumOutAero+sumOutProp+sumOutTab
        fnameSum = "MssnSum_"+self.aircraft.name.replace(" ","-")+".txt"
        return summaryOut, fnameSum
            
    def buildTimeHistory(self) -> str:
               
        # column headers
        lines = [
            f"{'Seg':<12}"                                  # Segment name
            f"{'Seg Time [min]':>16}{'Run Time [min]':>16}" # Time
            f"{'Seg Dist [nm]':>16}{'Run Dist [nm]':>16}"   # Distance
            f"{'Seg Fuel [lb]':>16}{'Run Fuel [lb]':>16}"   # Fuel
            f"{'Gross Wt [lb]':>16}"                        # Gross Weight
            f"{'Alt [ft]':>10}{'Mach':>8}{'KTAS':>8}{'KCAS':>8}" # Altitude and Speeds
            f"{'FN/eng [lb]':>10}{'Drag [lb]':>10}{'Fuel FLow [lb/hr]':>18}" # FN, Drag, WFtotal
            f"{'Ps Pot. [fpm]':>14}"
        ]
        
        runTime = 0
        runDist = 0
        runFuel = 0
        for seg in self.segment_results:
            for ii, segTH in enumerate(seg.history):
                runTime = runTime + segTH['time_min']
                runDist = runDist + segTH['distance_nm']
                runFuel = runFuel + segTH['Fuel_burn_lb']
                lines.append(
                    f"{seg.segment_name:<12}" # segment name
                    f"{segTH['time_min']:>16.1f}{runTime:>16.1f}" # Time
                    f"{segTH['distance_nm']:>16.1f}{runDist:>16.1f}" # Distance
                    f"{segTH['Fuel_burn_lb']:>16.1f}{runFuel:>16.1f}" # Fuel
                    f"{segTH['weight_lb']:>16.1f}" # Gross Weight
                    f"{segTH['altitude_ft']:>10.1f}{segTH['mach']:>8.4f}{segTH['tas_kt']:>8.1f}" # ALT / Mach / KTAS
                    f"{convert.ms_to_kt(convert.mach_to_cas(segTH['mach'], convert.ft_to_m(segTH['altitude_ft']),self.aircraft.DISAC)):>8.1f}" # KCAS
                    f"{segTH['thrust_lb']/self.aircraft.propulsion_model.num_engines:10.1f}"f"{segTH['drag_lb']:10.1f}"f"{segTH['fuel_flow_lbphr']:18.1f}"
                    f"{segTH['Ps_theor_fpm']:14.1f}"
                )
                
        # build text file
        timeHistoryOut = "\n".join(lines)
        fnameTH = "MssnTimeHist_"+self.aircraft.name.replace(" ","-")+".txt"
        return timeHistoryOut, fnameTH
    
    def writeFile(self,filename,text):
        with open(self.saveDir+"\\"+filename,"w") as file:
            print("Writing "+filename+"...")
            for line in text:
                file.write(line)

class Mission:
    """
    Mission runs the list of mission segments 
    """
    def __init__(self, aircraft: Aircraft, segments: List[MissionSegment], saveDir: str):
        self.aircraft = aircraft
        self.segments = segments
        self.saveDir = saveDir # None = no saved file

    # Evlauate the segment list beginning at some defined weight
    def run(self) -> MissionResult:
        start_weight_kg     = convert.lb_to_kg(self.aircraft.gross_weight_lb)
        weight_kg           = start_weight_kg
        segment_results     = []
        total_distance_nm   = 0
        total_time_s        = 0

        # Loop thru each mission segment, running segment-specific solver
        for segment in self.segments:
            result = segment.run(self.aircraft, weight_kg)
            segment_results.append(result)
            weight_kg           = result.end_weight_kg
            total_distance_nm   += result.distance_nm
            total_time_s        += result.time_s
            
        # place error message if there is insufficient fuel
        end_line = []
        if weight_kg < convert.lb_to_kg(self.aircraft.operating_empty_weight_lb):
            end_line = [
                f"\n\nWARNING: mission ends below zero-fuel weight "
                f"(Final Weight = {convert.kg_to_lb(weight_kg):.0f} lb < ZFW = {self.aircraft.zero_fuel_weight_lb:.0f} lb). "
            ]
        else:
            end_line = [
                f"\n\nFuel remaining at end of mission: {convert.kg_to_lb(weight_kg) - self.aircraft.zero_fuel_weight_lb:.0f} lb"
            ]

        # Output whole mission summary data
        return MissionResult(
            aircraft             = self.aircraft,
            end_weight_lb        = convert.kg_to_lb(weight_kg),
            total_fuel_burned_lb = convert.kg_to_lb(start_weight_kg - weight_kg),
            total_distance_nm    = total_distance_nm,
            total_time_s         = total_time_s,
            segment_results      = segment_results,
            saveDir              = self.saveDir,
            missionSuccess       = end_line,
        )