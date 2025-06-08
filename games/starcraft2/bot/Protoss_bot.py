import os
import random
import time
import copy
from typing import Set, List
import math
from sc2.bot_ai import BotAI

from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from games.starcraft2.utils.action_info import ActionDescriptions
from collections import Counter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import nest_asyncio

nest_asyncio.apply()

ATTACK_START_THRESHOLD = 70
ATTACK_STOP_THRESHOLD = 40
SOME_DELAY = 100
WAIT_ITER_FOR_KILL_VISIBLE = 200
WAIT_ITER_FOR_CLUSTER_ASSIGN = 400
CRITICAL_BUILDING_DISTANCE = 3  # Or another suitable value
# Define constants
DEFEND_UNIT_COUNT = 5  # Number of defensive units
DEFEND_UNIT_ASSIGN_INTERVAL = 200  # Interval steps for reassigning defensive units
MIN_DISTANCE = 3  # This value can be adjusted based on actual conditions
BUILDING_DISTANCE = 7
EXCLUDED_UNITS = {UnitTypeId.LARVA, UnitTypeId.CHANGELING, UnitTypeId.EGG}
SCOUTING_INTERVAL = 300


# import torch
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Protoss_Bot(BotAI):
    def __init__(self, transaction, lock, isReadyForNextStep):
        self.iteration = 0
        self.lock = lock
        self.transaction = transaction
        self.worker_supply = 12  # Number of workers
        self.army_supply = 0  # Army population
        self.base_pending = 0
        self.enemy_units_count = 0  # Number of enemy units
        # self.army_units_list=[]  # List of our army units
        # self.enemy_list = []
        self.base_count = 1  # Number of bases
        self.pylon_count = 0
        self.gas_buildings_count = 0
        self.gateway_count = 0
        self.forge_count = 0
        self.photon_cannon_count = 0
        self.shield_battery_count = 0
        self.warp_gate_count = 0
        self.cybernetics_core_count = 0
        self.twilight_council_count = 0
        self.robotics_facility_count = 0
        self.stargate_count = 0
        self.templar_archives_count = 0
        self.dark_shrine_count = 0
        self.robotics_bay_count = 0
        self.fleet_beacon_count = 0

        self.base_pending = 0
        self.action_dict = self.get_action_dict()

        self.rally_defend = False
        self.isReadyForNextStep = isReadyForNextStep
        self.is_attacking = False  # Determine if attacking
        self.assigned_to_clusters = False  # Determine if attack units are assigned
        self.temp1 = False  # Used to determine if an attack is initiated
        self.temp2 = False  # Used to determine if the enemy base is destroyed
        self.last_cluster_assignment_iter = 0
        self.clear_base_check_count = 0
        self.wait_iter_start = 0
        self.wait_iter_start_for_cluster = 0
        self.last_attack_visible_iter = 0
        self.last_cluster_assignment_iter = 0
        self.defend_units = []  # List of defensive units
        self.last_defend_units_assign_iter = 0  # Last iteration of defensive unit assignment
        self.temp_failure_list = []
        self.last_scouting_iteration = 0

        self.military_unit_types = {
            UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.VOIDRAY, UnitTypeId.STALKER,
            UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.OBSERVER,
            UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY, UnitTypeId.TEMPEST,
            UnitTypeId.ORACLE, UnitTypeId.COLOSSUS, UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM,
            UnitTypeId.IMMORTAL, UnitTypeId.CHANGELINGZEALOT
        }

    def is_position_valid_for_building(self, position: Point2) -> bool:
        """
        Check if the given position is suitable for building.
        :param position: The position to check.
        :return: Returns True if the position is suitable for building, otherwise False.
        """

        # Check the distance to other buildings
        # Iterate through all existing buildings
        for structure in self.structures:
            # If the distance between the given position and a building is less than the preset minimum distance + the building's radius, it is too close and may cause building overlap.
            if position.distance_to(structure.position) < BUILDING_DISTANCE + structure.radius:
                return False

        # Check the distance to resources
        # Iterate through all existing resources (minerals and gas)
        for resource in self.resources:
            # If the distance between the given position and a resource is less than the preset minimum distance + the resource's radius, it is too close and may hinder unit gathering.
            if position.distance_to(resource.position) < MIN_DISTANCE + 2:  # Adding 3 here ensures the pylon is a certain distance from resources
                return False

        # Check the distance to critical buildings
        # Iterate through all existing critical buildings, such as gas collectors and main bases
        critical_structures = self.structures({UnitTypeId.ASSIMILATOR, UnitTypeId.NEXUS})
        for critical_structure in critical_structures:
            # If the distance between the given position and a critical building is less than the preset minimum distance for critical buildings + the critical building's radius, it is too close.
            # This may hinder unit movement, especially between the base and gas collectors.
            if position.distance_to(
                    critical_structure.position) < CRITICAL_BUILDING_DISTANCE + critical_structure.radius:
                return False

        # Check the distance to buildings of the same type
        # Iterate through all existing buildings of certain specific types (such as Stargate, Robotics Facility, etc.)
        for building in self.structures(
                {UnitTypeId.STARGATE, UnitTypeId.ROBOTICSFACILITY, UnitTypeId.GATEWAY, ...}):
            # If the distance between the given position and a specific type of building is less than the preset minimum distance + the building's radius, it is too close.
            # This may cause building overlap or other undesirable behavior.
            if position.distance_to(building.position) < BUILDING_DISTANCE + building.radius:
                return False

        # If all the above conditions are met, the given position is suitable for building.
        return True

    def find_optimal_building_position_for_base(self, base_position: Point2, building_type: UnitTypeId,
                                                max_distance=15) -> Point2:
        """
        Find the optimal building position near the base.

        Args:
        - base_position (Point2): The position of the base we want to build around.
        - building_type (UnitTypeId): The type of building we want to construct.
        - max_distance (int): The maximum radius to search for a building position around the base.

        Returns:
        - Point2 or None: The optimal position for the new building, or None if no suitable position is found.
        """

        # Get all pylons
        pylons = self.structures(UnitTypeId.PYLON)

        # If there are no pylons, use the original method to find a building position near the Nexus
        if not pylons.exists:
            return self.original_building_position_search(base_position, building_type, max_distance)

        # Calculate scores for each pylon based on its distance to the Nexus and the number of nearby buildings
        pylon_scores = {}
        for pylon in pylons:
            distance_to_base = pylon.distance_to(base_position)
            nearby_buildings = sum(1 for building in self.structures if building.distance_to(pylon) < 12)
            # Scoring function: weight of 1 for distance to Nexus, weight of -2 for the number of nearby buildings
            score = 1 * distance_to_base - 2 * nearby_buildings
            pylon_scores[pylon] = score

        # Select the pylon with the highest score
        best_pylon = max(pylon_scores, key=pylon_scores.get)

        # Use the original method to find a building position near the best pylon
        best_position = self.original_building_position_search(best_pylon.position, building_type, max_distance)

        return best_position

    def original_building_position_search(self, base_position: Point2, building_type: UnitTypeId,
                                          max_distance=15) -> Point2:
        """
        Original method for finding a building position.
        """
        enemy_base_position = self.enemy_start_locations[0]
        weight_own_base = 1.0  # Positive weight: prefer positions farther from our base
        weight_enemy_base = -2.0  # Negative weight: avoid positions too close to the enemy base

        # Define a scoring function for each position
        def compute_score(pos):
            distance_to_own_base = pos.distance_to(base_position)
            distance_to_enemy_base = pos.distance_to(enemy_base_position)

            # Calculate score based on weighted distances
            score = weight_own_base * distance_to_own_base + weight_enemy_base / (distance_to_enemy_base + 1)
            return score

        for distance in range(1, max_distance + 1):
            candidate_positions = list(self.neighbors8(base_position, distance))
            valid_positions = [pos for pos in candidate_positions if self.is_position_valid_for_building(pos)]

            if valid_positions:
                best_position = max(valid_positions, key=compute_score)
                return best_position

        return None

    def find_best_base_for_building(self, building_type: UnitTypeId):
        base_positions = [base.position for base in self.townhalls]
        building_positions = [building.position for building in self.structures(building_type)]
        base_building_counts = {}
        for base_pos in base_positions:
            count = sum(1 for building_pos in building_positions if base_pos.distance_to(building_pos) <= 12)
            base_building_counts[base_pos] = count
        sorted_bases = sorted(base_building_counts.keys(), key=lambda base: base_building_counts[base])
        return sorted_bases[0] if sorted_bases else None

    async def handle_action_build_building(self, building_type: UnitTypeId, building_limits=None):
        """
        building_limits is a dictionary where the key is a time range and the value is the number of buildings allowed in that time range.
        For example, {"00:00-02:00": 1} means only 1 building is allowed between 0:00 and 2:00.

        """
        # If the cost cannot be afforded, return directly
        if not self.can_afford(building_type):
            print(f"Cannot afford {building_type}")
            return

        current_time = self.time_formatted  # Get the current game time

        # If building limits are provided and the current time is within the limit range
        if building_limits:
            for time_range, limit in building_limits.items():
                start_time, end_time = time_range.split('-')
                if start_time <= current_time <= end_time:
                    # Calculate the current number of buildings of the same type (including those under construction)
                    current_building_count = self.structures(building_type).amount + self.already_pending(building_type)

                    # If the current number of buildings has reached or exceeded the limit for this time period, do not build
                    if current_building_count >= limit:
                        print(f"Building limit reached for {building_type} for the time range {time_range}.")
                        return

        # --- The following is the code for finding building positions and initiating construction, consistent with the original ---

        if building_type in {UnitTypeId.PHOTONCANNON, UnitTypeId.SHIELDBATTERY}:
            # Determine the outer base
            enemy_start = self.enemy_start_locations[0]
            outermost_base = min(self.townhalls, key=lambda base: base.distance_to(enemy_start))

            # Find a nearby building position
            best_position = self.find_optimal_building_position_for_base(outermost_base.position, max_distance=8,
                                                                         building_type=building_type)

        else:
            best_base = self.find_best_base_for_building(building_type)
            if not best_base:
                print(f"No suitable base found for {building_type}")
                return

            # Prioritize finding the best position
            best_position = self.find_optimal_building_position_for_base(best_base, building_type=building_type)

            # If no suitable position is found at this best base, try to find a suitable position at other bases
            if not best_position:
                print(f"No suitable position found near best base for {building_type}. Trying alternative bases.")

                for base in self.townhalls:
                    if base.position != best_base.position:
                        best_position = self.find_optimal_building_position_for_base(base.position,
                                                                                     building_type=building_type)
                        if best_position:
                            break

            # If no ideal position is found at all bases, try to find a suboptimal position
            if not best_position:
                print("Trying suboptimal positions...")

                adjusted_building_distance = 10  # Can be adjusted as needed
                best_position = self.find_optimal_building_position_for_base(best_base.position,
                                                                             max_distance=adjusted_building_distance,
                                                                             building_type=building_type)

                # If still not found, try expanding the search range
                if not best_position:
                    print("Expanding search range...")
                    expanded_search_range = 30  # Can be adjusted as needed
                    best_position = self.find_optimal_building_position_for_base(best_base.position,
                                                                                 max_distance=expanded_search_range,
                                                                                 building_type=building_type)

        if not best_position:
            print(f"Still no suitable position found for {building_type}. Aborting.")
            return

        await self.build(building_type, near=best_position)
        print(f"Building {building_type}")

    def is_position_blocking_resources(self, position: Point2) -> bool:
        """
        Check if the specified position will block resource gathering.
        """
        for resource in self.resources:
            if position.distance_to(resource.position) < MIN_DISTANCE:
                return True
        return False

    def is_position_valid_for_pylon(self, position: Point2) -> bool:
        """
        Check if the specified position is suitable for building a pylon.
        """
        # Check the distance to other buildings, including other pylons
        for structure in self.structures:
            if position.distance_to(structure.position) < MIN_DISTANCE:
                return False

        # Specifically check the distance to ASSIMILATOR
        for assimilator in self.structures(UnitTypeId.ASSIMILATOR):
            if position.distance_to(assimilator.position) < MIN_DISTANCE * 1.5:
                return False

        # Check the distance to resources
        for resource in self.resources:
            if position.distance_to(resource.position) < MIN_DISTANCE:
                return False

        return True

    def find_optimal_pylon_position_for_base(self, base_position: Point2) -> Point2:
        """
        Find the optimal position for a pylon.
        """
        # Generate candidate positions for the base
        candidate_positions = [base_position] + [pos for i in range(1, 15) for pos in
                                                 self.neighbors8(base_position, distance=i)]

        # Filter out positions that will not block resource gathering
        valid_positions = [pos for pos in candidate_positions if
                           not self.is_position_blocking_resources(pos) and self.is_position_valid_for_pylon(pos)]

        # If no suitable position is found, return the base position directly
        if not valid_positions:
            return base_position

        pylons = self.structures(UnitTypeId.PYLON)
        pylon_positions = [pylon.position for pylon in pylons]

        # Use a sigmoid function to score each position
        def score_position(pos):
            total_score = 0
            for pylon_pos in pylon_positions:
                distance = pos.distance_to(pylon_pos)
                # This sigmoid function will give a score of 0.5 for a distance of 5
                score = 1 / (1 + math.exp(-distance + 5))
                total_score += score
            return total_score

        # Select the position with the highest score
        best_position = max(valid_positions, key=score_position)
        return best_position

    def find_best_base_for_pylon(self):
        """
        Find the best base position for building a pylon.
        """
        pylons = self.structures(UnitTypeId.PYLON)
        base_positions = [base.position for base in self.townhalls]
        pylon_positions = [pylon.position for pylon in pylons]
        base_pylon_counts = {}

        # Calculate the number of pylons near each base
        for base_pos in base_positions:
            count = sum(1 for pylon_pos in pylon_positions if base_pos.distance_to(pylon_pos) <= 10)
            base_pylon_counts[base_pos] = count

        # Return the base with the fewest nearby pylons
        sorted_bases = sorted(base_pylon_counts.keys(), key=lambda base: base_pylon_counts[base])
        return sorted_bases[0]

    def assign_defend_units(self, iteration):
        MILITARY_UNITS = self.get_military_units()

        # If the attack has not started, do not reassign defensive units
        if not self.temp1:
            return

        # Determine if defensive units need to be reassigned
        if not self.defend_units or iteration - self.last_defend_units_assign_iter >= DEFEND_UNIT_ASSIGN_INTERVAL:
            self.defend_units = random.sample(MILITARY_UNITS, DEFEND_UNIT_COUNT)
            self.last_defend_units_assign_iter = iteration

    def get_military_units(self):
        return self.units.of_type(self.military_unit_types)

    def get_enemy_unity(self):
        # Determine if there are enemy units
        if self.enemy_units:
            # Use list comprehension to get the name of each enemy unit, with the name format "enemy_" + unit type ID
            unit_names = ['enemy_' + str(unit.type_id) for unit in self.enemy_units]

            # Use Counter to count the number of each type of unit
            unit_type_amount = Counter(unit_names)

            # Print the current unit count statistics
            print(unit_type_amount)

            # Return the unit count statistics dictionary
            return unit_type_amount
        else:
            # If there are no enemy units, return an empty dictionary
            return {}

    def get_action_dict(self):
        action_description = ActionDescriptions('Protoss')
        action_dict = action_description.action_descriptions
        flat_dict = {}
        for key, value in action_dict.items():
            for inner_key, inner_value in value.items():
                flat_dict[inner_key] = inner_value
        return flat_dict

    def get_enemy_structure(self):
        # Determine if there are enemy structures
        if self.enemy_structures:
            # Use list comprehension to get the name of each enemy structure, with the name format "enemy_" + structure type ID
            structure_names = ['enemy_' + str(structure.type_id) for structure in self.enemy_structures]

            # Use Counter to count the number of each type of structure
            structure_type_amount = Counter(structure_names)

            # Print the current structure count statistics
            print(structure_type_amount)

            # Return the structure count statistics dictionary
        else:
            # If there are no enemy structures, return an empty dictionary
            return {}

    def get_information(self):
        """Retrieve game-related information and structure it."""
        information = {
            "resource": self._get_resource_information(),
            "building": self._get_building_information(),
            "unit": self._get_unit_information(),
            "planning": self._get_planning_information(),
            "research": self._get_research_information(),
        }

        information = self._update_enemy_information(information)
        return information

    def get_process_data(self):
        '''Intermediate result saving'''
        process_data={
            "iteration":self.iteration,
            "time":self.time,
            "supply_cap":self.transaction['information']['resource']['supply_cap'],
            "supply_used":self.transaction['information']['resource']['supply_used'],
            "minerals":self.minerals,
            "collected_minerals":self.state.score.collected_minerals,
            "spent_minerals":self.state.score.spent_minerals,
            "vespene":self.vespene,
            "collected_vespene":self.state.score.collected_vespene,
            "spent_vespene":self.state.score.spent_vespene,
            "score":self.state.score.score,
            "completed_tech":sum(1 for count in self.transaction['information']['building'].values() if count > 0)+sum(1 for researches in self.transaction['information']['research'].values() for status in researches.values() if status == 1)
        }
        print(process_data)
        return process_data
    def _get_resource_information(self):
        """Retrieve resource-related information."""
        self.worker_supply = self.workers.amount
        self.army_supply = self.supply_army
        # print("~~~~~~~~~~",self.state.score)
        return {
            'game_time': self.time_formatted,
            'worker_supply': self.worker_supply,
            'mineral': self.minerals,
            'gas': self.vespene,
            'supply_left': self.supply_left,
            'supply_cap': self.supply_cap,
            'supply_used': self.supply_used,
            'army_supply': self.army_supply,
        }

    def _get_building_information(self):
        """Retrieve building-related information."""
        self.base_count = self.structures(UnitTypeId.NEXUS).amount
        self.base_pending = self.already_pending(UnitTypeId.NEXUS)
        self.pylon_count = self.structures(UnitTypeId.PYLON).amount
        self.gas_buildings_count = self.structures(UnitTypeId.ASSIMILATOR).amount
        self.gateway_count = self.structures(UnitTypeId.GATEWAY).amount
        self.forge_count = self.structures(UnitTypeId.FORGE).amount
        self.photon_cannon_count = self.structures(UnitTypeId.PHOTONCANNON).amount
        self.shield_battery_count = self.structures(UnitTypeId.SHIELDBATTERY).amount
        self.warp_gate_count = self.structures(UnitTypeId.WARPGATE).amount
        self.cybernetics_core_count = self.structures(UnitTypeId.CYBERNETICSCORE).amount
        self.twilight_council_count = self.structures(UnitTypeId.TWILIGHTCOUNCIL).amount
        self.robotics_facility_count = self.structures(UnitTypeId.ROBOTICSFACILITY).amount
        self.stargate_count = self.structures(UnitTypeId.STARGATE).amount
        self.templar_archives_count = self.structures(UnitTypeId.TEMPLARARCHIVE).amount
        self.dark_shrine_count = self.structures(UnitTypeId.DARKSHRINE).amount
        self.robotics_bay_count = self.structures(UnitTypeId.ROBOTICSBAY).amount
        self.fleet_beacon_count = self.structures(UnitTypeId.FLEETBEACON).amount
        return {
            'nexus_count': self.base_count,
            'pylon_count': self.pylon_count,
            'gas_buildings_count': self.gas_buildings_count,
            'gateway_count': self.gateway_count,
            'forge_count': self.forge_count,
            'photon_cannon_count': self.photon_cannon_count,
            'shield_battery_count': self.shield_battery_count,
            'warp_gate_count': self.warp_gate_count,
            'cybernetics_core_count': self.cybernetics_core_count,
            'twilight_council_count': self.twilight_council_count,
            'robotics_facility_count': self.robotics_facility_count,
            'statgate_count': self.stargate_count,
            'templar_archives_count': self.templar_archives_count,
            'dark_shrine_count': self.dark_shrine_count,
            'robotics_bay_count': self.robotics_bay_count,
            'fleet_beacon_count': self.fleet_beacon_count,
        }

    def _get_unit_information(self):
        """Retrieve unit-related information."""
        return {
            "probe_count": self.units(UnitTypeId.PROBE).amount,
            'Zealot_count': self.units(UnitTypeId.ZEALOT).amount,
            'stalker_count': self.units(UnitTypeId.STALKER).amount,
            'sentry_count': self.units(UnitTypeId.SENTRY).amount,
            'adept_count': self.units(UnitTypeId.ADEPT).amount,
            'high_templar_count': self.units(UnitTypeId.HIGHTEMPLAR).amount,
            'dark_templar_count': self.units(UnitTypeId.DARKTEMPLAR).amount,
            'immortal_count': self.units(UnitTypeId.IMMORTAL).amount,
            'colossus_count': self.units(UnitTypeId.COLOSSUS).amount,
            'disruptor_count': self.units(UnitTypeId.DISRUPTOR).amount,
            'archon_count': self.units(UnitTypeId.ARCHON).amount,
            'observer_count': self.units(UnitTypeId.OBSERVER).amount,
            'warp_prism_count': self.units(UnitTypeId.WARPPRISM).amount,
            'phoenix_count': self.units(UnitTypeId.PHOENIX).amount,
            'voidray_count': self.units(UnitTypeId.VOIDRAY).amount,
            'Oracle_count': self.units(UnitTypeId.ORACLE).amount,
            'Carrier_count': self.units(UnitTypeId.CARRIER).amount,
            'tempest_count': self.units(UnitTypeId.TEMPEST).amount,
            'mothership_count': self.units(UnitTypeId.MOTHERSHIP).amount,
        }

    def _get_planning_information(self):
        """Retrieve planning-related information."""
        return {
            # Building related
            "planning_structure": {
                'planning_nexus_count': self.already_pending(UnitTypeId.NEXUS),
                'planning_pylon_count': self.already_pending(UnitTypeId.PYLON),
                'planning_gas_buildings_count': self.already_pending(UnitTypeId.ASSIMILATOR),
                'planning_gateway_count': self.already_pending(UnitTypeId.GATEWAY),
                'planning_forge_count': self.already_pending(UnitTypeId.FORGE),
                'planning_photon_cannon_count': self.already_pending(UnitTypeId.PHOTONCANNON),
                'planning_shield_battery_count': self.already_pending(UnitTypeId.SHIELDBATTERY),
                'planning_warp_gate_count': self.already_pending(UnitTypeId.WARPGATE),
                'planning_cybernetics_core_count': self.already_pending(UnitTypeId.CYBERNETICSCORE),
                'planning_twilight_council_count': self.already_pending(UnitTypeId.TWILIGHTCOUNCIL),
                'planning_robotics_facility_count': self.already_pending(UnitTypeId.ROBOTICSFACILITY),
                'planning_stargate_count': self.already_pending(UnitTypeId.STARGATE),
                'planning_templar_archives_count': self.already_pending(UnitTypeId.TEMPLARARCHIVE),
                'planning_dark_shrine_count': self.already_pending(UnitTypeId.DARKSHRINE),
                'planning_robotics_bay_count': self.already_pending(UnitTypeId.ROBOTICSBAY),
                'planning_fleet_beacon_count': self.already_pending(UnitTypeId.FLEETBEACON),
            },
            # Unit related
            "planning_unit": {
                'planning_probe_count': self.already_pending(UnitTypeId.PROBE),
                'planning_Zealot_count': self.already_pending(UnitTypeId.ZEALOT),
                'planning_stalker_count': self.already_pending(UnitTypeId.STALKER),
                'planning_sentry_count': self.already_pending(UnitTypeId.SENTRY),
                'planning_adept_count': self.already_pending(UnitTypeId.ADEPT),
                'planning_high_templar_count': self.already_pending(UnitTypeId.HIGHTEMPLAR),
                'planning_dark_templar_count': self.already_pending(UnitTypeId.DARKTEMPLAR),
                'planning_immortal_count': self.already_pending(UnitTypeId.IMMORTAL),
                'planning_colossus_count': self.already_pending(UnitTypeId.COLOSSUS),
                'planning_disruptor_count': self.already_pending(UnitTypeId.DISRUPTOR),
                'planning_archon_count': self.already_pending(UnitTypeId.ARCHON),
                'planning_observer_count': self.already_pending(UnitTypeId.OBSERVER),
                'planning_warp_prism_count': self.already_pending(UnitTypeId.WARPPRISM),
                'planning_phoenix_count': self.already_pending(UnitTypeId.PHOENIX),
                'planning_voidray_count': self.already_pending(UnitTypeId.VOIDRAY),
                'planning_Oracle_count': self.already_pending(UnitTypeId.ORACLE),
                'planning_Carrier_count': self.already_pending(UnitTypeId.CARRIER),
                'planning_tempest_count': self.already_pending(UnitTypeId.TEMPEST),
                'planning_mothership_count': self.already_pending(UnitTypeId.MOTHERSHIP),
            }
        }

    def _get_research_information(self):
        return {
            # warpgate
            "cybernetics_core": {
                'warpgate_research_status': self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH),

                # protoss air weapons

                'protoss_air_armor_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL1),
                'protoss_air_armor_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL2),
                'protoss_air_armor_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL3),

                # protoss air weapons
                "protoss_air_weapon_level_1_research_status": self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL1),
                "protoss_air_weapon_level_2_research_status": self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL2),
                "protoss_air_weapon_level_3_research_status": self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL3),
            },

            "forge": {
                # protoss ground armor

                'protoss_ground_armor_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL1),
                'protoss_ground_armor_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL2),
                'protoss_ground_armor_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL3),

                # protoss ground weapons

                'protoss_ground_weapon_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1),
                'protoss_ground_weapon_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2),
                'protoss_ground_weapon_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3),

                # protoss_shield

                'protoss_shield_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSSHIELDSLEVEL1),
                'protoss_shield_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSSHIELDSLEVEL2),
                'protoss_shield_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSSHIELDSLEVEL3),
            },

            # twilight council upgrades
            "twilight_council": {
                'blink_research_status': self.already_pending_upgrade(UpgradeId.BLINKTECH),
                'charge_research_status': self.already_pending_upgrade(UpgradeId.CHARGE),
                'resonating_glaives_research_status': self.already_pending_upgrade(UpgradeId.ADEPTPIERCINGATTACK),
                # adept attack speed
            },

            # robotics bay upgrades
            "robotics_bay": {
                'extended_thermal_lance_research_status': self.already_pending_upgrade(UpgradeId.EXTENDEDTHERMALLANCE),
                "GRAVITICDRIVE_research_status": self.already_pending_upgrade(UpgradeId.GRAVITICDRIVE),
                "OBSERVERGRAVITICBOOSTER_research_status": self.already_pending_upgrade(
                    UpgradeId.OBSERVERGRAVITICBOOSTER),
            },
            # fleet beacon upgrades
            "fleet_beacon": {
                "PHOENIXRANGEUPGRADE_research_status": self.already_pending_upgrade(UpgradeId.PHOENIXRANGEUPGRADE),
                # "TEMPESTGROUNDATTACKUPGRADE_research_status": self.already_pending_upgrade(
                #     UpgradeId.TEMPESTGROUNDATTACKUPGRADE),
                "VOIDRAYSPEEDUPGRADE_research_status": self.already_pending_upgrade(UpgradeId.VOIDRAYSPEEDUPGRADE),
            },
            # templar archives upgrades
            "templar_archives": {
                "PSISTORMTECH_research_status": self.already_pending_upgrade(UpgradeId.PSISTORMTECH),
            },
            # dark shrine upgrades
            "dark_shrine": {
                "SHADOWSTRIKE_research_status": self.already_pending_upgrade(UpgradeId.DARKTEMPLARBLINKUPGRADE), },
        }

    def _update_enemy_information(self, information):
        self.enemy_unit_type_amount = self.get_enemy_unity()
        self.enemy_structure_type_amount = self.get_enemy_structure()
        """
        Update the given information dictionary with enemy data.

        Args:
        - information (dict): Original information dictionary.

        Returns:
        - dict: Updated information dictionary.
        """
        # Initialize enemy information part if it doesn't exist.
        if 'enemy' not in information:
            information['enemy'] = {
                'unit': {},
                'structure': {}
            }

        # Try to add enemy unit types and amounts if they exist.
        try:
            if hasattr(self, 'enemy_unit_type_amount') and self.enemy_unit_type_amount:
                information['enemy']['unit'].update(self.enemy_unit_type_amount)
        except Exception as e:
            print(f"Error updating enemy unit information: {e}")

        # Try to add enemy structure types and amounts if they exist.
        try:
            if hasattr(self, 'enemy_structure_type_amount') and self.enemy_structure_type_amount:
                information['enemy']['structure'].update(self.enemy_structure_type_amount)
        except Exception as e:
            print(f"Error updating enemy structure information: {e}")

        return information

    async def defend(self):
        print("Defend:", self.rally_defend)
        if self.structures(UnitTypeId.NEXUS).exists and self.supply_army >= 2:
            for nexus in self.townhalls:
                if self.enemy_units.amount >= 2 and self.enemy_units.closest_distance_to(nexus) < 30:
                    self.rally_defend = True
                    for unit in self.units.of_type(
                            {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                             UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                             UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                             UnitTypeId.CARRIER,
                             UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                             UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                             UnitTypeId.CHANGELINGZEALOT}):
                        closed_enemy = self.enemy_units.sorted(lambda x: x.distance_to(unit))
                        unit.attack(closed_enemy[0])
                else:
                    self.rally_defend = False

            if self.rally_defend == True:
                map_center = self.game_info.map_center
                rally_point = self.townhalls.random.position.towards(map_center, distance=5)
                for unit in self.units.of_type(
                        {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                         UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                         UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                         UnitTypeId.CARRIER,
                         UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                         UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                         UnitTypeId.CHANGELINGZEALOT}):
                    if unit.distance_to(self.start_location) > 100 and unit not in self.unit_tags_received_action:
                        unit.move(rally_point)

    def record_failure(self, action, reason):
        self.temp_failure_list.append(f'Action failed: {self.action_dict[action]}, Reason: {reason}')

    async def handle_scouting(self, unit_type, action_id):
        """
        Use the specified unit for scouting.

        :param unit_type: The type of unit used for scouting.
        :param action_id: Action ID, used to record the reason for failure.
        """
        # Check if the specified interval has passed since the last scouting
        if self.iteration - self.last_scouting_iteration < SCOUTING_INTERVAL:
            return

        # If no scouting units are available, record the failure and return
        if not self.units(unit_type).exists:
            return self.record_failure(action_id, f'No {unit_type} available for scouting')

        # Prefer idle scouting units, if none, randomly select one
        if self.units(unit_type).idle.exists:
            scout_unit = random.choice(self.units(unit_type).idle)
        else:
            scout_unit = random.choice(self.units(unit_type))

        # Locate the scouting position, here simply choose the enemy's starting position
        target_location = self.enemy_start_locations[0]

        # Issue the scouting command
        scout_unit.attack(target_location)

        # Update the iteration count of the last scouting
        self.last_scouting_iteration = self.iteration

        print(f'{unit_type} scouting towards {target_location}')

    async def produce_bg_unit(self, action_id: int, unit_type: UnitTypeId, required_buildings: List[UnitTypeId] = None):
        print(f'action={action_id}')

        # Check the research status of Warp Gate
        warp_gate_research_status = self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH)

        if warp_gate_research_status == 1:
            # Check if there are pylons available
            if not self.structures(UnitTypeId.PYLON).ready.exists:
                return self.record_failure(action_id, 'No pylons available')

            # Get the pylon closest to the enemy's starting position as a reference point
            proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0]).position

            # Attempt to warp units from the Warp Gate
            await self.warp_unit(action_id, unit_type, proxy, required_buildings=required_buildings)
            print(f'Warping {unit_type.name}')
        else:
            # Attempt to produce units from the Gateway
            success = await self.train_from_gateway(action_id, unit_type, required_buildings)
            if success:
                print(f'Training {unit_type.name}')

    async def research_air_upgrade(self, action_id, upgrade_id, ability_id, description):
        print(f'action={action_id}')

        # Precondition check
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'Cybernetics Core does not exist')

        # Get a random instance of CYBERNETICSCORE
        by = self.structures(UnitTypeId.CYBERNETICSCORE).random
        abilities = await self.get_available_abilities(by)

        # Check if the specified upgrade can be researched
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.CYBERNETICSCORE).idle.exists and ability_id in abilities:
                by = self.structures(UnitTypeId.CYBERNETICSCORE).idle.random
                by.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Cybernetics Core is not idle')

    async def build_pylon_time_period(self, action_id, supply_threshold, pending_threshold, location_multiplier):
        if self.supply_left <= supply_threshold and self.already_pending(
                UnitTypeId.PYLON) <= pending_threshold and not self.supply_cap == 200:
            if self.can_afford(UnitTypeId.PYLON):
                base = self.townhalls.random
                place_position = base.position + Point2((0, self.Location * location_multiplier))
                await self.build(UnitTypeId.PYLON, near=place_position, placement_step=2)
                print('Building Pylon')
            else:
                return self.record_failure(action_id, 'Cannot afford Pylon')
        return self.record_failure(action_id, 'Not enough supply or too many pending Pylons')

    async def research_upgrade(self, action_id, upgrade_id, ability_id, description):
        print(f'action={action_id}')

        # Precondition check
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'Cybernetics Core does not exist')

        # Get a random instance of CYBERNETICSCORE
        by = self.structures(UnitTypeId.CYBERNETICSCORE).random
        abilities = await self.get_available_abilities(by)

        # Check if the specified upgrade can be researched
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and by.is_idle and ability_id in abilities:
                by.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Cybernetics Core is not idle')

    async def research_twilight_upgrade(self, action_id, upgrade_id, ability_id, description):
        print(f'action={action_id}')

        # Precondition check
        if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            return self.record_failure(action_id, 'Twilight Council does not exist')

        # Get a random instance of TWILIGHTCOUNCIL
        vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
        abilities = await self.get_available_abilities(vc)

        # Check if the specified upgrade can be researched
        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.TWILIGHTCOUNCIL).idle.exists and ability_id in abilities:
                vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).idle.random
                vc.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Twilight Council is not idle')

    async def research_fleetbeacon_upgrade(self, action_id, upgrade_id, ability_id, description):
        print(f'action={action_id}')

        # Precondition check
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'Probe does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.STARGATE).exists:
            return self.record_failure(action_id, 'Stargate does not exist')

        if not self.structures(UnitTypeId.FLEETBEACON).exists:
            return self.record_failure(action_id, 'FleetBeacon does not exist')

        # Get a random instance of FLEETBEACON
        vf = self.structures(UnitTypeId.FLEETBEACON).random
        abilities = await self.get_available_abilities(vf)

        # Check if the specified upgrade can be researched
        if not self.structures(UnitTypeId.FLEETBEACON).ready:
            return self.record_failure(action_id, 'FleetBeacon is not ready')

        if self.already_pending(upgrade_id) != 0:
            return self.record_failure(action_id, f'Upgrade {description} is already pending')

        if not self.can_afford(upgrade_id):
            return self.record_failure(action_id, f'Cannot afford {description}')

        if not self.structures(UnitTypeId.FLEETBEACON).idle.exists:
            return self.record_failure(action_id, 'FleetBeacon is not idle')

        if ability_id not in abilities:
            return self.record_failure(action_id, f'Ability {description} is not available')
        vf = self.structures(UnitTypeId.FLEETBEACON).idle.random
        vf.research(upgrade_id)
        print(description)

    async def research_forge_upgrade(self, action_id, upgrade_id, ability_id, description):
        print(f'action={action_id}')

        # Precondition check
        if not (self.structures(UnitTypeId.PYLON).exists
                and self.units(UnitTypeId.PROBE).exists
                and self.structures(UnitTypeId.NEXUS).exists
                and self.structures(UnitTypeId.FORGE).exists):
            return self.record_failure(action_id, 'Required structures or units do not exist')

        # Get a random instance of FORGE
        bf = self.structures(UnitTypeId.FORGE).random
        abilities = await self.get_available_abilities(bf)

        # Check if the specified upgrade can be researched
        if self.structures(UnitTypeId.FORGE).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.FORGE).idle.exists and ability_id in abilities:
                bf = self.structures(UnitTypeId.FORGE).idle.random
                bf.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Forge is not idle')

    async def research_roboticsbay_upgrade(self, action_id, upgrade_id, ability_id, description):
        print(f'action={action_id}')

        # Precondition check
        if not (self.structures(UnitTypeId.PYLON).exists
                and self.units(UnitTypeId.PROBE).exists
                and self.structures(UnitTypeId.NEXUS).exists
                and self.structures(UnitTypeId.ROBOTICSFACILITY).exists
                and self.structures(UnitTypeId.ROBOTICSBAY).exists):
            return self.record_failure(action_id, 'Required structures or units do not exist')

        # Get a random instance of ROBOTICSBAY
        vb = self.structures(UnitTypeId.ROBOTICSBAY).random
        abilities = await self.get_available_abilities(vb)

        # Check if the specified upgrade can be researched
        if self.structures(UnitTypeId.ROBOTICSBAY).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.ROBOTICSBAY).idle.exists and ability_id in abilities:
                vb = self.structures(UnitTypeId.ROBOTICSBAY).idle.random
                vb.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or RoboticsBay is not idle')

    async def apply_chronoboost(self, action_id, target_type, supply_requirement, description):
        print(f'action={action_id}')

        # Precondition check
        if not self.structures(target_type).exists:
            return self.record_failure(action_id, f'{target_type} does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        target = self.structures(target_type).random

        # Check if the target building is idle or already chronoboosted
        if target.is_idle or target.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
            return self.record_failure(action_id, f'{description} is idle or already chronoboosted')

        # Check the supply situation
        if self.supply_left < supply_requirement:
            return self.record_failure(action_id, f'Not enough supply for {description}')

        # Attempt to apply chronoboost
        nexuses = self.structures(UnitTypeId.NEXUS)
        abilities = await self.get_available_abilities(nexuses)
        for loop_nexus, abilities_nexus in zip(nexuses, abilities):
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target)
                print(f'Chronoboost applied to {description}')

        return self.record_failure(action_id, f'No available Nexus to chronoboost {description}')

    async def warp_unit(self, action_id: int, unit_type: UnitTypeId, initial_reference_point: Point2,
                        required_buildings: List[UnitTypeId] = None,
                        max_attempts=5) -> None:
        """
        Attempt to warp the specified unit near the reference point.
        :param action_id: ID of the action, used to record failure reasons.
        :param unit_type: The type of unit to warp.
        :param initial_reference_point: The reference point, usually the closest pylon to the enemy's starting location.
        :param max_attempts: Maximum number of attempts to warp.
        :param required_buildings: List of required tech buildings.
        """

        # If additional tech buildings are required, check them
        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        # Check if the unit type is allowed to warp
        if unit_type not in {UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.DARKTEMPLAR}:
            return self.record_failure(action_id, f"Unit type {unit_type.name} not warpable")

        # Check if there is a Warp Gate available
        if not self.structures(UnitTypeId.WARPGATE).exists:
            return self.record_failure(action_id, 'No Warp Gate available')

        # Select an idle Warp Gate
        idle_warpgates = self.structures(UnitTypeId.WARPGATE).ready.idle
        if not idle_warpgates:
            return self.record_failure(action_id, 'No idle Warp Gate available')
        warpgate = idle_warpgates.first
        warp_ability = getattr(AbilityId, f"WARPGATETRAIN_{unit_type.name}")

        # Get the available abilities of the Warp Gate
        abilities = await self.get_available_abilities(warpgate)

        # Check if the Warp Gate can warp the specified unit
        if warp_ability not in abilities:
            return self.record_failure(action_id, f'Cannot train {unit_type.name} from Warp Gate')
        elif not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')
        elif self.supply_left < self.calculate_supply_cost(unit_type):
            return self.record_failure(action_id, 'Not enough supply left')

        # Find the closest pylon with energy
        pylons = [p for p in self.structures(UnitTypeId.PYLON) if p.is_ready]
        pylons.sort(key=lambda p: p.distance_to(initial_reference_point))
        if not pylons:
            return self.record_failure(action_id, 'No powered pylons available')
        pylon = pylons[0]

        # Attempt to find a suitable position near the pylon to warp in
        for distance in range(1, 6):
            pos = pylon.position.random_on_distance(distance)
            placement = await self.find_placement(warp_ability, pos, placement_step=1)
            if placement and placement.distance_to(pylon.position) <= 5:
                warpgate.warp_in(unit_type, placement)
                print(f'Warped in {unit_type.name}')

        # If all attempts fail, record the failure reason
        return self.record_failure(action_id, f"Couldn't find a placement near pylon for {unit_type.name}")

    async def train_from_gateway(self, action_id: int, unit_type: UnitTypeId,
                                 required_buildings: List[UnitTypeId] = None):
        """
        Train the specified unit from the Gateway.

        :param action_id: ID of the action, used to record failure reasons.
        :param unit_type: The type of unit to train.
        :param required_buildings: List of additional buildings required to train this unit.
        :return: True if training was successful, otherwise False.
        """
        print(f'action={action_id}')

        # Check if there is a Gateway available
        if not self.structures(UnitTypeId.GATEWAY).exists:
            return self.record_failure(action_id, 'No Gateway available')

        # Use calculate_supply_cost function to get the supply required
        required_supply = self.calculate_supply_cost(unit_type)
        # Check if there is enough supply space
        if self.supply_left < required_supply:
            return self.record_failure(action_id, 'Not enough supply left')

        # If additional tech buildings are required, check them
        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        # Select an idle Gateway for unit training
        idle_gates = [gate for gate in self.structures(UnitTypeId.GATEWAY) if gate.is_idle]
        if not idle_gates:
            # If there are no idle Gateways, select one that is close to completing the task
            gates_sorted_by_build_progress = sorted(self.structures(UnitTypeId.GATEWAY), key=lambda g: g.build_progress,
                                                    reverse=True)
            gate = gates_sorted_by_build_progress[0]
        else:
            gate = idle_gates[0]

        # Check if there is enough resource
        if not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')

        # Train the unit
        gate.train(unit_type)
        print(f'Training {unit_type.name}')
        return True

    async def train_from_robotics(self, action_id: int, unit_type: UnitTypeId,
                                  required_buildings: List[UnitTypeId] = None,
                                  unit_limit: int = None) -> None:
        """
        Train the specified unit from the Robotics Facility, but only one unit at a time.
        """

        # If additional tech buildings are required, check them
        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        # Check if the Robotics Facility exists and is idle
        robotics_facilities = self.structures(UnitTypeId.ROBOTICSFACILITY)
        if robotics_facilities.empty:
            return self.record_failure(action_id, 'No Robotics Facility available')

        # Check if there are units being produced
        if robotics_facilities.filter(lambda facility: facility.is_idle).empty:
            # If all Robotics Facilities are busy, no new production task is assigned
            print('Robotics Facility is busy')
            return  # No failure recorded, as this is normal production state

        # If a unit limit is specified, check if it has been reached
        if unit_limit is not None and (self.units(unit_type).amount + self.already_pending(unit_type)) >= unit_limit:
            return self.record_failure(action_id, f'{unit_type.name} limit reached')

        # Check resource and supply availability
        if self.supply_left < self.calculate_supply_cost(unit_type):
            return self.record_failure(action_id, 'Not enough supply left')
        if not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')

        # Find an idle Robotics Facility and start training
        idle_facility = robotics_facilities.idle.first
        if idle_facility:
            idle_facility.train(unit_type)
            print(f'Training {unit_type.name}')
            return  # Production task has started, so normal return

        # In some extreme cases, if no idle facility is found, record failure
        return self.record_failure(action_id, 'No idle Robotics Facility available')

    async def train_from_stargate(self, action_id: int, unit_type: UnitTypeId,
                                  required_buildings: List[UnitTypeId] = None,
                                  unit_limit: int = None) -> None:
        """
        Train the specified unit from the Stargate, but only one unit at a time.
        """

        # If additional tech buildings are required, check them
        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        # Check if the Stargate exists and is idle
        stargates = self.structures(UnitTypeId.STARGATE)
        if stargates.empty:
            return self.record_failure(action_id, 'No Stargate available')

        # Check if there are units being produced
        if stargates.filter(lambda facility: facility.is_idle).empty:
            # If all Stargates are busy, no new production task is assigned
            print('Stargate is busy')
            return  # No failure recorded, as this is normal production state

        # If a unit limit is specified, check if it has been reached
        if unit_limit is not None and (self.units(unit_type).amount + self.already_pending(unit_type)) >= unit_limit:
            return self.record_failure(action_id, f'{unit_type.name} limit reached')

        # Check resource and supply availability
        if self.supply_left < self.calculate_supply_cost(unit_type):
            return self.record_failure(action_id, 'Not enough supply left')
        if not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')

        # Find an idle Stargate and start training
        idle_stargate = stargates.idle.first
        if idle_stargate:
            idle_stargate.train(unit_type)
            print(f'Training {unit_type.name}')
            return  # Production task has started, so normal return

        # In some extreme cases, if no idle facility is found, record failure
        return self.record_failure(action_id, 'No idle Stargate available')

    async def handle_action_attack(self, action_id, iteration):
        print(f'action={action_id}')
        await self.attack()

    async def handle_action_0(self):
        action_id = 0
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')

        for nexus in self.townhalls:
            if self.workers.amount + self.already_pending(UnitTypeId.PROBE) > 75:
                return self.record_failure(action_id, 'Too many probes (more than 75)')
            if self.supply_left <= 0:
                return self.record_failure(action_id, 'No supply left')
            if not self.can_afford(UnitTypeId.PROBE):
                return self.record_failure(action_id, 'Cannot afford Probe')
            if nexus.is_idle:
                nexus.train(UnitTypeId.PROBE)
                print('Training Probe')
                return

        return self.record_failure(action_id, 'All Nexus are busy')

    async def handle_action_1(self):
        action_id = 1
        print(f'action={action_id}')

        success = await self.produce_bg_unit(action_id, UnitTypeId.ZEALOT)
        if success:
            print('Training Zealot')

    async def handle_action_2(self):
        action_id = 2
        print(f'action={action_id}')

        # List the buildings required for training Adept
        required_buildings = [UnitTypeId.CYBERNETICSCORE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.ADEPT, required_buildings)
        if success:
            print('Training Adept')

    async def handle_action_3(self):
        action_id = 3
        print(f'action={action_id}')

        # List the buildings required for training Stalker
        required_buildings = [UnitTypeId.CYBERNETICSCORE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.STALKER, required_buildings)
        if success:
            print('Training Stalker')

    async def handle_action_4(self):
        action_id = 4
        print(f'action={action_id}')

        # List the buildings required for training Sentry
        required_buildings = [UnitTypeId.CYBERNETICSCORE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.SENTRY, required_buildings)
        if success:
            print('Training Sentry')

    async def handle_action_5(self):
        action_id = 5
        print(f'action={action_id}')

        # List the buildings required for training High Templar
        required_buildings = [UnitTypeId.TEMPLARARCHIVE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.HIGHTEMPLAR, required_buildings)
        if success:
            print('Training High Templar')

    async def handle_action_6(self):
        action_id = 6
        print(f'action={action_id}')

        # List the buildings required for training Dark Templar
        required_buildings = [UnitTypeId.DARKSHRINE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.DARKTEMPLAR, required_buildings)
        if success:
            print('Training Dark Templar')

    async def handle_action_7(self):
        action_id = 7
        print(f'action={action_id}')
        await self.train_from_stargate(action_id, UnitTypeId.VOIDRAY)

    async def handle_action_8(self):
        action_id = 8
        print(f'action={action_id}')
        await self.train_from_stargate(action_id, UnitTypeId.CARRIER)

    async def handle_action_9(self):
        action_id = 9
        print(f'action={action_id}')
        await self.train_from_stargate(action_id, UnitTypeId.TEMPEST)

    async def handle_action_10(self):
        action_id = 10
        print(f'action={action_id}')
        await self.train_from_stargate(action_id, UnitTypeId.ORACLE, unit_limit=1)

    async def handle_action_11(self):
        action_id = 11
        print(f'action={action_id}')
        await self.train_from_stargate(action_id, UnitTypeId.PHOENIX, unit_limit=4)

    async def handle_action_12(self):
        action_id = 12
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')

        if not self.structures(UnitTypeId.STARGATE).exists:
            return self.record_failure(action_id, 'No Stargate available')

        if not self.structures(UnitTypeId.FLEETBEACON).exists:
            return self.record_failure(action_id, 'Fleet Beacon not available')

        if self.supply_left < 10:
            return self.record_failure(action_id, 'Not enough supply left')

        nexuses = self.structures(UnitTypeId.NEXUS)
        abilities = await self.get_available_abilities(nexuses)

        if AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP not in abilities:
            return self.record_failure(action_id, 'Cannot train Mothership')

        if not self.can_afford(UnitTypeId.MOTHERSHIP):
            return self.record_failure(action_id, 'Cannot afford Mothership')

        for base in self.townhalls:
            if base.is_idle:
                base.train(UnitTypeId.MOTHERSHIP)
                reward = 0.010
                print('Training Mothership')
                return reward

        return self.record_failure(action_id, 'No idle Nexus available')

    async def handle_action_13(self):
        action_id = 13
        print(f'action={action_id}')
        await self.train_from_robotics(action_id, UnitTypeId.OBSERVER, unit_limit=2)
        print('Training Observer')
        return

    async def handle_action_14(self):
        action_id = 14
        print(f'action={action_id}')

        await self.train_from_robotics(action_id, UnitTypeId.IMMORTAL)
        print('Training Immortal')
        return

    async def handle_action_15(self):
        action_id = 15
        print(f'action={action_id}')
        await self.train_from_robotics(action_id, UnitTypeId.WARPPRISM, unit_limit=1)
        print('Training Warp Prism')
        return

    async def handle_action_16(self):
        action_id = 16
        print(f'action={action_id}')
        required_buildings = [UnitTypeId.ROBOTICSBAY]
        await self.train_from_robotics(action_id, UnitTypeId.COLOSSUS, required_buildings=required_buildings)
        print('Training Colossus')
        return

    async def handle_action_17(self):
        action_id = 17
        print(f'action={action_id}')
        required_buildings = [UnitTypeId.DISRUPTOR]
        await self.train_from_robotics(action_id, UnitTypeId.DISRUPTOR, required_buildings=required_buildings,
                                       unit_limit=1)
        print('Training Disruptor')
        return

    async def handle_action_18(self):
        action_id = 18
        print(f'action={action_id}')

        if self.units(UnitTypeId.HIGHTEMPLAR).exists:
            hts = self.units(UnitTypeId.HIGHTEMPLAR)
            for ht in hts:
                ht(AbilityId.MORPH_ARCHON)
            return 0.005

        elif self.units(UnitTypeId.DARKTEMPLAR).exists:
            dts = self.units(UnitTypeId.DARKTEMPLAR)
            for dt in dts:
                dt(AbilityId.MORPH_ARCHON)
            print('Synthesizing Archon')
            return 0.005

        return self.record_failure(action_id, 'No High Templar or Dark Templar available for Archon morph')

    async def handle_action_19(self):
        action_id = 19
        print(f'action={action_id}')

        # If there is no Nexus or Probe, do not proceed
        if not (self.structures(UnitTypeId.NEXUS).exists and self.units(UnitTypeId.PROBE).exists):
            return self.record_failure(action_id, 'No Nexus or Probe available')

        # Set supply threshold and pending Pylon threshold based on current time
        supply_left_threshold, pending_pylon_threshold = 3, 2
        current_time = self.time_formatted

        if '06:00' <= current_time <= '07:00':
            supply_left_threshold, pending_pylon_threshold = 5, 4
        elif '06:00' <= current_time <= '08:00':
            supply_left_threshold, pending_pylon_threshold = 5, 3
        elif '08:00' <= current_time <= '10:00':
            supply_left_threshold, pending_pylon_threshold = 7, 4
        elif '10:00' <= current_time:
            supply_left_threshold, pending_pylon_threshold = 7, 4

        # If the conditions are met to build a Pylon, execute the construction
        if (
                self.supply_left <= supply_left_threshold and
                self.already_pending(UnitTypeId.PYLON) <= pending_pylon_threshold and
                not self.supply_cap == 200  # Ensure supply cap is not 200
        ):
            if not self.can_afford(UnitTypeId.PYLON):
                return self.record_failure(action_id, 'Cannot afford Pylon')

            # Find the best base and position to build a Pylon
            best_base = self.find_best_base_for_pylon()
            best_position = self.find_optimal_pylon_position_for_base(best_base)

            # Attempt to build a Pylon
            construction_result = await self.build(UnitTypeId.PYLON, near=best_position, placement_step=2)
            if not construction_result:  # If construction fails
                return self.record_failure(action_id, 'Could not build Pylon for unknown reasons')

        else:
            # If the conditions are not met to build a Pylon
            return self.record_failure(action_id, 'Not the right conditions to build a Pylon')

    async def handle_action_20(self):
        action_id = 20
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.ASSIMILATOR):
            return self.record_failure(action_id, 'Cannot afford Assimilator')
        if not self.vespene_geyser.exists:
            return self.record_failure(action_id, 'No Vespene Geyser available')
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')

        # Get the number of gas geysers being built
        building_assimilators_num = self.already_pending(UnitTypeId.ASSIMILATOR)

        # If the number of gas geysers being built is less than 2, we can try to build a new one
        if building_assimilators_num < 2:
            for nexus in self.townhalls:
                for vespene in self.vespene_geyser.closer_than(10, nexus):
                    # Check if there are workers already moving to build the gas geyser
                    moving_workers = [worker for worker in self.workers.closer_than(5, vespene) if worker.is_moving]
                    if moving_workers:
                        continue  # If there are workers already moving to build, we skip this gas geyser

                    # If there are no other Assimilators nearby, try to build a new one on this gas geyser
                    if self.can_afford(UnitTypeId.ASSIMILATOR) and not self.structures(
                            UnitTypeId.ASSIMILATOR).closer_than(2, vespene):
                        await self.build(UnitTypeId.ASSIMILATOR, vespene)
                        print('Building gas geyser')
                        return  # Exit the function immediately after starting construction, ensuring only one Assimilator is built at a time
        else:
            print('Already building 2 Assimilators, waiting...')  # If there are already 2 being built, wait

    async def handle_action_21(self):
        action_id = 21
        print(f'action={action_id}')

        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.NEXUS):
            return self.record_failure(action_id, 'Cannot afford Nexus')
        if self.can_afford(UnitTypeId.NEXUS) and self.already_pending(UnitTypeId.NEXUS) <= 1 and self.structures(
                UnitTypeId.NEXUS).amount <= 8:
            await self.expand_now()
            print('Expanding base')

    async def handle_action_22(self):
        action_id = 22
        print(f'action={action_id}')
        building_limits = {
            '00:00-03:00': 1,  # Only 1 Robotics Facility can be built from 0:00 to 5 minutes
            '03:00-05:00': 3,  # Only 2 Robotics Facilities can be built from 5 minutes to 10 minutes
            '05:00-07:00': 6,  # Only 3 Robotics Facilities can be built from 10 minutes to 15 minutes
            '07:00-10:00': 8,  # Only 4 Robotics Facilities can be built from 15 minutes to 20 minutes
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.GATEWAY):
            return self.record_failure(action_id, 'Cannot afford Gateway')
        await self.handle_action_build_building(UnitTypeId.GATEWAY, building_limits=building_limits)
        print('Building BG')

    async def handle_action_23(self):
        action_id = 23
        print(f'action={action_id}')
        building_limits = {
            '00:00-05:00': 1,
            '05:00-20:00': 2,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.GATEWAY).exists:
            return self.record_failure(action_id, 'No Gateway available')
        if not self.can_afford(UnitTypeId.CYBERNETICSCORE):
            return self.record_failure(action_id, 'Cannot afford Cybernetics Core')
        await self.handle_action_build_building(UnitTypeId.CYBERNETICSCORE, building_limits=building_limits)
        print('Building BY')

    async def handle_action_24(self):
        action_id = 24
        print(f'action={action_id}')
        building_limits = {
            '00:00-05:00': 1,
            '05:00-10:00': 2,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.FORGE):
            return self.record_failure(action_id, 'Cannot afford Forge')
        await self.handle_action_build_building(UnitTypeId.FORGE, building_limits=building_limits)
        print('Building BF')

    async def handle_action_25(self):
        action_id = 25
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.can_afford(UnitTypeId.TWILIGHTCOUNCIL):
            return self.record_failure(action_id, 'Cannot afford Twilight Council')
        await self.handle_action_build_building(UnitTypeId.TWILIGHTCOUNCIL)
        print('Building VC')

    async def handle_action_26(self):
        action_id = 26
        print(f'action={action_id}')
        building_limits = {
            '00:00-05:00': 1,  # Only 1 Robotics Facility can be built from 0:00 to 5 minutes
            '05:00-10:00': 2,  # Only 2 Robotics Facilities can be built from 5 minutes to 10 minutes
            '10:00-15:00': 3,  # Only 3 Robotics Facilities can be built from 10 minutes to 15 minutes
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.can_afford(UnitTypeId.ROBOTICSFACILITY):
            return self.record_failure(action_id, 'Cannot afford Robotics Facility')
        await self.handle_action_build_building(UnitTypeId.ROBOTICSFACILITY, building_limits=building_limits)
        print('Building VR')

    async def handle_action_27(self):
        action_id = 27
        print(f'action={action_id}')
        buildind_limits = {
            '00:00-05:00': 1,  # Only 1 Stargate can be built from 0:00 to 5 minutes
            '05:00-10:00': 3,  # Only 2 Stargates can be built from 5 minutes to 10 minutes
            '10:00-15:00': 4,  # Only 3 Stargates can be built from 10 minutes to 15 minutes
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.can_afford(UnitTypeId.STARGATE):
            return self.record_failure(action_id, 'Cannot afford Stargate')
        await self.handle_action_build_building(UnitTypeId.STARGATE, building_limits=buildind_limits)
        print('Building VS')

    async def handle_action_28(self):
        action_id = 28
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            return self.record_failure(action_id, 'No Twilight Council available')
        if not self.can_afford(UnitTypeId.TEMPLARARCHIVE):
            return self.record_failure(action_id, 'Cannot afford Templar Archive')
        await self.handle_action_build_building(UnitTypeId.TEMPLARARCHIVE)
        print('Building VT')

    async def handle_action_29(self):
        action_id = 29
        print(f'action={action_id}')
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            return self.record_failure(action_id, 'No Twilight Council available')
        if not self.can_afford(UnitTypeId.DARKSHRINE):
            return self.record_failure(action_id, 'Cannot afford Dark Shrine')
        await self.handle_action_build_building(UnitTypeId.DARKSHRINE)

        print('Building VD')

    async def handle_action_30(self):
        action_id = 30
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.ROBOTICSFACILITY).exists:
            return self.record_failure(action_id, 'No Robotics Facility available')
        if not self.can_afford(UnitTypeId.ROBOTICSBAY):
            return self.record_failure(action_id, 'Cannot afford Robotics Bay')
        await self.handle_action_build_building(UnitTypeId.ROBOTICSBAY)
        print('Building VB')

    async def handle_action_31(self):
        action_id = 31
        print(f'action={action_id}')
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.STARGATE).exists:
            return self.record_failure(action_id, 'No Stargate available')
        if not self.can_afford(UnitTypeId.FLEETBEACON):
            return self.record_failure(action_id, 'Cannot afford Fleet Beacon')
        await self.handle_action_build_building(UnitTypeId.FLEETBEACON)
        print('Building VF')

    async def handle_action_32(self):
        action_id = 32
        print(f'action={action_id}')
        building_limits = {
            '00:00-05:00': 1,  # Only 1 Photon Cannon can be built from 0:00 to 5 minutes
            '05:00-10:00': 3,  # Only 2 Photon Cannons can be built from 5 minutes to 10 minutes
            '10:00-15:00': 5,  # Only 3 Photon Cannons can be built from 10 minutes to 15 minutes
            '15:00-20:00': 7,  # Only 4 Photon Cannons can be built from 15 minutes to 20 minutes
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.FORGE).exists:
            return self.record_failure(action_id, 'No Forge')
        if self.structures(UnitTypeId.PHOTONCANNON).amount + self.already_pending(UnitTypeId.PHOTONCANNON) > 3:
            return self.record_failure(action_id, 'Photon Cannon limit reached')
        if not self.can_afford(UnitTypeId.PHOTONCANNON):
            return self.record_failure(action_id, 'Cannot afford  Photon Cannon')
        await self.handle_action_build_building(UnitTypeId.PHOTONCANNON, building_limits=building_limits)
        print('Building BC')

    async def handle_action_33(self):
        action_id = 33
        print(f'action={action_id}')
        building_limits = {
            '00:00-05:00': 1,  #
            '05:00-10:00': 3,  #
            '10:00-15:00': 5,  #
            '15:00-20:00': 7,  #
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.FORGE).exists:
            return self.record_failure(action_id, 'No Forge')
        if self.structures(UnitTypeId.SHIELDBATTERY).amount + self.already_pending(UnitTypeId.SHIELDBATTERY) > 3:
            return self.record_failure(action_id, 'Shield Battery limit reached')
        if not self.can_afford(UnitTypeId.SHIELDBATTERY):
            return self.record_failure(action_id, 'Cannot afford Shield Battery')
        await self.handle_action_build_building(UnitTypeId.SHIELDBATTERY, building_limits=building_limits)
        print('Building BB')

    async def handle_action_34(self):
        action_id = 34
        print(f'action={action_id}')

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'Cybernetics Core does not exist')

        by = self.structures(UnitTypeId.CYBERNETICSCORE).random
        abilities = await self.get_available_abilities(by)
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(UpgradeId.WARPGATERESEARCH) == 0:
            if self.can_afford(UpgradeId.WARPGATERESEARCH) and by.is_idle and AbilityId.RESEARCH_WARPGATE in abilities:
                by.research(UpgradeId.WARPGATERESEARCH)
                print('Researching Warp Gate')
                return 0.010

        return self.record_failure(action_id, 'Cannot afford Warp Gate research or Cybernetics Core is not idle')

    async def handle_action_35(self):
        upgrade_id = UpgradeId.PROTOSSAIRWEAPONSLEVEL1
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1
        description = 'Research Air 1 Attack'
        return await self.research_upgrade(35, upgrade_id, ability_id, description)

    async def handle_action_36(self):
        upgrade_id = UpgradeId.PROTOSSAIRWEAPONSLEVEL2
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2
        description = 'Research Air 2 Attack'
        return await self.research_upgrade(36, upgrade_id, ability_id, description)

    async def handle_action_37(self):
        upgrade_id = UpgradeId.PROTOSSAIRWEAPONSLEVEL3
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3
        description = 'Research Air 3 Attack'
        return await self.research_upgrade(37, upgrade_id, ability_id, description)

    async def handle_action_38(self):
        upgrade_id = UpgradeId.PROTOSSAIRARMORSLEVEL1
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1
        description = 'Research Air 1 Defense'
        return await self.research_upgrade(38, upgrade_id, ability_id, description)

    async def handle_action_39(self):
        upgrade_id = UpgradeId.PROTOSSAIRARMORSLEVEL2
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2
        description = 'Research Air 2 Defense'
        return await self.research_upgrade(39, upgrade_id, ability_id, description)

    async def handle_action_40(self):
        upgrade_id = UpgradeId.PROTOSSAIRARMORSLEVEL3
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3
        description = 'Research Air 3 Defense'
        return await self.research_upgrade(40, upgrade_id, ability_id, description)

    async def handle_action_41(self):
        upgrade_id = UpgradeId.ADEPTPIERCINGATTACK
        ability_id = AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES
        description = 'Research Adept Attack Speed'
        return await self.research_twilight_upgrade(41, upgrade_id, ability_id, description)

    async def handle_action_42(self):
        upgrade_id = UpgradeId.BLINKTECH
        ability_id = AbilityId.RESEARCH_BLINK
        description = 'Research Blink'
        return await self.research_twilight_upgrade(42, upgrade_id, ability_id, description)

    async def handle_action_43(self):
        upgrade_id = UpgradeId.CHARGE
        ability_id = AbilityId.RESEARCH_CHARGE
        description = 'Research Charge'
        return await self.research_twilight_upgrade(43, upgrade_id, ability_id, description)

    async def handle_action_44(self):
        return await self.research_forge_upgrade(44, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1, 'Research Ground 1 Attack')

    async def handle_action_45(self):
        return await self.research_forge_upgrade(45, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2, 'Research Ground 2 Attack')

    async def handle_action_46(self):
        return await self.research_forge_upgrade(46, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3, 'Research Ground 3 Attack')

    async def handle_action_47(self):
        return await self.research_forge_upgrade(47, UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1, 'Research Ground 1 Defense')

    async def handle_action_48(self):
        return await self.research_forge_upgrade(48, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2, 'Research Ground 2 Defense')

    async def handle_action_49(self):
        return await self.research_forge_upgrade(49, UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3, 'Research Ground 3 Defense')

    async def handle_action_50(self):
        return await self.research_forge_upgrade(50, UpgradeId.PROTOSSSHIELDSLEVEL1,
                                                 AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1, 'Research 1 Shield')

    async def handle_action_51(self):
        return await self.research_forge_upgrade(51, UpgradeId.PROTOSSSHIELDSLEVEL2,
                                                 AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2, 'Research 2 Shield')

    async def handle_action_52(self):
        return await self.research_forge_upgrade(52, UpgradeId.PROTOSSSHIELDSLEVEL3,
                                                 AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3, 'Research 3 Shield')

    async def handle_action_53(self):
        return await self.research_roboticsbay_upgrade(53, UpgradeId.EXTENDEDTHERMALLANCE,
                                                       AbilityId.RESEARCH_EXTENDEDTHERMALLANCE, 'Research Colossus Range')

    async def handle_action_54(self):
        return await self.research_roboticsbay_upgrade(54, UpgradeId.GRAVITICDRIVE, AbilityId.RESEARCH_GRAVITICDRIVE,
                                                       'Research Prism Speed')

    async def handle_action_55(self):
        return await self.research_roboticsbay_upgrade(55, UpgradeId.OBSERVERGRAVITICBOOSTER,
                                                       AbilityId.RESEARCH_GRAVITICBOOSTER, 'Research OB Speed')

    async def handle_action_56(self):
        action_id = 56
        print(f'action={action_id}')
        # Precondition check
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'Probe does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.TEMPLARARCHIVE).exists:
            return self.record_failure(action_id, 'Templar archive does not exist')

        # Get a random instance of TEMPLARARCHIVE
        vt = self.structures(UnitTypeId.TEMPLARARCHIVE).random
        abilities = await self.get_available_abilities(vt)
        upgrade_id = UpgradeId.PSISTORMTECH
        description = 'psi storm'
        ability_id = AbilityId.RESEARCH_PSISTORM
        # Check if the specified upgrade can be researched
        if not self.structures(UnitTypeId.TEMPLARARCHIVE).ready:
            return self.record_failure(action_id, 'Templar archive is not ready')

        if self.already_pending(upgrade_id) != 0:
            return self.record_failure(action_id, f'Upgrade {description} is already pending')

        if not self.can_afford(upgrade_id):
            return self.record_failure(action_id, f'Cannot afford {description}')

        if not vt.is_idle:
            return self.record_failure(action_id, 'Templar archive is not idle')

        if ability_id not in abilities:
            return self.record_failure(action_id, f'Ability {description} is not available')

        vt.research(upgrade_id)
        print(description)
        return 0.010

    async def handle_action_57(self):
        return await self.research_fleetbeacon_upgrade(57, UpgradeId.VOIDRAYSPEEDUPGRADE,
                                                       AbilityId.FLEETBEACONRESEARCH_RESEARCHVOIDRAYSPEEDUPGRADE,
                                                       'Research Void Ray Speed')

    async def handle_action_58(self):
        return await self.research_fleetbeacon_upgrade(58, UpgradeId.PHOENIXRANGEUPGRADE,
                                                       AbilityId.RESEARCH_PHOENIXANIONPULSECRYSTALS, 'Research Phoenix Range')

    async def handle_action_59(self):
        # return await self.research_fleetbeacon_upgrade(59, UpgradeId.TEMPESTGROUNDATTACKUPGRADE,
        #                                                AbilityId.FLEETBEACONRESEARCH_TEMPESTRESEARCHGROUNDATTACKUPGRADE,
        #                                                'Research Storm on Buildings')
        return

    async def handle_action_60(self):
        return await self.handle_scouting(UnitTypeId.PROBE, 60)

    async def handle_action_61(self):
        return await self.handle_scouting(UnitTypeId.OBSERVER, 61)

    async def handle_action_62(self):
        return await self.handle_scouting(UnitTypeId.ZEALOT, 62)

    async def handle_action_63(self):
        return await self.handle_scouting(UnitTypeId.PHOENIX, 63)

    async def handle_action_64(self):
        return await self.handle_action_attack(64, iteration=self.iteration)

    async def handle_action_65(self):
        action_id = 65
        print(f'action={action_id}')

        # Check if army supply is greater than 0
        if self.supply_army <= 0:
            return self.record_failure(action_id, 'Army supply is 0 or less')

        # Define the set of units to consider for retreating
        retreating_units = {
            UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER,
            UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
            UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
            UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
            UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
            UnitTypeId.CHANGELINGZEALOT
        }

        # Get units to retreat
        units_to_retreat = self.units.of_type(retreating_units)
        if not units_to_retreat.exists:
            return self.record_failure(action_id, 'No units available to retreat')

        try:
            # Choose the closest nexus to move to
            closest_nexus = units_to_retreat.closest_to(self.enemy_start_locations[0])
            for retreat_unit in units_to_retreat:
                retreat_unit.move(closest_nexus)
            print('Retreat initiated')
        except Exception as e:
            return self.record_failure(action_id, f'Retreat failed due to: {e}')

    async def handle_action_66(self):
        print("chronoboost nexus")
        return await self.apply_chronoboost(66, UnitTypeId.NEXUS, 2, 'base')

    async def handle_action_67(self):
        print("chronoboost cybernetics core")
        return await self.apply_chronoboost(67, UnitTypeId.CYBERNETICSCORE, 0, 'cybernetics core (by)')

    async def handle_action_68(self):
        print("chronoboost twilight council")
        return await self.apply_chronoboost(68, UnitTypeId.TWILIGHTCOUNCIL, 4, 'twilight council (vc)')

    async def handle_action_69(self):
        print("chronoboost stargate")
        return await self.apply_chronoboost(69, UnitTypeId.STARGATE, 4, 'stargate (vs)')

    async def handle_action_70(self):
        print("chronoboost forge")
        return await self.apply_chronoboost(70, UnitTypeId.FORGE, 4, 'forge (bf)')

    async def handle_action_71(self):
        print("empty action")
        pass

    async def on_step(self, iteration: int):
        try:
            self.iteration = iteration
            if self.time_formatted == '00:00':
                if self.start_location == Point2((160.5, 46.5)):
                    self.Location = -1  # detect location
                else:
                    self.Location = 1

            # Get information
            information = self.get_information()
            await self.defend()

            # Lock and read action
            with self.lock:
                self.transaction['information'] = information
            print("action", self.transaction['action'])
            while self.transaction['action'] is None:
                time.sleep(0.001)
            action = self.transaction['action']

            # Handle chat commands
            if self.transaction['output_command_flag'] == True:
                command = self.transaction['command']
                if command is None:
                    message = "Welcome to StarCraft II"
                # elif isinstance(command, list):  # Check if command is a list
                #     message = "\n".join(command)  # Join list elements with newline separator
                # else:
                #     message = command
                else:
                    message = self.action_dict[action]
                await self.chat_send(message)

            # Call the corresponding action handling function
            method_name = f'handle_action_{action}'
            method = getattr(self, method_name, None)
            if method and callable(method):
                await method()
            else:
                print(f'Error: Method {method_name} does not exist!')
            if self.time_formatted >= "12:00" and self.supply_used >= 100:
                await self.attack()
            # Perform regular operations
            await self.distribute_workers()

            # Update transaction dictionary
            with self.lock:
                self.transaction['action'] = None
                self.transaction['reward'] = 0  # You may need to calculate the true reward here
                self.transaction['iter'] = iteration
                self.transaction['action_failures'] = copy.deepcopy(self.temp_failure_list)
                self.transaction['action_executed'] = copy.deepcopy(self.action_dict[action])
                # 2024-8-21 Add intermediate information
                self.transaction['process_data'] = self.get_process_data()

                #
                print("====================self.temp_failure_list", self.temp_failure_list)
                print("====================self.transaction['action_failures']", self.transaction['action_failures'])
                print("====================self.transaction['action_executed']", self.transaction['action_executed'])
                self.temp_failure_list.clear()  # Clear temporary list

            self.isReadyForNextStep.set()
        except Exception as e:
            print("===========",e)

    async def attack(self):
        if self.army_supply >= 10:
            attack_units = [UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                            UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                            UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                            UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                            UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                            UnitTypeId.CHANGELINGZEALOT]
            if any(self.units(unit_type).amount > 0 for unit_type in attack_units):
                enemy_start_location_cleared = not self.enemy_units.exists and self.is_visible(
                    self.enemy_start_locations[0])

                if enemy_start_location_cleared:
                    for unit_type in attack_units:
                        await self.assign_units_to_resource_clusters(unit_type)
                else:
                    for unit_type in attack_units:
                        units = self.units(unit_type).idle
                        for unit in units:
                            unit.attack(self.enemy_start_locations[0])

    async def assign_units_to_resource_clusters(self, unit_type):
        units = self.units(unit_type).idle
        resource_clusters = self.expansion_locations_list

        if units.exists and resource_clusters:
            # Assign each unit a random target resource point
            for unit in units:
                target = random.choice(resource_clusters)
                unit.attack(target)

    @staticmethod
    def neighbors4(position, distance=1) -> Set[Point2]:
        """
        Returns the four adjacent positions (top, bottom, left, right) to the given position.

        Args:
        - position (Point2): The reference position.
        - distance (int, default=1): The distance from the reference position.

        Returns:
        - Set[Point2]: A set containing the four adjacent positions.
        """
        p = position
        d = distance
        return {
            Point2((p.x - d, p.y)),  # Left
            Point2((p.x + d, p.y)),  # Right
            Point2((p.x, p.y - d)),  # Bottom
            Point2((p.x, p.y + d))  # Top
        }

    def neighbors8(self, position, distance=1) -> Set[Point2]:
        """
        Returns the eight adjacent positions (top, bottom, left, right, and the four diagonals) to the given position.

        Args:
        - position (Point2): The reference position.
        - distance (int, default=1): The distance from the reference position.

        Returns:
        - Set[Point2]: A set containing the eight adjacent positions.
        """
        p = position
        d = distance
        # Get the four direct neighbors (top, bottom, left, right)
        direct_neighbors = self.neighbors4(position, distance)
        # Get the four diagonal neighbors
        diagonal_neighbors = {
            Point2((p.x - d, p.y - d)),  # Bottom-left
            Point2((p.x - d, p.y + d)),  # Top-left
            Point2((p.x + d, p.y - d)),  # Bottom-right
            Point2((p.x + d, p.y + d))  # Top-right
        }
        return direct_neighbors | diagonal_neighbors






