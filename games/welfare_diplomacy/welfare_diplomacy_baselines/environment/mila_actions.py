# Copyright 2021 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for converting to the action format used by Pacquette et al."""

import sys

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import collections
from typing import List, Set, Union
import immutabledict

from environment import action_list, action_utils, province_order
from environment import observation_utils as utils

_tag_to_area_id = immutabledict.immutabledict(
    province_order.province_name_to_id(province_order.MapMDF.BICOASTAL_MAP)
)
_area_id_to_tag = immutabledict.immutabledict(
    {v: k for k, v in _tag_to_area_id.items()}
)

_fleet_adjacency = immutabledict.immutabledict(
    {a: tuple(b) for a, b in province_order.fleet_adjacency_map().items()}
)


# Some province names are different for MILA action strings:
_DM_TO_MILA_TAG_MAP = immutabledict.immutabledict(dict(ECH="ENG", GOB="BOT", GOL="LYO"))


def mila_area_string(
    unit_type: utils.UnitType, province_tuple: utils.ProvinceWithFlag
) -> str:
    """Gives the string MILA actions use to represent the area.

    If the Unit type is fleet and the province is bicoastal, then the coast flag
    will be used to determine which coast, and the returned string will be e.g.
    STP/NC or STP/SC. If the main area is wanted, use UnitType.ARMY.

    Args:
      unit_type: What type of unit is in the province, used to determine coast
      province_tuple: the province and coast flag describing the area

    Returns:
      The string used in the MILA action format to describe the area.
    """
    province_id = province_tuple[0]
    if unit_type == unit_type.FLEET:
        area_index = utils.area_index_for_fleet(province_tuple)
    else:
        area_index = 0

    area_id = utils.area_from_province_id_and_area_index(province_id, area_index)

    province_tag = _area_id_to_tag[area_id]
    return _DM_TO_MILA_TAG_MAP.get(province_tag, province_tag)


def mila_unit_string(
    unit_type: utils.UnitType, province_tuple: utils.ProvinceWithFlag
) -> str:
    return ["A %s", "F %s"][unit_type.value] % mila_area_string(
        unit_type, province_tuple
    )


def possible_unit_types(province_tuple: utils.ProvinceWithFlag) -> Set[utils.UnitType]:
    """The unit types can occupy this province."""
    if province_tuple[1] > 0:
        # Must be fleet in Bicoastal province
        return {utils.UnitType.FLEET}
    province_type = utils.province_type_from_id(province_tuple[0])
    if province_type == utils.ProvinceType.LAND:
        return {utils.UnitType.ARMY}
    elif province_type == utils.ProvinceType.SEA:
        return {utils.UnitType.FLEET}
    else:
        return {utils.UnitType.ARMY, utils.UnitType.FLEET}


def possible_unit_types_movement(
    start_province_tuple: utils.ProvinceWithFlag,
    dest_province_tuple: utils.ProvinceWithFlag,
) -> Set[utils.UnitType]:
    """Returns what unit types can move from the start to the destination.

    Args:
      start_province_tuple: the province the unit starts in.
      dest_province_tuple: the province the unit moves to.

    Returns:
      Set of unit types that could make this move.
    """
    possible_types = set()
    if utils.UnitType.ARMY in possible_unit_types(
        start_province_tuple
    ) & possible_unit_types(dest_province_tuple):
        possible_types.add(utils.UnitType.ARMY)

    # Check if a fleet can actually make the journey.
    start_area_id = utils.area_from_province_id_and_area_index(
        start_province_tuple[0], utils.area_index_for_fleet(start_province_tuple)
    )
    dest_area_id = utils.area_from_province_id_and_area_index(
        dest_province_tuple[0], utils.area_index_for_fleet(dest_province_tuple)
    )
    if dest_area_id in _fleet_adjacency[start_area_id]:
        possible_types.add(utils.UnitType.FLEET)

    return possible_types


def possible_unit_types_support(
    start_province_tuple: utils.ProvinceWithFlag,
    dest_province_tuple: utils.ProvinceWithFlag,
) -> Set[utils.UnitType]:
    """Returns what unit types can support the destination from the start area.

    Args:
      start_province_tuple: the province the unit starts in.
      dest_province_tuple: the province the support is offered to.

    Returns:
      Set of unit types that could make this support.
    """
    possible_types = set()
    if utils.UnitType.ARMY in possible_unit_types(
        start_province_tuple
    ) & possible_unit_types((dest_province_tuple[0], 0)):
        possible_types.add(utils.UnitType.ARMY)

    # Check if a fleet can actually reach the province (on either coast if
    # bicoastal).
    start_area_id = utils.area_from_province_id_and_area_index(
        start_province_tuple[0], utils.area_index_for_fleet(start_province_tuple)
    )
    for area_index in (
        [1, 2]
        if utils.province_type_from_id(dest_province_tuple[0])
        == utils.ProvinceType.BICOASTAL
        else [0]
    ):
        dest_area_id = utils.area_from_province_id_and_area_index(
            dest_province_tuple[0], area_index
        )
        if dest_area_id in _fleet_adjacency[start_area_id]:
            possible_types.add(utils.UnitType.FLEET)

    return possible_types


def action_to_mila_actions(
    action: Union[action_utils.Action, action_utils.ActionNoIndex],
) -> List[str]:
    """Returns all Mila action strings an action can correspond to.

    Note that MILA's action list did not include all convoy related actions, as
    it omitted those which were either irrelevant, or corresponded to very long
    convoys. We did not apply this filtering, and if such an action is provided,
    then the function, actions will be returned in the same format as MILA
    actions, but that do not appear in the action_list.MILA_ACTIONS_LIST

    Args:
      action: The action to write down

    Returns:
      Action in the MILA notation.
    """
    order, p1, p2, p3 = action_utils.action_breakdown(action)
    # order: an integer between 0 and 13
    # p1, p2, p3: tuples of (province_id, coast indicator bit)

    mila_action_strings = set()

    # DeepMind action space does not specify the coast of the unit acting or being
    # supported, unless the action is build. In order to correctly construct all
    # possible actions, we will add the maybe missing area_flag=1 versions of
    # actions when the province is bicoastal.
    # Because the bicoastal provinces are far apart, only one of p1, p2 or p3 can
    # be a bicoastal province, so we only ever need to change a single province
    # flag.
    possible_orders_incl_flags = [(order, p1, p2, p3)]

    # If not build:
    if order not in {action_utils.BUILD_ARMY, action_utils.BUILD_FLEET}:
        # If unit acting is in bicostal province, add order with other flag? Is 0 by default
        if utils.province_type_from_id(p1[0]) == utils.ProvinceType.BICOASTAL:
            possible_orders_incl_flags.append((order, (p1[0], 1), p2, p3))

    # If unit is supporting a hold:
    if order == action_utils.SUPPORT_HOLD:
        if utils.province_type_from_id(p2[0]) == utils.ProvinceType.BICOASTAL:
            possible_orders_incl_flags.append((order, p1, (p2[0], 1), p3))

    # If unit is supporting a move:
    if order == action_utils.SUPPORT_MOVE_TO:
        # If supported unit is in a bicoastal province, it could be on the other coast:
        if utils.province_type_from_id(p3[0]) == utils.ProvinceType.BICOASTAL:
            possible_orders_incl_flags.append((order, p1, p2, (p3[0], 1)))
        # If destination is bicoastal province, add other coast:
        if utils.province_type_from_id(p2[0]) == utils.ProvinceType.BICOASTAL:
            possible_orders_incl_flags.append((order, p1, (p2[0], 1), p3))

    for order, p1, p2, p3 in possible_orders_incl_flags:
        if order == action_utils.HOLD:
            for unit_type in possible_unit_types(p1):
                unit_string = mila_unit_string(unit_type, p1)
                mila_action_strings.add(f"{unit_string} H")
        elif order == action_utils.CONVOY:
            mila_action_strings.add(
                f"{mila_unit_string(utils.UnitType.FLEET, p1)} C "
                f"{mila_unit_string(utils.UnitType.ARMY, p3)} - "
                f"{mila_area_string(utils.UnitType.ARMY, p2)}"
            )
        elif order == action_utils.CONVOY_TO:
            mila_action_strings.add(
                f"{mila_unit_string(utils.UnitType.ARMY, p1)} - "
                f"{mila_area_string(utils.UnitType.ARMY, p2)} VIA"
            )
        elif order == action_utils.MOVE_TO:
            for unit_type in possible_unit_types_movement(p1, p2):
                mila_action_strings.add(
                    f"{mila_unit_string(unit_type, p1)} - "
                    f"{mila_area_string(unit_type, p2)}"
                )
        elif order == action_utils.SUPPORT_HOLD:
            for acting_unit_type in possible_unit_types_support(p1, p2):
                for supported_unit_type in possible_unit_types(p2):
                    mila_action_strings.add(
                        f"{mila_unit_string(acting_unit_type, p1)} S "
                        f"{mila_unit_string(supported_unit_type, p2)}"
                    )
        elif order == action_utils.SUPPORT_MOVE_TO:
            for acting_unit_type in possible_unit_types_support(p1, p2):
                # My own code - theirs was wrong

                for supported_unit_type in possible_unit_types_movement(p3, p2):
                    # I think you do specify destination coast! This overcoms one bug
                    mila_action_strings.add(
                        f"{mila_unit_string(acting_unit_type, p1)} S "
                        f"{mila_unit_string(supported_unit_type, p3)} - "
                        f"{mila_area_string(supported_unit_type, p2)}"
                    )

                    if supported_unit_type != utils.UnitType.ARMY:
                        mila_action_strings.add(
                            f"{mila_unit_string(acting_unit_type, p1)} S "
                            f"{mila_unit_string(supported_unit_type, p3)} - "
                            f"{mila_area_string(utils.UnitType.ARMY, p2)}"
                        )
        elif order == action_utils.RETREAT_TO:
            for unit_type in possible_unit_types_movement(p1, p2):
                mila_action_strings.add(
                    f"{mila_unit_string(unit_type, p1)} R "
                    f"{mila_area_string(unit_type, p2)}"
                )
        elif order == action_utils.DISBAND:
            for unit_type in possible_unit_types(p1):
                mila_action_strings.add(f"{mila_unit_string(unit_type, p1)} D")
        elif order == action_utils.BUILD_ARMY:
            mila_action_strings.add(f"{mila_unit_string(utils.UnitType.ARMY, p1)} B")
        elif order == action_utils.BUILD_FLEET:
            mila_action_strings.add(f"{mila_unit_string(utils.UnitType.FLEET, p1)} B")
        elif order == action_utils.REMOVE:
            for unit_type in possible_unit_types(p1):
                mila_action_strings.add(f"{mila_unit_string(unit_type, p1)} D")
        elif order == action_utils.WAIVE:
            mila_action_strings.add("WAIVE")
        else:
            raise ValueError("Unrecognised order %s " % order)

    return list(mila_action_strings)


# Build Inverse Mapping
_mila_action_to_deepmind_actions = collections.defaultdict(set)
for _action in action_list.POSSIBLE_ACTIONS:
    _mila_action_list = action_to_mila_actions(_action)
    for _mila_action in _mila_action_list:
        _mila_action_to_deepmind_actions[_mila_action].add(_action)
_mila_action_to_deepmind_actions = immutabledict.immutabledict(
    {k: frozenset(v) for k, v in _mila_action_to_deepmind_actions.items()}
)


def mila_action_to_possible_actions(mila_action: str) -> List[action_utils.Action]:
    """Converts a MILA action string to all deepmind actions it could refer to."""
    if mila_action not in _mila_action_to_deepmind_actions:
        raise ValueError("Unrecognised MILA action %s" % mila_action)
    else:
        return list(_mila_action_to_deepmind_actions[mila_action])


def mila_action_to_action(
    mila_action: str, season: utils.Season
) -> action_utils.Action:
    """Converts mila action and its phase to the deepmind action."""
    mila_actions = mila_action_to_possible_actions(mila_action)
    if len(mila_actions) == 1:
        return mila_actions[0]
    else:
        order, _, _, _ = action_utils.action_breakdown(mila_actions[0])
        if order == action_utils.REMOVE:
            if season.is_retreats():
                return mila_actions[1]
            else:
                return mila_actions[0]
        elif order == action_utils.DISBAND:
            if season.is_retreats():
                return mila_actions[0]
            else:
                return mila_actions[1]
        else:
            assert False, "Unexpected: only Disband/Remove ambiguous in MILA actions."


# --- MY CODE BELOW ---

_MILA_TO_DM_TAG_MAP = immutabledict.immutabledict(
    {v: k for k, v in _DM_TO_MILA_TAG_MAP.items()}
)


def mila_to_dm_area(mila_string):
    """Converts a MILA area tag to the corresponding DM area ID."""
    province_tag = _MILA_TO_DM_TAG_MAP.get(mila_string, mila_string)
    area_id = _tag_to_area_id[province_tag]
    return area_id


def possible_orders_by_loc_to_power(possible_orders_by_loc: dict, game):
    """Takes a dictionary of possible orders by location and returns a dictionary of possible orders by power (in the MILA format).

    Args:
      possible_orders_by_loc: a dictionary with locations as keys, and their respective list of possible orders as values. From game.get_all_possible_orders()
      game: the current game instance.

    Returns:
      A dictionary of power: possible orders (in the MILA format).
    """

    assert (
        possible_orders_by_loc == game.get_all_possible_orders()
    ), "Orders should be the same as game.get_all_possible_orders()"

    # Convert dictionary of orders by location to dictionary of orders by power.
    powers = game.powers.values()
    possible_orders_by_power = {power: [] for power in powers}

    if game.phase_type == "R" and game.dislodged:
        # In RETREATS phases, only possible moves are by dislodged units.
        for power in powers:
            for unit in power.retreats:
                loc = unit[2:]
                possible_orders_by_power[power] += possible_orders_by_loc[loc]
    else:
        # Add all orders for non-dislodged units on board.
        for power in powers:
            for unit in power.units:
                loc = unit[2:]
                possible_orders_by_power[power] += possible_orders_by_loc[loc]
        # In BUILD phases, there can be orders for as-of-yet non-existent units.
        if game.phase_type == "A":
            for power in powers:
                # Check if power can build.
                build_count = len(power.units) - len(power.centers)
                if build_count > 0:
                    # Add build orders for power's unoccupied home locs.
                    for loc in game._build_sites(power):
                        possible_orders_by_power[power] += possible_orders_by_loc[loc]

    return possible_orders_by_power


def mila_to_dm_possible_orders(possible_orders_by_loc, game):
    """Takes a dictionary of possible orders by location (in the MILA format) and returns a dictionary of possible orders by power (in the DM format).

    Args:
      possible_orders_by_loc: dictionary from game.get_all_possible_orders()
      game: the current game instance.
      season: a utils.Season

    Returns:
      A dictionary of possible orders by power (in the DM format)."""

    powers = game.powers.values()
    season = mila_to_dm_season(game)

    possible_orders_by_power = possible_orders_by_loc_to_power(
        possible_orders_by_loc, game
    )

    legal_actions = {power: [] for power in powers}
    for power, orders in possible_orders_by_power.items():
        for order in orders:
            action = mila_action_to_action(order, season)
            legal_actions[power].append(action)

    # Sort actions and remove duplicates
    for power in powers:
        legal_actions[power] = list(set(legal_actions[power]))
        legal_actions[power].sort()

    return legal_actions


def mila_to_dm_season(game):
    """Extracts season information from game.

    Args:
      game: the current game instance.

    Returns:
      A utils.Season object."""

    # game.phase: string containing long rep of current phase
    if "SPRING" in game.phase:
        season = "SPRING"
    elif "FALL" in game.phase:
        season = "AUTUMN"
    elif "WINTER" in game.phase:
        # MILA calls 'BUILDS' phase 'WINTER ADJUSTMENT'
        season = "BUILDS"
    else:
        raise ValueError(f"{game.phase} does not contain expected season information.")

    # game.phase_type: 'M' for Movement, 'R' for Retreats, 'A' for Adjustment, '-' for non-playing
    if "M" in game.phase_type:
        season = getattr(utils.Season, season + "_MOVES")
    elif "R" in game.phase_type:
        season = getattr(utils.Season, season + "_RETREATS")
    elif "A" in game.phase_type:
        season = getattr(utils.Season, season)
    else:
        raise ValueError("Not a valid phase type.")

    return season


def resolve_mila_orders(orders: list, game) -> str:
    """Takes a list of candidate MILA orders for an action and the game state from the MILA environment, and finds the correct order.

    Args:
      orders: list of strings, e.g., 'F MAR D'
      game: the current game instance

    Returns:
      A string with the correct order."""

    legal_orders = game.get_all_possible_orders()
    order_resolved = False

    units_by_power = [
        list(set(power.units) | set(power.retreats.keys()))
        for power in game.powers.values()
    ]
    units_on_board = [unit for power_units in units_by_power for unit in power_units]

    if len(orders) > 1:
        for order in orders:
            word = order.split()
            loc = word[1]
            if loc in legal_orders and order in legal_orders[loc]:
                correct_order = order
                order_resolved = True
                break
        assert (
            order_resolved
        ), "Couldn't find correct order; candidate orders are {}".format(orders)
    else:
        correct_order = orders[0]

    return correct_order
