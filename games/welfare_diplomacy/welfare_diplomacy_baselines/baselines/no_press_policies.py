""""Code for no-press policies to play Welfare Diplomacy against language models.

These policies are modifications on a network policy trained for Standard Diplomacy.
The algorithm used for the network policy is FPPI-2.
See https://arxiv.org/abs/2006.04635 for more details."""

import os
import numpy as np
from functools import partial
import logging
import argparse
from typing import Any, Sequence, Tuple

import sys

sys.path.append(
    "/Users/hannaherlebach/research/welfare-diplomacy/welfare_diplomacy_baselines/"
)

from games.welfare_diplomacy.welfare_diplomacy_baselines.network import config, network_policy, parameter_provider
from games.welfare_diplomacy.welfare_diplomacy_baselines.environment import diplomacy_state, game_runner, mila_actions, action_utils, human_readable_actions
from games.welfare_diplomacy.welfare_diplomacy_baselines.environment import observation_utils as utils

from games.welfare_diplomacy.welfare_diplomacy_baselines.baselines import disband_policies

# from games.welfare_diplomacy.diplomacy.engine.map import Map
# welfare_map = Map("standard_welfare")  # Errors on first repo use

from games.welfare_diplomacy.diplomacy.engine.game import Game
from games.welfare_diplomacy.diplomacy.engine.map import Map
welfare_map = Map("standard_welfare")

class SwitchPolicy:
    """A policy that switches from the network policy to disbanding policy after a certain number of years.
    If year_to_switch = n, then the disbanding policy will start taking effect in the adjustment phase of year n.

    Args:
        disband_policy: instance of the chosen disbanding policy
        year_to_switch: int, year in which to switch to disbanding policy"""

    def __init__(self, disband_policy, year_to_switch=None):
        self.disband = False
        self.disband_policy = disband_policy
        self.network_policy = get_network_policy_instance()
        self.year = 0

        if year_to_switch is None:
            year_to_switch = float("inf")
        self.year_to_switch = year_to_switch

    def reset(self):
        self.network_policy.reset()
        self.disband_policy.reset()

    def actions(self, slots_list, observation, legal_actions):
        season = observation.season
        if season == utils.Season.BUILDS:
            if self.year == self.year_to_switch:
                self.disband = True
            self.year += 1

        if not self.disband:
            # Run network policy
            actions = self.network_policy.actions(
                slots_list, observation, legal_actions
            )
        else:
            # Run disband policy
            actions = self.disband_policy.actions(
                slots_list, observation, legal_actions
            )

        return actions
    
    
class ExploiterPolicy:
    """Policy that plays cooperatively until there are fewer than a certain number of units on the board, then switches to a network policy to gain supply centres, and then disbands again.
    
    Args:
        unit_threshold: the number of units on the board below which the policy will switch to the network policy.
        center_threshold: number of centers the exploiter accrues before disbanding
        disband_policy: the disband policy
        power: Power instance"""
    def __init__(self, unit_threshold: int, center_threshold: int, power_ix: int, disband_policy):

        self.unit_threshold = unit_threshold
        self.center_threshold = center_threshold
        self.power_ix = power_ix
        self.disband_policy = disband_policy
        self.exploiter_policy = get_network_policy_instance()
        self.exploit = False
        self.disbanding = False
        
        power = sorted(welfare_map.powers)[power_ix] # power name string
        self.enemy_homes = [home for power_name, homes in welfare_map.homes.items() if power_name != power for home in homes]
        
    def reset(self):
        self.exploiter_policy.reset()
        self.disband_policy.reset()

    def actions(self, slots_list, observation, legal_actions):
        board = observation.board
        total_units = np.sum(board[:72,:2]) + np.any(board[72:75,:2]) + np.any(board[75:78,:2]) + np.any(board[78:81,:2])
        power_units = np.sum(board[:72, 3+self.power_ix]) + np.any(board[72:75, 3+self.power_ix]) + np.any(board[75:78, 3+self.power_ix]) + np.any(board[78:81, 3+self.power_ix])
        enemy_units = total_units - power_units
        power_centers = np.sum(board[:72, 27+self.power_ix]) + np.any(board[72:75, 27+self.power_ix]) + np.any(board[75:78, 27+self.power_ix]) + np.any(board[78:81, 27+self.power_ix])

        if self.exploit:
            # Run network policy until power has enough supply centers

            # if self.disbanding:
            #     return self.disband_policy.actions(slots_list, observation, legal_actions)
            if power_centers > self.center_threshold:
                print('start disbanding!')
                print(f'power has {power_centers} centers')
                self.disbanding = True
                return self.disband_policy.actions(slots_list, observation, legal_actions)
            else:
                print('exploiting!')
                return self.exploiter_policy.actions(slots_list, observation, legal_actions)

        if enemy_units < self.unit_threshold:
            # Run network policy
            print('start exploiting!')
            print(f'total units on board are {total_units}')
            print(f'enemy units on board are {enemy_units}')
            self.exploit = True
            return self.exploiter_policy.actions(slots_list, observation, legal_actions)
        else:
            # Play cooperative policy, not sure what this should be - maybe a filtered network policy?
            exploiter_actions, step_outs = self.exploiter_policy.actions(slots_list, observation, legal_actions)
            final_actions = []
            for action in exploiter_actions[0]:
                # Convert to string
                action_string = human_readable_actions.action_string(action, board)
                word = action_string.split()
                if len(word[-1]) >= 3:
                    target_province = word[-1][:3]
                    if target_province in self.enemy_homes:
                        # Don't take actions that target enemy homes
                        print(f"I'm being cooperative removing order {action_string}!")
                        continue
                    else:
                        final_actions.append(action)
            return [final_actions], step_outs
        

class HybridExploiterPolicy:
    """Policy played an exploiter agent once it switches to exploitation mode. Plays network policy until it has a certain number of supply centers, then starts disbanding. To be combined with an APIAgent pre-exploiting.
    
    Args:
        center_threshold: number of centers the exploiter accrues before disbanding
        disband_policy: the disband policy
        power_ix: Power index"""
    
    def __init__(self, center_threshold: int, power_ix: int, disband_policy):

        self.center_threshold = center_threshold
        self.power_ix = power_ix
        self.disband_policy = disband_policy
        self.exploiter_policy = get_network_policy_instance()
        self.disbanding = False # use if you want to switch unconditionally
        
    def reset(self):
        self.exploiter_policy.reset()
        self.disband_policy.reset()

    def actions(self, slots_list, observation, legal_actions):
        board = observation.board
        centers = np.sum(board[:72, 27+self.power_ix]) + np.any(board[72:75, 27+self.power_ix]) + np.any(board[75:78, 27+self.power_ix]) + np.any(board[78:81, 27+self.power_ix])

        if centers > self.center_threshold: # or self.disbanding==True
            print('start disbanding!')
            print(f'power has {centers} centers')
            self.disbanding = True
            return self.disband_policy.actions(slots_list, observation, legal_actions)
        else:
            print('exploiting!')
            return self.exploiter_policy.actions(slots_list, observation, legal_actions)



# Make sure files containing parameters are in welfare_diplomacy_baselines/network_parameters
network_parameter_path = os.path.join(os.path.join(os.path.dirname(__file__), ".."), "network_parameters")

def get_network_policy_instance(
    algorithm="FPPI2", file_path=network_parameter_path
):
    """Returns a network policy instance.

    By default all experiments should use the FFPI-2 parameters, but the SL parameters are also available.

    Args:
        algorithm: str in ['SL', 'FFPI2']
        file_path: str, path to directory containing the parameters"""

    if algorithm == "SL":
        params = "sl_params.npz"
    elif algorithm == "FPPI2":
        params = "fppi2_params.npz"
    else:
        raise ValueError("Algorithm must be SL or FFPI2.")
    with open(os.path.join(file_path, params), "rb") as f:
        provider = parameter_provider.ParameterProvider(f)

    network_info = config.get_config()
    network_handler = parameter_provider.SequenceNetworkHandler(
        network_cls=network_info.network_class,
        network_config=network_info.network_kwargs,
        parameter_provider=provider,
        rng_seed=42,
    )
    network_policy_instance = network_policy.Policy(
        network_handler=network_handler,
        num_players=7,
        temperature=0.1,
        calculate_all_policies=False,
    )

    return network_policy_instance

def smart_disband_policy(max_years=10):
    policy = disband_policies.SmartDisbandPolicy(
        num_to_disband=1, max_years=max_years
    )
    return policy


policy_map = {
    0: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=0),
    1: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=1),
    2: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=2),
    3: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=3),
    10: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=0
    ),
    11: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=1
    ),
    12: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=2
    ),
    13: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=3
    ),
    20: lambda: SwitchPolicy(
        disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=0
    ),
    21: lambda: SwitchPolicy(
        disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=1
    ),
    22: lambda: SwitchPolicy(
        disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=2
    ),
    23: lambda: SwitchPolicy(
        disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=3
    ),
    30: lambda: ExploiterPolicy(
        9, 12, 0, disband_policies.InstantDisbandPolicy()
    ),
    31: lambda: ExploiterPolicy(
        9, 12, 1, disband_policies.InstantDisbandPolicy()
    ),
    32: lambda: ExploiterPolicy(
        9, 12, 2, disband_policies.InstantDisbandPolicy()
    ),
    33: lambda: ExploiterPolicy(
        6, 12, 3, disband_policies.InstantDisbandPolicy()
    ),
    34: lambda: ExploiterPolicy(
        6, 12, 4, disband_policies.InstantDisbandPolicy()
    ),
    35: lambda: ExploiterPolicy(
        6, 12, 5, disband_policies.InstantDisbandPolicy()
    ),
    36: lambda: ExploiterPolicy(
        6, 12, 6, disband_policies.InstantDisbandPolicy()
    ),
    40: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(num_to_disband=1), year_to_switch=0
    ),
    41: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(num_to_disband=1), year_to_switch=1
    ),
    42: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(num_to_disband=1), year_to_switch=2
    ),
    43: lambda: SwitchPolicy(
        disband_policies.RandomDisbandPolicy(num_to_disband=1), year_to_switch=3
    ),

}


def smart_policy_experiment():
    logging.basicConfig(filename='smart_disband_experiments.log', filemode='a', level=logging.INFO)
    
    # Get each power to play smart policy
    for year in range(4):
        for pow in range(7):
            logging.info(f"Power {pow} plays SwitchPolicy with SmartDisbandPolicy(num_to_disband=1) from year {year}")
            game_instance = Game(map_name="standard_welfare")
            initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)
            test_policy = SwitchPolicy(disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=year)
            bg_policy = get_network_policy_instance()
            policies = (test_policy, bg_policy)
            slots_to_policies = [1] * 7
            slots_to_policies[pow] = 0
            trajectory = game_runner.run_game(
                state=initial_state,
                policies=policies,
                slots_to_policies=slots_to_policies,
                max_years=10,
            )

            logging.info(f"Power {pow} plays SwitchPolicy with RandomDisbandPolicy(num_to_disband=1) from year {year}")
            game_instance = Game(map_name="standard_welfare")
            initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)
            test_policy = SwitchPolicy(disband_policies.RandomDisbandPolicy(num_to_disband=1), year_to_switch=year)
            bg_policy = get_network_policy_instance()
            policies = (test_policy, bg_policy)
            slots_to_policies = [1] * 7
            slots_to_policies[pow] = 0
            trajectory = game_runner.run_game(
                state=initial_state,
                policies=policies,
                slots_to_policies=slots_to_policies,
                max_years=10,
            )

def exploiter_test():
    game_instance = Game(map_name="standard_welfare")
    initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)
    background_policy = disband_policies.RandomDisbandPolicy(p=0.2, max_years=10)
    test_policy = ExploiterPolicy(9, 12, 0, disband_policies.SmartDisbandPolicy(num_to_disband=1))
    policies = (test_policy, background_policy)
    slots_to_policies = [0, 1, 1, 1, 1, 1, 1]
    trajectory = game_runner.run_game(
        state=initial_state,
        policies=policies,
        slots_to_policies=slots_to_policies,
        max_years=10
        )

# Test on DM setup
def main():
    exploiter_test()
    # smart_policy_experiment()
    # game_instance = Game(map_name="standard_welfare")
    # initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)
    # # policies = (
    # #     policy_map[30](),
    # #     policy_map[31](),
    # #     policy_map[32](),
    # #     policy_map[33](),
    # #     policy_map[34](),
    # #     policy_map[35](),
    # #     policy_map[36](),
    # # )
    # # slots_to_policies = [0, 1, 2, 3, 4, 5, 6]
    # policies = (policy_map[40](),)
    # slots_to_policies = [0] * 7
    # trajectory = game_runner.run_game(
    #     state=initial_state,
    #     policies=policies,
    #     slots_to_policies=slots_to_policies,
    #     max_years=5,
    # )

if __name__ == "__main__":
    main()
