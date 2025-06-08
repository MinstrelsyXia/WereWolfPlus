"""Run as a notebook.

This notebook allows you to run a series of 1v6 games of one focal agent vs a population of identical background agents, varying the number of turns after which the policies switch from the zero-sum (network) policy to a random disbanding policy.

Parameters to specify:
    alg: string denoting the network policy, in {'FPPI2', 'SL'}
    max_length: integer denoting number of turns to be played per game
    p: float for probability of disbanding each unit in the random disbanding policy
    focal: integer denoting the power index of the focal agent (in [Austria, England, France, Germany, Italy, Russia, Turkey])
    step: integer denoting step size in the sweep over number of turns after which to switch policies

All policies use the same network policy and disband policy; all that varies is the number of turns after which the policies switch.
"""
#%%
import logging
logging.basicConfig(filename='equilibrium_experiments_years_2.log', filemode='a', level=logging.INFO)

import sys
sys.path.append('/Users/hannaherlebach/research/welfare-diplomacy/welfare_diplomacy_baselines/')

import os
import numpy as np
import nashpy as nash
from tqdm import tqdm
from functools import partial
from typing import Sequence

from network import parameter_provider, config, network_policy
from baselines import disband_policies as dp
from environment import diplomacy_state, game_runner, action_utils
from environment import observation_utils as utils

from games.welfare_diplomacy.diplomacy.engine.game import Game
#%%

def print_payoff_matrix(matrix):
    # Assuming matrix shape is m x n x 2
    m, n, _ = matrix.shape
    
    # Print header
    print(" " * 10, end="")  # Some spacing
    for j in range(n):
        print(f"Player 2 Action {j + 1}", end="\t")
    print()  # New line

    # Print rows with payoffs
    for i in range(m):
        print(f"Player 1 Action {i + 1} |", end="\t")
        for j in range(n):
            print(tuple(matrix[i, j]), end="\t")
        print()  # New line

#%%
def nash_social_welfare(payoffs: Sequence[int]):
    """Calculates the Nash Social Welfare (i.e., geometric mean) of a sequence of payoffs."""
    welfare = 1
    for payoff in payoffs:
        welfare *= payoff
    return welfare ** (1/len(payoffs))

#%%
# Make sure files containing parameters are in welfare_diplomacy_baselines/network_parameters
network_parameter_path = os.path.join(os.path.join(os.path.dirname(__file__), ".."), "network_parameters")

def get_network_policy_instance(algorithm='SL', file_path=network_parameter_path):
    """Returns a network policy instance, using SL or FFPI-2 parameters.
    
    Args:
        algorithm: str in ['SL', 'FFPI2']"""

    if algorithm=='SL':
        params = 'sl_params.npz'
    elif algorithm=='FPPI2':
        params = 'fppi2_params.npz'
    else:
        raise ValueError('Algorithm must be SL or FFPI2.')
    with open(os.path.join(file_path, params), 'rb') as f:
        provider = parameter_provider.ParameterProvider(f)
    
    network_info = config.get_config()
    network_handler = parameter_provider.SequenceNetworkHandler(
        network_cls=network_info.network_class,
        network_config=network_info.network_kwargs,
        parameter_provider=provider,
        rng_seed=42)
    network_policy_instance = network_policy.Policy(
        network_handler=network_handler,
        num_players=7,
        temperature=0.1,
        calculate_all_policies=False)
    
    return network_policy_instance
#%%
class SwitchTurnPolicy:
    """Policy which switches from zero-sum policy to disbanding policy.
    
    Args:
        network_policy: a network policy instance.
        k: the year in which to switch to disbanding (from the BUILDS phase). If k=-1, always plays disbanding policy.
        hard_coded_policy: the disbanding policy."""
    def __init__(self, network_policy, k, hard_coded_policy):
        self.network_policy = network_policy # a network policy instance
        self.k = k # number of turns after which to switch
        self.hard_coded_policy = hard_coded_policy # a hard-coded policy instance
        self.turn_num = 0
        self.year = 0

    def actions(self, slots_list, observation, legal_actions):
        """Plays the network policy and switches to disbanding at the start of BUILDS in the kth year."""
        season = observation.season
        if self.year > self.k or self.year == self.k and season == utils.Season.BUILDS:
            actions = self.hard_coded_policy.actions(slots_list, observation, legal_actions)
        else:
            actions = self.network_policy.actions(slots_list, observation, legal_actions)
        if season == utils.Season.BUILDS:
            self.year += 1
        self.turn_num +=1
        return actions

    def reset(self):
        self.network_policy.reset()
        self.hard_coded_policy.reset()

#%%
# Specify & log experiment parameters
alg = 'FPPI2'
max_years = 5
disband = 'instant'
p = 0.5
focal = 1
step = 1

params = {
    'alg': alg,
    'max_length': max_years, # max number of years to play
    'p': p, # for random disband policy
    'disband': disband,
    'focal': focal,
    'step': step # years to step each iteration of the game
}

for key, value in params.items():
    logging.info(f"{key} = {value}")

#%%

# Create an array to store the payoffs.
payoff_matrix = np.zeros((max_years, max_years, 7))

# Sweep over i and j for focal and background policies.
for i in range(0, max_years, step): # focal 
    for j in range(0, max_years, step): # background       
        print(f'Focal: switches after {i} years')
        print(f'Background: switches after {j} years')

        # Create a new game instance
        game_instance = Game()
        initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)

        # Focal policy
        network_policy_instance = get_network_policy_instance(alg)
        if disband == 'instant':
            disband_policy = dp.InstantDisbandPolicy()
        elif disband == 'random':
            disband_policy = dp.RandomDisbandPolicy(p=p)
        else:
            raise ValueError('Disband policy must be instant or random.')
        focal_policy = SwitchTurnPolicy(network_policy_instance, i, disband_policy)

        # Background policy
        network_policy_instance = get_network_policy_instance(alg)
        if disband == 'instant':
            disband_policy = dp.InstantDisbandPolicy()
        elif disband == 'random':
            disband_policy = dp.RandomDisbandPolicy(p=p)
        else:
            raise ValueError('Disband policy must be instant or random.')
        background_policy = SwitchTurnPolicy(network_policy_instance, j, disband_policy)

        # Run game
        policies = (background_policy, focal_policy)
        slots_to_policies = [int(i==focal) for i in range(7)]
        
        trajectory = game_runner.run_game(
            state = initial_state,
            policies = policies,
            slots_to_policies = slots_to_policies,
            max_length = None,
            max_years = max_years
            )
        
        # Store payoffs (i.e., final welfare points)
        payoffs = np.array(trajectory.returns)
        payoff_matrix[i,j] = payoffs

        logging.info("Payoffs for focal policy switching in Year %d and background policy switching in Year %d: %s", i, j, payoffs)

logging.info(f"Raw Payoff Matrix:\n{payoff_matrix}")
payoff_matrix = np.array(payoff_matrix)

#%%
"""Getting payoff matrices."""
# Slice matrix only to include games played
payoff_matrix_reduced = payoff_matrix[list(range(0, max_years, step)), :,:]
payoff_matrix_reduced = payoff_matrix_reduced[:, list(range(0, max_years, step)),:]

# Calculate NSW for everyone
nsw_everyone = np.apply_along_axis(nash_social_welfare, axis=-1, arr=payoff_matrix_reduced)
logging.info(f"Nash Social Welfare for everyone:\n{nsw_everyone}")

# Calculate NSW for background players
background_slots = [i for i in range(7) if slots_to_policies[i]==0]
focal_payoffs = payoff_matrix_reduced[:,:, focal]
background_payoffs = payoff_matrix_reduced[:,:, background_slots]
averaged_background_payoffs = np.apply_along_axis(nash_social_welfare, axis=-1, arr=background_payoffs)
# Last dimension is (focal payoff, averaged background payoff)
nsw_background = np.stack((focal_payoffs, averaged_background_payoffs), axis=-1)
print_payoff_matrix(nsw_background)
logging.info(f"Focal payoff against Nash Social Welfare of background:\n{nsw_background}")

#%%
"""Equilibrium analysis."""

# Focal against NSW background
final_focal_payoffs = nsw_background[:,:,0]
final_background_payoffs = nsw_background[:,:,1]
matrix_game = nash.Game(final_focal_payoffs, final_background_payoffs)

# Get a generator of all equilibria
equilibria = matrix_game.support_enumeration()

for n, equ in enumerate(equilibria): # eq is a tuple of arrays
    focal_equ, background_equ = equ
    print(f"Equilibrium {n}:")
    # Equilibrium strategy with prob for each k to switch on
    focal_strategy = {step * i: probs for i, probs in enumerate(focal_equ)}
    background_strategy = {step * i: probs for i, probs in enumerate(background_equ)}

    logging.info("Equilibrium focal strategy %d", n)
    for k, v in focal_strategy.items():
        logging.info("Disband in Year %d with probability %d", k, v)
    output = ",".join(f"\n\t Year {k} with probability {v}" for k, v in focal_strategy.items())
    print(f"Focal: disband in {output}")

    logging.info("Equilibrium background strategy %d", n)
    for k, v in background_strategy.items():
        logging.info("Disband in Year %d with probability %d", k, v)
    output = ",".join(f"\n\tYear {k} with probability {v}" for k, v in background_strategy.items())
    print(f"Background: disband in {output}")

#%%
"""Optimal social welfare analysis."""

optimal_game = np.unravel_index(np.argmax(nsw_everyone), nsw_everyone.shape)
logging.info("Optimal social welfare: when focal switches in Year %d and background switches in Year %d", optimal_game[0]*step, optimal_game[1]*step)
print(f"Optimal social welfare when focal switches in Year {optimal_game[0]*step} and background switches in Year {optimal_game[1]*step}")
#%%
