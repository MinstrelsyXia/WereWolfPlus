import abc
import collections
import functools
from typing import Any, Dict, Sequence, Tuple

from absl.testing import absltest
import numpy as np
import tree
import dill

from environment import diplomacy_state
from environment import game_runner
from environment import observation_utils as utils
from network import config
from network import network_policy
from network import parameter_provider

import tests.observation_test as observation_test

from games.welfare_diplomacy.diplomacy.engine.game import Game

# Create game instance
game = Game(map_name='standard_welfare')

class WelfareObservationTest(observation_test.ObservationTest):

    def get_diplomacy_state(self) -> diplomacy_state.DiplomacyState:
        state = diplomacy_state.WelfareDiplomacyState(game)
        return state
  
    def get_parameter_provider(self) -> parameter_provider.ParameterProvider:
        if SL:
            params = 'sl_params.npz'
        else:
            params = 'fppi2_params.npz'
        with open(file_path + params, 'rb') as f:
            provider = parameter_provider.ParameterProvider(f)
        return provider
  
    def get_reference_observations(self) -> Sequence[collections.OrderedDict]:
        with open(file_path + 'observations.npz', 'rb') as f:
            observations = dill.load(f)
        return observations
    
    def get_reference_legal_actions(self) -> Sequence[np.ndarray]:
        with open(file_path + 'legal_actions.npz', 'rb') as f:
            legal_actions = dill.load(f)
        return legal_actions
  
    def get_reference_step_outputs(self) -> Sequence[Dict[str, Any]]:
        with open(file_path + 'step_outputs.npz', 'rb') as f:
            step_outputs = dill.load(f)
        return step_outputs
  
    def get_actions_outputs(self) -> Sequence[collections.OrderedDict]:
        with open(file_path + 'actions_outputs.npz', 'rb') as f:
            actions_outputs = dill.load(f)
        return actions_outputs
    

if __name__ == '__main__':
    absltest.main()