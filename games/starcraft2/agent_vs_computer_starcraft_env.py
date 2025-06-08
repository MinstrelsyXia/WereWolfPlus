import multiprocessing
import time
import datetime
from typing import (
    Optional,
)
import gym
import os
from gym import spaces
from sc2 import maps
from sc2.data import Race
from sc2.player import Bot, Computer
from sc2.main import run_game
from games.starcraft2.bot.Protoss_bot import Protoss_Bot
from games.starcraft2.bot.Zerg_bot import Zerg_Bot
from games.starcraft2.utils.action_info import ActionDescriptions
from games.starcraft2.sc2_config import LADDER_MAP_2023
from games.starcraft2.sc2_config import map_difficulty, map_race, map_ai_build

class AgentVSComputerStarcraftEnv(gym.Env):
    """
    This is a pure language state for the StarCraft II environment.

    Attributes:
    player_race: The race of the player. This environment only supports 'Protoss' or 'Zerg'.
    map_name: The name of the map for the bot to choose. Usually it is a string.
    opposite_race: The race of the opponent.
    lock: The lock for the transaction.
    transaction: The transaction between the bot and the environment(StarCraft II).
                 This includes
                 information (language-based data about the game state),
                 reward (the reward obtained from the last action),
                 action (the last action performed by the bot),
                 done (a flag indicating if the game is over),
                 result (the result of the game, win or lose),
                 and iter (the current step of the game).
    p: The process for the bot to run.
    isReadyForNextStep: The flag for the bot to tell the environment that it is ready for the next step.
    isReadyForReset: The flag for the bot to tell the environment that it is ready for a reset.
    game_over: A flag to track whether the game is over or not. This helps to manage the lifecycle of 'p'.

    """
    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self._time_step = 1  # Corresponding to wandb fig
        self.map_name = args.game.game_map
        assert self.map_name in LADDER_MAP_2023
        if "_VS_" in args.game.players:
            players=args.game.players.split("_VS_")
        assert len(players) == 2
        self.player_race = players[0]
        self.opposite_race = players[1]
        # todo assert
        self.difficulty = args.game.game_difficulty
        self.lock = multiprocessing.Manager().Lock()
        self.transaction = multiprocessing.Manager().dict()
        self.transaction.update(
            {'information': [], 'reward': 0, 'action': None,
             'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
             'action_executed': [], 'action_failures': [], 'process_data':None,'_time_step':self._time_step})
        self.isReadyForNextStep = multiprocessing.Event()
        self.game_end_event = multiprocessing.Event()
        self.game_over = multiprocessing.Value('b', False)  # Add a new flag to track if the game is over
        self.done_event = multiprocessing.Event()  
        self.p = None
        self.check_process(reset=True)
        self.check_process()
        self.action_space = spaces.Discrete(self.calculate_action_space(self.player_race))
        self.observation_space = spaces.Dict({
            "player_race": spaces.Text(max_length=20),  # Terran,Protoss, Zerg, Random
            "opposite_race": spaces.Text(max_length=20),  # Terran,Protoss, Zerg, Random
            "map_name": spaces.Text(max_length=20),  # Map name
            "information": spaces.Dict({
                "observation1": gym.spaces.Discrete(10),
                "observation2": gym.spaces.Box(low=0, high=1, shape=(3, 3)),
            }),  # Information about the game state
        })

        self.logger.info("=" * 5 + f"AgentVSComputerStarcraftEnv Init Successfully!: " + "=" * 5)

    def calculate_action_space(self, player_race):
        action_description = ActionDescriptions(player_race)
        action_list = action_description.action_descriptions
        return len(action_list)

    def check_process(self, reset=False):
        """

        This function checks the current process.
        If the process is still alive and the game is not over, it simply returns.
        If the game is over, it terminates and joins the process.
        If reset is True, it resets the transaction, clears the 'isReadyForReset' flag,
        starts a new process based on the player's race, and starts the new process.
        """
        if self.p is not None:
            if self.p.is_alive():
                if not self.game_over.value:  # Check if the game is over
                    return  # If the game is not over, just return and do not restart the process
                self.p.terminate()
            self.p.join()
        if reset:
            self.transaction.update(
                {'information': [], 'reward': 0, 'action': None,
                 'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
                 'action_executed': [], 'action_failures': [], 'process_data':None,'_time_step':self._time_step })
            self.game_end_event.clear()  # Clear the game_end_event
            if self.player_race == 'Protoss':
                self.p = multiprocessing.Process(target=protoss_agent_vs_build_in, args=(
                    self.transaction, self.lock, self.map_name, self.isReadyForNextStep, self.game_end_event,
                    self.done_event, self.opposite_race, self.difficulty,self.args.game.game_ai_build,self.args.game.asynch_mode, self.args.eval.output_path))
            elif self.player_race == 'Zerg':
                self.p = multiprocessing.Process(target=zerg_agent_vs_build_in, args=(
                    self.transaction, self.lock, self.map_name, self.isReadyForNextStep, self.game_end_event,
                    self.done_event, self.opposite_race, self.difficulty,self.args.game.game_ai_build,self.args.game.asynch_mode, self.args.eval.output_path))
            else:
                raise ValueError("Invalid race. Only 'Protoss' and 'Zerg' are supported.")
            self.logger.info("=" * 5 + f"game run asynch mode: {self.args.game.asynch_mode}" + "=" * 5)
            self.p.start()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        This function resets the environment.
        It calls 'check_process' with reset=True, waits until the environment is ready for a reset,
        and then returns the new state, reward, 'done' flag, and game result.
        return : state,info
        state include player race,opposite_race , map_name, information
        info is just for debug
        """
        # Check the process status and reset.
        self.check_process(reset=True)

        # Clear the game end event flag.
        self.game_end_event.clear()

        # Define and return the state dictionary, which includes the player's race, opponent's race, map name, game information, etc.
        state = {
            'player_race': self.player_race,  # Player's race
            'opposite_race': self.opposite_race,  # Opponent's race
            'map_name': self.map_name,  # Map name
            'information': self.transaction['information'],  # Game information
            'action_executed': self.transaction['action_executed'],  # List of executed actions
            'action_failures': self.transaction['action_failures'],  # List of failed actions
            '_time_step': self._time_step
        }

        return state, None

    def step(self, action):
        """
        This function performs one step in the environment using the provided action.
        It sets the 'action' in the transaction, waits until the environment is ready for the next step,
        and then gets the next state.
        If the game is done, it sets the 'game_over' flag.
        If the game is not over, it calls 'check_process' to possibly start a new process.
        It then returns the next state, reward, 'done' flag, and game result.

        return : state, reward, done, result
        state include player race,opposite_race , map_name, information
        information is a dict contains the information of the game
        """
        # Lock resources to ensure that no other thread/process changes the transaction state during this code execution.
        with self.lock:
            # If the action is a tuple with 3 elements, unpack and set them as action, command, and command flag.
            if isinstance(action, tuple) and len(action) == 4:
                print("================================and len(action) == 4=")
                action_, command, command_flag,match_data = action
                self.transaction['action'] = action_
                self.transaction['command'] = command
                self.transaction['output_command_flag'] = command_flag
            else:
                print("================================and tuple")
                # If not a tuple, directly set the action as the transaction action.
                self.transaction['action'] = action
                self.transaction['command'] = None
                self.transaction['output_command_flag'] = False

        # Wait for the game to end or be ready for the next step.
        while not (self.done_event.is_set() or self.isReadyForNextStep.is_set()):
            time.sleep(0.0001)

        # If the game ends, clear related events and set the reward.
        if self.done_event.is_set():
            self.done_event.clear()
            self.isReadyForNextStep.clear()
            self.game_over.value = True
            if self.transaction['result'].name == 'Victory':
                self.transaction['reward'] += 50
        elif self.isReadyForNextStep.is_set():
            # If the game hasn't ended but is ready for the next step, clear related events and check the process.
            self.isReadyForNextStep.clear()
            self.check_process()

        print('Result before returning:', self.transaction['result'])

        result = self.transaction['result']
        result_str = str(result) if result is not None else None

        # Define the next state.
        state = {
            'player_race': self.player_race,
            'opposite_race': self.opposite_race,
            'map_name': self.map_name,
            'information': self.transaction['information'],
            'action_executed': self.transaction['action_executed'],
            'action_failures': self.transaction['action_failures'],
            'process_data':self.transaction['process_data'],
            '_time_step':self._time_step
        }

        # Ensure each part of the state is serializable.
        for key, value in state.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, (int, float, str, bool, type(None))):
                        value[sub_key] = str(sub_value)
                state[key] = value

        return state, self.transaction['reward'], self.transaction['done'], result_str, None

    def render(self, mode='human'):
        return None

    def close(self):
        return None


def protoss_agent_vs_build_in(transaction, lock, map_name, isReadyForNextStep, game_end_event, done_event,
                              opposite_race, difficulty,ai_build,asy_mode, save_path):

    # Create a replay folder
    replay_folder = os.path.join(save_path)

    # If the directory does not exist, create it
    if not os.path.exists(replay_folder):
        os.makedirs(replay_folder)

    cur_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Run the game
    result = run_game(maps.get(map_name),
                      [Bot(Race.Protoss, Protoss_Bot(transaction, lock, isReadyForNextStep)),
                       Computer(map_race(opposite_race), map_difficulty(difficulty), map_ai_build(ai_build))],
                      realtime=asy_mode,
                      save_replay_as=f'{replay_folder}/{map_name}_player_Protoss_VS_BUILD_IN_AI_{difficulty}_{opposite_race}_{cur_time}.SC2Replay')
    with lock:
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~",result)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~",result.value)
        transaction['done'] = True
        transaction['result'] = result
    done_event.set()  # Set done_event when the game is over
    game_end_event.set()  # Set game_end_event when the game is over

def zerg_agent_vs_build_in(transaction, lock, map_name, isReadyForNextStep, game_end_event, done_event,
                           opposite_race, difficulty,ai_build,asy_mode, save_path):
    # Create a replay folder
    replay_folder = os.path.join(save_path)
    if not os.path.exists(replay_folder):
        try:
            os.makedirs(replay_folder)

        except OSError:
            print(f"create dictionary {replay_folder} failure,please check and run program again.")
            return
    cur_time=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result = run_game(maps.get(map_name),
                      [Bot(Race.Zerg, Zerg_Bot(transaction, lock, isReadyForNextStep)),
                       Computer(map_race(opposite_race), map_difficulty(difficulty), map_ai_build(ai_build))],
                      realtime=asy_mode,
                      save_replay_as=f'{replay_folder}/{map_name}_Player_Zerg_VS_BUILD_IN_AI_{difficulty}_{opposite_race}_{cur_time}.SC2Replay')

    with lock:
        transaction['done'] = True
        transaction['result'] = result
    # print("transaction done:", transaction['done'])
    done_event.set()  # Set done_event when the game is over

    # print("game end")
    game_end_event.set()  # Set game_end_event when the game is over
