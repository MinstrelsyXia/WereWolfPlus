import multiprocessing
import time
from typing import (
    Optional,
)
import gym
import os
from gym import spaces
from sc2 import maps
from sc2.data import Race
from sc2.player import Bot, Human
from sc2.main import run_game
from games.starcraft2.utils.action_info import ActionDescriptions
from games.starcraft2.sc2_config import LADDER_MAP_2023
from games.starcraft2.sc2_config import map_race
from games.starcraft2.bot.Protoss_bot import Protoss_Bot


class AgentVSHumanStarcraftEnv(gym.Env):
    """
    This is a pure language state for the StarCraft II environment.

    Attributes:
    map_pool: The pool of maps for the bot to choose from. Usually it is a list.
    map_idx: The index of the chosen map in the map pool.
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


    这是一个纯自然语言状态的星际争霸II环境。请注意,我们实现了Human vs LLM agent。
    相关变量:
    map_pool:代理可以选择的地图池。通常是一个列表。
    map_idx:地图池中所选地图的索引。
    player_race:玩家的种族。此环境仅支持“Protoss”或“Zerg”。
    map_name:选择的地图的名称。通常是一个字符串。
    opposite_race:对手的种族。
    lock:信息交流的锁。
    transaction:代理与环境（星际争霸II）之间的交易。这包括
                信息（关于游戏状态的基于语言的数据），
                奖励（从上次行动获得的奖励），
                行动（代理执行的上次行动），
                完成（指示游戏是否结束的标志），
                结果（游戏的结果，赢或输），
                iter（游戏的当前步骤）。
    p:代理运行的进程。
    isReadyForNextStep:代理告诉环境它已经准备好进行下一步的标志。
    isReadyForReset:代理告诉环境它已经准备好进行重置的标志。
    game_over:一个标志，用于跟踪游戏是否结束。这有助于管理“p”的生命周期。

    """

    def __init__(self, args):
        self.args = args
        self.map_name = args.game.game_map
        assert self.map_name in LADDER_MAP_2023
        self.player_race = args.game.players[0]
        self.args.Human_race=args.game.players[1]
        self.lock = multiprocessing.Manager().Lock()
        self.transaction = multiprocessing.Manager().dict()
        self.transaction.update(
            {'information': [], 'reward': 0, 'action': None,
             'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
             'action_executed': [], 'action_failures': [], 'process_data':None})
        self.isReadyForNextStep = multiprocessing.Event()
        self.game_end_event = multiprocessing.Event()
        self.game_over = multiprocessing.Value('b', False)  # Add a new flag to track if the game is over
        self.done_event = multiprocessing.Event()  # 新增
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

        这个函数检查当前进程。
        如果进程仍然活着，游戏还没有结束，它只是返回。
        如果游戏结束，它终止并加入进程。
        如果重置为True，它将重置事务，清除“isReadyForReset”标志，
        根据玩家的种族启动一个新的进程，并启动新的进程。

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
                 'action_executed': [], 'action_failures': [], 'process_data':None })
            self.game_end_event.clear()  # Clear the game_end_event
            if self.player_race == 'Protoss':
                self.p = multiprocessing.Process(target=protoss_agent_vs_human, args=(
                    self.transaction, self.lock, self.map_name, self.isReadyForNextStep, self.game_end_event,
                    self.done_event, self.args))
            else:
                raise ValueError("Invalid race. Only 'Protoss' LLM Agent is supported.")
            self.p.start()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Your implementation here

        """
        This function resets the environment.
        It calls 'check_process' with reset=True, waits until the environment is ready for a reset,
        and then returns the new state, reward, 'done' flag, and game result.
        return : state,info
        state include player race,opposite_race , map_name, information
        info is just for debug

        这个函数重置环境。
        它调用“check_process”重置=True，等待环境准备重置，
        然后返回新的状态，和 info
        返回：状态，
        状态包括玩家种族，对手种族，地图名称，
        info 仅用于调试


        """
        # 检查进程状态，并重置。
        self.check_process(reset=True)

        # 清除游戏结束的事件标记。
        self.game_end_event.clear()

        # 定义并返回状态字典，包含了玩家的种族、对手的种族、地图名、游戏信息等。
        state = {
            'player_race': self.player_race,  # 玩家种族
            'opposite_race': self.args.Human_race,  # 对手种族
            'map_name': self.map_name,  # 地图名称
            'information': self.transaction['information'],  # 游戏信息
            'action_executed': self.transaction['action_executed'],  # 执行过的动作列表
            'action_failures': self.transaction['action_failures']  # 失败的动作列表
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

        这个函数使用提供的动作在环境中执行一步。
        它在事务中设置“action”,获取command和command flag，等待环境准备下一步，
        其中action是一个元组，包含动作的id，语言模型的指令，是否有指令,即(action_, command, command_flag)

        然后获取下一个状态。
        如果游戏结束，它设置“game_over”标志。
        如果游戏没有结束，它调用“check_process”来可能启动一个新的进程。
        然后返回下一个状态，奖励，'done'标志和游戏结果,info

        返回：状态，奖励，完成，结果
        状态包括玩家种族，对手种族，地图名称，信息
        信息是一个包含游戏信息的字典
        terminated :表示游戏是否结束
        result : 游戏结果
        info : 用于调试的信息


        """
        # 锁定资源，确保在这段代码执行期间，没有其他线程/进程会更改交易状态。
        with self.lock:
            # 如果动作是一个包含3个元素的元组，则解包并将其分别设置为动作、命令和命令标志。
            if isinstance(action, tuple) and len(action) == 3:
                action_, command, command_flag = action
                self.transaction['action'] = action_
                self.transaction['command'] = command
                self.transaction['output_command_flag'] = command_flag
            else:
                # 如果不是元组，直接将动作设置为交易动作。
                self.transaction['action'] = action
                self.transaction['command'] = None
                self.transaction['output_command_flag'] = False

        # 等待游戏结束或准备好下一步。
        while not (self.done_event.is_set() or self.isReadyForNextStep.is_set()):
            time.sleep(0.0001)

        # 如果游戏结束，则清除相关事件并设置奖励。
        if self.done_event.is_set():
            self.done_event.clear()
            self.isReadyForNextStep.clear()
            self.game_over.value = True
            if self.transaction['result'].name == 'Victory':
                self.transaction['reward'] += 50
        elif self.isReadyForNextStep.is_set():
            # 如果游戏没有结束，但是准备好了下一步，则清除相关事件并检查进程。
            self.isReadyForNextStep.clear()
            self.check_process()

        print('Result before returning:', self.transaction['result'])

        result = self.transaction['result']
        result_str = str(result) if result is not None else None

        # 定义下一个状态。
        state = {
            'player_race': self.player_race,
            'opposite_race': self.args.Human_race,  # 对手种族
            'map_name': self.map_name,
            'information': self.transaction['information'],
            'action_executed': self.transaction['action_executed'],
            'action_failures': self.transaction['action_failures'],
            'process_data':self.transaction['process_data']
        }

        # 确保每个状态的部分都是可序列化的。
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

def protoss_agent_vs_human(transaction, lock, map_name, isReadyForNextStep, game_end_event, done_event, args):
    # 计算log的基础目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    opposite_race = args.Human_race
    # 根据agent_type决定保存路径
    base_dir = os.path.join(current_dir, '..', '..', 'log', "vshuman", f'{args.agent[0].agent_type}_log')

    # 创建replay的文件夹
    replay_folder = os.path.join(base_dir, f"game_{args.current_time}")

    # 如果目录不存在，创建它
    if not os.path.exists(replay_folder):
        os.makedirs(replay_folder)

    map = map_name

    temp_replay_path = f'{replay_folder}/{args.current_time}_{map}_player_race_PROTOSS_VS_Human_{opposite_race}_temp.SC2Replay'

    # 运行游戏
    result = run_game(maps.get(map),
                      [Human(map_race(opposite_race)),
                       Bot(Race.Protoss, Protoss_Bot(transaction, lock, isReadyForNextStep))],
                      realtime=False,
                      save_replay_as=temp_replay_path)
    # 优化result
    formatted_result = format_result_for_filename(result)
    # 游戏结束后重命名重放文件

    final_replay_path = f'{replay_folder}/{args.current_time}_{map}_player_race_PROTOSS_VS_Human_{opposite_race}_{formatted_result}.SC2Replay'
    os.rename(temp_replay_path, final_replay_path)
    with lock:
        transaction['done'] = True
        transaction['result'] = result
    done_event.set()  # Set done_event when the game is over
    game_end_event.set()  # Set game_end_event when the game is over

def format_result_for_filename(result):
    # 将结果转换为字符串，并移除不允许的字符
    result_str = str(result)
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        result_str = result_str.replace(char, '')
    return result_str