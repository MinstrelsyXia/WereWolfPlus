import gym
import multiprocessing
from gym import spaces
import time
from sc2 import maps
from sc2.player import Bot
from sc2.main import run_game
import os
import datetime
from games.starcraft2.sc2_config import map_race
from games.starcraft2.bot.Protoss_bot import Protoss_Bot
from games.starcraft2.bot.Zerg_bot import Zerg_Bot
from games.starcraft2.utils.action_info import ActionDescriptions
from typing import (
    Optional,
)
from games.starcraft2.sc2_config import LADDER_MAP_2023
def clean_filename(filename):
    """ 清理文件名中的特殊字符 """
    return filename.replace('<', '').replace('>', '').replace(':', '').replace('"', '').replace('/', '').replace('\\', '').replace('|', '').replace('?', '').replace('*', '')


class AgentVSAgentStarcraftEnv(gym.Env):
    def __init__(self, args):
        # 初始化
        self.args = args
        self.logger = self.args.logger
        if "_VS_" in args.game.players:
            players = args.game.players.split("_VS_")
        assert len(players) == 2
        self.player1_race = players[0]
        self.player2_race = players[1]

        self.map_name = args.game.game_map
        assert self.map_name in LADDER_MAP_2023

        # 为每个智能体创建独立的锁和transaction
        self.lock1 = multiprocessing.Manager().Lock()
        self.lock2 = multiprocessing.Manager().Lock()

        self.transaction1 = multiprocessing.Manager().dict()
        self.transaction2 = multiprocessing.Manager().dict()

        for transaction in [self.transaction1, self.transaction2]:
            transaction.update(
                {'information': [], 'reward': 0, 'action': None,
                 'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
                 'action_executed': [], 'action_failures': [], })
        # 同步机制
        self.isReadyForNextStep1 = multiprocessing.Event()
        self.isReadyForNextStep2 = multiprocessing.Event()
        # 游戏结束标志
        self.game_end_event1 = multiprocessing.Event()
        self.game_end_event2 = multiprocessing.Event()
        # 游戏结束标志
        self.game_over1 = multiprocessing.Value('b', False)
        self.game_over2 = multiprocessing.Value('b', False)
        # 重置标志
        self.done_event1 = multiprocessing.Event()
        self.done_event2 = multiprocessing.Event()

        # 启动智能体进程
        self.p1 = None
        self.p2 = None

        # 设置动作空间
        self.action1_space = spaces.Discrete(self.calculate_action_space(self.player1_race))
        self.action2_space = spaces.Discrete(self.calculate_action_space(self.player2_race))

        # 设置观测空间
        self.observation1_space = spaces.Dict({
            "player_race": spaces.Text(max_length=20),  # Terran,Protoss, Zerg, Random
            "opposite_race": spaces.Text(max_length=20),  # Terran,Protoss, Zerg, Random
            "map_name": spaces.Text(max_length=20),  # Map name
            "information": spaces.Dict({
                "observation1": gym.spaces.Discrete(10),
                "observation2": gym.spaces.Box(low=0, high=1, shape=(3, 3)),
            }),  # Information about the game state
        })
        self.observation2_space = spaces.Dict({
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

    def check_process(self, agent_num, reset=False):
        # 根据传入的agent_num决定是操作哪个智能体的变量和进程
        p = self.p1 if agent_num == 1 else self.p2
        game_end_event = self.game_end_event1 if agent_num == 1 else self.game_end_event2
        lock = self.lock1 if agent_num == 1 else self.lock2
        transaction = self.transaction1 if agent_num == 1 else self.transaction2
        isReadyForNextStep = self.isReadyForNextStep1 if agent_num == 1 else self.isReadyForNextStep2
        done_event = self.done_event1 if agent_num == 1 else self.done_event2

        # 如果对应的进程存在且仍在运行
        if p is not None:
            if p.is_alive():
                # 如果游戏没有结束则直接返回，不重新启动进程
                if not game_end_event.is_set():
                    return
                # 如果游戏结束，则终止进程
                p.terminate()
            # 等待进程结束
            p.join()

        # 如果需要重置环境
        if reset:
            # 更新transaction字典，清空或重置一些字段
            self.transaction1.update(
                {'information': [], 'reward': 0, 'action': None,
                 'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
                 'action_executed': [], 'action_failures': [], })
            self.transaction2.update(
                {'information': [], 'reward': 0, 'action': None,
                 'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
                 'action_executed': [], 'action_failures': [], })

            # 清除游戏结束的事件标记
            self.game_end_event1.clear()
            self.game_end_event2.clear()

            # 启动新的agent_vs_agent进程
            p = multiprocessing.Process(target=agent_vs_agent, args=(
                self.transaction1, self.transaction2, self.lock1, self.lock2,
                self.isReadyForNextStep1, self.isReadyForNextStep2,
                self.game_end_event1, self.game_end_event2,
                self.done_event1, self.done_event2,
                self.map_name, self.player1_race,self.player2_race,self.args.game.asynch_mode,self.args.eval.output_path
            ))
            p.start()

            # 更新p1和p2
            self.p1 = p
            self.p2 = p
            self.logger.info("=" * 5 + f"game run asynch mode: {self.args.game.asynch_mode}" + "=" * 5)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.check_process(1, reset=True)
        self.check_process(2, reset=True)

        # 重置游戏结束标志
        self.game_end_event1.clear()
        self.game_end_event2.clear()

        state1 = {
            'player_race': self.player1_race,  # 玩家种族
            'opposite_race': self.player2_race,  # 对手种族
            'map_name': self.map_name,  # 地图名称
            'information': self.transaction1['information'],  # 游戏信息
            'action_executed': self.transaction1['action_executed'],  # 执行过的动作列表
            'action_failures': self.transaction1['action_failures'],  # 失败的动作列表
            'process_data': None

        }

        state2 = {
            'player_race': self.player2_race,  # 玩家种族
            'opposite_race': self.player1_race,  # 对手种族
            'map_name': self.map_name,  # 地图名称
            'information': self.transaction2['information'],  # 游戏信息
            'action_executed': self.transaction2['action_executed'],  # 执行过的动作列表
            'action_failures': self.transaction2['action_failures'],  # 失败的动作列表
            'process_data': None

        }

        return state1, state2

    def step(self, actions):
        # 这里假设actions是一个包含两个元素的列表，actions[0]是agent1的动作，actions[1]是agent2的动作
        actions_dict = {1: actions[0], 2: actions[1]}

        # 对每个智能体进行动作处理
        for agent_num, action in actions_dict.items():
            transaction = self.transaction1 if agent_num == 1 else self.transaction2
            lock = self.lock1 if agent_num == 1 else self.lock2

            with lock:
                if isinstance(action, tuple) and len(action) == 4:
                    action_, command, command_flag,match_data = action
                    transaction['action'] = action_
                    transaction['command'] = command
                    transaction['output_command_flag'] = command_flag
                else:
                    transaction['action'] = action
                    transaction['command'] = None
                    transaction['output_command_flag'] = False

        states, rewards, dones, infos = {}, {}, {}, {}

        for agent_num in [1, 2]:
            done_event = self.done_event1 if agent_num == 1 else self.done_event2
            isReadyForNextStep = self.isReadyForNextStep1 if agent_num == 1 else self.isReadyForNextStep2
            transaction = self.transaction1 if agent_num == 1 else self.transaction2
            player_race = self.player1_race if agent_num == 1 else self.player2_race
            opposite_race = self.player2_race if agent_num == 1 else self.player1_race
            game_over = self.game_over1 if agent_num == 1 else self.game_over2
            while not (done_event.is_set() or isReadyForNextStep.is_set()):
                time.sleep(0.0001)

            if done_event.is_set():
                done_event.clear()
                isReadyForNextStep.clear()
                game_over.value = True
                if transaction['result'].name == 'Victory':
                    transaction['reward'] += 50
            elif isReadyForNextStep.is_set():
                isReadyForNextStep.clear()
                self.check_process(agent_num)

            result = transaction['result']
            result_str = str(result) if result is not None else None

            states[agent_num] = {
                'player_race': player_race,
                'opposite_race': opposite_race,
                'map_name': self.map_name,
                'information': transaction['information'],
                'action_executed': transaction['action_executed'],
                'action_failures': transaction['action_failures'],
                'process_data': transaction['process_data']
            }

            for key, value in states[agent_num].items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if not isinstance(sub_value, (int, float, str, bool, type(None))):
                            value[sub_key] = str(sub_value)
                    states[agent_num][key] = value

            rewards[agent_num] = transaction['reward']
            dones[agent_num] = transaction['done']
            infos[agent_num] = result_str
        # print("states:", states)
        # print(type(states))
        #
        # print("rewards:", rewards)
        # print(type(rewards))
        # print("dones:", dones)
        # print(type(dones))
        print("===========================================infos:", infos)
        print("===========================================infos[1]:", infos[1])
        print("===========================================infos[2]:", infos[2])
        print("===========================================dones:", dones)
        print("===========================================dones[1]:", dones[1])
        print("===========================================dones[2]:", dones[2])
        # print(type(infos))

        return states, rewards, dones, infos, None

    def render(self, mode='human'):
        return None

    def close(self):
        return None


def agent_vs_agent(transaction1, transaction2, lock1, lock2,
                   isReadyForNextStep1, isReadyForNextStep2,
                   game_end_event1, game_end_event2,
                   done_event1, done_event2,
                   map_name, player_race, opp_player_race,asy_mode, save_path):
    # 创建replay的文件夹
    replay_folder = os.path.join(save_path)

    # 如果目录不存在，创建它
    if not os.path.exists(replay_folder):
        os.makedirs(replay_folder)

    if player_race == "Protoss":
        agent1 = Protoss_Bot
    elif player_race == "Zerg":
        agent1 = Zerg_Bot
    else:
        raise ValueError("Now we only support Protoss and Zerg")
    if opp_player_race == "Protoss":
        agent2 = Protoss_Bot
    elif opp_player_race == "Zerg":
        agent2 = Zerg_Bot
    else:
        raise ValueError("Now we only support Protoss and Zerg")
    cur_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # 运行游戏并获取结果
    result = run_game(maps.get(map_name),
                      [Bot(map_race(player_race), agent1(transaction1, lock1, isReadyForNextStep1)),
                       Bot(map_race(opp_player_race), agent2(transaction2, lock2, isReadyForNextStep2))],
                      realtime=asy_mode,
                      save_replay_as=f'{replay_folder}/{map_name}_player_{player_race}_VS_{opp_player_race}_{cur_time}.SC2Replay')
    print("===============================realtime:".format(asy_mode))
    with lock1:
        transaction1['done'] = True
        transaction1['result'] = result[0]

    with lock2:
        transaction2['done'] = True
        transaction2['result'] = result[1]

    done_event1.set()  # Set done_event for agent1 when the game is over
    done_event2.set()  # Set done_event for agent2 when the game is over

    game_end_event1.set()  # Set game_end_event for agent1 when the game is over
    game_end_event2.set()  # Set game_end_event for agent2 when the game is over
