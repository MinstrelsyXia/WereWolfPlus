import json
import threading
from threading import Thread
import time
from box import Box
import yaml
import os
from agent_manager.agents.streetfight3_agent.agent import Robot,KEN_GREEN, KEN_RED
from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward

class StreetFight3Agent(object):

    def __init__(self, config, args,**kwargs):
        self.args = args
        self.logger = self.args.logger
        self.prompt_constructor = config.prompt
        self.model = config.llm_model

        self.prompt_constructor_opp = config.get('prompt_opp')
        self.model_opp = config.get('llm_model_opp')
        print("====")
        self.render = True
        self.splash_screen = False
        self.save_game = True
        self.characters = ["Ken", "Ken"]
        self.super_arts = [3, 3]
        self.outfits = [1, 3]
        self.frame_shape = [0, 0, 0]
        self.seed = 42
        self.observation = None
        self.info = None
        self.player_1 = Robot(
            action_space=None,
            character="Ken",
            side=0,
            character_color=KEN_RED,
            ennemy_color=KEN_GREEN,
            only_punch=os.getenv("TEST_MODE", False),
            model=self.model,
            prompt_templete=self.prompt_constructor,
            player_nb=1,
            weave_prj_name=args.eval.weave_prj_name,
            logger=self.logger
        )
        self.player_2 = Robot(
            action_space=None,
            character="Ken",
            side=1,
            character_color=KEN_GREEN,
            ennemy_color=KEN_RED,
            sleepy=os.getenv("TEST_MODE", False),
            model=self.model_opp,
            prompt_templete=self.prompt_constructor_opp,
            player_nb=2,
            weave_prj_name=args.eval.weave_prj_name,
            logger=self.logger
        )
        self.actions = {
            "agent_0": 0,
            "agent_1": 0,
        }
        self.reward = 0.0
        self.asy_running = args.game.asynch_mode
        self.player1_running = True
        self.player2_running = True
        # print("====self.asy_running===",self.asy_running)
        self.generate_times = 1
        self.grounding_errors = 0
        self.opp_generate_times = 1
        self.opp_grounding_errors = 0

        ## trajectory
        self.trajectory: Trajectory = []
        self.cur_time_step = 0

        self.logger.info("=" * 5 + f"StreetFight3Agent Init Successfully!: " + "=" * 5)

    def plan_act(self):

        # Observe the environment
        self.player_1.observe(self.observation, self.actions, self.reward)
        # Plan
        self.player_1.plan()
        # Act
        self.actions["agent_0"] = self.player_1.act()

        # Observe the environment
        self.player_2.observe(self.observation, self.actions, -self.reward)
        # Plan
        self.player_2.plan()
        # Act
        self.actions["agent_1"] = self.player_2.act()

    def step(self, observation,reward=0.0):
        """

        :param observation:
        :return:
        """

        self.observation = observation
        self.reward += reward
        if not self.asy_running:
            # print("not asy_running")
            self.plan_act()

        actions = self.actions.copy()
        # print("==========================self.actions==========================")
        # print("self.actions", self.actions)
        # print("==========================self.actions==========================")

        if "agent_0" not in actions:
            actions["agent_0"] = 0
        if "agent_1" not in actions:
            actions["agent_1"] = 0
        # print("========")
        # print("actions:",self.actions)
        # print("========")

        return actions

    def start_player_planAndAct(self):
        print("asy_running: True")
        player1_thread = PlanAndActPlayer1(game=self)
        player1_thread.start()
        player2_thread = PlanAndActPlayer2(game=self)
        player2_thread.start()

    def set_trajectory_reward(self,env,role,score):
        reward = set_reward(env,role,score)
        if role=="player":
            self.player_1.trajectory.append(reward)
        else:
            self.player_2.trajectory.append(reward)

    def save_trajectory(self,role,save_path,name):
        item_id=name.split("/")[-1].strip()+"_"+role
        output_path = os.path.join(save_path,item_id+".json")
        if role=="player":
            temp_traj=self.player_1.trajectory
        else:
            temp_traj=self.player_2.trajectory
        react_data={"item_id":item_id,"conversation":temp_traj[:-1],"rewards":temp_traj[-1]}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)

class PlanAndAct(Thread):
    def __init__(self, game):
        self.running = True
        self.game = game
        Thread.__init__(self, daemon=True)
class PlanAndActPlayer1(PlanAndAct):
    def run(self) -> None:
        while self.running:
            if "agent_0" not in self.game.actions:
                # Plan
                self.game.player_1.plan()
                # Act
                self.game.actions["agent_0"] = self.game.player_1.act()
                # print(" ays player 1 =====", self.game.actions["agent_0"])
                # Observe the environment
                self.game.player_1.observe(self.game.observation, self.game.actions, self.game.reward)

class PlanAndActPlayer2(PlanAndAct):
    def run(self) -> None:
        while self.running:
            if "agent_1" not in self.game.actions:
                # Plan
                self.game.player_2.plan()
                # Act
                self.game.actions["agent_1"] = self.game.player_2.act()
                # print(" ays player 2 =====", self.game.actions["agent_1"])
                                # Observe the environment
                self.game.player_2.observe(self.game.observation, self.game.actions, -self.game.reward)


