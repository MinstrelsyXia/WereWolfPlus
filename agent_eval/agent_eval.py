import os.path
import traceback
from datetime import datetime

import wandb
import weave
from tqdm import tqdm
from utils import utils
import json
import numpy as np
from agent_manager.agents.welfare_diplomacy.data_types import (
    AgentResponse,
    AgentParams,
    MessageSummaryHistory,
    PromptAblation,
)
from games.welfare_diplomacy.diplomacy import GamePhaseData, Message  # 仅用于wd游戏
import shutil
from tasks_config import WANDB_ENTITY,WEAVE_OPEN

class AgentEval(object):

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.history_tracker = HistoryTracker(args)  # Initialize history_tracker
        self.env = self.init_env(args)  # Initialize environment
        self.agent_list = self.init_agent(args)  # Initialize agent
        print(self.args.game.get('game_turn'))
        if self.args.game.get('game_turn'):
            assert self.args.game.game_turn>=1
        wandb.init(project=self.args.eval.weave_prj_name, entity=WANDB_ENTITY, name="match_" + str(self.args.match_idx),reinit=True)

        if WEAVE_OPEN:
            if not ((self.args.game.game_name.__contains__("StreetFight3") and self.args.game.asynch_mode) or self.args.game.game_name.__contains__("Civ")):
                self.logger.info("=" * 5 + f"{self.args.eval.weave_prj_name} weave Init Successfully!: " + "=" * 5)
                weave.init(self.args.eval.weave_prj_name)
        self.logger.info("=" * 5 + f"AgentEval Init Successfully!: " + "=" * 5)

    def init_env(self, args):
        """
        Initialize game environment
        """
        env = utils.load_env(args)
        return env


    def init_agent(self, args):
        """
        Initialize all agents
        """
        agents =  utils.load_agent(args)
        for agent in agents:
            from agent_manager.agents.were_wolf.were_wolf_agent import WereWolfAgent
            if isinstance(agent, WereWolfAgent):
                agent.init_message_pool(self.env._get_messagePool())
        return agents
    

    def play(self):
        self.logger.info("=" * 5 + f"AgentEval play() beginning: " + "=" * 5)
        if self.args.game.game_name.__contains__("WereWolf"):
            self.were_wolf_play()
            return
        if self.args.game.game_name.__contains__("WelfareDiplomacy"):
            self.wd_play()
            return
        if self.args.game.game_name.__contains__("Civ"):
            self.civ_play()
            return
        if self.args.game.game_name.__contains__("Stratego"):
            self.stratego_play()
            return
        if self.args.game.game_name.__contains__("StreetFight3"):
            self.streetfight3_play()
            return
        if self.args.game.game_name.__contains__("Starcraft"):
            self.sc_play()
            return

    def sc_play(self):
        if len(self.agent_list) == 1:
            observation, _ = self.env.reset()
            done = False
            
            match_info = {
                "agents": self.args.agent[0].agent_name,
                "observations": [],
                "actions": [],
                "analysis": [],  
            }
            match_iter = 0
            time_step = 1
            while not done:
                self.env._time_step = time_step
                action = self.agent_list[0].step(observation)
                print(f"===============action==========={action}")

                observation, reward, done, result, info = self.env.step(action)
                time_step += 1

                # Record match process
                match_info["observations"].append(observation)
                match_info["actions"].append(action)

                if done:
                    if isinstance(action, tuple) and len(action) == 4:
                        _, _, _, match_data = action
                    else:
                        match_data = {"match_grounding_acc": 0.0}
                    if result is not None:
                        if result == "Result.Victory":
                            match_data['result'] = 1
                        elif result == "Result.Defeat":
                            match_data['result'] = 2
                        elif result == "Result.Tie":
                            match_data['result'] = 3
                        else:
                            match_data['result'] = 4
                    else:
                        match_data['result'] = 4
                    # Save trajectory
                    self.agent_list[0].set_trajectory_reward("Starcraft2", "player", match_data['result'])
                    self.agent_list[0].save_trajectory("player", self.args.eval.output_path, self.args.eval.weave_prj_name + "_" + str(self.args.match_idx))
                    match_data['observation'] = observation
                    self.history_tracker.extract_match_info(match_data, done)
                    self.history_tracker.save_result()
                    break
                match_iter += 1
            # self.history_tracker.add_match(match_info)

        elif len(self.agent_list) == 2:
            if self.env:
                observations = self.env.reset()
                agent1_obs, agent2_obs = observations
                dones = {'1': False, '2': False}

                # Token not considered
                match_info = {
                    "agents": [agent.agent_name for agent in self.agent_list],
                    "observations": [],
                    "actions": [],
                    "analysis": [],  # Store analysis results
                }

                while not all(dones.values()):
                    action1 = self.agent_list[0].step(agent1_obs)
                    action2 = self.agent_list[1].step(agent2_obs, wandb_flag=False)

                    new_observations, rewards, new_dones, results, infos = self.env.step([action1, action2])

                    agent1_obs = new_observations[1]
                    agent2_obs = new_observations[2]
                    dones = new_dones

                    # Record match process
                    match_info["observations"].append((agent1_obs, agent2_obs))
                    match_info["actions"].append((action1, action2))

                    if all(dones.values()):
                        if isinstance(action1, tuple) and len(action1) == 4:
                            _, _, _, match_data = action1
                        else:
                            match_data = {"match_grounding_acc": 0.0}
                        result = results[1]
                        done = dones[1]
                        if result is not None:
                            if result == "Result.Victory":
                                match_data['result'] = 1
                            elif result == "Result.Defeat":
                                match_data['result'] = 2
                            elif result == "Result.Tie":
                                match_data['result'] = 3
                            else:
                                match_data['result'] = 4
                        else:
                            match_data['result'] = 4
                        match_data['observation'] = agent1_obs
                        self.history_tracker.extract_match_info(match_data, done)
                        self.history_tracker.save_result()
                        break

    def were_wolf_play(self):
        self.env.reset()
        # Create an agent for eneach player
        player_agent_dict = {}
        for name, player in self.env.game.state.players.items():
            for agent in self.agent_list:
                if agent.agent_role == player.role and agent.agent_name is None:
                    player.model = agent.model.model_name
                    print("====player.model==", player.model)
                    player_agent_dict[name] = agent
                    agent.agent_name = name
                    break
        observations, _ = self.env.step('')  # 第一步，初始化
        # Provide action based on observation
        while not self.env.game.state.winner:
            # 主要是这里的函数，分别调用 were_wolf_agent.py 和 werewolf_env.py
            try:
                action_agent = player_agent_dict[observations['player_name']] # 现在正在执行的玩家
            except KeyError as e:
                self.logger.error(f"KeyError: {e}. Player {observations['player_name']} not found in player_agent_dict {player_agent_dict.keys()}.")
                continue
            action = action_agent.step(observations) # player执行动作
            observations, winner = self.env.step(action, observations) # 环境更新，是更新observation和workflow
        tqdm.write("Game is complete!")
        if self.env.game.state.winner:
            info = {}
            if self.env.game.state.winner == "Werewolves":
                info['player'] = 0
            else:
                info['player'] = 1
            # Save trajectory
            for name, player_agent in player_agent_dict.items():
                if name not in self.env.game.this_round.players:
                    player_agent.set_trajectory_reward("WereWolf", player_agent.agent_role, 0)
                else:
                    if player_agent.agent_role != "Werewolf":
                        player_agent.set_trajectory_reward("WereWolf", player_agent.agent_role, info['player'])
                    else:
                        player_agent.set_trajectory_reward("WereWolf", player_agent.agent_role, 1 - info['player'])

                player_agent.save_trajectory(player_agent.agent_role, name, self.args.eval.output_path, self.args.eval.weave_prj_name + "_" + str(self.args.match_idx))

            info['match_turns'] = self.env.game.current_round_num + 1
            self.history_tracker.extract_match_info(info, True)
            self.history_tracker.save_result()
        log_directory = self.env.log_directory()
        self.env.save_game(self.env.state, self.env.game.logs, log_directory)
        print(self.env.game.state.winner)

    def streetfight3_play(self):
        observation, info = self.env.reset(seed=42)
        reward = 0.0
        self.agent_list[0].player_1.observe(observation, {}, reward)
        self.agent_list[0].player_2.observe(observation, {}, reward)
        # print("========self.agent_list[0].asy_running:",self.agent_list[0].asy_running)
        if self.agent_list[0].asy_running:
            self.agent_list[0].start_player_planAndAct()
        sum_turns = 0
        max_timer = 0
        p1_health = 10000
        p2_health = 10000
        player1_last_health = 160
        player2_last_health = 160
        player1_cur_skill = {'skill': [], 'cmd': None, 'health': 0, 'his_skill': [], 'his_health': []}
        player2_cur_skill = {'skill': [], 'cmd': None, 'health': 0, 'his_skill': [], 'his_health': []}
        player1_success_hit = 0
        player1_super_success_hit = 0
        player1_super_cmd_num = 0
        player1_cmd_num = 0
        player2_success_hit = 0
        player2_cmd_num = 0
        player2_super_success_hit = 0
        player2_super_cmd_num = 0
        time_step = 1
        while True:
            self.env._time_step = time_step
            # (Optional) Environment rendering
            self.env.render()
            actions = self.agent_list[0].step(observation, reward)

            # Environment stepping
            observation, reward, terminated, truncated, info = self.env.step(actions)
            time_step += 1
            if "agent_0" in self.agent_list[0].actions:
                del self.agent_list[0].actions["agent_0"]
            if "agent_1" in self.agent_list[0].actions:
                del self.agent_list[0].actions["agent_1"]
            sum_turns += 1

            from agent_manager.agents.streetfight3_agent.agent.config import META_INSTRUCTIONS
            move_list = [move.lower() for move in META_INSTRUCTIONS]

            p1_health = int(observation['P1']['health']) if p1_health > observation['P1']['health'] else p1_health
            p2_health = int(observation['P2']['health']) if p2_health > observation['P2']['health'] else p2_health

            if actions['agent_0'] == 0:
                player1_cur_skill['health'] = player2_last_health - p2_health if player2_last_health > 0 else 0
                if len(player1_cur_skill['skill']) > 0:
                    for skill in self.agent_list[0].player_1.next_skills:
                        flag = False
                        for k, v in skill.items():
                            if player1_cur_skill['skill'] == v:
                                player1_cur_skill['cmd'] = k
                                # print(
                                #     f"============{player1_cur_skill['skill']} | {player1_cur_skill['cmd']} |{player1_cur_skill['health']} ")
                                k = k.lower()
                                # if k in move_list and (k not in['move closer','move away','jump closer','jump away']):
                                if k in move_list and (k not in['move away','jump away']):
                                    player1_cmd_num+=1
                                    if player1_cur_skill['health']>0:
                                        player1_success_hit+=1
                                if k in ['megafireball','super attack 2','super attack 3','super attack 4']:
                                    player1_super_cmd_num+=1
                                    if player1_cur_skill['health'] > 0:
                                        player1_super_success_hit+=1
                                flag=True
                        if flag:
                            break

                player1_cur_skill['skill'] = []
                player1_cur_skill['cmd'] = None

                player1_last_health = p1_health
                player2_last_health = p2_health

            else:
                player1_cur_skill['skill'].append(actions['agent_0'])
            if actions['agent_1'] == 0:
                if len(player2_cur_skill['skill'])>0:
                    for skill in self.agent_list[0].player_2.next_skills:
                        flag=False
                        for k,v in skill.items():
                            if player2_cur_skill['skill']==v:
                                player2_cur_skill['cmd']=k
                                # print(
                                #     f"======333333333333======{player2_cur_skill['skill']} | {player2_cur_skill['cmd']} |{player2_cur_skill['health']} ")
                                flag=True
                        if flag:
                            break
                player2_cur_skill['skill'] = []
                player2_cur_skill['health'] = player2_last_health - p2_health if player2_last_health > 0 else 0
                player2_last_health = p2_health
            else:
                player2_cur_skill['skill'].append(actions['agent_1'])

            player1_cur_skill['his_skill'].append(actions['agent_0'])
            player1_cur_skill['his_health'].append(p1_health)

            player2_cur_skill['his_skill'].append(actions['agent_1'])
            player2_cur_skill['his_health'].append(p2_health)
            # print(f"==========player_1========={self.agent_list[0].player_1.next_skills}")
            # print(f"==========player_2========={self.agent_list[0].player_2.next_skills}")

            AHR=player1_success_hit/player1_cmd_num if player1_cmd_num>0 else 0
            SMHR=player1_super_success_hit/player1_super_cmd_num if player1_super_cmd_num>0 else 0
            HCR=(160-p1_health)/(99-observation["timer"][0]) if (99-observation["timer"][0])>0 else 0
            # print(f"=================AHR:{AHR}  | SMHR:{SMHR} | HCR:{HCR}")
            wandb.log({"player_health": p1_health, "opp_player_health": p2_health,
                       "AHR":AHR,"SMHR":SMHR,"HCR":HCR},step=time_step)

            grounding_acc = 1 - self.agent_list[0].player_1.grounding_errors / self.agent_list[
                0].player_1.generate_times
            opp_grounding_acc = 1 - self.agent_list[0].player_2.grounding_errors / self.agent_list[
                0].player_2.generate_times
            wandb.log({"grounding_acc": grounding_acc, "opp_grounding_errors_rate": opp_grounding_acc},step=time_step)
            p1_wins = observation["P1"]["wins"][0]
            p2_wins = observation["P2"]["wins"][0]
            timer = 99-observation["timer"][0]
            max_timer=timer if max_timer<timer else max_timer

            # print(f"===================match_time_use:  {max_timer} ")
            if p1_wins == 1 or p2_wins == 1:
                self.agent_list[0].player1_running = False
                self.agent_list[0].player2_running = False
                match_info={}
                if p1_wins == 1 :
                    match_info['player']=1
                    match_info['opp_player']=0
                elif p2_wins == 1:
                    match_info['player'] = 0
                    match_info['opp_player'] = 1
                else:
                    match_info['player'] = 0
                    match_info['opp_player'] = 0
                ## save trajectory
                self.agent_list[0].set_trajectory_reward("Streetfight3", "player", match_info['player'])
                self.agent_list[0].set_trajectory_reward("Streetfight3", "opp_player", match_info['opp_player'])
                self.agent_list[0].save_trajectory("player",self.args.eval.output_path, self.args.eval.weave_prj_name +"_"+str(self.args.match_idx))
                self.agent_list[0].save_trajectory("opp_player",self.args.eval.output_path, self.args.eval.weave_prj_name +"_"+str(self.args.match_idx))

                match_info['match_turns']=sum_turns
                match_info['match_time_use']=int(max_timer)
                # grounding_acc = 1 - self.agent_list[0].player_1.grounding_errors / self.agent_list[0].player_1.generate_times
                # opp_grounding_acc = 1 - self.agent_list[0].player_2.grounding_errors / self.agent_list[0].player_2.generate_times
                match_info['grounding_acc']=grounding_acc
                match_info['opp_grounding_acc']=opp_grounding_acc
                self.history_tracker.extract_match_info(match_info, True)
                self.history_tracker.save_result()
                break

        self.env.close()
        
    def wd_play(self):
        self.env.reset()
        time_step = 0
        message_summary_history = MessageSummaryHistory()
        if not self.env.game.no_press:
            for power_name in self.env.game.powers.keys():
                message_summary_history[power_name] = []
        # Convert the comma-separated strings to Enum members
        self.args.prompt_ablations="" # default value
        prompt_ablations = self.args.prompt_ablations.split(",")
        prompt_ablations = [
            PromptAblation[ablation.upper()]
            for ablation in prompt_ablations
            if ablation != ""
        ]
        # Uppercase the exploiter powers
        self.args.exploiter_prompt="" # default value
        self.args.exploiter_powers="" # default value
        exploiter_powers = self.args.exploiter_powers.split(",")
        exploiter_powers = [power.upper() for power in exploiter_powers if power != ""]

        # Instantiating base agents
        power_name_to_agent={}
        for agent in self.agent_list:
            power_name_to_agent[agent.agent_power_name.upper()]=agent
        if self.env.super_exploiter_powers:
            for power_name in self.env.super_exploiter_powers:
                power_name_to_agent[power_name].change_exploiter_agent()

        # Initialize global counters
        game_completion_error_traces: list[list[str]] = []
        game_num_completion_errors: int = 0
        game_tokens_prompt_sum: int = 0
        game_tokens_completion_sum: int = 0
        game_messages_public_ratio_list: list[float] = []
        game_message_similarity_list: list[float] = []
        # Log the initial state of the game
        rendered_with_orders = self.env.game.render(incl_abbrev=True)
        log_object = {
            "_progress/year_fractional": 0.0,
            "board/rendering_with_orders": wandb.Html(rendered_with_orders),
            "board/rendering_state": wandb.Html(rendered_with_orders),
        }
        for power in self.env.game.powers.values():
            short_name = power.name[:3]
            log_object[f"score/units/{short_name}"] = len(power.units)
            log_object[f"score/welfare/{short_name}"] = power.welfare_points
            log_object[f"score/centers/{short_name}"] = len(power.centers)
        welfare_list = [power.welfare_points for power in self.env.game.powers.values()]
        # log_object["welfare/hist"] = wandb.Histogram(welfare_list)

        wandb.log(log_object,step=time_step)

        self.logger.info(f"Starting game with map {self.args.game.game_map} and ending after {self.args.game.game_max_years} years with {self.args.game.game_max_message_rounds} message rounds per phase .")
        progress_bar_phase = tqdm(total=self.env.simulation_max_years * 3, desc="🔄️ Phases")
        while not self.env.game.is_game_done:
            time_step += 1
            self.env._time_step = time_step
            self.logger.info(f"🕰️  Beginning phase {self.env.game.get_current_phase()}")
            phase_orders_total_num = 0
            phase_orders_valid_num = 0
            valid_valid_order_ratios_list = []
            phase_num_valid_completions = 0
            phase_num_completion_errors = 0
            phase_message_total = 0
            # (power_name, message_round, agent_name, agent_response, invalid orders)
            phase_agent_response_history: list[
                tuple[str, int, str, AgentResponse, list[str]]
            ] = []
            turn_order = {}
            phase_completion_times_sec_list = []
            phase_prompt_tokens_list = []
            phase_completion_tokens_list = []
            phase_total_tokens_list = []
            phase_message_history: list[tuple(str, int, str, str, str)] = []

            # During Retreats, only 1 round of completions without press
            num_of_message_rounds = (
                1
                if self.env.game.no_press
                else self.args.game.game_max_message_rounds
                if self.env.game.phase_type != "R"
                else 1
            )
            num_completing_powers = (
                len(self.env.game.powers)
                if self.env.game.phase_type != "R"
                else len([power for power in self.env.game.powers.values() if power.retreats])
            )

            # Cache the list of possible orders for all locations
            possible_orders = self.env.game.get_all_possible_orders()

            progress_bar_messages = tqdm(
                total=num_of_message_rounds * num_completing_powers, desc="Messages"
            )
            for message_round in range(1, num_of_message_rounds + 1):
                # Randomize order of powers
                powers_items = list(self.env.game.powers.items())
                np.random.shuffle(powers_items)

                self.logger.info(f" Beginning message round {message_round}/{num_of_message_rounds}. Completion ordering: {', '.join([name for name, _ in powers_items])}")

                # power: Power
                for power_name, power in powers_items:
                    # # Skip no-press powers until final message round
                    # if (
                    #         power_name in no_press_powers
                    #         and message_round < num_of_message_rounds
                    # ):
                    #     continue

                    # On retreat phases, skip powers that have no retreats to make
                    if self.env.game.phase_type == "R" and not power.retreats:
                        continue

                    # Prompting the model for a response
                    agent = power_name_to_agent[power_name]
                    agent.set_time_step(time_step)
                    from agent_manager.agents.welfare_diplomacy.agents import AgentCompletionError
                    try:
                        observations=AgentParams(
                                power=power,
                                game=self.env.game,
                                message_summary_history=message_summary_history,
                                possible_orders=possible_orders,
                                current_message_round=message_round,
                                max_message_rounds=num_of_message_rounds,
                                final_game_year=self.env.final_game_year,
                                prompt_ablations=prompt_ablations,
                                exploiter_prompt=self.args.exploiter_prompt,
                                exploiter_powers=exploiter_powers,
                            )
                        agent_response: AgentResponse = agent.step(observations)
                    except AgentCompletionError as exc:
                        # If the agent fails to complete, we need to log the error and continue
                        phase_num_completion_errors += 1
                        game_num_completion_errors += 1
                        progress_bar_messages.update(1)
                        continue
                    if self.env.game.no_press:
                        assert not agent_response.messages, agent_response.messages
                    phase_completion_times_sec_list.append(
                        agent_response.completion_time_sec
                    )
                    phase_prompt_tokens_list.append(agent_response.prompt_tokens)
                    phase_completion_tokens_list.append(agent_response.completion_tokens)
                    phase_total_tokens_list.append(agent_response.total_tokens)
                    game_tokens_prompt_sum += agent_response.prompt_tokens
                    game_tokens_completion_sum += agent_response.completion_tokens
                    if self.env.game.phase_type == "R":
                        if len(agent_response.messages) > 0:
                            self.logger.Warning("No messages are allowed during retreats, clearing.")
                            agent_response.messages = {}
                    phase_num_valid_completions += 1
                    now = datetime.now()
                    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    agent_log_string = f"⚙️  {current_time} {power_name} {self.env.game.get_current_phase()} Round {message_round}: Agent {agent} took {agent_response.completion_time_sec:.2f}s to respond."
                    if isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled):
                        agent_log_string += f"\nReasoning: {agent_response.reasoning}\nOrders: {agent_response.orders}\nMessages: {agent_response.messages}"
                    utils.log_info(
                        self.logger,
                        agent_log_string,
                    )
                    # Check how many of the orders were valid
                    num_valid_orders = 0
                    invalid_orders = []
                    for order in agent_response.orders:
                        if "WAIVE" in order or "VOID" in order:
                            utils.log_warning(
                                self.logger,
                                f"Order '{order}' should not be generated by agent",
                            )
                            num_valid_orders += 1
                            invalid_orders.append(order)
                            continue
                        word = order.split()
                        if len(word) < 2:
                            utils.log_warning(
                                self.logger,
                                f"Order needs to be longer than 1 word",
                            )
                            num_valid_orders += 1
                            invalid_orders.append(order)
                            continue
                        location = word[1]
                        if (
                                location in possible_orders
                                and order in possible_orders[location]
                        ):
                            num_valid_orders += 1
                        else:
                            invalid_orders.append(order)
                    num_orders = len(agent_response.orders)
                    valid_order_ratio = (
                        num_valid_orders / num_orders if num_orders > 0 else None
                    )
                    valid_order_display_percent = (
                        valid_order_ratio * 100.0
                        if valid_order_ratio is not None
                        else np.NaN
                    )
                    if isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled):
                        utils.log_info(
                            self.logger,
                            f"✔️  {power_name} valid orders: {num_valid_orders}/{num_orders} = {valid_order_display_percent:.2f}%"
                            + (
                                f". Invalid Orders: {invalid_orders}"
                                if invalid_orders
                                else ""
                            ),
                        )
                    phase_orders_total_num += num_orders
                    phase_orders_valid_num += num_valid_orders
                    if valid_order_ratio is not None:
                        valid_valid_order_ratios_list.append(valid_order_ratio)

                    phase_agent_response_history.append(
                        (
                            power_name,
                            message_round,
                            str(agent),
                            agent_response,
                            invalid_orders,
                        )
                    )
                    valid_od = [od for od in agent_response.orders if od not in invalid_orders]
                    turn_order[power_name] = valid_od
                    # Set orders, clearing first due to multiple message rounds
                    self.env.game.set_orders(power_name, [])
                    try:
                        self.env.game.set_orders(power_name, agent_response.orders)
                    except Exception as exc:
                        # If the agent gave an invalid order, we need to log the error and continue
                        phase_num_completion_errors += 1
                        game_num_completion_errors += 1
                        exception_trace = "".join(
                            traceback.TracebackException.from_exception(exc).format()
                        )
                        utils.log_error(
                            self.logger,
                            f"🚨 {power_name} {self.env.game.get_current_phase()} Round {message_round}: Agent {wandb.config.agent_model} gave an invalid order ({phase_num_completion_errors} errors this phase). Skipping. Exception:\n{exception_trace}",
                        )
                        # Log the error to Weights & Biases
                        game_completion_error_traces.append(
                            [
                                self.env.game.get_current_phase(),
                                message_round,
                                power_name,
                                exception_trace,
                            ]
                        )
                        wandb.log(
                            {
                                "completion_error_table": wandb.Table(
                                    columns=["phase", "round", "power", "exception"],
                                    data=game_completion_error_traces,
                                )
                            }
                        )
                        progress_bar_messages.update(1)
                        continue

                    # Send messages
                    for recipient, message in agent_response.messages.items():
                        self.env.game.add_message(
                            Message(
                                sender=power_name,
                                recipient=recipient,
                                message=message,
                                phase=self.env.game.get_current_phase(),
                            )
                        )
                        phase_message_history.append(
                            (
                                self.env.game.get_current_phase(),
                                message_round,
                                power_name,
                                recipient,
                                message,
                            )
                        )
                        phase_message_total += 1

                    progress_bar_messages.update(1)
            # Render saved orders and current turn message history before processing
            rendered_with_orders = self.env.game.render(incl_abbrev=True)
            messages_table = wandb.Table(
                columns=["phase", "round", "sender", "recipient", "message"],
                data=[
                    [phase, round, sender, recipient, message]
                    for (phase, round, sender, recipient, message) in phase_message_history
                ],
            )
            # Save summaries of the message history
            if not self.env.game.no_press:
                for power_name, power in tqdm(
                        self.env.game.powers.items(), desc="✍️ Summarizing messages"
                ):
                    phase_message_summary = self.env.message_summarizer.summarize(
                        AgentParams(
                            game=self.env.game,
                            power=power,
                            final_game_year=self.env.final_game_year,
                            prompt_ablations=prompt_ablations,
                            exploiter_prompt=self.args.exploiter_prompt,
                            exploiter_powers=self.env.exploiter_powers,
                            # Unused params
                            message_summary_history={},
                            possible_orders={},
                            current_message_round=-1,
                            max_message_rounds=-1,
                        ),turn_orders=turn_order
                    )
                    message_summary_history[power_name].append(phase_message_summary)
                    game_tokens_prompt_sum += phase_message_summary.prompt_tokens
                    game_tokens_completion_sum += phase_message_summary.completion_tokens
                        # Advance the game simulation to the next phase
            try:
                allied_pairs= {}
                nego_pairs= {}
                betr_pairs= {}
                nego_nums = 0
                alli_nums = 0
                betr_nums = 0
                for power, summarys in message_summary_history.items():
                    for summ in summarys:
                        if summ.phase not in allied_pairs.keys():
                            allied_pairs[summ.phase] = []
                        if summ.phase not in nego_pairs.keys():
                            nego_pairs[summ.phase] = []
                        if summ.phase not in betr_pairs.keys():
                            betr_pairs[summ.phase] = []
                        # print(power, "===", type(summ.summary))
                        json_completion = summ.summary
                        start = json_completion.index("{")
                        end = json_completion.rindex("}") + 1  # +1 to include the } in the slice
                        json_completion = json_completion[start:end]
                        json_summ = json.loads(json_completion)
                        print("======", json_summ)
                        if "negotiations_powers" in json_summ.keys():
                            nego_nums += len(json_summ['negotiations_powers'])
                            for nego in json_summ['negotiations_powers']:
                                nego_one = set([power.lower(), nego.lower()])
                                if nego_one not in nego_pairs[summ.phase]:
                                    nego_pairs[summ.phase].append(nego_one)
                        if "allied_powers" in json_summ.keys():
                            alli_nums += len(json_summ['allied_powers'])
                            for alli in json_summ['allied_powers']:
                                alli_one = set([power.lower(), alli.lower()])
                                if alli_one not in allied_pairs[summ.phase]:
                                    allied_pairs[summ.phase].append(alli_one)
                        if "betrayal_powers" in json_summ.keys():
                            betr_nums += len(json_summ['betrayal_powers'])
                            for betr in json_summ['betrayal_powers']:
                                betr_one = set([power.lower(), betr.lower()])
                                if betr_one not in betr_pairs[summ.phase]:
                                    betr_pairs[summ.phase].append(betr_one)

                nego_nums=0
                for pha,val in nego_pairs.items():
                    nego_nums+=len(val)
                alli_nums = 0
                for pha, val in allied_pairs.items():
                    alli_nums += len(val)
                betr_nums = 0
                for pha, val in betr_pairs.items():
                    betr_nums += len(val)
                alli_succ_rate=alli_nums/nego_nums
                betr_rate=betr_nums/alli_nums
                log_sjnl = {
                    "social/allied_succ_rate": alli_succ_rate,
                    "social/betrayal_rate": betr_rate
                }
                wandb.log(log_sjnl,step=time_step)
                show_heads=["phase"]
                full_heads=[]
                all_powers=list(self.env.game.powers.keys())
                for i in range(len(all_powers)):
                    for j in range(i+1,len(all_powers)):
                        short_join_name=all_powers[i][:3]+"-"+all_powers[j][:3]
                        full_join_name=all_powers[i].lower()+"-"+all_powers[j].lower()
                        print(short_join_name)
                        show_heads.append(short_join_name)
                        full_heads.append(full_join_name)
                data_lines=[]
                for phase,alli in allied_pairs.items():
                    print(phase,"===",alli)
                    data_line = [phase]
                    for col in full_heads:
                        alli_pair_tmp=set(col.split("-"))
                        if alli_pair_tmp in alli:
                            data_line.append('T')
                        else:
                            data_line.append('F')
                    data_lines.append(data_line)
                #  calc sum of allied_power
                allied_power_sum = {}
                for i in range(len(full_heads)):
                    alli_pair = full_heads[i]
                    if alli_pair not in allied_power_sum.keys():
                        allied_power_sum[alli_pair] = 0
                    for line in data_lines:
                        if line[i + 1] == 'T':
                            allied_power_sum[alli_pair] += 1
                # calc avg of allied time per power
                avg_alli_num={}
                for k, v in power_name_to_agent.items():
                    power = k.lower()
                    if power not in avg_alli_num.keys():
                        avg_alli_num[power]=0
                    temp_alli=[]
                    temp_sum=0
                    for p,s in allied_power_sum.items():
                        if power in p:
                            if p not in temp_alli and s>0:
                                temp_alli.append(p)
                            temp_sum+=s
                    if len(temp_alli)>0:
                        avg_alli_num[power]=temp_sum/len(temp_alli)
                # calc avg for model
                model_alli={}

                avg_model_alli={}
                for k, v in power_name_to_agent.items():
                    power = k.lower()
                    model_name=v.model.model_name
                    if model_name not in model_alli.keys():
                        model_alli[model_name]=[]
                        avg_model_alli[model_name]=0
                    model_alli[model_name].append(power)
                for m,p in model_alli.items():
                    temp_p=[]
                    sum_all=0
                    for c in p:
                        if avg_alli_num[c]>0:
                            temp_p.append(c)
                            sum_all+=avg_alli_num[c]
                    if len(temp_p)>0:
                        avg_model_alli[m]= sum_all/len(temp_p)

                log_model_allied_avg={}
                for k,v in avg_model_alli.items():
                    log_model_allied_avg[f"score/model_allied_avg_turns/{k}"] = v
                wandb.log(log_model_allied_avg,step=time_step)
                wandb.log({
                    "social/allied_power": wandb.Table(columns=show_heads,data=data_lines,)
                            },step=time_step)
            except Exception as e:
                print(" wandb log error: ",e)
            self.env.game.process()
            phase: GamePhaseData = self.env.game.get_phase_history()[-1]
            # Check whether to end the game
            if int(self.env.game.phase.split()[1]) - 1900 > self.env.simulation_max_years:
                self.env.game._finish([])
            rendered_state = self.env.game.render(incl_abbrev=True)
            from agent_manager.agents.welfare_diplomacy.utils import get_phase_fractional_years_passed
            log_object = {
                "_progress/year_fractional": get_phase_fractional_years_passed(phase),
                "board/rendering_with_orders": wandb.Html(rendered_with_orders),
                "board/rendering_state": wandb.Html(rendered_state)
            }
            for power in self.env.game.powers.values():
                short_name = power.name[:3]
                if phase.name[-1] == "A" or phase.name[-1] == "R":
                    # Centers/welfare/units only change after adjustments or sometimes retreats
                    log_object[f"score/units/{short_name}"] = len(power.units)
                    log_object[f"score/welfare/{short_name}"] = power.welfare_points
                    log_object[f"score/centers/{short_name}"] = len(power.centers)

            wandb.log(log_object,step=time_step)
            # Update the progress bar based on how many turns have progressed (just counting M and A)
            new_phase_type = self.env.game.phase_type
            if new_phase_type == "M":
                # Any to M, update 1
                progress_bar_phase.update(1)
            elif new_phase_type == "A":
                # M or R to A, update 1
                progress_bar_phase.update(1)
            elif new_phase_type == "R":
                # Retreats, don't count it
                pass
            else:
                self.logger.info( f"Unknown phase type {new_phase_type}")
        for name,pagent in power_name_to_agent.items():
            pagent.save_trajectory( pagent.agent_power_name ,self.args.eval.output_path, self.args.eval.weave_prj_name +"_"+str(self.args.match_idx))

    def stratego_play(self):
        if self.env:
            obs = self.env.reset()
        sum_turns=0
        live_pieces_rate=0
        live_pieces_score=0
        critical_live_pieces_rate=0
        opp_live_pieces_rate = 0
        opp_live_pieces_score = 0
        opp_critical_live_pieces_rate = 0
        error_times=0
        time_step=1
        while True:
            try:
                self.env._time_step = time_step
                action = self.agent_list[0].step(self.env,obs)
                obs, rew, done, info = self.env.step(action_dict=action)
                time_step+=1
                if self.agent_list[0].current_player==1:
                    pieces_state=self.agent_list[0].get_live_pieces_state()
                    live_pieces_rate = pieces_state["live_pieces_rate"]
                    live_pieces_score = pieces_state["live_pieces_score"]
                    critical_live_pieces_rate = pieces_state["critical_live_pieces_rate"]
                    opp_live_pieces_rate = pieces_state["opp_live_pieces_rate"]
                    opp_live_pieces_score = pieces_state["opp_live_pieces_score"]
                    opp_critical_live_pieces_rate = pieces_state["opp_critical_live_pieces_rate"]
                    wandb.log(pieces_state,step=time_step)
                grounding_acc=1-self.agent_list[0].grounding_errors/self.agent_list[0].generate_times
                opp_grounding_acc=1-self.agent_list[0].opp_grounding_errors/self.agent_list[0].opp_generate_times
                sum_turns+=1
                wandb.log({"grounding_acc": grounding_acc, "opp_grounding_errors_rate": opp_grounding_acc},step=time_step)
                if done["__all__"]:
                    if rew.get(1,0)>rew.get(-1,0):
                        info['player']=1
                        info['opp_player']=0
                    elif rew.get(1,0)<rew.get(-1,0):
                        info['player'] = 0
                        info['opp_player'] = 1
                    else:
                        info['player'] = 0
                        info['opp_player'] = 0
                    ## save trajectory
                    self.agent_list[0].set_trajectory_reward("Stratego","player",info['player'])
                    self.agent_list[0].set_trajectory_reward("Stratego","opp_player",info['opp_player'])
                    self.agent_list[0].save_trajectory("player",self.args.eval.output_path, self.args.eval.weave_prj_name +"_"+str(self.args.match_idx))
                    self.agent_list[0].save_trajectory("opp_player",self.args.eval.output_path, self.args.eval.weave_prj_name +"_"+str(self.args.match_idx))

                    info['match_turns']=sum_turns
                    info['grounding_acc']=grounding_acc
                    info['opp_grounding_acc']=opp_grounding_acc
                    info['match_live_pieces_rate']=live_pieces_rate
                    info['match_live_pieces_score']=int(live_pieces_score)
                    info['match_critical_live_pieces_rate']=critical_live_pieces_rate
                    info['match_opp_live_pieces_rate'] = opp_live_pieces_rate
                    info['match_opp_live_pieces_score'] = int(opp_live_pieces_score)
                    info['match_opp_critical_live_pieces_rate'] = opp_critical_live_pieces_rate
                    self.history_tracker.extract_match_info(info, True)
                    self.history_tracker.save_result()
                    self.logger.info("=" * 5 + f"AgentEval stratego  end: " + "=" * 5)
                    break
            except Exception as e:
                print(e)
                self.logger.info("stratego_play wrong, {}".format(e))
                error_times+=1
                if error_times>10:
                    raise
                else:
                    continue

    def civ_play(self):
        observations, info = self.env.reset(10)
        done = False
        step = 0
        while not done :
            try:
                action = self.agent_list[0].act(observations, info)
                observations, reward, terminated, truncated, info = self.env.step(
                    action)
                done = terminated or truncated

                step += 1
                self.logger.info(utils.print_step(f'Step: {step}, Turn: {info["turn"]}, ' +
                           f'Reward: {reward}, Terminated: {terminated}, ' +
                           f'Truncated: {truncated}'))
                # game_results = self.env.env.get_game_results()
                # print('game results:', game_results)
                players, tags, turns, evaluations = self.env.env.evaluate_game()
                # print(f'Players: {players}, Tags: {tags}, Turns: {turns}, Evaluations: {evaluations}')
                grounding_acc = (1 - self.agent_list[0].invalid_action_num/self.agent_list[0].gen_action_num) if self.agent_list[0].gen_action_num!=0 else 1
                # print(f"=========grounding_acc:{grounding_acc}")

                # Initialize current step evaluation dictionary
                current_step_evaluation = {}

                if players is not None :
                    wandb.log({"grounding_acc": grounding_acc}, step=info["turn"])
                    for player_id, player_info in players.items():
                        player_name = player_info['name']
                        if player_name not in ["myagent","Myagent"] :
                            continue

                        # Get metrics
                        score = evaluations['score'][player_id][-1]
                        population = evaluations['population'][player_id][-1]
                        economics = evaluations['economics'][player_id][-1]
                        production = evaluations['production'][player_id][-1]
                        gold = evaluations['gold'][player_id][-1]
                        cities = evaluations['cities'][player_id][-1]
                        land_area = evaluations['land_area'][player_id][-1]
                        settled_area = evaluations['settled_area'][player_id][-1]
                        military_units = evaluations['military_units'][player_id][-1]
                        wonders = evaluations['wonders'][player_id][-1]
                        units_killed = evaluations['units_killed'][player_id][-1]
                        units_lost = evaluations['units_lost'][player_id][-1]
                        researched_techs = evaluations['researched_techs'][player_id][-1]
                        research_speed = evaluations['research_speed'][player_id][-1]

                        # Strategic planning ability
                        # EGR (Economic Growth Rate) economics / game_time
                        # egr = (economics/info["turn"]) if info["turn"]!=0 else 0
                        egr = economics
                        # CER (City Expansion Rate) cities / game_time
                        cer = (cities) if info["turn"] != 0 else 0
                        # TRP (Technology Research Progress) (researched_techs / total_techs) * 100 %
                        trp = researched_techs * research_speed
                        # LUR (Land Utilization Rate) (settled_area / land_area) * 100 %
                        lur = (settled_area / land_area) if land_area != 0 else 0
                        # MGR (Military Growth Rate) (military_units / game_time)
                        # mgr = (military_units/info["turn"]) if info["turn"]!=0 else 0
                        mgr = military_units
                        # WBR (Wonder Building Ratio) (wonders / total_wonders_available) * 100 %
                        # wbr = (wonders/info["turn"]) if info["turn"]!=0 else 0
                        wbr = wonders

                        # Calculate Economic Growth Index (EGI)
                        egi = economics + production + gold

                        # City Expansion Rate CER

                        # Calculate Urbanization Index (UI)
                        ui = (0.5 * (population / cities) + 0.3 * (settled_area / cities) + 0.2 * (land_area / cities)) if cities != 0 else 0

                        # Calculate Technology Progress Index (TPI)
                        tpi = researched_techs * research_speed

                        # Calculate Military Power Index (MPI)
                        mpi = military_units

                        # Calculate Combat Efficiency Ratio (CER)
                        cer = units_killed / (units_lost + 1)

                        # Calculate Technology Progress Index (TPI)
                        tpi = researched_techs * research_speed

                        # Calculate Cultural Development Index (CVP)
                        cvp = wonders

                        # Save calculation results
                        current_step_evaluation[player_name] = {
                            'Economics Growth Index (EGI)': egi,
                            'Urbanization Index (UI)': ui,
                            'Military Power Index (MPI)': mpi,
                            'Combat Efficiency Ratio (CER)': cer,
                            'Technology Progress Index (TPI)': tpi,
                            'Cultural Victory Potential (CVP)': cvp,
                            'score': score
                        }

                        # Save all basic metrics and calculated metrics
                        wandb_log_data = {
                            # Basic metrics
                            f'{player_name}/Population': population,
                            f'{player_name}/Economics': economics,
                            f'{player_name}/Production': production,
                            f'{player_name}/Gold': gold,
                            f'{player_name}/Cities': cities,
                            f'{player_name}/Land Area': land_area,
                            f'{player_name}/Settled Area': settled_area,
                            f'{player_name}/Military Units': military_units,
                            f'{player_name}/Wonders': wonders,
                            f'{player_name}/Units Killed': units_killed,
                            f'{player_name}/Units Lost': units_lost,
                            f'{player_name}/Researched Techs': researched_techs,
                            f'{player_name}/Research Speed': research_speed,

                            # Calculated metrics
                            f'{player_name}/EGI': egi,
                            f'{player_name}/UI': ui,
                            f'{player_name}/MPI': mpi,
                            f'{player_name}/CER': cer,
                            f'{player_name}/TPI': tpi,
                            f'{player_name}/CVP': cvp,

                            f'{player_name}/egr': egr,
                            f'{player_name}/cer': cer,
                            f'{player_name}/trp': trp,
                            f'{player_name}/lur': lur,
                            f'{player_name}/mgr': mgr,
                            f'{player_name}/wbr': wbr
                        }

                        # Log to wandb
                        wandb.log(wandb_log_data, step=info["turn"])

                # Test
                for player, metrics in current_step_evaluation.items():
                    print(f"Player: {player}")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value}")

            except Exception as e:
                print(repr(e))
                raise e
        self.agent_list[0].save_trajectory("player", self.args.eval.output_path, self.args.eval.weave_prj_name + "_" + str(self.args.match_idx))
        self.env.close()
        '''
        players, tags, turns, evaluations = self.env.env.evaluate_game()
        '''
        game_results = self.env.env.get_game_results()
        self.history_tracker.extract_match_info(game_results, done)
        self.history_tracker.save_result()
        # self.env.env.plot_game_scores()
        print('game results:', game_results)

    def reset(self):
        if self.env:
            self.env.reset()

    def summarize(self, path=""):
        """
        Summarize and save match history
        """
        if len(self.agent_list) == 2:
            filename = f"{path}/{self.agent_list[0].agent_name}_{self.agent_list[1].agent_name}_{self.agent_list[0].model.nick_name}_{self.agent_list[1].model.nick_name}.json"
        else:
            filename = f"{path}/{self.agent_list[0].agent_name}_{self.agent_list[0].model.nick_name}.json"

        self.history_tracker.save_as_json(filename)

    def save_result_to_jsonl(self, path):
        if os.path.exists(path):
            with open(path, 'a') as f:
                f.writelines(json.dumps(self.history_tracker.to_dict()) + '\n')
        else:
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")

    def save_match_result(self):
        self.history_tracker.save_as_json()

    def get_eval_result(self):
        return self.history_tracker.get_eval_name()[0]

    def close(self):
        wandb.finish()


class HistoryTracker:
    def __init__(self, args) -> None:
        self.args = args
        self.game_config = {}
        self.matches = []
        self.match_idx = args.match_idx
        self.agents = set()
        self.agents_config = []
        self.models_config = []
        self.process_info={'count':0,'pop_utilization':0.0}
        self.starcraft_eval_matrix={'PBR':0.0,'RUR':0.0,"APU":0.0,'TR':0.0,'grounding_acc':0.0}
        self.civ_eval_matrix={}
        self.stratego_eval_matrix={}
        self.streetfight3_eval_matrix={}
        self.werewolf_eval_matrix={}
        self.welfare_diplomacy_eval_matrix={}
        self.init_output_file()

    def init_output_file(self):
        folder_name=self.args.game.game_name
        if not self.args.get("eval"):
            return
        ret_path=os.path.join(self.args.eval.output_path,folder_name)
        if not os.path.exists(ret_path):
            os.makedirs(ret_path)
        self.match_info_path=os.path.join(ret_path,"match_info.txt")
        self.eval_matrix_path=os.path.join(ret_path,"eval_result.txt")
        # save config files
        eval_config_path=self.args.eval_config
        shutil.copy(eval_config_path,ret_path)
        if self.args.get("agent")[0].get("agent_model_config"):
            llm_config_path=os.path.join('configs/llm_configs/', self.args.get("agent")[0].get("agent_model_config"))
            if os.path.exists(llm_config_path):
                shutil.copy(llm_config_path,ret_path)
        print("==== 1. success: save the eval config files=====")


    def get_all_matches(self):
        return self.matches

    def set_game_config(self, config):
        self.game_config = config

    def add_agents_config(self, config):
        self.agents_config.append(config)

    def add_models_config(self, config):
        self.models_config.append(config)

    def extract_match_info(self,info,done):
        if self.args.game.game_name.__contains__("Starcraft"):
            process_data = info['observation']['process_data']
            self.process_info['count']+=1
            self.process_info['pop_utilization']=process_data['supply_used']/max(process_data['supply_cap']+self.process_info['pop_utilization'],1)
            # if process_data['supply_cap'] >= 196:
            if True:
                self.process_info['pop_cap_time'] = process_data['time']
                self.process_info['pop_cap_iter'] = process_data['iteration']
                self.process_info['pop_cap'] = process_data['supply_cap']
                self.process_info['pop_used'] = process_data['supply_used']
                self.process_info['pop_cap_spent_minerals'] = process_data['spent_minerals']
                self.process_info['pop_cap_spent_gas'] = process_data['spent_vespene']
                print("==========process_info",self.process_info)
            if done:
                self.process_info['game_over_time'] = process_data['time']
                self.process_info['game_over_iter'] = process_data['iteration']
                self.starcraft_eval_matrix['PBR']=self.process_info['pop_cap_iter']/max(self.process_info['game_over_iter'],1)
                self.starcraft_eval_matrix['RUR']=(self.process_info['pop_cap_spent_minerals']+self.process_info['pop_cap_spent_gas'])/max(self.process_info['game_over_time'],1)
                self.starcraft_eval_matrix['APU']=self.process_info['pop_utilization']/max(self.process_info['count'],1)
                self.starcraft_eval_matrix['TR']=process_data['completed_tech']
                self.starcraft_eval_matrix['result']=info['result']
                self.starcraft_eval_matrix['grounding_acc']=info['match_grounding_acc']
                print("==========starcraft_eval_matrix",self.starcraft_eval_matrix)
        if self.args.game.game_name.__contains__("Civ"):
            if done:
                self.civ_eval_matrix=info
                print("==========civ_eval_matrix",self.civ_eval_matrix)
        if self.args.game.game_name.__contains__("Stratego"):
            if done:
                self.stratego_eval_matrix=info
                print("==========stratego_eval_matrix",self.stratego_eval_matrix)
        if self.args.game.game_name.__contains__("StreetFight"):
            if done:
                self.streetfight3_eval_matrix=info
                print("==========streetfight3_eval_matrix", self.streetfight3_eval_matrix)
        if self.args.game.game_name.__contains__("WereWolf"):
            if done:
                self.werewolf_eval_matrix=info
                print("==========werewolf_eval_matrix", self.werewolf_eval_matrix)

    def get_eval_name(self):
        if self.args.game.game_name.__contains__("Starcraft"):
            return [self.starcraft_eval_matrix]
        if self.args.game.game_name.__contains__("Civ"):
            return [self.civ_eval_matrix]
        if self.args.game.game_name.__contains__("Stratego"):
            return [self.stratego_eval_matrix]
        if self.args.game.game_name.__contains__("StreetFight"):
            return [self.streetfight3_eval_matrix]
        if self.args.game.game_name.__contains__("WereWolf"):
            return [self.werewolf_eval_matrix]
        if self.args.game.game_name.__contains__("WelfareDiplomacy"):
            return [self.welfare_diplomacy_eval_matrix]

    def save_result(self):
        data={
            "match_idx":self.match_idx,
           "eval_matrix": self.get_eval_name()
        }
        json_data = json.dumps(data, indent=2)
        # Save JSON to a file
        with open(self.eval_matrix_path, 'a+') as json_file:
            json_file.write(json_data)

    def add_match_info(self, match):
        self.matches.append(match)

    def get_token_size(self):
        self.token_size = sum([s.get_token_size() for s in self.matches])
        return self.token_size

    def to_dict(self):
        return {
            "match_idx":self.match_idx,
           "matches": [m.to_dict() for m in self.matches]
        }

    def __json__(self):
        return self.to_dict()

    def clear(self):
        '''
        This function will clear all steps and agents
        '''
        self.matches.clear()
        self.agents.clear()

    def save_as_json(self):
        '''
        outout a json file containing agents' name and steps
        '''
        data = self.to_dict()
        json_data = json.dumps(data, indent=2)
        # Save JSON to a file
        with open(self.match_info_path, 'a+') as json_file:
            json_file.write(json_data)