import json
import os
import time

import numpy as np
from datetime import datetime

import wandb
import weave

from games.starcraft2.summarize.L1_summarize import generate_summarize_L1
from games.starcraft2.summarize.gpt_test.L2_summarize import L2_summary
from collections import deque
from games.starcraft2.utils.action_extractor import *
from games.starcraft2.utils.action_vector_test import ActionDBManager
from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward

class ChatGPTAgent:
    """
    ChatGPTAgent
    """

    def __init__(self, llm_model, prompt_constructor,logger,
                 action_interval=10, chunk_window=5, action_window=10, action_mix_rate=0.5,
                 last_k=5, prompt_type='v4'):
        self.logger=logger
        self.llm_model = llm_model
        self.action_interval = action_interval  # Execute a real action every few steps
        self.current_step = 0  # Current step count
        self.prompt_constructor = prompt_constructor
        self.action_queue = deque()
        self.summary_queue = deque()
        self.executed_actions_queue = deque()
        self.failed_actions_queue = deque()
        self.chunk_window = chunk_window
        self.action_window = action_window
        self.action_mix_rate = action_mix_rate
        self.last_k = last_k
        self.action_description = self.prompt_constructor.action_description
        self.action_dict = self.action_description.action_descriptions
        self.temp_command = "temp command"  # Used to store temporary command
        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Get current time
        self.empty_action_idx = self.prompt_constructor.action_description.get_empty_action_idx()
        self.system_prompt = self.prompt_constructor.system_prompt
        self.example_prompt = self.prompt_constructor.example_prompt
        self.L2 = L2_summary(llm_model=self.llm_model, system_prompt=self.system_prompt,
                             example_prompt=self.example_prompt, chunk_window=self.chunk_window,
                             prompt_type=prompt_type,logger=self.logger)
        self.base_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'gpt4_log')
        self.action_db_manager = self.init_action_db()
        # Create a separate folder for each game, using the current time as a unique identifier
        self.game_folder = os.path.join(self.base_save_dir, f"game_{self.current_time}")

        # Create the directory if it does not exist
        if not os.path.exists(self.game_folder):
            os.makedirs(self.game_folder)
        self.valid_action_num = 0
        self.action_time_dict = {}
        self.failaction_time_dict = {}
        self.success_exe_action_num = 0
        self.agent_name=llm_model.model_name
        ## trajectory
        self.trajectory: Trajectory = []
        self.cur_time_step = 0

    def init_action_db(self):
        relative_path_parts = ["..", "..", "..", "utils", "actionvdb", "action_vdb"]
        action_vdb_path = os.path.join(*relative_path_parts)
        self.action_db = ActionDBManager(db_path=action_vdb_path)
        if self.prompt_constructor.org_race == "Protoss":
            self.action_db.initialize_collection("protoss_actions")
            # from games.starcraft2.utils.action_vector_test import configure_protoss_actions
            # configure_protoss_actions(self.action_db.db_path)
            return self.action_db
        elif self.prompt_constructor.org_race == "Zerg":
            self.action_db.initialize_collection("zerg_actions")
            # from games.starcraft2.utils.action_vector_test import configure_zerg_actions
            # configure_zerg_actions(self.action_db.db_path)
            return self.action_db
        else:
            raise ValueError("Not support Race")

    def _save_data_to_file(self, data, filename):
        """
        General data saving method
        :param data: Data to be saved
        :param filename: Name of the file to save
        :return: None
        """
        full_path = os.path.join(self.game_folder, filename)
        with open(full_path, "a") as file:
            json.dump(data, file)
            file.write("\n")

    def preprocess_actions(self):
        # Convert executed actions to a list without 'EMPTY ACTION'
        executed_actions = [action for action in self.executed_actions_queue if action != "EMPTY ACTION"]

        # Convert failed actions to a structured format
        failed_actions_list = self.failed_actions_queue
        failed_actions_structured = []
        for failure in failed_actions_list:
            for f in failure:
                failed_actions_structured.append(f)

        return executed_actions, failed_actions_structured

    def _save_raw_observation_to_file(self, raw_observation):
        """
        Save observation information
        :param raw_observation:
        :return: None
        """
        filename = "raw_observation.json"
        self._save_data_to_file(raw_observation, filename)

    def _save_action_executed_to_file(self, action_executed):
        """
        Save executed action information
        :param action_executed:
        :return: None
        """
        filename = "action_executed.json"
        self._save_data_to_file(action_executed, filename)

    def _save_action_failures_to_file(self, action_failures):
        """
        Save failed action information
        :param action_failures:
        :return: None
        """
        filename = "action_failures.json"
        self._save_data_to_file(action_failures, filename)

    def _save_L1_observation_to_file(self, L1_observation):
        """
        Save information after L1 summarize
        :param L1_observation: Information after L1 summarize
        :return: None
        """
        filename = "L1_observation.json"
        self._save_data_to_file(L1_observation, filename)

    def _save_commander_to_file(self, commander):
        """
        Save command information output by GPT
        :param commander: Command information
        :return: None
        """
        filename = "commander.json"
        self._save_data_to_file(commander, filename)

    def _save_combined_input_to_file(self, combined_input):
        """
        Save input during LLM decision-making
        :param combined_input: Input during LLM decision-making
        :return: None
        """
        filename = "combined_input.json"
        self._save_data_to_file(combined_input, filename)

    def _get_next_action(self):
        """
        Get the next action
        :return: Next action
        """
        # Check if there are actions in the queue
        if self.action_queue:
            # Return the first action and remove it from the queue
            return self.action_queue.popleft()
        else:
            empty_idx = self.empty_action_idx
            # If there are no actions, return empty action
            return empty_idx

    def extract_actions_from_command(self, command):
        if isinstance(command, list):
            command = " ".join(command)
        self.action_extractor = ActionExtractor(self.action_dict)
        action_ids, valid_actions = extract_actions_from_command(command,action_dict=self.action_dict, action_extractor=self.action_extractor,
                                                                 empty_idx=self.empty_action_idx,
                                                                 action_db_manager=self.action_db_manager,
                                                                 llm_model=self.llm_model,)
        return action_ids, valid_actions

    def _wandb_save_log(self, observation):
        # Extract key information from observation data
        research_data = observation['information']['research']  # Research data
        process_data = observation['process_data']  # Process data

        # Initialize the total number of started technologies
        total_research_count = 0
        # Count the total number of technologies
        for category, research in research_data.items():
            for key, value in research.items():
                if value != 0:
                    total_research_count += 1

        # Strategic planning capability evaluation metrics
        RPM = process_data['collected_minerals'] + process_data['collected_vespene']
        EER = (process_data['collected_minerals'] + process_data['spent_vespene']) * 100 / max((
                process_data['collected_minerals'] + process_data['collected_vespene']), 1)
        SUR = process_data['supply_used'] / max(process_data['supply_cap'], 1)
        TRR = process_data['completed_tech'] / max(total_research_count, 1)
        wandb.log({
            "Strategic planning capabilities/RPM": RPM,
            "Strategic planning capabilities/EER": EER,
            "Strategic planning capabilities/SUR": SUR,
            "Strategic planning capabilities/TRR": TRR,
        },step=self.cur_time_step)

    def _wandb_save_log_decision(self,observation,action,wandb_flag):
        cur_time = int(observation['process_data']['time'])
        # Log the raw observation, level 1 summary, command, and action to Weights & Biases.
        for k in list(self.action_dict.keys()):
            if action in list(self.action_dict[k].keys()):
                if self.action_dict[k][action] != "EMPTY ACTION":
                    self.valid_action_num += 1

                    if cur_time in self.action_time_dict.keys():
                        self.action_time_dict[cur_time] += 1
                    else:
                        self.action_time_dict[cur_time] = 1
        valid_action_num_per_minute = 0
        rekey_action_time_dict = []
        for k, v in self.action_time_dict.items():
            if cur_time - k > 60:
                rekey_action_time_dict.append(k)
            else:
                valid_action_num_per_minute += v
        for k in rekey_action_time_dict:
            del self.action_time_dict[k]

        if cur_time in self.failaction_time_dict.keys():
            self.failaction_time_dict[cur_time] += len(observation['action_failures'])
        else:
            self.failaction_time_dict[cur_time] = 1

        rekey_failures_action_dict = []
        failures_action_num_per_minute = 0
        for k, v in self.failaction_time_dict.items():
            if cur_time - k > 60:
                rekey_failures_action_dict.append(k)
            else:
                failures_action_num_per_minute += v
        for k in rekey_failures_action_dict:
            del self.failaction_time_dict[k]
        self.success_exe_action_num = self.valid_action_num - len(observation['action_failures'])
        print("================self.valid_action_num:", self.valid_action_num)
        print("================self.success_exe_action_num:", self.success_exe_action_num)
        # Real-time decision-making capability metrics
        # 1. APM = total_actions / game_time_per_minutes
        # 2. EPM = effective_operations / game_time_per_minutes
        print("================observation['process_data']", observation['process_data'])
        print("================observation['process_data']['time']", observation['process_data']['time'])
        APM_avg = self.valid_action_num / max((observation['process_data']['time'] / 60), 1)
        EPM_avg = self.success_exe_action_num / max((observation['process_data']['time'] / 60), 1)
        APM = valid_action_num_per_minute
        EPM = failures_action_num_per_minute
        if wandb_flag:
            wandb.log({
                "real_decision_capability/APM": APM,
                "real_decision_capability/APM_avg": APM_avg,
                "real_decision_capability/EPM": EPM,
                "real_decision_capability/EPM_avg": EPM_avg
            }, step=self.cur_time_step)
        # Learning capability
        match_grounding_acc = self.success_exe_action_num / max(self.valid_action_num, 1)
        if wandb_flag:
            wandb.log({
                "learning_capability/match_grounding_acc": match_grounding_acc,
            },step=self.cur_time_step)
        match_data = {
            "match_grounding_acc": match_grounding_acc
        }
        return match_data

    def action(self, observation,wandb_flag=True):
        """
        This function generates the next action for the ChatGPT agent to take.

        :param observation: The current observation from the environment.

        :return: The action that the ChatGPT agent should take, along with a command and a flag indicating whether a new command was generated.
        """

        # Extract the raw observation from the list and save it to a file.
        player_race = observation['player_race']
        opposite_race = observation['opposite_race']
        self.cur_time_step=observation.get('_time_step', 1)

        map_name = observation['map_name']
        raw_observation = observation['information']
        action_executed = observation['action_executed']
        action_failures = observation['action_failures']

        self.executed_actions_queue.append(action_executed)  # Store executed_action
        self.failed_actions_queue.append(action_failures)  # Store failed_action

        self._save_raw_observation_to_file(raw_observation)
        # Save executed actions and failed actions data
        self._save_action_executed_to_file(action_executed)
        self._save_action_failures_to_file(action_failures)

        # If the raw observation is a dictionary, generate a level 1 summary and save it to a file. Otherwise, return the next action.
        if isinstance(raw_observation, dict):
            print("======================isinstance(raw_observation, dict)========")
            L1_observation = generate_summarize_L1(raw_observation)
            self._save_L1_observation_to_file(L1_observation)
        else:
            print("======================return self._get_next_action()========")
            return self._get_next_action()

        # Add the new level 1 summary to the queue of summaries.
        self.summary_queue.append(L1_observation)  # Store L1_summary

        # Initialize the command and the command flag. The command will contain the output of the level 2 summary model, and the flag will be True if a new command was generated.
        command = None  # Initialize command
        command_flag = False  # Initialize command_flag

        # If the current step is a multiple of the action interval and the summary queue is not empty, generate a level 2 summary and get a command.
        if self.current_step % self.action_interval == 0 and self.summary_queue:  # Execute every few steps
            # Convert the summary queue to a list and get the last k level 1 summaries.
            summaries = [list(self.summary_queue)]
            last_k_L1_summaries = self.L2.get_latest_k_messages(summaries, self.last_k)  # Get the latest k L1_summaries
            executed, failed = self.preprocess_actions()  # Preprocess executed actions and failed actions

            combined_input = {
                'L1_summaries': last_k_L1_summaries,
                'executed_actions': executed,
                'failed_actions': failed
            }

            self._save_combined_input_to_file(combined_input)

            # Generate a level 2 summary and get a command.
            L2_summaries = self.L2.query(combined_input)
            command = L2_summaries
            print("=========================command:", command)

            # Save the command to a file and print it.
            self._save_commander_to_file(command)
            self.temp_command = command

            # Extract the action ids and values from the command.
            action_ids, action_values = self.extract_actions_from_command(
                command)  # Extract action_ids and action_values from command

            self.L2.update_traj_action(str(action_values))
            # Mix the extracted actions with empty actions based on the mix rate.
            mixed_actions = self.mix_actions(action_ids)
            print("=========================mixed_actions:", mixed_actions)

            # Add the mixed actions to the action queue.
            self.action_queue.extend(mixed_actions)

            # Clear the summary queue for the next cycle.
            # Clear the executed_queue and failed_queue for the next cycle.
            self.summary_queue.clear()
            self.executed_actions_queue.clear()
            self.failed_actions_queue.clear()

            # A new command was generated in this step, so set the command flag to True.
            command_flag = True

        #
        self._wandb_save_log(observation)
        # Increment the current step.
        self.current_step += 1

        # Get the next action from the action queue.
        action = self._get_next_action()

        match_data = self._wandb_save_log_decision(observation, action, wandb_flag)
        print("=========match_data=================")
        print(match_data)
        print("=========match_data=================")
        # Return the action, command, and command flag. The environment can use the command and command flag to display the command and whether it was newly generated.
        return action, command, command_flag, match_data

    def mix_actions(self, real_actions):
        """
        Mix real actions and empty actions to execute a real action every few steps, with the rest being empty actions.
        This achieves action sparsity, alleviating the computational pressure on LLM, allowing LLM and the game engine to interact more normally.

        :param real_actions:
        :return:
        """
        empty_action = self.empty_action_idx
        mixed_actions = []
        # Calculate the number of real actions in each action_window
        num_real_actions = int(self.action_window * self.action_mix_rate)

        # Check the boundary conditions:
        # if the action_mix_rate is too high that requires more real_actions than we have, adjust it.
        if num_real_actions > len(real_actions):
            num_real_actions = len(real_actions)

        # The indices in the window where the real action should be placed
        real_action_indices = np.linspace(start=0, stop=self.action_window - 1, num=num_real_actions, dtype=int)

        for i in range(self.action_window):
            if i in real_action_indices and real_actions:
                mixed_actions.append(real_actions.pop(0))
            else:
                mixed_actions.append(empty_action)

        return mixed_actions
