import os.path
import re
import json
from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward
import weave

from agent_manager.prompts.starcraft2_prompt import Template



class L2_summary:
    """
    L2_summary class

    """

    def __init__(self, llm_model, system_prompt, example_prompt, chunk_window,prompt_type,logger):
        """
        Initialization
        :param llm_model:
        :param system_prompt:
        :param example_prompt:
        :param chunk_window: # Size of the summary window
        """
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.example_prompt = example_prompt
        self.chunk_window = chunk_window
        self.logger=logger
        self.chatbot=llm_model
        self.template = Template()
        self.prompt_type = prompt_type
        ## trajectory
        self.trajectory: Trajectory = []
        self.cur_time_step=1


    def split_into_chunks(self, L1_summaries):
        """
        Split L1_summaries into chunks based on self.chunk_window size

        :param L1_summaries:
        :return:
        """
        if not isinstance(L1_summaries, list):
            raise TypeError("Input must be a list of L1 summaries.")
        self.L1_summaries = L1_summaries
        return [self.L1_summaries[i:i + self.chunk_window] for i in range(0, len(self.L1_summaries), self.chunk_window)]

    def construct_query_message(self,user_input):
        query_message = [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": self.example_prompt[0]},
                         {"role": "assistant", "content": self.example_prompt[1]},
                         {"role": "user", "content": user_input}]
        return query_message
    def get_latest_k_messages(self, chunks, k):
        """
        Get the latest K messages

        :param chunks:
        :param k:
        :return:
        """
        if not chunks:
            raise ValueError("Input must be a non-empty list of chunks.")
        if not all(isinstance(chunk, list) for chunk in chunks):
            raise TypeError("Input must be a list of chunks.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        latest_k_messages = []
        for chunk in chunks:
            # Select the latest k pieces of information from each block
            latest_messages = chunk[-k:]
            latest_k_messages.append(latest_messages)
        return latest_k_messages

    @weave.op()
    def query(self, inputs):
        """
        Perform L2summary request

        First check if it is empty
        Then check if it is [[],[]], i.e., a list of chunks
        After checking, each chunk will be summarized by LLM
        The summarized L2_summary will be added to a list
        Finally, return this list of L2_summaries

        :param chunks:
        :return:
        """
        chunks = inputs['L1_summaries']
        executed_actions = inputs['executed_actions']
        failed_actions = inputs['failed_actions']
        # print("chunks_type", type(chunks))
        # print("type_executed_actions", type(executed_actions))
        if executed_actions:
            pass
            # print("type_executed_actions[0]", type(executed_actions[0]))
        else:
            pass
            # print("executed_actions is empty!")
        if failed_actions:
            # print("type_failed_actions[0]", type(failed_actions[0]))
            pass
        else:
            pass
            # print("failed_actions is empty!")
        # print("type_failed_actions", type(failed_actions))
        if not chunks:
            raise ValueError("Input must be a non-empty list of chunks.")
        if not all(isinstance(chunk, list) for chunk in chunks):
            raise TypeError("Input must be a list of chunks.")

        L2_summaries = []

        if self.prompt_type == "v1":
            for chunk in chunks:
                # Fill content based on template
                chunks_str = "\n".join(f"chunk{i}: {item}" for i, item in enumerate(chunk))
                # Query using the populated template
                formatted_input = self.template.input_template_v1.format(
                    num_rounds=len(chunks),
                    chunks_str=chunks_str,
                )
                # Build query prompt
                formatted_input=self.construct_query_message(formatted_input)
                # Query using the populated template
                L2_summary = self.chatbot.query(formatted_input)
                L2_summaries.append(L2_summary)
        elif self.prompt_type == "v2":
            for chunk in chunks:
                # Fill content based on template
                chunks_str = "\n".join(f"chunk{i}: {item}" for i, item in enumerate(chunk))
                # For executed_actions and failed_actions, we need to specially handle the nested list structure
                executed_actions_str = "\n".join(
                    " ".join(str(sub_action) for sub_action in action) for action in executed_actions)

                failed_actions_str = "\n".join(
                    " ".join(str(sub_action) for sub_action in action) for action in failed_actions)

                # Use template fill
                formatted_input = self.template.input_template_v2.format(
                    num_rounds=len(chunks),
                    chunks_str=chunks_str,
                    executed_actions_str=executed_actions_str,
                    failed_actions_str=failed_actions_str
                )
                # Build query prompt
                formatted_input = self.construct_query_message(formatted_input)
                # Query using the populated template
                L2_summary = self.chatbot.query(formatted_input)
                L2_summaries.append(L2_summary)
        elif self.prompt_type=="v3":
            for chunk in chunks:
                # Fill content based on template
                chunks_str = "\n".join(f"chunk{i}: {item}" for i, item in enumerate(chunk))
                # Use template fill
                formatted_input = self.template.input_template_v3.format(
                    num_rounds=len(chunks),
                    chunks_str=chunks_str,
                )
                # formatted_input+='\n## IMPORTANT The Decisions output must be well-formatted!'
                # print("input: ",formatted_input)
                # Build query prompt
                formatted_input = self.construct_query_message(formatted_input)
                # Query using the populated template
                L2_summary = self.chatbot.query(formatted_input)
                self.logger.info(f"=============model:{self.chatbot.model_name}===============")
                self.logger.info(f"====model_input:{formatted_input}")
                self.logger.info(f"====model_output:{L2_summary}")
                L2_summaries.append(L2_summary)
        elif self.prompt_type=="v4":
            for chunk in chunks:
                # Fill content based on template
                chunks_str = "\n".join(f"chunk{i}: {item}" for i, item in enumerate(chunk))
                # Build query prompt
                chunks_str = self.construct_query_message(chunks_str)
                if self.chatbot.model_name=='o1-mini':
                    chunks_str[0]['role']='user'
                    L2_summary,_ = self.chatbot.query_single_turn_o1(chunks_str)
                else:
                    L2_summary = self.chatbot.query(chunks_str)
                    print("L2_summary before:",L2_summary)
                if self.chatbot.model_name.__contains__("llama3.1:70b"):
                    post_prompt = """
                    Input:\n  {{L2_summary}}  
                    Output: extract the action decision and change format of the string to the following:
                    ```Decisions:
                    0: <BUILD CYBERNETICSCORE>
                    1: <BUILD PYLON>
                    2: <BUILD FORGE>
                    3: <RESEARCH WARPGATERESEARCH>
                    4: <CHRONOBOOST NEXUS>```
                    specific decisions must match from the action dictionary{'TRAIN UNIT': {0: 'TRAIN PROBE', 1: 'TRAIN ZEALOT', 2: 'TRAIN ADEPT', 3: 'TRAIN STALKER', 4: 'TRAIN SENTRY', 5: 'TRAIN HIGHTEMPLAR', 6: 'TRAIN DARKTEMPLAR', 7: 'TRAIN VOIDRAY', 8: 'TRAIN CARRIER', 9: 'TRAIN TEMPEST', 10: 'TRAIN ORACLE', 11: 'TRAIN PHOENIX', 12: 'TRAIN MOTHERSHIP', 13: 'TRAIN OBSERVER', 14: 'TRAIN IMMORTAL', 15: 'TRAIN WARPPRISM', 16: 'TRAIN COLOSSUS', 17: 'TRAIN DISRUPTOR', 18: 'MORPH ARCHON'}, 'BUILD STRUCTURE': {19: 'BUILD PYLON', 20: 'BUILD ASSIMILATOR', 21: 'BUILD NEXUS', 22: 'BUILD GATEWAY', 23: 'BUILD CYBERNETICSCORE', 24: 'BUILD FORGE', 25: 'BUILD TWILIGHTCOUNCIL', 26: 'BUILD ROBOTICSFACILITY', 27: 'BUILD STARGATE', 28: 'BUILD TEMPLARARCHIVE', 29: 'BUILD DARKSHRINE', 30: 'BUILD ROBOTICSBAY', 31: 'BUILD FLEETBEACON', 32: 'BUILD PHOTONCANNON', 33: 'BUILD SHIELDBATTERY'}, 'RESEARCH TECHNIQUE': {34: 'RESEARCH WARPGATERESEARCH', 35: 'RESEARCH PROTOSSAIRWEAPONSLEVEL1', 36: 'RESEARCH PROTOSSAIRWEAPONSLEVEL2', 37: 'RESEARCH PROTOSSAIRWEAPONSLEVEL3', 38: 'RESEARCH PROTOSSAIRARMORSLEVEL1', 39: 'RESEARCH PROTOSSAIRARMORSLEVEL2', 40: 'RESEARCH PROTOSSAIRARMORSLEVEL3', 41: 'RESEARCH ADEPTPIERCINGATTACK', 42: 'RESEARCH BLINKTECH', 43: 'RESEARCH CHARGE', 44: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1', 45: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2', 46: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3', 47: 'RESEARCH PROTOSSGROUNDARMORSLEVEL1', 48: 'RESEARCH PROTOSSGROUNDARMORSLEVEL2', 49: 'RESEARCH PROTOSSGROUNDARMORSLEVEL3', 50: 'RESEARCH PROTOSSSHIELDSLEVEL1', 51: 'RESEARCH PROTOSSSHIELDSLEVEL2', 52: 'RESEARCH PROTOSSSHIELDSLEVEL3', 53: 'RESEARCH EXTENDEDTHERMALLANCE', 54: 'RESEARCH GRAVITICDRIVE', 55: 'RESEARCH OBSERVERGRAVITICBOOSTER', 56: 'RESEARCH PSISTORMTECH', 57: 'RESEARCH VOIDRAYSPEEDUPGRADE', 58: 'RESEARCH PHOENIXRANGEUPGRADE', 59: 'RESEARCH TEMPESTGROUNDATTACKUPGRADE'}, 'OTHER ACTION': {60: 'SCOUTING PROBE', 61: 'SCOUTING OBSERVER', 62: 'SCOUTING ZEALOT', 63: 'SCOUTING PHOENIX', 64: 'MULTI-ATTACK', 65: 'MULTI-RETREAT', 66: 'CHRONOBOOST NEXUS', 67: 'CHRONOBOOST CYBERNETICSCORE', 68: 'CHRONOBOOST TWILIGHTCOUNCIL', 69: 'CHRONOBOOST STARGATE', 70: 'CHRONOBOOST FORGE', 71: 'EMPTY ACTION'}}. This dictionary comprises four categories of actions: unit production, building construction, technology research, and other actions. Remember to align these decisions with the current stage of the game, and avoid proposing actions that are not currently feasible.
                    """
                    post_prompt = self.format_prompt(post_prompt, {"L2_summary": L2_summary})
                    post_messages = [{"role": "user", "content": post_prompt}]
                    L2_summary = self.chatbot.query(post_messages)
                self.logger.info(f"=============model:{self.chatbot.model_name}===============")
                self.logger.info(f"====model_input:{chunks_str}")
                self.logger.info(f"====model_output:{L2_summary}")
                ## trajectory
                state_info = set_state_info(from_="Starcraft2", role="player", step=self.cur_time_step, content=chunks_str[3]["content"],
                                            system_content=chunks_str[0]["content"], user_content=chunks_str[3]["content"])
                self.trajectory.append(state_info)
                action = set_action_info(from_=self.chatbot.model_name, role="player", step=self.cur_time_step,
                                         content="", other_content=L2_summary)
                self.trajectory.append(action)
                L2_summaries.append(L2_summary)
        return L2_summaries
    def update_traj_action(self,content):
        if len(self.trajectory) >= 1:
            self.trajectory[-1]['content']=content

    def set_cur_time_step(self,time_step):
        self.cur_time_step=time_step

    def set_trajectory_reward(self,env,role,score):
        reward = set_reward(env,role,score)
        self.trajectory.append(reward)

    def save_trajectory(self,role, save_path,name):
        item_id=name.split("/")[-1].strip()+"_"+role
        output_path = os.path.join(save_path,item_id+".json")
        temp_traj=[]
        for traj in self.trajectory:
            if traj["role"]==role:
                temp_traj.append(traj)
        react_data={"item_id":item_id,"conversation":temp_traj[:-1],"rewards":temp_traj[-1]}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)

    def format_prompt(self,prompt_template, worldstate) -> str:
        import jinja2
        return jinja2.Template(prompt_template).render(worldstate)