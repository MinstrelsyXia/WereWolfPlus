import json

import os

from agent_manager.prompts.werewolf_prompt import ACTION_PROMPTS_AND_SCHEMAS # 这里存了所有prompt（动作相对应的提示）
import jinja2
from games.werewolf.lm import LmLog
from games.werewolf.utils import parse_json
import random
from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward

class WereWolfAgent(object):

    def __init__(self, config,args=None, **kwargs):
        self.args = args
        self.logger = self.args.logger
        self.prompt_constructor = config.prompt
        self.model = config.llm_model
        self.agent_idx=kwargs['idx']
        self.cur_agent_config=self.args.agent[self.agent_idx]
        self.agent_role=self.cur_agent_config['agent_nick']
        self.agent_name=None
        self.use_message_pool = self.cur_agent_config.get('use_message_pool', False)
        ## trajectory
        self.trajectory: Trajectory = []
        self.cur_time_step = 0

        self.logger.info("=" * 5 + f"WereWolfAgent "+self.agent_role+" Init Successfully!: " + "=" * 5)

    def init_message_pool(self, message_pool):
        if self.use_message_pool:
            self.message_pool = message_pool

    def step(self, observations):
        """
        :param observations:
        {
        "name": self.name,
        "role": self.role,
        "round": self.gamestate.round_number,
        "observations": formatted_observations,
        "remaining_players": ", ".join(remaining_players),
        "debate": formatted_debate,
        "bidding_rationale": self.bidding_rationale,
        "debate_turns_left": MAX_DEBATE_TURNS - len(formatted_debate),
        "personality": self.personality,
        "num_players": NUM_PLAYERS,
        "num_villagers": NUM_PLAYERS - 4,
    }
        :return:
        """
        game_state = observations['game_state']
        self.cur_time_step = game_state['round']
        action = observations['action']
        options = observations['options']
        if options:
            game_state["options"] = (", ").join(options)
        else:
            game_state["options"] = ""
        prompt_template, response_schema = ACTION_PROMPTS_AND_SCHEMAS[action]

        result_key, allowed_values = (
            (action, options)
            if action in ["vote", "remove", "investigate", 
                          "protect", "save", "poison", "shoot", "bid",
                          "run_for_sheriff","elect","determine_statement_order","badge_flow","pseudo_vote"]
            else (None, None)
        )

        # Set temperature based on allowed_values
        temperature = 0.5 if allowed_values else 1.0


        if action == "vote" and self.message_pool != None:
            # 构建查询文本
            summarize_template, _  = ACTION_PROMPTS_AND_SCHEMAS['summarize']
            game_state_my = {
                "name": game_state["name"],
                "role": game_state["role"],
                "round": game_state["round"],
                "observations": game_state.get("observations", ""),
                "remaining_players": game_state.get("remaining_players", ""),
                "debate": game_state.get("debate", ""),
                "bidding_rationale": game_state.get("bidding_rationale", ""),
                "debate_turns_left": game_state.get("debate_turns_left", 0),
                "personality": game_state.get("personality", ""),
                "num_players": game_state.get("num_players", 0),
                "num_villagers": game_state.get("num_villagers", 0),
                "action": "summarize"
            }
            summarize_prompt = self.format_prompt(summarize_template, game_state_my)
            messages = [{"role": "user", "content": summarize_prompt}]
            
            query = None
            for _ in range(30):
                try:
                    raw_resp, _ = self.model.query_single_turn_gen(messages)
                    try:
                        result = parse_json(raw_resp)
                    except:
                        if self.model.model_name.__contains__("llama3.1"):
                            # 处理llama3.1的特殊情况
                            post_prompt = """
                            Input:\n  {{raw_resp}}  
                            Output: change the format of the input string to the following:
                            ```json
                            {
                                "reasoning": "string", // reasoning part.
                                "summary": "string" // Summary part
                            } 
                            """
                            post_prompt = self.format_prompt(post_prompt, {"raw_resp": raw_resp})
                            post_messages = [{"role": "user", "content": post_prompt}]
                            raw_resp, _ = self.model.query_single_turn_gen(post_messages)
                            result = parse_json(raw_resp)
                    
                    # 记录日志
                    self.logger.info(f"=============player:{game_state_my['role']}--{game_state_my['name']}=action:{action}=============")
                    self.logger.info(f"====model_output:{result}")
                    # 更新trajectory
                    state_info = set_state_info(
                        from_="WereWolf", 
                        role=game_state_my['role'], 
                        step=self.cur_time_step,
                        content=summarize_prompt,
                        system_content="",
                        user_content=summarize_prompt
                    )
                    self.trajectory.append(state_info)
                    
                    action_info = set_action_info(
                        from_=self.model.model_name, 
                        role=game_state_my['role'], 
                        step=self.cur_time_step,
                        content=result, 
                        other_content=raw_resp
                    )
                    self.trajectory.append(action_info)
                    
                    # 检查并获取summary
                    if result and "summary" in result:
                        query = result["summary"]
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error generating summary: {e}")
                    continue
            # query  = "Who should I vote?"
                # 使用生成的summary作为query
            if query:
                similar_exps = self.message_pool.get_vote_experience(
                    agent_name=game_state['name'],
                    role=game_state['role'],
                    query=query,
                    k=3
                )
                
                if similar_exps:
                    good_exps, bad_exp = similar_exps
                    # 将相似经验添加到prompt中
                    exp_text = "\n\nSIMILAR PAST EXPERIENCES:\n"
                    # 添加好的经验
                    for exp, sim in good_exps:
                        exp_text += f"- Good Experience (similarity: {sim:.2f}): {exp.content[0]}\n"
                        exp_text += f"  Action: {exp.content[1]}\n"
                        exp_text += f"  Reward: {exp.content[2]}\n\n"
                    # 添加坏的经验作为反面例子
                    if len(bad_exp) > 0:
                        exp_text += f"- Bad Experience (similarity: {bad_exp[1]:.2f}): {bad_exp[0].content[0]}\n"
                        exp_text += f"  Action: {bad_exp[0].content[1]}\n"
                        exp_text += f"  Reward: {bad_exp[0].content[2]}\n\n"
                    
                    # 将经验添加到game_state中
                    game_state["experiences"] = exp_text
                else:
                    game_state["experiences"] = ""

        prompt = self.format_prompt(prompt_template, game_state)
        messages=[{"role": "user", "content": prompt}]
        raw_responses = []
        for _ in range(30):
            raw_resp = None
            try:
                raw_resp,_=self.model.query_single_turn_gen(messages)
                try:
                    result = parse_json(raw_resp)
                except:
                    if self.model.model_name.__contains__("llama3.1") and action=="summarize":

                        print("================before ==============",raw_resp)
                        post_prompt="""
                        Input:\n  {{raw_resp}}  
                        Output: change the format of the input string to the following:
                        ```json
                        {
                            "reasoning": "string", // reasoning part.
                            "summary": "string" // Summary part
                        } 
                        """
                        post_prompt = self.format_prompt(post_prompt, {"raw_resp":raw_resp})
                        post_messages = [{"role": "user", "content": post_prompt}]
                        raw_resp, _ = self.model.query_single_turn_gen(post_messages)
                        # raw_resp = self.model.query_single_turn(post_messages)
                        # raw_resp = raw_resp.choices[0].message.content
                        print("================after ==============",raw_resp)
                    result = parse_json(raw_resp)
                    print("================final ==============",raw_resp)
                self.logger.info(f"=============model:{self.model.model_name}===============")
                self.logger.info(f"=============player:{game_state['role']}--{game_state['name']}=action:{action}=============")
                self.logger.info(f"====model_input:{messages}")
                self.logger.info(f"====model_output:{result}")
                ## trajectory
                state_info = set_state_info(from_="WereWolf", 
                                        role=game_state['role'],                   
                                        step=self.cur_time_step,
                                        content=prompt,
                                        system_content="",
                                        user_content=prompt)
                self.trajectory.append(state_info)
                action = set_action_info(from_=self.model.model_name, role=game_state['role'], step=self.cur_time_step,content=result, other_content=raw_resp)
                self.trajectory.append(action)
                log = LmLog(prompt=prompt, raw_resp=raw_resp, result=result)

                if result and result_key:
                    result = result.get(result_key)
                    # if 'no' in result.lower() or 'None' in result.lower():
                    #     # 如果女巫不救人或者不毒人
                    #     return None, log

                if allowed_values is None or result in allowed_values:
                    return result, log

            except Exception as e:
                print(f"Retrying due to Exception in were_wolf_agent: {e}")
            raw_responses.append(raw_resp)
        if allowed_values is not None and len(allowed_values)>0:
            ret=random.choice(allowed_values)
        else:
            ret=None
        return ret, LmLog(
            prompt=prompt, raw_resp="-------".join(raw_responses), result=None
        )

    def set_trajectory_reward(self, env, role, score):
        reward = set_reward(env, role, score)
        self.trajectory.append(reward)

    def save_trajectory(self, role,role_name,save_path, name):
        item_id = name.split("/")[-1].strip() + "_" + role + "_" + role_name
        output_path = os.path.join(save_path,item_id+".json")
        temp_traj = []
        for traj in self.trajectory:
            if traj["role"] == role:
                temp_traj.append(traj)

        react_data = {"item_id": item_id, "conversation": temp_traj[:-1], "rewards": temp_traj[-1]}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)

    def format_prompt(self,prompt_template, worldstate) -> str:
        return jinja2.Template(prompt_template).render(worldstate)
