import json
import time
import random
import numpy as np
import marko
import os
import yaml
from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward


class StrategoAgent(object):

    def __init__(self, config, args=None, **kwargs):
        self.args = args
        self.logger = self.args.logger
        self.prompt_constructor = config.prompt
        self.model = config.llm_model

        self.prompt_constructor_opp = config.get('prompt_opp')
        self.model_opp = config.get('llm_model_opp')
        self.generate_times = 1
        self.grounding_errors = 0
        self.opp_generate_times = 1
        self.opp_grounding_errors = 0
        self.live_pieces_num=40
        self.live_pieces_score=148
        self.critical_live_pieces_num=4
        self.opp_live_pieces_num = 40
        self.opp_live_pieces_score = 148
        self.opp_critical_live_pieces_num = 4
        self.player_moves_his=[]
        self.opp_moves_his=[]

        ## trajectory
        self.trajectory: Trajectory=[]
        self.cur_time_step=0

        self.logger.info("=" * 5 + f"StrategoAgent Init Successfully!: " + "=" * 5)

    def step(self, env, observations):
        """

        :param observations:
        :return:
        """
        self.cur_time_step=env._time_step
        assert len(observations.keys()) == 1
        self.current_player = list(observations.keys())[0]
        assert self.current_player == 1 or self.current_player == -1

        gameState, his_valid_moves_pre,valid_move,pieces_state = get_gameState(env, self.current_player)
        live_pieces_num=pieces_state["live_pieces_num"]
        critical_live_pieces_num=pieces_state["critical_live_pieces_num"]
        live_pieces_score=pieces_state["live_pieces_score"]
        opp_live_pieces_num=pieces_state["opp_live_pieces_num"]
        opp_critical_live_pieces_num=pieces_state["opp_critical_live_pieces_num"]
        opp_live_pieces_score=pieces_state["opp_live_pieces_score"]
        # lastest n operation
        his_str="\n## History moves: (history of the last 5 moves,The smaller the number, the closer it is to the current.)\n"
        his_list=[]
        if self.current_player == 1:
            his_list=self.player_moves_his
        else:
            his_list=self.opp_moves_his
        num=1
        for i in range(len(his_list)-1,-1,-1):
            his_str+=str(num)+". "+his_list[i]+"\n"
            num+=1
        if len(his_list)==0:
            his_str += "There is no historical moves at this time\n"
        gameState+=his_str
        print(f"====self.current_player:{self.current_player}")
        self.logger.info("=" * 5 + f"current state: " + "=" * 5)
        self.logger.info(gameState)
        if self.current_player==1:
            self.live_pieces_num=live_pieces_num
            self.live_pieces_score=live_pieces_score
            self.critical_live_pieces_num=critical_live_pieces_num
            self.opp_live_pieces_num = opp_live_pieces_num
            self.opp_live_pieces_score = opp_live_pieces_score
            self.opp_critical_live_pieces_num = opp_critical_live_pieces_num
            self.logger.info("=" * 5+f"player: live pieces num : {self.live_pieces_num} | live pieces score : {self.live_pieces_score}| critical live pieces num : {self.critical_live_pieces_num} ")
            self.logger.info("=" * 5+f"opponent: live pieces num : {self.opp_live_pieces_num} | live pieces score : {self.opp_live_pieces_score}| critical live pieces num : {self.opp_critical_live_pieces_num} ")
        action_cmd = self.llm_gen(gameState, valid_move,his_valid_moves_pre)
        action = env.base_env.get_action_1d_index_from_positions(*action_cmd)
        print(f"Player {self.current_player} made move {action}")
        return {self.current_player: action}

    def record_llm_gen(self, model):
        if model == self.model:
            self.generate_times += 1
        elif model == self.model_opp:
            self.opp_generate_times += 1

    def record_llm_grounding_errors(self, model):
        if model == self.model:
            self.grounding_errors += 1
        elif model == self.model_opp:
            self.opp_grounding_errors += 1

    def llm_gen(self, gameState, valid_move,his_valid_moves_pre):

        generate_times = 0
        while True:
            if self.current_player == 1:
                sys_prompt = self.prompt_constructor.sys_prompt
                model = self.model
                role="player"
            elif self.prompt_constructor_opp is not None:
                sys_prompt = self.prompt_constructor_opp.sys_prompt
                model = self.model_opp
                role="opp_player"
            else:
                raise Exception("prompt_constructor_opp and model_opp is None")
            messages = [
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": gameState,
                }
            ]
            raw_resp,_=model.query_single_turn_gen(messages)
            if model.model_name.__contains__("llama3.1:70b"):
                # re_flection = {'role': "user",
                #       'content': f"""Please adjust the result : {raw_resp} , and then generate your action as follows: \n```json\n{{\n  'reasoning": "string", // Explain your macro strategy and infer the intention of your opponent\'s operation and adjust your strategy .\n  "move": "3 1 4 1" // the move that you choose without any commentary. Choose from Valid moves\n}}."""}
                # messages.append(re_flection)
                post_prompt = """
                Input:\n  {{raw_resp}}  
                Output: change the format of the input string to the following:
                ```json
                {
                    "reasoning": "string", // reasoning part.
                    "move": "string" // the move that you choose without any commentary. Choose from Valid moves. the formate like "3 1 4 1"
                } 
                """
                post_prompt = self.format_prompt(post_prompt, {"raw_resp": raw_resp})
                post_messages = [{"role": "user", "content": post_prompt}]
                raw_resp, _ = model.query_single_turn_gen(post_messages)
            # print("==============:raw_resp:",raw_resp)
            llm_responce = parse_json(raw_resp)

            self.logger.info(f"=============model:{model.model_name}===============")
            self.logger.info(f"====model_input:{messages}")
            self.logger.info(f"====model_output:{llm_responce}")
            responce=llm_responce['move']
            responce = responce.replace("  ", " ")
            responce = responce.replace("\n", "")

            self.logger.info(f"====responce:{responce}")
            self.record_llm_gen(model)

            if len(responce) == 7:
                action_key = responce[0] + ',' + responce[2]
                if action_key in valid_move.keys() and [int(responce[4]), int(responce[6])] in valid_move[action_key]:
                    print(model.model_name, "generate action is valid")
                    key=(int(responce[0]), int(responce[2]))
                    his_move = his_valid_moves_pre[key] + responce[4:]
                    if self.current_player == 1:
                        if len(self.player_moves_his) >= 5:
                            self.player_moves_his = self.player_moves_his[1:]
                        self.player_moves_his.append(his_move)
                    else:
                        if len(self.opp_moves_his) >= 5:
                            self.opp_moves_his = self.opp_moves_his[1:]
                        self.opp_moves_his.append(his_move)
                    break
                else:
                    print("{} not in {} ".format(responce, valid_move))
                    self.record_llm_grounding_errors(model)
                    time.sleep(1)
                    generate_times += 1
                    if generate_times > 3:
                        time.sleep(10)
                    if generate_times >= 10:
                        rand_cmd = random_choice(valid_move)
                        self.logger.warning(
                            f"generate times is more than 50 times, random action for the game,{rand_cmd}")
                        return rand_cmd
                    continue
            else:
                print(f"{responce}|len(responce){len(responce)} != 7,continue")
                self.record_llm_grounding_errors(model)

        print(responce)

        action_cmd = [int(t) for t in responce.split(" ")]
        print("================real", action_cmd)
        if self.current_player==-1:
            action_cmd=[9-int(t) for t in action_cmd]
        print("================final", action_cmd)
        ## trajectory
        state_info = set_state_info(from_="Stratego", role=role,step=self.cur_time_step, content=gameState,system_content=sys_prompt,user_content=gameState)
        self.trajectory.append(state_info)
        action = set_action_info(from_=model.model_name, role=role, step=self.cur_time_step, content=str(action_cmd),other_content=llm_responce)
        self.trajectory.append(action)
        print("===================")
        print(state_info)
        print(action)
        print("===================")
        return action_cmd

    def get_live_pieces_state(self):
        pieces_state={
            "live_pieces_rate":self.live_pieces_num/40,
            "live_pieces_score":self.live_pieces_score,
            "critical_live_pieces_rate":self.critical_live_pieces_num/4,
            "opp_live_pieces_rate": self.opp_live_pieces_num / 40,
            "opp_live_pieces_score": self.opp_live_pieces_score,
            "opp_critical_live_pieces_rate": self.opp_critical_live_pieces_num / 4,
        }
        return pieces_state
    def format_prompt(self,prompt_template, worldstate) -> str:
        import jinja2
        return jinja2.Template(prompt_template).render(worldstate)

    def set_trajectory_reward(self,env,role,score):
        reward = set_reward(env,role,score)
        self.trajectory.append(reward)

    def save_trajectory(self,role,save_path,name):
        item_id=name.split("/")[-1].strip()+"_"+role
        output_path = os.path.join(save_path,item_id+".json")
        temp_traj=[]
        for traj in self.trajectory:
            if traj["role"]==role:
                temp_traj.append(traj)
        react_data={"item_id":item_id,"conversation":temp_traj[:-1],"rewards":temp_traj[-1]}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)
def random_choice(valid_move):
    random_rc = random.choice(list(valid_move.keys()))
    random_xy = random.choice(valid_move[random_rc])
    action_cmd = [int(t) for t in random_rc.split(",")] + random_xy
    return action_cmd


def _player_index(player):
    # player 1 returns 0
    # player -1 returns 1
    return (player - 1) // -2


def get_gameState(env, current_player):
    piece_content_dict = {1: 'Spy', 2: 'Scout', 3: 'Miner', 4: 'Sergeant', 5: 'Lieutenant', 6: 'Captain', 7: 'Major',
                          8: 'Colonel', 9: 'General', 10: 'Marshall', 11: 'Flag', 12: 'Bomb'}
    piece_dict = {1: 's', 2: '¹', 3: '²', 4: '3', 5: '4', 6: '5', 7: '6',
                  8: '7', 9: '8', 10: '9', 11: '¶', 12: 'o'}
    # ['', '¶', 's', '¹', '²', '3', '4', '5', '6', '7', '8', '9', 'o']
    # player_perspective_state = env.base_env.get_state_from_player_perspective(state=env.state, player=current_player)
    current_player_valid_move = env.base_env.get_dict_of_valid_moves_by_position(state=env.state, player=current_player)
    opp_player=-1
    if current_player==1:
        layer_idx=0
        opp_layer_idx=1
        opp_part_layer_idx=4
    else:
        layer_idx=1
        opp_layer_idx=0
        opp_part_layer_idx=3

    friend_board = env.state[layer_idx]
    enemy_board = env.state[opp_layer_idx]
    live_pieces_num=np.sum(friend_board>0)
    critical_live_pieces_num=np.sum(friend_board[ (friend_board>7) & (friend_board<11) ]>0)
    live_pieces_score=np.sum(friend_board[(friend_board>0) & (friend_board<11)])
    opp_live_pieces_num = np.sum(enemy_board > 0)
    opp_critical_live_pieces_num = np.sum(enemy_board[(enemy_board > 7) & (enemy_board < 11)] > 0)
    opp_live_pieces_score = np.sum(enemy_board[(enemy_board > 0) & (enemy_board < 11)])
    pieces_state={
        "live_pieces_num":live_pieces_num,
        "critical_live_pieces_num":critical_live_pieces_num,
        "live_pieces_score":live_pieces_score,
        "opp_live_pieces_num":opp_live_pieces_num,
        "opp_critical_live_pieces_num":opp_critical_live_pieces_num,
        "opp_live_pieces_score":opp_live_pieces_score
    }
    # friend_board = player_perspective_state[current_player, :, :]
    # enemy_board = player_perspective_state[(current_player - 1) // -2, :, :]
    enemy_part_view = env.state[opp_part_layer_idx]
    his_valid_moves={}
    valid_moves_str = ''
    for k, v in current_player_valid_move.items():
        position = eval(k)
        piece_idx = friend_board[position[0], position[1]]

        valid_moves_str += piece_content_dict[piece_idx] + " 'R(" + piece_dict[
            piece_idx] + ")' at position '" + k + "'  could move to"
        if len(v) > 1:
            valid_moves_str += " any of"
        valid_moves_str += ": " + ', '.join(' '.join(str(i) for i in item) for item in v)
        valid_moves_str += "\n"

        his_valid_moves[position]=piece_content_dict[piece_idx] + " 'R(" + piece_dict[
            piece_idx] + ")' at position '" + k + "'  moved to "

    gameState = ''
    boardState = [['  ', ' c0 ', ' c1 ', ' c2 ', ' c3 ', ' c4 ', ' c5 ', ' c6 ', ' c7 ', ' c8 ', ' c9 ']]
    gameState += "## Board State:\n"
    gameState += '\n'.join([', '.join(map(str, sublist)) for sublist in boardState]) + "\n"

    board_view = friend_board.copy()
    obstacle_view = env.state[2, :, :]
    for r in range(10):
        for c in range(10):
            if obstacle_view[r, c] == 1:
                board_view[r, c] = '-102'
            elif enemy_part_view[r, c] == 13:
                board_view[r, c] = '-101'
            elif enemy_part_view[r, c] > 0 and enemy_part_view[r, c] < 13:
                board_view[r, c] = -enemy_part_view[r, c]
    board_view_str = ''
    for r in range(10):
        board_view_str += "r{}".format(r)
        # print("r{}".format(r))
        for c in range(10):
            if board_view[r, c] == -101:
                show_str = 'B(#)'
                board_view_str += ", {}".format(show_str)
            elif board_view[r, c] == -102:
                show_str = "~~~~"
                board_view_str += ", {}".format(show_str)
            elif board_view[r, c] == 0:
                show_str = "...."
                board_view_str += ", {}".format(show_str)
            elif board_view[r, c] < 0 and board_view[r, c] > -13:
                show_str = -board_view[r, c]
                board_view_str += ", B({})".format(piece_dict[show_str])
            else:
                show_str = board_view[r, c]
                board_view_str += ", R({})".format(piece_dict[show_str])
            # print(", R{}".format(show_str), end='')
        board_view_str += "\n"
    gameState += board_view_str + "\n\n"
    gameState += "## Valid moves: \n" + valid_moves_str + "\n"
    gameState += '''## IMPORTANT \n The selection of 'r c' you make must choose from "position" of the **Valid moves**  ,and the 'x y' choose from "move to" of the **Valid moves**'''
    # print(gameState)

    return gameState, his_valid_moves,current_player_valid_move,pieces_state





def parse_json(text: str) :
    result_json = parse_json_markdown(text)

    if not result_json:
        result_json = parse_json_str(text)
    return result_json


def parse_json_markdown(text: str) :
    ast = marko.parse(text)

    for c in ast.children:
        # find the first json block (```json or ```JSON)
        if hasattr(c, "lang") and c.lang.lower() == "json":
            json_str = c.children[0].children
            return parse_json_str(json_str)

    return None


def parse_json_str(text: str) :
    try:
        # use yaml.safe_load which handles missing quotes around field names.
        result_json = yaml.safe_load(text)
    except yaml.parser.ParserError:
        return None

    return result_json