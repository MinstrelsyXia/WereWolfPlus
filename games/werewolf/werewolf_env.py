import json
import random
from typing import List, Tuple
import enum
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import wandb
from tqdm import tqdm
import gym
from games.werewolf.runner import initialize_players,initialize_players_diy
from games.werewolf.model import State
from games.werewolf.game import GameMaster
from games.werewolf.config import _THREADS
from games.werewolf.model import Round, RoundLog, State, VoteLog, PseudoVoteLog, ElectLog, Player
from games.werewolf.chatarena.message import MessagePool, Message
from games.werewolf.lm import LmLog

class WereWolfEnv(gym.Env):
    """ WereWolfEnv

        The goal of hotter colder is to *****

        After each step the agent receives an observation of:
        *****

        The rewards is calculated as:
        *****
        """
    def __init__(self, args):
        # 初始化
        self.args = args
        self.logger = self.args.logger
        self.game=None
        self.state=None
        self.round_debate_players = []
        self.round_vote_players = []
        self.round_votes = {}
        self.round_votes_log = []
        self.round_summ_players = []
        # 警长相关
        self.round_elect_players = []
        self.round_pseudo_vote_players=[]
        self.round_pseudo_votes = {}
        self.round_pseudo_votes_log = []
        self.round_elect = {}
        self.round_elect_log = []
        self.statement_order=None
        if args.has_sheriff==False:
            self.action_seq=['new_round','eliminate','protect','save','poison','unmask','resolve_night_phase','check_for_winner', 'check_for_shoot','check_for_winner','run_day_phase','vote','exile','check_for_winner','check_for_shoot','check_for_winner','run_summaries']
        else:
            self.action_seq=['new_round','eliminate','protect','save','poison','unmask','resolve_night_phase','check_for_winner', 'check_for_shoot','check_for_winner','badge_flow_1',"run_for_sheriff","sheriff_debate","elect","determine_statement_order",'run_day_phase','pseudo_vote','sheriff_summarize', 'vote','exile','check_for_winner','check_for_shoot','check_for_winner','badge_flow_2','run_summaries']
        self.action_seq_pointer=-1
        self.key_role_num=0 # init 更新
        self.vote_wolf=0
        self.vote_exile_wolf=0
        self.valid_roundvote_times=0
        self.good_vote_times=0
        self.valid_vote_times=0
        self.all_vote_times=0
        self.cur_alive_wolf_name = []
        self.cur_alive_good_name=[]
        
        # 其他指标统计
        self.seer_inv_wolf=0
        self.witch_save_good=0
        self.witch_poison_bad=0
        self.hunter_shoot_wolf=0
        self.guard_the_attacked=0
        self.guard_good=0
        self.bad_players=[]
        self.wolf_surviving_round=0
        self.logger.info("=" * 5 + f"WereWolfEnv Init Successfully!: " + "=" * 5)
        if self.args.use_message_pool == True:
            self.messagePool = MessagePool(self.args.message_pool_args)
        else:
            self.messagePool = None
        self.current_round_experiences = []  # 用于存储当前回合的经验消息
    
    def _get_messagePool(self):
        return self.messagePool



    def _get_messagePool(self):
        return self.messagePool
    
    def _init_players_v2(self):
        """根据配置文件初始化所有玩家角色，保持与原 _init_players 完全相同的变量赋值"""
        from games.werewolf.model import Guard, Seer, Hunter, Witch
        from games.werewolf.model import SEER, WITCH
        from games.werewolf.model import State
        from games.werewolf.model import Villager
        from games.werewolf.model import WEREWOLF
        from games.werewolf.model import Werewolf
        from games.werewolf.config import get_player_names
        # 获取玩家名称并随机打乱
        player_names = get_player_names()
        random.shuffle(player_names)
        
        # 初始化角色容器
        seer = None
        guard = None
        hunter = None
        witch = None
        villagers = []
        werewolves = []
        
        # 为角色分配玩家名称和模型
        name_pool = player_names.copy()  # 创建副本以防修改原始列表
        
        # 按照配置分配模型
        role_models = {}
        for agent_config in self.args.agent:
            role = agent_config['agent_nick']
            model = agent_config['agent_model_config']
            if role not in role_models:
                role_models[role] = model
        
        # 确保关键角色存在
        if 'Seer' in role_models:
            seer = Seer(name=name_pool.pop(), model=role_models['Seer'])
        
        if 'Guard' in role_models:
            guard = Guard(name=name_pool.pop(), model=role_models['Guard'])
        
        if 'Hunter' in role_models:
            hunter = Hunter(name=name_pool.pop(), model=role_models['Hunter'])
        
        if 'Witch' in role_models:
            witch = Witch(name=name_pool.pop(), model=role_models['Witch'])
        
        # 分配狼人
        werewolf_count = sum(1 for agent in self.args.agent if agent['agent_nick'] == 'Werewolf')
        for _ in range(werewolf_count):
            if len(name_pool) > 0:
                werewolves.append(Werewolf(name=name_pool.pop(), model=role_models.get('Werewolf')))
        
        # 分配村民
        villager_count = sum(1 for agent in self.args.agent if agent['agent_nick'] == 'Villager')
        for _ in range(villager_count):
            if len(name_pool) > 0:
                villagers.append(Villager(name=name_pool.pop(), model=role_models.get('Villager')))
        
        # 初始化所有玩家的游戏视图
        all_players = []
        if seer: all_players.append(seer)
        if guard: all_players.append(guard)
        if hunter: all_players.append(hunter)
        if witch: all_players.append(witch)
        all_players.extend(werewolves)
        all_players.extend(villagers)
        
        all_player_names = [player.name for player in all_players]
        
        for player in all_players:
            other_wolf = (
                next((w.name for w in werewolves if w.name != player.name), None)
                if isinstance(player, Werewolf)
                else None
            )
            tqdm.tqdm.write(f"{player.name} has role {player.role}")
            player.initialize_game_view(
                current_players=all_player_names,
                round_number=0,
                other_wolf=other_wolf,
            )
        
        # 使用与原函数相同的 session_id
        session_id = "10"  # 与原函数一致
        
        # 创建游戏状态
        self.state = State(
            villagers=villagers,
            werewolves=werewolves,
            seer=seer,
            guard=guard,
            hunter=hunter,
            witch=witch,
            session_id=session_id,
        )
        
        # 设置阵营信息，与原函数保持一致
        self.total_wolf = len(werewolves)
        
        # 注意保持与原函数相同的顺序
        self.good_players = []
        if seer: self.good_players.append(seer.name)
        if guard: self.good_players.append(guard.name) 
        if witch: self.good_players.append(witch.name)
        if hunter: self.good_players.append(hunter.name)
        self.good_players.extend([villager.name for villager in villagers])
        
        self.bad_players = [wolf.name for wolf in werewolves]
        
        # 返回与initialize_players相同的返回值，保持接口一致
        return seer, guard, hunter, witch, villagers, werewolves


    def _init_players(self):
        seer, guard, hunter, witch, villagers, werewolves = initialize_players_diy(args=self.args)
        session_id = "10"  # You might want to make this unique per game
        self.state = State(
            villagers=villagers,
            werewolves=werewolves,
            seer=seer,
            guard=guard,
            hunter=hunter,
            witch=witch,
            session_id=session_id,
        )
        self.total_wolf=len(werewolves)
        self.bad_players=[wolf.name for wolf in werewolves]
        self.key_role_num=sum([1 if player else 0 for player in  [seer,guard,hunter,witch]  ])

    def reset(self):
        if self.game is not None:
            self.state=None
            self.game=None
            self.action_seq_pointer = -1
            self.round_debate_players = []
            self.round_vote_players = []
            self.round_votes = {}
            self.round_votes_log = []
            self.round_summ_players = []
        self._init_players()
        self.game = GameMaster(self.state, num_threads=_THREADS)
        self.logger.info("=" * 5 + f"WereWolfEnv Reset successfully!: " + "=" * 5)

    def reset_new_round(self):
        # 新来一轮，可以从player属性判断是否用了药/子弹
        self.game.state.rounds.append(Round())
        self.game.logs.append(RoundLog())
        self.game.this_round.players = (
            list(self.game.state.players.keys())
            if self.game.current_round_num == 0
            else self.game.state.rounds[self.game.current_round_num - 1].players.copy()
        )
        self.logger.info(f"STARTING ROUND: {self.game.current_round_num}")
        self.action_seq_pointer += 1
        cur_action = self.action_seq[self.action_seq_pointer]
        # Execute Night Phase
        if cur_action == "eliminate":
            message = "The Werewolves are picking someone to remove from the game."
            self.logger.info(message)
            return self.game.eliminate_pre()

    def _wandb_log(self):
        # IRP(Identity Inference Accuracy) (correct_identifications / total_identification_attempts) * 100%
        irp=self.vote_wolf/self.good_vote_times if self.good_vote_times!=0 else 0

        # IUR(Information utilization) (effective_use_of_information / total_available_information) * 100%
        wandb.log({"irp":irp},step=self.game.current_round_num)
        self.logger.info(f"IRP: {irp}")

    def to_dict(self,o: Any) -> Union[Dict[str, Any], List[Any], Any]:
        return json.loads(JsonEncoder().encode(o))
    def save_game(self,state: State, logs: List[RoundLog], directory: str):
        """Save the current game state to a specified file.

        This function serializes the game state to JSON and writes it to the
        specified file. If an error message is provided, it adds the error
        message to the current round of the game state before saving.

        Args:
          state: Instance of the `State` class.
          logs: Logs of the  game.
          directory: where to save the game.
        """
        os.makedirs(directory, exist_ok=True)

        partial_game_state_file = f"{directory}/game_partial.json"
        if state.error_message:
            game_file = partial_game_state_file
        else:
            game_file = f"{directory}/game_complete.json"
            # Remove the partial game file if it exists
            if os.path.exists(partial_game_state_file):
                os.remove(partial_game_state_file)

        log_file = f"{directory}/game_logs.json"

        with open(game_file, "w") as file:
            json.dump(state.to_dict(), file, indent=4)

        with open(log_file, "w") as file:
            json.dump(self.to_dict(logs), file, indent=4)
    def _wandb_ksr(self,key_role_alive):
        # KSR(Key role survival rate) (key_role_survived / total_key_role_games) * 100%
        ksr=1
        if isinstance(key_role_alive,int):
            ksr=key_role_alive/self.key_role_num if self.key_role_num!=0 else 0
        # VSS(Key voting success rate) (successful_votes / total_critical_votes) * 100%
        vss=self.vote_exile_wolf/self.valid_roundvote_times if self.valid_roundvote_times!=0 else 0
        wandb.log({"ksr":ksr,"vss":vss},step=self.game.current_round_num)
        self.logger.info(f"""KSR: {ksr}%, VSS: {vss}, round: {self.game.current_round_num}%""")

    def _wandb_seer(self):
        wandb.log({"seer_kpi":self.seer_inv_wolf/self.total_wolf},step=self.game.current_round_num)
        self.logger.info(f"seer_kpi: {self.seer_inv_wolf/self.total_wolf}, round: {self.game.current_round_num+1}")
    
    def _wandb_witch(self):
        potion_used = float(self.game.state.witch.has_poisoned+ self.game.state.witch.has_saved)
        if potion_used==0:
            return
        wandb.log({"witch_kpi":(self.witch_save_good+self.witch_poison_bad)/potion_used},step=self.game.current_round_num)
        self.logger.info(f"witch_kpi: {(self.witch_save_good+self.witch_poison_bad)/potion_used}, round: {self.game.current_round_num+1}")
    
    def _wandb_hunter(self):
        wandb.log({"hunter_kpi":self.hunter_shoot_wolf},step=self.game.current_round_num)
        self.logger.info(f"hunter_kpi: {self.hunter_shoot_wolf}, round: {self.game.current_round_num+1}")

    def _wandb_guard(self,alpha=0.5):
        guard_kpi=alpha*self.guard_good+(1-alpha)*self.guard_the_attacked
        wandb.log({"guard_kpi":guard_kpi/(self.game.current_round_num+1)},step=self.game.current_round_num)
        self.logger.info(f"guard_kpi: {guard_kpi/(self.game.current_round_num+1)}, round: {(self.game.current_round_num+1)}")
    
    def _wandb_wolves(self,alpha=0.5):
        num_surviving_wolves= len([wolf for wolf in self.game.state.werewolves if wolf.name in self.game.this_round.players])
        num_total_wolves=len(self.bad_players)
        self.wolf_surviving_round += num_surviving_wolves
        wolves_kpi = alpha * (num_surviving_wolves / num_total_wolves) + (1 - alpha) * (self.wolf_surviving_round /num_total_wolves / (self.game.current_round_num + 1))
        wandb.log({"wolves_kpi": wolves_kpi}, step=self.game.current_round_num)
        self.logger.info(f"wolves_kpi: {wolves_kpi}, round: {self.game.current_round_num+1}")

    def _wandb_sheriff(self):
        # 信任度与改票率
        if self.args.has_sheriff==False or self.game.this_round.sheriff is None:
            return
        votes=self.game.this_round.votes[-1]
        pseudo_votes=self.game.this_round.pseudo_votes[-1]
        sheriff=self.game.this_round.sheriff
        sheriff_vote=votes[sheriff]
        count=0
        count_star=0
        for player in self.game.this_round.players:
            if player==sheriff:
                continue
            else:
                if (pseudo_votes[player]==sheriff_vote) and (pseudo_votes[player]!=votes[player]):
                    # 跟警长改票
                    count+=1
                elif (pseudo_votes[player]!=votes[player]):
                    count_star+=1
        decision_change=count/len(self.game.this_round.players)
        decision_change_star=count_star/len(self.game.this_round.players)
        # wandb.log({"decision_change": decision_change,"decision_change_star": decision_change_star},step=self.game.current_round_num)
        self.logger.info(f"Round {self.game.current_round_num} - Decision Change: {decision_change:.2f}, Decision Change Star: {decision_change_star:.2f}")
        # wandb.log({'sheriff':sheriff},step=self.game.current_round_num)
        self.logger.info(f"Round {self.game.current_round_num} - Sheriff: {sheriff}")


    def step(self,action,pre_observations=None):
        self.action_seq_pointer += 1
        cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "new_round":
            observations=self.reset_new_round()
            if observations is not None:
                return observations,None

        # 1. input None and return the observed value
        # 2. input action and return the next observed value
        # 解决上一个，处理当前，执行下一个
        if cur_action=="protect":
            # 广播狼刀人的结果
            eliminated,log=action 
            self.game.eliminate_post(pre_observations,eliminated,log)
            if self.game.state.guard is None or self.game.state.guard.name not in self.game.this_round.players:
                # 如果不在，跳过，执行下一个save
                self.action_seq_pointer =self.action_seq.index("save")
                cur_action = self.action_seq[self.action_seq_pointer]  # action的名字
            else:
                # 如果在，执行protect
                # options=list(self.game.state.guard.gamestate.current_players)
                last_guarded=None
                if self.game.state.guard.guarded !=[]:
                    last_guarded = self.game.state.guard.guarded[-1]
                options = [player for player in self.game.this_round.players if player != last_guarded ]
                random.shuffle(options)
                observations = {
                    "player_name": self.game.state.guard.name,
                    "game_state": self.game.state.guard._get_game_state(),
                    "action": "protect",
                    "options": options
                }
                return observations,None
        if cur_action=="save":
            if pre_observations is not None and pre_observations['action']=="protect":
                # 如果守卫还活着，广播守卫的结果
                protected,log=action
                self.game.protect_post(protected,log) 
                if protected not in self.bad_players:
                    self.guard_good += 1
                if protected==self.game.this_round.eliminated:
                    self.guard_the_attacked += 1
                self._wandb_guard()
            if self.game.state.witch is None or self.game.state.witch.name not in self.game.this_round.players or self.game.this_round.eliminated is None or self.game.state.witch.has_saved == True:
                # 女巫不在，执行下一个posion
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                self.game.state.witch._add_observation("During the"
                                                        f" night, werewolves decided to"
                                                        f" eliminate {self.game.this_round.eliminated}.")
                options = ["Yes", "No"]
                observations = {
                    "player_name": self.game.state.witch.name,
                    "game_state": self.game.state.witch._get_game_state(),
                    "action": "save",
                    "options": options
                }
                return observations,None
        if cur_action=="poison":
            if pre_observations is not None and pre_observations['action']=="save":
                #  如果女巫在，广播女巫救人的结果
                saved,log=action
                self.game.save_post(saved,log)
                if saved not in self.bad_players:
                    self.witch_save_good += 1
            if self.game.state.witch is None or self.game.state.witch.name not in self.game.this_round.players or self.game.state.witch.has_poisoned == True:
                # 如果女巫不在，或者已经毒过人了，执行下一个unmask
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                # 女巫毒人
                options=[player for player in self.game.this_round.players if player != self.game.state.witch.name]
                random.shuffle(options)
                options.append("No")
                observations = {
                    "player_name": self.game.state.witch.name,
                    "game_state": self.game.state.witch._get_game_state(),
                    "action": "poison",
                    "options": options
                }
                return observations,None
        if cur_action=="unmask":
            if pre_observations is not None and pre_observations['action']=="poison":
                poison, log = action
                self.game.poison_post(poison, log)
                if poison in self.bad_players:
                    self.witch_poison_bad+=1
                self._wandb_witch()
            if self.game.state.seer is None or self.game.state.seer.name not in self.game.this_round.players:
                # 预言家死掉了，执行下一个：处理夜晚
                self.action_seq_pointer = self.action_seq.index("resolve_night_phase")
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                options = [
                    player
                    for player in self.game.state.seer.gamestate.current_players
                    if player != self.game.state.seer.name and player not in self.game.state.seer.previously_unmasked.keys()
                ]
                random.shuffle(options)
                observations = {
                    "player_name": self.game.state.seer.name,
                    "game_state": self.game.state.seer._get_game_state(),
                    "action": "investigate",
                    "options": options
                }
                return observations,None
        if cur_action=="resolve_night_phase":
            if pre_observations is not None and pre_observations['action']=="investigate":
                unmask, log = action
                self.game.this_round_log.investigate = log
                # 执行下一个，判断是否为猎人
                if unmask is not None:
                    self.game.this_round.unmasked = unmask
                    self.game.state.seer.reveal_and_update(unmask, self.game.state.players[unmask].role)
                else:
                    raise ValueError("Unmask function did not return a valid player.")
                if unmask in self.bad_players:
                    self.seer_inv_wolf += 1
                self._wandb_seer()
            self.game.resolve_night_phase()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="check_for_winner":
            self.game.check_for_winner()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="check_for_shoot":
            if self.game.state.hunter is None:
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            elif self.game.this_round.eliminated != self.game.this_round.protected and self.game.this_round.eliminated != self.game.this_round.saved:
                # 有人真的被狼刀了
                if self.game.this_round.eliminated == self.game.state.hunter.name:
                    # 猎人被狼刀了
                    self.game.state.hunter._add_observation(
                        f"You were eliminated by the werewolves. You can now shoot someone."
                    )
                    options = [
                        player
                        for player in self.game.this_round.players
                        if player != self.game.state.hunter.name
                    ]
                    random.shuffle(options)
                    observations = {
                        "player_name": self.game.state.hunter.name,
                        "game_state": self.game.state.hunter._get_game_state(),
                        "action": "shoot",
                        "options": options
                    }
                    return observations,None
                # 其他情况，执行下一个
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                # 其他情况，执行下一个
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]

        if cur_action=="check_for_winner":
            # 执行下一个走白天流程
            if pre_observations is not None and pre_observations['action']=="shoot":
                shoot, log = action
                self.game.shoot_post(shoot, log)
                if shoot in self.bad_players:
                    self.hunter_shoot_wolf += 1
                self._wandb_hunter()
            self.game.check_for_winner()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        
        if cur_action=="badge_flow_1":
            if self.game.current_round_num==0:
                # 首回合不执行badge_flow 进入警长竞选
                self.action_seq_pointer+=1
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                if self.game.this_round.eliminated == self.game.last_round.sheriff and self.game.this_round.eliminated is not None :
                    # 警长被杀：badge flow
                    self.game.badge_flow()
                    cur_action='determine_statement_order'
                    self.action_seq_pointer=self.action_seq.index(cur_action)-1 # 每次step开头加一，这样保证它会跳转到determine_statement_order
                    if self.game.last_round.sheriff == None:
                        print("error occurs")
                    player=self.state.players[self.game.last_round.sheriff]
                    options=[ player_name for player_name in player.gamestate.current_players
                            if player_name != player.name       ]
                    observations={
                        'player_name':player.name,
                        'game_state':player._get_game_state(),
                        "action":'badge_flow',
                        'options': options
                    }
                    return observations,None
                else:
                    if self.game.last_round.sheriff == None:
                        # 不存在警长 跳过整个
                        cur_action = 'run_day_phase'
                        self.action_seq_pointer =self.action_seq.index(cur_action)
                    else:
                        # 存在警长：
                        # 警长没有被杀
                        self.game.determine_sheriff(sheriff=self.game.last_round.sheriff,badge_flow=False)
                        cur_action='determine_statement_order'
                        self.action_seq_pointer=self.action_seq.index(cur_action) # 不返回 直接跳转到determine_statement_order
        if cur_action=="run_for_sheriff":
            if pre_observations is not None and pre_observations['action'] == "run_for_sheriff":
                run_for_sheriff, log=action
                pre_player_name=pre_observations['player_name']
                pre_player=self.game.state.players[pre_player_name]
                if run_for_sheriff=='True' or run_for_sheriff == True:
                    self.game.add_round_candidate(pre_player)
            next_decider=None
            for decider in self.game.this_round.players:
                if decider not in self.round_vote_players:
                    next_decider = decider
                    self.round_vote_players.append(decider)
                    break
            if next_decider is not None:
                player = self.state.players[next_decider]
                # options=["True", "False",True,False]
                options = ['True', 'False']
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "run_for_sheriff",
                    "options": options
                }
                self.action_seq_pointer -= 1
                return observations,None
            else:
                self.round_vote_players=[]
                self.candidates=self.game.this_round.sheriff_candidates
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="sheriff_debate":
            # 警长竞选辩论
            # if pre_observations is not None and pre_observations['action'] == "run_for_sheriff":
            #     run_for_sheriff, log=action
            #     pre_player_name=pre_observations['player_name']
            #     pre_player=self.game.state.players[pre_player_name]
            #     if run_for_sheriff=='True' or run_for_sheriff == True:
            #         self.game.add_round_candidate(pre_player)
            if pre_observations is not None and pre_observations['action']=="sheriff_debate":
                # 更新上一个人的发言信息
                result, log=action
                pre_speaker=pre_observations['player_name']
                if result is not None:
                    dialogue = result.get("say", None)
                    self.game.this_round_log.debate.append((pre_speaker, log))
                    self.game.this_round.debate.append([pre_speaker, dialogue])
                    tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    # update all player gamestate
                    for name in self.game.this_round.players:
                        player = self.game.state.players[name]
                        if player.gamestate:
                            player.gamestate.update_debate(pre_speaker, dialogue)
                        else:
                            raise ValueError(f"{name}.gamestate needs to be initialized.")
                else:
                    raise ValueError(
                        f"{pre_speaker} did not return a valid dialouge from sheriff_debate()."
                    )

            # 2. Select the player without debate
            # 没人选或大家都上警时，没有警长
            # if len(self.round_elect_players) == 0 or len(self.round_elect_players) == len(self.game.this_round.players):
                
            #     self.game.this_round.sheriff_candidates=random.choice(self.game.this_round.players)
            if self.candidates==[] or len(self.candidates) == len(self.game.this_round.players):
                # 检查是所有人都上警/都不上警，还是正常竞选结束
                if len(self.game.this_round.sheriff_candidates) == len(self.game.this_round.players) or len(self.game.this_round.sheriff_candidates) == 0:
                    self.game.no_sheriff()
                    self.action_seq=['new_round','eliminate','protect','unmask','resolve_night_phase','check_for_winner','run_day_phase','vote','exile','check_for_winner','run_summaries']
                    pnt = self.action_seq.index('run_day_phase')
                    self.action_seq_pointer = pnt
                    # 跳过警长竞选和警长总结阶段 这对吗 应该直接从actionseq里移除这些阶段？
                    # while cur_action in ["sheriff_debate", "elect", "determine_statement_order"]:
                    #     self.action_seq_pointer += 1
                    #     cur_action = self.action_seq[self.action_seq_pointer]
                else:
                    # 正常竞选结束，进入投票阶段
                    self.action_seq_pointer += 1
                    cur_action = self.action_seq[self.action_seq_pointer]
            # 警长竞选debate阶段？
            else:
                # 下一个人发言
                player_name=self.candidates[0]
                player=self.state.players[player_name]
                self.candidates=self.candidates[1:]
                observations={
                    'action':cur_action,
                    'player_name':player_name,
                    "game_state": player._get_game_state(),
                    'options': []
                }
                self.action_seq_pointer -= 1
                return observations,None   
        if cur_action=="elect":
            if pre_observations is not None and pre_observations['action'] == "elect":
                # 更新一下前一个人的选警长信息
                vote, log=action
                pre_player_name=pre_observations['player_name']
                pre_player=self.game.state.players[pre_player_name]
                if vote is not None:
                    pre_player._add_observation(
                        f"Before the debate, I elected {vote} to be the sheriff."
                    )
                self.round_elect_log.append(ElectLog(pre_player_name, vote, log))

                if vote is not None:
                    self.round_elect[pre_player_name]=vote
                else:
                    self.game.this_round.elect.append(self.round_elect)
                    self.game.this_round_log.elect.append(self.round_elect_log)
                    self.round_elect = {}
                    self.round_elect_log = []
                    raise ValueError(f"{pre_player_name} vote did not return a valid player.")
            
            # If every player has spoken, then vote
            # Loop through the votes
            next_voter=None
            for voter in self.game.this_round.players:
                # voter = self.game.state.players[name]
                if voter in self.game.this_round.sheriff_candidates:#警长不可投票
                    continue
                if voter not in self.round_elect_players:
                    next_voter = voter
                    self.round_elect_players.append(voter)
                    break
            if next_voter is not None:
                player = self.state.players[next_voter]
                options=[ player_name for player_name in self.game.this_round.sheriff_candidates
                        if player_name != player.name       ]
                random.shuffle(options)
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "elect",
                    "options": options
                }
                self.action_seq_pointer -= 1
                return observations,None
            else:
                # 选警长结束
                self.round_elect_players=[]
                self.game.this_round.elect.append(self.round_elect)
                self.game.this_round_log.elect.append(self.round_elect_log)
                for player, vote in self.game.this_round.elect[-1].items():
                    tqdm.write(f"{player} planed to vote {vote} as the sheriff")
                self.round_elect_log=[]
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="determine_statement_order":
            if pre_observations is not None and (pre_observations['action']=="badge_flow"):
                sheriff,log=action
                self.game.determine_sheriff(sheriff=sheriff,badge_flow=True)
            if pre_observations is not None and (pre_observations['action']=="elect"):
                self.game.determine_sheriff()
                
            # self.game.determine_statement_order()
            # self.statement_order=self.game.this_round.statement_order
            # self.action_seq_pointer += 1
            # cur_action = self.action_seq[self.action_seq_pointer]
            player:Player=self.game.state.players[self.game.this_round.sheriff]
            order=player.gamestate.legal_order(player.name)
            observations={
                        'player_name':player.name,
                        'game_state':player._get_game_state(),
                        "action":'determine_statement_order',
                        'options': order,
                    }
            return observations,None         
        if cur_action=="run_day_phase":
            if pre_observations is not None and pre_observations['action']=="determine_statement_order":
                order, log=action
                self.game.determine_statement_order(order)
                self.statement_order=self.game.this_round.statement_order
            # 1. Process the previous debate results first
            if pre_observations is not None and pre_observations['action']=="debate":
                result, log=action
                pre_speaker=pre_observations['player_name']
                if result is not None:
                    dialogue = result.get("say", None)
                    self.game.this_round_log.debate.append((pre_speaker, log))
                    self.game.this_round.debate.append([pre_speaker, dialogue])
                    # tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    # 修改为:
                    try:
                        tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    except UnicodeEncodeError:
                        # 替换掉无法显示的Unicode字符
                        safe_dialogue = dialogue.encode('ascii', 'replace').decode('ascii')
                        tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {safe_dialogue}")
                    # update all player gamestate
                    for name in self.game.this_round.players:
                        player = self.game.state.players[name]
                        if player.gamestate:
                            player.gamestate.update_debate(pre_speaker, dialogue)
                        else:
                            raise ValueError(f"{name}.gamestate needs to be initialized.")
                else:
                    dialogue="The player says nothing due to the connection error."
                    self.game.this_round_log.debate.append((pre_speaker, log))
                    self.game.this_round.debate.append([pre_speaker, dialogue])
                    # tqdm.write(f"{pre_speaker} did not return a valid dialouge from debate(), the result is {result}")
                    # tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    try:
                        tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    except UnicodeEncodeError:
                        # 替换掉无法显示的Unicode字符
                        safe_dialogue = dialogue.encode('ascii', 'replace').decode('ascii')
                        tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {safe_dialogue}")
                    # update all player gamestate
                    for name in self.game.this_round.players:
                        player = self.game.state.players[name]
                        if player.gamestate:
                            player.gamestate.update_debate(pre_speaker, dialogue)
                        else:
                            raise ValueError(f"{name}.gamestate needs to be initialized.")

            # 2. Select the player without debate
            next_speaker=None
            if self.statement_order==None or self.statement_order==[]:
                next_speaker=None
                for speaker in self.game.this_round.players:
                    # speaker = self.game.state.players[name]
                    if speaker not in self.round_debate_players:
                        next_speaker=speaker
                        self.round_debate_players.append(speaker)
                        break
                if next_speaker is not None:
                    player = self.state.players[next_speaker]
                    observations = {
                        "player_name": player.name,
                        "game_state": player._get_game_state(),
                        "action": "debate",
                        "options": []
                    }
                    self.action_seq_pointer -= 1
                    return observations,None
                else:
                    self.action_seq_pointer += 1
                    cur_action = self.action_seq[self.action_seq_pointer]
                    self.round_debate_players=[]
            else:
                next_speaker=self.statement_order[0]
                self.statement_order=self.statement_order[1:]
                if next_speaker != self.game.this_round.sheriff: # 让警长最后发言
                    player = self.state.players[next_speaker]
                    observations = {
                        "player_name": player.name,
                        "game_state": player._get_game_state(),
                        "action": "debate",
                        "options": []
                    }
                    self.action_seq_pointer -= 1
                    return observations,None
                else:
                    # 跳到pseudo vote
                    self.action_seq_pointer = self.action_seq.index("pseudo_vote")
                    cur_action = self.action_seq[self.action_seq_pointer]
                    self.statement_order=None
        if cur_action=="pseudo_vote":
            if self.game.this_round.sheriff == None:
                cur_action = "vote"
                self.action_seq_pointer=self.action_seq.index(cur_action)
            else:        
                if pre_observations is not None and pre_observations['action'] == "pseudo_vote":
                    # 更新一下前一个人的伪投票信息
                    vote, log=action
                    pre_player_name=pre_observations['player_name']
                    pre_player=self.game.state.players[pre_player_name]
                    if vote is not None:
                        pre_player._add_observation(
                            f"Before the debate, I plan to remove {vote} from the game."
                        )
                    self.round_pseudo_votes_log.append(PseudoVoteLog(pre_player_name, vote, log))

                    if vote is not None:
                        self.round_pseudo_votes[pre_player_name]=vote
                    else:
                        self.game.this_round.pseudo_votes.append(self.round_pseudo_votes)
                        self.game.this_round_log.pseudo_votes.append(self.round_pseudo_votes_log)
                        self.round_pseudo_votes = {}
                        self.round_pseudo_votes_log = []
                        raise ValueError(f"{pre_player_name} vote did not return a valid player.")
                
                # If every player has spoken, then vote
                # Loop through the votes
                next_voter=None
                for voter in self.game.this_round.players:
                    # voter = self.game.state.players[name]
                    if voter not in self.round_pseudo_vote_players:
                        next_voter = voter
                        self.round_pseudo_vote_players.append(voter)
                        break
                if next_voter is not None:
                    player = self.state.players[next_voter]
                    # options=[ player_name for player_name in player.gamestate.current_players if player_name != player.name]
                    options = player.gamestate.current_players

                    random.shuffle(options)
                    observations = {
                        "player_name": player.name,
                        "game_state": player._get_game_state(),
                        "action": "pseudo_vote",
                        "options": options
                    }
                    self.action_seq_pointer -= 1
                    return observations,None
                else:
                    # 伪投票结束
                    self.round_pseudo_vote_players=[]
                    self.game.this_round.pseudo_votes.append(self.round_pseudo_votes)
                    self.game.this_round_log.pseudo_votes.append(self.round_pseudo_votes_log)
                    for player, vote in self.game.this_round.pseudo_votes[-1].items():
                        tqdm.write(f"{player} planed to vote to remove {vote}")
                    self.round_pseudo_votes_log=[]
                    self.action_seq_pointer += 1
                    cur_action = self.action_seq[self.action_seq_pointer]
                    
        if cur_action=='sheriff_summarize':
            if self.game.this_round.sheriff is None:
                self.action_seq_pointer = self.action_seq.index("vote")
                cur_action = self.action_seq[self.action_seq_pointer]
            else:
                sheriff=self.game.state.players[self.game.this_round.sheriff]
                player=sheriff
                observations = {
                        "player_name": player.name,
                        "game_state": player._get_game_state(),
                        "action": "sheriff_summarize",
                        "options": []
                    }
                return observations,None
                            
        if cur_action=="vote":
            # 1. Process the vote return value first
            if pre_observations is not None and pre_observations['action'] == "sheriff_summarize":
                # 更新警长发言
                result, log=action
                pre_speaker=pre_observations['player_name']
                if result is not None:
                    dialogue = result.get("say", None)
                    self.game.this_round_log.debate.append((pre_speaker, log))
                    self.game.this_round.debate.append([pre_speaker, dialogue])
                    tqdm.write(f"{pre_speaker} ({self.game.state.players[pre_speaker].role}): {dialogue}")
                    # update all player gamestate
                    for name in self.game.this_round.players:
                        player = self.game.state.players[name]
                        if player.gamestate:
                            player.gamestate.update_debate(pre_speaker, dialogue)
                        else:
                            raise ValueError(f"{name}.gamestate needs to be initialized.")
                        
            # 在投票阶段开始时初始化存活玩家列表
            if len(self.round_vote_players) == 0:
                # 记录当前存活的狼人和好人
                cur_alive_wolf = [w for w in self.game.state.werewolves if w.name in self.game.this_round.players]
                self.cur_alive_wolf_name = [w.name for w in cur_alive_wolf]
                self.cur_alive_good_name = [w for w in self.game.this_round.players if w not in self.cur_alive_wolf_name]

            if pre_observations is not None and pre_observations['action'] == "vote":
                vote, log=action
                pre_player_name=pre_observations['player_name']
                pre_player=self.game.state.players[pre_player_name]

                ###! message pool starting
                # 获取最后一次有效的reflection
                reflection = ""
                if isinstance(log, LmLog) and log.result:
                    reflection = log.result.get("reasoning", "")
                elif isinstance(log, LmLog) and log.raw_resp:
                    json_strs = log.raw_resp.split('-------')
                    for json_str in reversed(json_strs):
                        try:
                            result = json.loads(json_str)
                            if "reasoning" in result:
                                reflection = result["reasoning"]
                                break
                        except json.JSONDecodeError:
                            continue
                
                # 计算reward
                reward = 0
                is_choose = 0
                
                # 如果投票成功投中了狼人
                if vote in self.cur_alive_wolf_name:
                    reward = 1000 - self.game.current_round_num
                    is_choose = 1
                # 如果投票投中了好人
                elif vote in self.cur_alive_good_name:
                    reward = self.game.current_round_num
                    is_choose = 1
                
                # 创建经验消息
                experience = Message(
                    agent_name=pre_player_name,
                    content=(reflection, vote, reward, is_choose),
                    turn=self.game.current_round_num,
                    stage="vote",
                    role=pre_player.role,
                    msg_type="exp"
                )
                
                # 添加到临时列表
                self.current_round_experiences.append(experience)
                self._update_and_save_experiences()






                ###! message pool ending
                if vote is not None:
                    pre_player._add_observation(
                        f"After the debate, I voted to remove {vote} from the game."
                    )
                self.round_votes_log.append(VoteLog(pre_player_name, vote, log))

                if vote is not None:
                    self.round_votes[pre_player_name]=vote
                else:
                    self.game.this_round.votes.append(self.round_votes)
                    self.game.this_round_log.votes.append(self.round_votes_log)
                    self.round_votes = {}
                    self.round_votes_log = []
                    raise ValueError(f"{pre_player_name} vote did not return a valid player.")
            
            # If every player has spoken, then vote
            # Loop through the votes
            next_voter=None
            for voter in self.game.this_round.players:
                # voter = self.game.state.players[name]
                if voter not in self.round_vote_players:
                    next_voter = voter
                    self.round_vote_players.append(voter)
                    break
            if next_voter is not None:
                player = self.state.players[next_voter]
                options=[ player_name for player_name in player.gamestate.current_players
                        if player_name != player.name       ]
                random.shuffle(options)
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "vote",
                    "options": options
                }
                self.action_seq_pointer -= 1
                return observations,None
            else:
                self.round_vote_players=[]
                self.game.this_round.votes.append(self.round_votes)
                self.game.this_round_log.votes.append(self.round_votes_log)
                # ====record vote for wandb====
                cur_alive_wolf=[w for w in self.game.state.werewolves if w.name in self.game.this_round.players]
                self.cur_alive_wolf_name=[w.name for w in cur_alive_wolf]
                self.cur_alive_good_name=[w for w in self.game.this_round.players if w not in self.cur_alive_wolf_name]
                for k,v in self.round_votes.items():
                    if k in self.cur_alive_good_name:
                        self.good_vote_times+=1
                        if v in self.cur_alive_wolf_name:
                            self.vote_wolf+=1
                self.all_vote_times+=1
                self._wandb_log()
                # ====record vote for wandb====
                for player, vote in self.game.this_round.votes[-1].items():
                    tqdm.write(f"{player} voted to remove {vote}")
                self.round_votes_log=[]
                self.action_seq_pointer = self.action_seq.index("exile")
                cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=="exile":
            self.game.exile()
            # record

            if self.game.this_round.exiled :
                self.valid_roundvote_times+=1
                if self.game.this_round.exiled in self.cur_alive_wolf_name:
                    self.vote_exile_wolf+=1
            key_role_alive=0
            if self.game.state.seer and self.game.state.seer.name in self.game.this_round.players:
                key_role_alive+=1
            if self.game.state.guard and self.game.state.guard.name in self.game.this_round.players:
                key_role_alive+=1
            # 添加关键人物 女巫 猎人
            if self.game.state.hunter and self.game.state.hunter.name in self.game.this_round.players:
                key_role_alive+=1
            if self.game.state.witch and self.game.state.witch.name in self.game.this_round.players:
                key_role_alive+=1
            self._wandb_ksr(key_role_alive)
            self._wandb_wolves()
            self._wandb_sheriff()
            
            
            

            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "check_for_winner":
            self.game.check_for_winner()
            self.action_seq_pointer += 1
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "check_for_shoot":
            # Process the returned results first
            # if pre_observations is not None and pre_observations['action'] == "summarize":
            #     result, log=action
            #     pre_player_name = pre_observations['player_name']
            #     pre_player = self.game.state.players[pre_player_name]
            #     if result is not None:
            #         summary = result.get("summary", None)
            #         if summary is not None:
            #             summary = summary.strip('"')
            #             pre_player._add_observation(f"Summary: {summary}")
            #         tqdm.write(f"{pre_player_name} summary: {summary}")
            #         self.game.this_round_log.summaries.append((pre_player_name, log))
            if self.game.state.hunter is None:
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
            elif self.game.this_round.exiled==self.game.state.hunter.name:
                # 猎人被狼刀了
                self.game.state.hunter._add_observation(
                    f"You were eliminated by the werewolves. You can now shoot someone."
                )
                options = [
                    player
                    for player in self.game.this_round.players
                    if player != self.game.state.hunter.name
                ]
                random.shuffle(options)
                observations = {
                    "player_name": self.game.state.hunter.name,
                    "game_state": self.game.state.hunter._get_game_state(),
                    "action": "shoot",
                    "options": options
                }
                return observations,None
            else:
                # 其他情况，执行下一个
                self.action_seq_pointer += 1
                cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "check_for_winner":
            if pre_observations is not None and pre_observations['action']=="shoot":
                shoot, log = action
                self.game.shoot_post(shoot, log)
                if shoot in self.bad_players:
                    self.hunter_shoot_wolf += 1
                self._wandb_hunter()
            self.game.check_for_winner()
            self.action_seq_pointer = self.action_seq.index("badge_flow_2")
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action=='badge_flow_2':
            # 警长被投出去了
            if self.game.this_round.exiled == self.game.this_round.sheriff or self.game.this_round.shot==self.game.this_round.sheriff:
                # badge flow
                player=self.state.players[self.game.this_round.sheriff]
                self.game.badge_flow()
                options=[ player_name for player_name in player.gamestate.current_players
                        if player_name != player.name       ]
                observations={
                    'player_name':player.name,
                    'game_state':player._get_game_state(),
                    "action":'badge_flow',
                    'options': options
                }
                return observations,None
            self.action_seq_pointer=self.action_seq.index("run_summaries")
            cur_action = self.action_seq[self.action_seq_pointer]
        if cur_action == "run_summaries":
            # Process the returned results first
            if pre_observations is not None and (pre_observations['action']=="badge_flow"):
                sheriff,log=action
                self.game.determine_sheriff(sheriff=sheriff,badge_flow=True)
            if pre_observations is not None and pre_observations['action'] == "summarize":
                result, log=action
                pre_player_name = pre_observations['player_name']
                pre_player = self.game.state.players[pre_player_name]
                if result is not None:
                    summary = result.get("summary", None)
                    if summary is not None:
                        summary = summary.strip('"')
                        pre_player._add_observation(f"Summary: {summary}")
                    try:
                        tqdm.write(f"{pre_player_name} ({pre_player.role}) summary: {summary}")
                    except UnicodeEncodeError:
                        # 替换掉无法显示的Unicode字符
                        safe_summary = summary.encode('ascii', 'replace').decode('ascii')
                        tqdm.write(f"{pre_player_name} ({pre_player.role}) summary: {safe_summary}")
                    # tqdm.write(f"{pre_player_name} summary: {summary}")
                    self.game.this_round_log.summaries.append((pre_player_name, log))
            
            # To summarize
            next_summ_player = None
            for summ_player in self.game.this_round.players:
                if summ_player not in self.round_summ_players:
                    next_summ_player = summ_player
                    self.round_summ_players.append(summ_player)
                    break
            if next_summ_player is not None:
                player = self.state.players[next_summ_player]
                observations = {
                    "player_name": player.name,
                    "game_state": player._get_game_state(),
                    "action": "summarize",
                    "options": []
                }
                self.action_seq_pointer -= 1
                return observations, None
            else:
                self.round_summ_players=[]
                if self.game.state.winner:
                    tqdm.write(f"Round {self.game.current_round_num} is complete.")
                    self.game.this_round.success = True
                    return None, self.game.state.winner
                else:
                    for name in self.game.this_round.players:
                        if self.game.state.players[name].gamestate:
                            self.game.state.players[name].gamestate.round_number = (
                                    self.game.current_round_num + 1
                            )
                            self.game.state.players[name].gamestate.clear_debate()
                    self.game.current_round_num += 1
                    self.action_seq_pointer=0
                    observations = self.reset_new_round()
                    if observations is not None:
                        return observations, None

    def _update_and_save_experiences(self):
        """
        更新当前回合所有经验的reward并保存到MessagePool
        """
        if self.messagePool == None:
            print("no message pool")
            return
        winner_role = "Werewolf" if self.game.state.winner == "Werewolves" else "Villager"
        
        for exp in self.current_round_experiences:
            # 根据玩家角色和获胜方更新reward
            if exp.role == winner_role:
                # 获胜方获得更高reward
                exp.content = (exp.content[0], exp.content[1], 1000 - self.game.current_round_num, exp.content[3])
            else:
                # 失败方获得较低reward
                exp.content = (exp.content[0], exp.content[1], self.game.current_round_num, exp.content[3])
            
            # 添加到MessagePool
            self.messagePool.append_message(exp)
        
        # 保存经验到文件
        self.messagePool.save_exps_to(is_incremental=True)
        
        # 清空临时列表
        self.current_round_experiences = []

    def log_directory(self) -> str:
        import datetime
        pacific_timezone = datetime.timezone(datetime.timedelta(hours=-8))
        timestamp = datetime.datetime.now(pacific_timezone).strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        directory = f"{os.getcwd()}/games/werewolf/visual/logs/{session_id}"
        return directory

    def render(self):
        self.logger.warnning("WereWolfEnv has no render!!!")
        return None
class JsonEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, enum.Enum):
      return o.value
    if isinstance(o, set):
      return list(o)
    return o.__dict__