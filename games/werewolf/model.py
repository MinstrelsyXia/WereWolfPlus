# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from games.werewolf.lm import LmLog, generate
from agent_manager.prompts.werewolf_prompt import ACTION_PROMPTS_AND_SCHEMAS 
from games.werewolf.utils import Deserializable  
from games.werewolf.config import  MAX_DEBATE_TURNS, NUM_PLAYERS, NUM_VILLAGERS

# Role names
VILLAGER = "Villager"  # 村民
WEREWOLF = "Werewolf"  # 狼人
SEER = "Seer"   # 预言家
# DOCTOR = "Doctor"  # 原来的医生，改为守卫
WITCH = "Witch" # 女巫 
GUARD = "Guard" # 守卫
HUNTER = "Hunter"  # 猎人
# Seer：investigate √
# Witch：save √，poison √
# Hunter：shoot
# Guard：protect √


def group_and_format_observations(observations):
  """Groups observations by round and formats them for output.

  Args:
      observations: A list of strings, where each string starts with "Round X:".

  Returns:
      A list of strings, where each string represents the formatted observations
      for a round.
  """
  # 结构化过去的observation
  # 输入是一个str list，每个元素格式为 "Round X: observation content"
  grouped = {}
  for obs in observations:
    round_num = int(obs.split(":", 1)[0].split()[1]) # get round number
    obs_text = obs.split(":", 1)[1].strip().replace('"', "") # get observation content and remove quotes
    grouped.setdefault(round_num, []).append(obs_text) # 改成结构：round_num: [obs_text](把一轮的observation按顺序整理到一起)

  formatted_obs = []
  for round_num, round_obs in sorted(grouped.items()):
    formatted_round = f"Round {round_num}:\n"
    formatted_round += "\n".join(f"   - {obs}" for obs in round_obs)
    formatted_obs.append(formatted_round)
    # 重新整理成：Round i\n -obs\n - obs\n -obs\n...

  return formatted_obs


# JSON serializer that works for nested classes
class JsonEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, enum.Enum):
      return o.value
    if isinstance(o, set):
      return list(o)
    return o.__dict__

def to_dict(o: Any) -> Union[Dict[str, Any], List[Any], Any]:
  return json.loads(JsonEncoder().encode(o))

class GameView:
  """Represents the state of the game for each player."""
  # 游戏视野 of every player 
  def __init__(
      self,
      round_number: int,
      current_players: List[str],
      other_wolf: Optional[str] = None,
  ):
    self.round_number: int = round_number # 当前轮次
    self.current_players: List[str] = current_players  # 当前玩家列表
    self.debate: List[tuple[str, str]] = []  # 元组列表（玩家，发言）
    self.other_wolf: Optional[str] = other_wolf # 其他狼人玩家的名字(?)
    self.sheriff: Optional[str] = None
    self.sheriff_candidates: List[str] = []

  def update_debate(self, author: str, dialogue: str):
    """Adds a new dialogue entry to the debate."""
    self.debate.append((author, dialogue))

  def clear_debate(self):
    """Clears all entries from the debate."""
    self.debate.clear()

  def remove_player(self, player_to_remove: str):
    """Removes a player from the list of current players."""
    if player_to_remove not in self.current_players:
      print(f"Player {player_to_remove} not in current players:"f" {self.current_players}" )
      return
    self.current_players.remove(player_to_remove)

  def to_dict(self) -> Any:
    return to_dict(self)
  
  # 警长相关代码
  def add_sheriff(self, sheriff: str):
    """Removes a player from the list of current players."""
    if sheriff not in self.current_players:
      print(f"Player {sheriff} not in current players:"f" {self.current_players}" )
      return
    self.sheriff = sheriff
  
  def add_candidates(self, candidate:str):
    if candidate not in self.current_players:
      print(f"Player {candidate} not in current players:"f" {self.current_players}" )
    else:
      self.sheriff_candidates.append(candidate)
  
  def legal_order(self, myname) -> List[str]:
    """Returns the name of a player who is left in the game."""
    my_idx=self.current_players.index(myname)
    if my_idx==0:
      left_idx=-1
      right_idx=1
    elif my_idx==len(self.current_players)-1:
      left_idx=my_idx-1
      right_idx=0
    else:
      left_idx=my_idx-1
      right_idx=my_idx+1
    left_order=[]
    right_order=[]
    for i in range(len(self.current_players)):
      left_order.append((left_idx-i)%len(self.current_players))
      right_order.append((right_idx+i)%len(self.current_players))

    left_name_order = "[" + ", ".join(self.current_players[left_order[i]] for i in range(len(left_order))) + "]"
    right_name_order = "[" + ", ".join(self.current_players[right_order[i]] for i in range(len(right_order))) + "]"

    return [left_name_order, right_name_order]

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    return cls(**data)


class Player(Deserializable):
  """Represents a player in the game."""

  def __init__(
      self,
      name: str,
      role: str,
      model: Optional[str] = None,
      personality: Optional[str] = "",
      num_players: int = NUM_PLAYERS, # NUM_PLAYERS是假的
      num_villagers: int = NUM_VILLAGERS, # NUM_VILLAGERS是假的

  ):
    self.name = name
    self.role = role
    self.personality = personality  # 性格
    self.model = model  #？
    self.observations: List[str] = []  # 观察内容的str列表
    self.bidding_rationale = ""  # 投票理由？
    self.gamestate: Optional[GameView] = None
    self.num_players = num_players  # 玩家数量
    self.num_villagers = num_villagers  # 村民数量
    self.max_debate_turns = num_players  # 最大辩论轮次
    self.is_sheriff=False

  def initialize_game_view(
      self, round_number, current_players, other_wolf=None
  ) -> None:
    # 初始化游戏视野
    self.gamestate = GameView(round_number, current_players, other_wolf)

  def _add_observation(self, observation: str):
    """Adds an observation for the given round."""
    # 添加观察内容到observation列表
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )  # 要获得轮次
    self.observations.append(
        f"Round {self.gamestate.round_number}: {observation}"
    )  # 记下来每轮的 observation（字符串）

  def add_announcement(self, announcement: str):
    """Adds the current game announcement to the player's observations."""
    self._add_observation(f"Moderator Announcement: {announcement}")  # 记录主持人公告

  def _get_game_state(self) -> Dict[str, Any]:
    """Gets the current game state from the player's perspective."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    remaining_players = [
        f"{player} (You)" if player == self.name else player
        for player in self.gamestate.current_players
    ]
    random.shuffle(remaining_players)  # 一个随机打乱的玩家列表（包括自己）
    formatted_debate = [
        f"{author} (You): {dialogue}"
        if author == self.name
        else f"{author}: {dialogue}"
        for author, dialogue in self.gamestate.debate
    ]  # 每个人的debate
    formatted_observations = group_and_format_observations(self.observations) # 结构化过去所有的observation（为啥不能结构化一次再存回去？或者存的时候就结构化）
    return {
        "name": self.name,
        "role": self.role,
        "round": self.gamestate.round_number,
        "observations": formatted_observations,
        "remaining_players": ", ".join(remaining_players),
        "debate": formatted_debate,
        "bidding_rationale": self.bidding_rationale,
        "debate_turns_left": self.num_players - len(formatted_debate),  # 剩几轮游戏
        "personality": self.personality, # 你的性格
        "num_players": self.num_players,
        "num_villagers": self.num_villagers, # 4根据游戏修正
    }

  def _generate_action(
      self,
      action: str,
      options: Optional[List[str]] = None,
  ) -> tuple[Any , LmLog]:
    """Helper function to generate player actions."""
    # 输入动作：选择prompt format
    # 输入需要的options
    game_state = self._get_game_state() # 根据过去全部信息选择动作
    if options:
      game_state["options"] = (", ").join(options) # 添加options
    prompt_template, response_schema = ACTION_PROMPTS_AND_SCHEMAS[action]  # 按照动作选择prompt

    # 原版里没放 debate, summarize
    result_key, allowed_values = (
        (action, options)
        if action in ["vote", "remove", "investigate", "protect", "bid",
                       "save", "poison", "shoot", 
                       "pseudo_vote","elect","determine_statement_order",
                      "sheriff_summarize","badge_flow","sheriff_debate","run_for_sheriff"]
        else (None, None)
    )  # investigate对应Seer？ protect对应Guard

    # Set temperature based on allowed_values
    temperature = 0.5 if allowed_values else 1.0

    return generate(
        prompt_template,
        response_schema,
        game_state,
        model=self.model,
        temperature=temperature,
        allowed_values=allowed_values,
        result_key=result_key,
    )

  # 下面根据可选的所有动作定义具体的函数，包括如何传递需要的prompt format keys
  # 总的player，特定角色的放在特定角色中定义
  def vote(self) -> tuple[str , LmLog]:
    """Vote for a player."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    options = [
        player
        for player in self.gamestate.current_players
        if player != self.name
    ]
    random.shuffle(options)
    vote, log = self._generate_action("vote", options)
    if vote is not None and len(self.gamestate.debate) == self.max_debate_turns:
      self._add_observation(
          f"After the debate, I voted to remove {vote} from the game."
      )
    return vote, log

  def bid(self) -> tuple[int , LmLog]:
    """Place a bid."""
    bid, log = self._generate_action("bid", options=["0", "1", "2", "3", "4"])
    if bid is not None:
      bid = int(bid)
      self.bidding_rationale = log.result.get("reasoning", "")
    return bid, log

  def debate(self) -> tuple[str , LmLog]:
    """Engage in the debate."""
    result, log = self._generate_action("debate", [])
    if result is not None:
      say = result.get("say", None)
      return say, log
    return result, log

  def summarize(self) -> tuple[str , LmLog]:
    """Summarize the game state."""
    result, log = self._generate_action("summarize", [])
    if result is not None:
      summary = result.get("summary", None)
      if summary is not None:
        summary = summary.strip('"')
        self._add_observation(f"Summary: {summary}")
      return summary, log
    return result, log

  def to_dict(self) -> Any:
    return to_dict(self)
  
  def run_for_sheriff(self) -> tuple[bool , LmLog]:
    """Run for sheriff"""
    result, log = self._generate_action("run_for_sheriff", [])
    return result,log
  
  def sheriff_debate(self) -> tuple[str , LmLog]:
    """Engage in the debate."""
    result, log = self._generate_action("sheriff_debate", [])
    if result is not None:
      say = result.get("say", None)
      return say, log
    return result, log
  
  def elect(self) -> tuple[str , LmLog]:
    """Vote for a player."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    options = [
        player
        for player in self.gamestate.sheriff_candidates
        if player != self.name
    ]
    random.shuffle(options)
    elect, log = self._generate_action("elect", options)
    if elect is not None and len(self.gamestate.debate) == MAX_DEBATE_TURNS:
      self._add_observation(
          f"I elect {elect} to be the sheriff of this round."
      )
    return elect, log
  
  def determine_statement_order(self) -> tuple[str , LmLog]:
    """Determine the statement order."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    # left or right
    your_idx=self.gamestate.current_players.index(self.name)
    if your_idx==0:
      left_idx=-1
      right_idx=1
    elif your_idx==len(self.gamestate.current_players)-1:
      left_idx=your_idx-1
      right_idx=0
    else:
      left_idx=your_idx-1
      right_idx=your_idx+1
    left_order=[left_idx]
    right_order=[right_idx]
    for i in range(len(self.gamestate.current_players)):
      left_order.append((left_idx-i)%len(self.gamestate.current_players))
      right_order.append((right_idx+i)%len(self.gamestate.current_players))

    left_order_name=[self.gamestate.current_players[left_order[i]] for i in range(len(left_order))]
    right_order_name=[self.gamestate.current_players[right_order[i]] for i in range(len(right_order))]
    options=[left_order_name, right_order_name]
    order,log=self._generate_action("determine_statement_order", options)
    if order is not None and len(self.gamestate.debate) == MAX_DEBATE_TURNS:
      self._add_observation(
          f"I, the sheriff, decide that the statement order is {order}."
      )
    return order, log
  
  def pseudo_vote(self) -> tuple[str , LmLog]:
    """Vote for a player."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    options = [
        player
        for player in self.gamestate.current_players
        if player != self.name
    ]
    random.shuffle(options)
    vote, log = self._generate_action("pseudo_vote", options)
    return vote, log
  
  def sheriff_summarize(self) -> tuple[str,LmLog]:
    """sheriff Engage in the debate."""
    assert self.is_sheriff
    result, log = self._generate_action("sheriff_summarize", [])
    if result is not None:
      say = result.get("say", None)
      return say, log
    return result, log

  def badge_flow(self) -> tuple[str,LmLog]:
    """sheriff decide who is the next sheriff"""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    options = [
        player
        for player in self.gamestate.current_players
        if player != self.name
    ]
    random.shuffle(options)
    next_sheriff, log = self._generate_action("badge_flow", options)
    return next_sheriff, log
  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    name = data["name"]
    role = data["role"]
    model = data.get("model", None)
    o = cls(name=name, role=role, model=model)
    o.gamestate = data.get("gamestate", None)
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o


class Villager(Player):
  """Represents a Villager in the game."""

  def __init__(
      self,
      name: str,
      model: Optional[str] = None,
      personality: Optional[str] = None,
      num_players: int = NUM_PLAYERS, # NUM_PLAYERS是假的
      num_villagers: int = NUM_VILLAGERS, 
      
  ):
    super().__init__(
        name=name, role=VILLAGER, model=model, personality=personality, num_players=num_players, num_villagers=num_villagers
    )

  @classmethod
  def from_json(cls, data: dict[Any, Any]):
    name = data["name"]
    model = data.get("model", None)
    o = cls(name=name, model=model)
    o.gamestate = data.get("gamestate", None)
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o


class Werewolf(Player):
  """Represents a Werewolf in the game."""

  def __init__(
      self,
      name: str,
      model: Optional[str] = None,
      personality: Optional[str] = None,
      num_players: int = NUM_PLAYERS, # NUM_PLAYERS是假的
      num_villagers: int = NUM_VILLAGERS, 
  ):
    super().__init__(
        name=name, role=WEREWOLF, model=model, personality=personality, num_players=num_players, num_villagers=num_villagers
    )

  def _get_game_state(self, **kwargs) -> Dict[str, Any]:
    """Gets the current game state, including werewolf-specific context."""
    state = super()._get_game_state(**kwargs)
    state["werewolf_context"] = self._get_werewolf_context()
    return state

  def eliminate(self) -> tuple[str , "LmLog"]:
    """Choose a player to eliminate."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )

    options = [
        player
        for player in self.gamestate.current_players
        if player != self.name and player != self.gamestate.other_wolf
    ]
    random.shuffle(options)
    eliminate, log = self._generate_action("remove", options)
    return eliminate, log

  def _get_werewolf_context(self):
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )

    if self.gamestate.other_wolf in self.gamestate.current_players:
      context = f"\n- The other Werewolf is {self.gamestate.other_wolf}."
    else:
      context = (
          f"\n- The other Werewolf, {self.gamestate.other_wolf}, was exiled by"
          " the Villagers. Only you remain."
      )

    return context

  @classmethod
  def from_json(cls, data: dict[Any, Any]):
    name = data["name"]
    model = data.get("model", None)
    o = cls(name=name, model=model)
    o.gamestate = data.get("gamestate", None)
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o


class Seer(Player):
  """Represents a Seer in the game."""

  def __init__(
      self,
      name: str,
      model: Optional[str] = None,
      personality: Optional[str] = None,
      num_players: int = NUM_PLAYERS,  # 默认12人
      num_villagers: int = NUM_VILLAGERS,  # 默认4个村民
  ):
    super().__init__(name=name, role=SEER, model=model, personality=personality, num_players=num_players, num_villagers=num_villagers)
    self.previously_unmasked: Dict[str, str] = {}

  def unmask(self) -> tuple[str , LmLog]:
    """Choose a player to unmask."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )

    options = [
        player
        for player in self.gamestate.current_players
        if player != self.name and player not in self.previously_unmasked.keys()
    ]
    random.shuffle(options)
    return self._generate_action("investigate", options)

  def reveal_and_update(self, player, role):
    self._add_observation(
        f"During the night, I decided to investigate {player} and learned they are a {role}."
    )
    self.previously_unmasked[player] = role

  @classmethod
  def from_json(cls, data: dict[Any, Any]):
    name = data["name"]
    model = data.get("model", None)
    o = cls(name=name, model=model)
    o.previously_unmasked = data.get("previously_unmasked", {})
    o.gamestate = data.get("gamestate", None)
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o


class Guard(Player):
  """Represents a Guard in the game."""

  def __init__(
      self,
      name: str,
      model: Optional[str] = None,
      personality: Optional[str] = None,
      num_players: int = NUM_PLAYERS, # NUM_PLAYERS是假的
      num_villagers: int = NUM_VILLAGERS,
  ):
    super().__init__(
        name=name, role=GUARD, model=model, personality=personality, num_players=num_players, num_villagers=num_villagers
    )
    self.guarded = []

  def protect(self) -> tuple[str , LmLog]:
    # 这个函数原来叫save,在同级game.py
    """Choose a player to protect."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )

    options = list(self.gamestate.current_players)
    random.shuffle(options)
    protected, log = self._generate_action("protect", options)
    if protected is not None:
      self._add_observation(f"During the night, I chose to protect {protected}")
      self.guarded.append(protected)
    else:
      self.guarded.append('None')
    return protected, log

  @classmethod
  def from_json(cls, data: dict[Any, Any]):
    name = data["name"]
    model = data.get("model", None)
    o = cls(name=name, model=model)
    o.gamestate = data.get("gamestate", None)
    o.guarded = data.get("guarded", [])
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o


# new: Witch
class Witch(Player):
  """Represents a Witch in the game."""

  def __init__(
      self,
      name: str,
      model: Optional[str] = None,
      personality: Optional[str] = None,
      num_players: int = NUM_PLAYERS, # NUM_PLAYERS是假的
      num_villagers: int = NUM_VILLAGERS,
  ):
    super().__init__(
        name=name, role=WITCH, model=model, personality=personality, num_players=num_players, num_villagers=num_villagers
    )
    self.poison_player = None
    self.save_player = None
    self.has_poisoned = False  # 是否已经毒过人
    self.has_saved = False  # 是否已经救过人

  def poison(self) -> tuple[str , LmLog]:
    # 毒人
    # killed_name
    """Choose a player to poison."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    killed_name = None  # 这个是被毒的人
    options = list([player for player in self.gamestate.current_players if player!=killed_name])  # 可选的人，需要把已经毒了的人去掉
    random.shuffle(options)
    poisoned, log = self._generate_action("poison", options)
    if poisoned is not None:
      self.poison_player = poisoned
      self._add_observation(f"During the night, I chose to poison {poisoned}. I ran out of my poison potion.")
      return poisoned, log
    return None, log  # 没毒
    

  def save(self) -> tuple[str , LmLog]:
    # 救人，options设置为被杀的人即可
    # killed_name
    """Choose whether to save or not."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )
    # options = list(self.gamestate.current_players)
    # random.shuffle(options)
    killed_name = None
    saved, log = self._generate_action("protect",options=[killed_name])
    if saved:
      self.save_player = killed_name
      self._add_observation(f"During the night, I chose to save {killed_name}. I ran out of my antidote.")
      return killed_name, log
    return None, log  # 没救

  @classmethod
  def from_json(cls, data: dict[Any, Any]):
    name = data["name"]
    model = data.get("model", None)
    o = cls(name=name, model=model)
    o.gamestate = data.get("gamestate", None)
    o.save_player = data.get("save_player", None)
    o.poison_player = data.get("poison_player", None)
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o

class Hunter(Player):
  """Represents a Hunter in the game."""

  def __init__(
      self,
      name: str,
      model: Optional[str] = None,
      personality: Optional[str] = None,
      num_players: int = NUM_PLAYERS, # NUM_PLAYERS是假的
      num_villagers: int = NUM_VILLAGERS,
  ):
    super().__init__(
        name=name, role=HUNTER, model=model, personality=personality, num_players=num_players, num_villagers=num_villagers
    )
    self.shot_player = None  # 被猎人射杀的玩家

  def shoot(self) -> tuple[str , LmLog]:
    """Choose a player to shoot."""
    if not self.gamestate:
      raise ValueError(
          "GameView not initialized. Call initialize_game_view() first."
      )

    options = list(self.gamestate.current_players)
    random.shuffle(options)
    shot_player, log = self._generate_action("protect", options)
    if shot_player is not None:
      self.shot_player = shot_player
      self._add_observation(f"Before I dead, I chose to shoot {shot_player}")
    return shot_player, log

  @classmethod
  def from_json(cls, data: dict[Any, Any]):
    name = data["name"]
    model = data.get("model", None)
    o = cls(name=name, model=model)
    o.gamestate = data.get("gamestate", None)
    o.shot_player = data.get("shot_player", None)
    o.bidding_rationale = data.get("bidding_rationale", "")
    o.observations = data.get("observations", [])
    return o


class Round(Deserializable):
  """Represents a round of gameplay in Werewolf.

  Attributes:
    players: List of player names in this round.
    eliminated: Who the werewolves killed during the night phase.
    unmasked: Who the Seer unmasked during the night phase.
    protected: Who the Guard saved during the night phase.
    exiled: Who the players decided to exile after the debate.
    debate: List of debate tuples of player name and what they said during the
      debate.
    votes:  Who each player voted to exile after each line of dialogue in the
      debate.
    bids: What each player bid to speak next during each turn in the debate.
    success (bool): Indicates whether the round was completed successfully.

  Methods:
    to_dict: Returns a dictionary representation of the round.
  """

  def __init__(self):
    self.players: List[str] = []
    self.eliminated: str  = None  # 狼人今晚杀的人
    self.unmasked: str  = None  # 预言家查验的人
    self.protected: str  = None  # 守卫保护的人
    self.exiled: str  = None  # 玩家投票放逐的人
    self.saved: str = None  # 女巫救的人
    self.poisoned: str = None  # 女巫毒的人
    self.shot: str = None  # 猎人是否出局并射杀了人
    self.debate: List[Tuple[str, str]] = []
    self.votes: List[Dict[str, str]] = []
    self.bids: List[Dict[str, int]] = []
    self.success: bool = False
    self.pseudo_votes: List[Dict[str, str]] = []
    self.elect: List[Dict[str, str]] = []
    self.sheriff: str = None
    self.statement_order: List[str] = None
    self.sheriff_candidates: List[str] = []

  def to_dict(self):
    return to_dict(self)

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    o = cls()
    o.players = data["players"]
    o.eliminated = data.get("eliminated", None)
    o.unmasked = data.get("unmasked", None)
    o.protected = data.get("protected", None)
    o.exiled = data.get("exiled", None)
    o.saved = data.get("saved", None)  # 女巫救的人
    o.poisoned = data.get("poisoned", None)  # 女巫毒的人
    o.shot = data.get("shot", None)  # 猎人射杀的人
    o.debate = data.get("debate", [])
    o.votes = data.get("votes", [])
    o.bids = data.get("bids", [])
    o.success = data.get("success", False)
    return o


class State(Deserializable):
  """Represents a game session.

  Attributes:
    session_id: Unique identifier for the game session.
    players: List of players in the game.
    seer: The player with the seer role.
    guard: The player with the guard role.
    villagers: List of players with the villager role.
    werewolves: List of players with the werewolf role.
    rounds: List of Rounds in the game.
    error_message: Contains an error message if the game failed during
      execution.
    winner: Villager or Werewolf

  Methods:
    to_dict: Returns a dictionary representation of the game.
  """

  def __init__(
      self,
      session_id: str,
      seer: Optional[Seer] = None,
      guard: Optional[Guard] = None,
      witch: Optional[Witch] = None,
      hunter: Optional[Hunter] = None,
      villagers: List[Villager] = [],
      werewolves: List[Werewolf] = [],
  ):
    self.session_id: str = session_id
    self.seer: Optional[Seer] = seer
    self.guard: Optional[Guard] = guard
    self.witch: Optional[Witch] = witch
    self.hunter: Optional[Hunter] = hunter
    self.villagers: List[Villager] = villagers
    self.werewolves: List[Werewolf] = werewolves
    self.players: Dict[str, Player] = {
      player.name: player
      for player in self.villagers
      + self.werewolves
      + [self.guard, self.seer, self.witch, self.hunter]
      if player is not None
    }
    self.rounds: List[Round] = []
    self.error_message: str = ""
    self.winner: str = ""

  def to_dict(self):
    return to_dict(self)

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    werewolves = []
    for w in data.get("werewolves", []):
      werewolves.append(Werewolf.from_json(w))

    villagers = []
    for v in data.get("villagers", []):
      villagers.append(Villager.from_json(v))

    guard = Guard.from_json(data.get("guard")) if data.get("guard") else None
    seer = Seer.from_json(data.get("seer")) if data.get("seer") else None
    witch = Witch.from_json(data.get("witch")) if data.get("witch") else None
    hunter = Hunter.from_json(data.get("hunter")) if data.get("hunter") else None

    players = {}
    for p in werewolves + villagers + [guard, seer, witch, hunter]:
      if p is not None:
        players[p.name] = p

    o = cls(
        data.get("session_id", ""),
        seer,
        guard,
        witch,
        hunter,
        villagers,
        werewolves,
    )
    rounds = []
    for r in data.get("rounds", []):
      rounds.append(Round.from_json(r))

    o.rounds = rounds
    o.error_message = data.get("error_message", "")
    o.winner = data.get("winner", "")
    return o


class VoteLog(Deserializable):

  def __init__(self, player: str, voted_for: str, log: LmLog):
    self.player = player
    self.voted_for = voted_for
    self.log = log

  def to_dict(self):
    return to_dict(self)

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    player = data.get("player", None)
    voted_for = data.get("voted_for", None)
    log = LmLog.from_json(data.get("log", None))
    return cls(player, voted_for, log)

class PseudoVoteLog(Deserializable):

  def __init__(self, player: str, voted_for: str, log: LmLog):
    self.player = player
    self.voted_for = voted_for
    self.log = log

  def to_dict(self):
    return to_dict(self)

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    player = data.get("player", None)
    voted_for = data.get("pseudo_voted_for", None)
    log = LmLog.from_json(data.get("log", None))
    return cls(player, voted_for, log)

class ElectLog(Deserializable):
  def __init__(self, player: str, voted_for: str, log: LmLog):
    self.player = player
    self.voted_for = voted_for
    self.log = log

  def to_dict(self):
    return to_dict(self)

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    player = data.get("player", None)
    voted_for = data.get("elect_for", None)
    log = LmLog.from_json(data.get("log", None))
    return cls(player, voted_for, log)

class RoundLog(Deserializable):
  """Represents the logs of a round of gameplay in Werewolf.

  Attributes:
    eliminate: Logs from the eliminate action taken by werewolves.
    investigate: Log from the invesetigate action taken by the seer.
    protect: Log from the protect action taken by the guard.
    bid: Logs from the bidding actions. The 1st element in the list is the bidding logs
      for the 1st debate turn, the 2nd element is the logs for the 2nd debate
      turn, and so on. Every player bids to speak on every turn, so the element
      is a list too. The tuple contains the name of the player and the log of
      their bidding.
    debate: Logs of the debates. Each round has multiple debate turbns, so it's a
      list. Each element is a tuple - the 1st element is the name of the player
      who spoke at this turn, and the 2nd element is the log.
    vote: Log of the votes. A list of logs, one for every player who voted. The
      1st element of the tuple is the name of the player, and the 2nd element is
      the log.
    summaries: Logs from the summarize step. Every player summarizes their
      observations at the end of a round before they vote. Each element is a
      tuple where the 1st element is the name of the player, and the 2nd element
      is the log
  """

  def __init__(self):
    self.eliminate: LmLog  = None
    self.investigate: LmLog  = None
    self.protect: LmLog  = None
    self.save: LmLog = None  # added
    self.poison: LmLog = None  #added
    self.shoot: LmLog = None # added
    self.bid: List[List[Tuple[str, LmLog]]] = []
    self.debate: List[Tuple[str, LmLog]] = []
    self.votes: List[List[VoteLog]] = []
    self.summaries: List[Tuple[str, LmLog]] = []
    self.pseudo_votes: List[List[PseudoVoteLog]] = []
    self.elect: List[List[ElectLog]] = []

  def to_dict(self):
    return to_dict(self)

  @classmethod
  def from_json(cls, data: Dict[Any, Any]):
    o = cls()

    eliminate = data.get("eliminate", None)
    investigate = data.get("investigate", None)
    protect = data.get("protect", None)
    save = data.get("save", None)
    poison = data.get("poison", None)
    shoot = data.get("shoot", None)

    if eliminate:
      o.eliminate = LmLog.from_json(eliminate)
    if investigate:
      o.investigate = LmLog.from_json(investigate)
    if protect:
      o.protect = LmLog.from_json(protect)
    if save:
      o.save = LmLog.from_json(save)
    if poison:
      o.poison = LmLog.from_json(poison)
    if shoot:
      o.shoot = LmLog.from_json(shoot)

    for votes in data.get("votes", []):
      v_logs = []
      o.votes.append(v_logs)
      for v in votes:
        v_logs.append(VoteLog.from_json(v))

    for r in data.get("bid", []):
      r_logs = []
      o.bid.append(r_logs)
      for player in r:
        r_logs.append((player[0], LmLog.from_json(player[1])))

    for player in data.get("debate", []):
      o.debate.append((player[0], LmLog.from_json(player[1])))

    for player in data.get("summaries", []):
      o.summaries.append((player[0], LmLog.from_json(player[1])))


    for pseudo_votes in data.get("pseudo_votes", []):
      p_logs = []
      o.pseudo_votes.append(p_logs)
      for p in pseudo_votes:
        p_logs.append(PseudoVoteLog.from_json(p))
    for elects in data.get("elect", []):
      e_logs = []
      o.elect.append(e_logs)
      for e in elects:
        e_logs.append(ElectLog.from_json(e))
      
    return o
