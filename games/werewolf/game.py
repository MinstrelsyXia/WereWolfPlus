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

"""Werewolf game."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import random
from typing import List

import tqdm

from games.werewolf.model import Round, RoundLog, State, VoteLog
from games.werewolf.config import  MAX_DEBATE_TURNS, RUN_SYNTHETIC_VOTES
import re

def get_max_bids(d):
  """Gets all the keys with the highest value in the dictionary."""
  max_value = max(d.values())
  max_keys = [key for key, value in d.items() if value == max_value]
  return max_keys


class GameMaster:
  # 游戏上帝
  def __init__(
      self,
      state: State,
      num_threads: int = 1,
  ) -> None:
    """Initialize the Werewolf game.

    Args:
    """
    self.state = state
    self.current_round_num = len(self.state.rounds) if self.state.rounds else 0
    self.num_threads = num_threads
    self.logs: List[RoundLog] = []

  @property
  def this_round(self) -> Round:
    return self.state.rounds[self.current_round_num]
  @property
  def last_round(self) -> Round:
    return self.state.rounds[self.current_round_num-1]
  
  @property
  def this_round_log(self) -> RoundLog:
    return self.logs[self.current_round_num]

  def eliminate(self):
    """Werewolves choose a player to eliminate."""
    werewolves_alive = [
        w for w in self.state.werewolves if w.name in self.this_round.players
    ]
    wolf = random.choice(werewolves_alive)
    eliminated, log = wolf.eliminate()
    self.this_round_log.eliminate = log
    if eliminated is not None:
      self.this_round.eliminated = eliminated
      tqdm.tqdm.write(f"{wolf.name} eliminated {eliminated}")
      for wolf in werewolves_alive:
        wolf._add_observation(
            "During the"
            f" night, {'we' if len(werewolves_alive) > 1 else 'I'} decided to"
            f" eliminate {eliminated}."
        )
    else:
      raise ValueError("Eliminate did not return a valid player.")
  def eliminate_pre(self):
    """Werewolves choose a player to eliminate."""
    werewolves_alive = [
        w for w in self.state.werewolves if w.name in self.this_round.players
    ]
    wolf = random.choice(werewolves_alive)
    # eliminated, log = wolf.eliminate()
    observations={
      "player_name":wolf.name,
      "game_state" : wolf._get_game_state(),
      "action" : "remove",
      "options" : [player for player in wolf.gamestate.current_players
        if player != wolf.name and player != wolf.gamestate.other_wolf ]
    }
    return observations
  def eliminate_post(self,observations,eliminated,log):
    werewolves_alive = [
      w for w in self.state.werewolves if w.name in self.this_round.players
    ]
    wolf_name=observations['player_name']
    self.this_round_log.eliminate = log
    if eliminated is not None:
      self.this_round.eliminated = eliminated
      tqdm.tqdm.write(f"{wolf_name} eliminated {eliminated}")
      for wolf in werewolves_alive:
        wolf._add_observation(
            "During the"
            f" night, {'we' if len(werewolves_alive) > 1 else 'I'} decided to"
            f" eliminate {eliminated}."
        )
    else:
      raise ValueError("Eliminate did not return a valid player.")

  def protect(self):
    """Guard chooses a player to protect."""
    if self.state.guard.name not in self.this_round.players:
      return  # Guard no longer in the game

    protect, log = self.state.guard.protect()
    self.this_round_log.protect = log

    if protect is not None:
      self.this_round.protected = protect
      tqdm.tqdm.write(f"{self.state.guard.name} protected {protect}")
    else:
      raise ValueError("Protect did not return a valid player.")
  def protect_post(self,protect,log):
    """Guard chooses a player to protect."""

    # protect, log = self.state.guard.protect()
    self.this_round_log.protect = log

    if protect is not None:
      self.this_round.protected = protect
      tqdm.tqdm.write(f"{self.state.guard.name} protected {protect}")
      self.state.guard._add_observation(f"I protected {protect}")
      self.state.guard.guarded.append(protect)  # 记录被保护的玩家
    else:
      raise ValueError("Protect did not return a valid player.")

  def save(self):
    """Witch chooses whether to save or not."""
    if self.state.witch.name not in self.this_round.players:
      return  # Witch no longer in the game
    if self.state.witch.save_player is not None:
      tqdm.tqdm.write(f"{self.state.witch.name} has already saved someone.")
      return   

    save, log = self.state.witch.save()  # 返回救的人的名字
    self.this_round_log.save = log

    if save is not None:
      self.this_round.saved = save
      tqdm.tqdm.write(f"{self.state.witch.name} saved {save}")
    else:
      raise ValueError("Witch did not save player.")
  def save_post(self,save,log):
    """Witch chooses whether to save or not."""

    # protect, log = self.state.guard.protect()
    self.this_round_log.save = log

    if "yes" in save.lower():
      self.this_round.saved = self.this_round.eliminated
      tqdm.tqdm.write(f"{self.state.witch.name} saved {self.this_round.eliminated}")
      self.state.witch.has_saved = True  # 标记女巫已经救过人了
    else:
      tqdm.tqdm.write(f"{self.state.witch.name} didn't save {self.this_round.eliminated}")

  def poison(self):
    """Witch chooses whether to poison someone."""
    if self.state.witch.name not in self.this_round.players:
      return  # Widch no longer in the game
    if self.state.witch.poison_player is not None:
      # 如果已经毒过人了，则不能再毒人
      tqdm.tqdm.write(f"{self.state.witch.name} has already poisoned someone.")
      return

    poison, log = self.state.witch.poison()  # 返回毒的人的名字
    self.this_round_log.poison = log

    if poison is not None:
      self.this_round.poisoned = poison
      tqdm.tqdm.write(f"{self.state.witch.name} poisoned {poison}")
    else:
      raise ValueError("Witch did not poison player.")

  def poison_post(self,poison,log):
    """Witch chooses whether to poison someone."""

    # protect, log = self.state.guard.protect()
    self.this_round_log.poison = log

    if poison != "No":
      self.this_round.poisoned = poison
      tqdm.tqdm.write(f"{self.state.witch.name} poisoned {poison}")
      self.state.witch._add_observation(f"I poisoned {poison}")
      self.state.witch.has_poisoned = True
    else:
      tqdm.tqdm.write(f"{self.state.witch.name} didn't poison anyone.")
      self.state.witch._add_observation(
          f"I didn't poison anyone."
      )

  def check_for_shoot(self):
    """Hunter chooses whether to shoot."""
    # 如果此轮杀的不是hunter/被保护/被救，则跳过
    if self.this_round.eliminated != self.state.hunter.name or\
        self.this_round.protected == self.state.hunter.name or \
        self.this_round.saved == self.state.hunter.name:
      return
    if self.this_round.poisoned == self.state.hunter.name:
      tqdm.tqdm.write(f"{self.state.hunter.name} is poisoned and cannot shoot.")
      return

    if self.state.hunter.name == self.this_round.eliminated:
      pass  # 如果猎人是在该轮被杀了
    elif self.state.hunter.name not in self.this_round.players:
      return  # Hunter no longer in the game（非该轮杀的）

    # 杀的是hunter
    shoot, log = self.state.hunter.shoot()  # 返回射击的人的名字
    self.this_round_log.shoot = log

    if shoot is not None:
      self.this_round.shot = shoot
      tqdm.tqdm.write(f"{self.state.hunter.name} shot {shoot}")
    else:
      raise ValueError("Hunter did not shoot.")
  def shoot_post(self,shoot,log):
    """Hunter chooses whether to shoot."""

    # protect, log = self.state.guard.protect()
    self.this_round_log.shoot = log

    if shoot is not None:
      self.this_round.shot = shoot
      self.this_round.players.remove(shoot)  # 移除
      announcement = (
          f"The Hunter {self.state.hunter.name} removed {shoot} from the game."
      )
      tqdm.tqdm.write(announcement)

      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.remove_player(shoot)
        player.add_announcement(announcement)
    else:
      raise ValueError("Hunter did not shoot.")

  def unmask(self):
    """Seer chooses a player to unmask."""
    if self.state.seer.name not in self.this_round.players:
      return  # Seer no longer in the game

    unmask, log = self.state.seer.unmask()
    self.this_round_log.investigate = log

    if unmask is not None:
      self.this_round.unmasked = unmask
      self.state.seer.reveal_and_update(unmask, self.state.players[unmask].role)
    else:
      raise ValueError("Unmask function did not return a valid player.")

  def _get_bid(self, player_name):
    """Gets the bid for a specific player."""
    player = self.state.players[player_name]
    bid, log = player.bid()
    if bid is None:
      raise ValueError(
          f"{player_name} did not return a valid bid. Find the raw response"
          " in the `bid` field in the log"
      )
    if bid > 1:
      tqdm.tqdm.write(f"{player_name} bid: {bid}")
    return bid, log

  def get_next_speaker(self):
    """Determine the next speaker based on bids."""
    previous_speaker, previous_dialogue = (
        self.this_round.debate[-1] if self.this_round.debate else (None, None)
    )

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_bids = {
          player_name: executor.submit(self._get_bid, player_name)
          for player_name in self.this_round.players
          if player_name != previous_speaker
      }

      bid_log = []
      bids = {}
      try:
        for player_name, bid_task in player_bids.items():
          bid, log = bid_task.result()
          bids[player_name] = bid
          bid_log.append((player_name, log))
      except TypeError as e:
        print(e)
        raise e

    self.this_round.bids.append(bids)
    self.this_round_log.bid.append(bid_log)

    potential_speakers = get_max_bids(bids)
    # Prioritize mentioned speakers if there's previous dialogue
    if previous_dialogue:
      potential_speakers.extend(
          [name for name in potential_speakers if name in previous_dialogue]
      )

    random.shuffle(potential_speakers)
    return random.choice(potential_speakers)

  def run_summaries(self):
    """Collect summaries from players after the debate."""

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_summaries = {
          name: executor.submit(self.state.players[name].summarize)
          for name in self.this_round.players
      }

      for player_name, summary_task in player_summaries.items():
        summary, log = summary_task.result()
        tqdm.tqdm.write(f"{player_name} summary: {summary}")
        self.this_round_log.summaries.append((player_name, log))

  def run_day_phase(self):
    """Run the day phase which consists of the debate and voting."""

    for idx in range(MAX_DEBATE_TURNS):
      next_speaker = self.get_next_speaker()
      if not next_speaker:
        raise ValueError("get_next_speaker did not return a valid player.")

      player = self.state.players[next_speaker]
      dialogue, log = player.debate()
      if dialogue is None:
        raise ValueError(
            f"{next_speaker} did not return a valid dialouge from debate()."
        )

      self.this_round_log.debate.append((next_speaker, log))
      self.this_round.debate.append([next_speaker, dialogue])
      tqdm.tqdm.write(f"{next_speaker} ({player.role}): {dialogue}")

      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.update_debate(next_speaker, dialogue)
        else:
          raise ValueError(f"{name}.gamestate needs to be initialized.")

      if idx == MAX_DEBATE_TURNS - 1 or RUN_SYNTHETIC_VOTES:
        votes, vote_logs = self.run_voting()
        self.this_round.votes.append(votes)
        self.this_round_log.votes.append(vote_logs)

    for player, vote in self.this_round.votes[-1].items():
      tqdm.tqdm.write(f"{player} voted to remove {vote}")

  def run_voting(self):
    """Conduct a vote among players to exile someone."""
    vote_log = []
    votes = {}

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_votes = {
          name: executor.submit(self.state.players[name].vote)
          for name in self.this_round.players
      }

      for player_name, vote_task in player_votes.items():
        vote, log = vote_task.result()
        vote_log.append(VoteLog(player_name, vote, log))

        if vote is not None:
          votes[player_name] = vote
        else:
          self.this_round.votes.append(votes)
          self.this_round_log.votes.append(vote_log)
          raise ValueError(f"{player_name} vote did not return a valid player.")

    return votes, vote_log

  def exile(self):
    """Exile the player who received the most votes."""

    candidates=list(self.this_round.votes[-1].values())
    # 每个人两票
    candidates=candidates+candidates
    if self.this_round.sheriff in self.this_round.votes[-1].keys():
      #警长再加一票 1.5:1
      candidates.append(self.this_round.votes[-1][self.this_round.sheriff])


    most_voted, vote_count = Counter(
        candidates
    ).most_common(1)[0]
    self.this_round.exiled = most_voted # 随机选

    if self.this_round.exiled is not None and (self.this_round.exiled in self.this_round.players):
      exiled_player = self.this_round.exiled
      self.this_round.players.remove(exiled_player)
      announcement = (
          f"The majority voted to remove {exiled_player} from the game."
      )
    else:
      announcement = (
          "A majority vote was not reached, so no one was removed from the"
          " game."
      )

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate and self.this_round.exiled is not None :
        player.gamestate.remove_player(self.this_round.exiled)
      player.add_announcement(announcement)
    tqdm.tqdm.write(announcement)
  
  def no_sheriff(self):
    self.this_round.sheriff = None

  def badge_flow(self):
          return False
  
  def determine_sheriff(self, sheriff=None, badge_flow=False):
    """Elect the player who received the most votes to be the sheriff"""
    if sheriff is None:
        # 检查是否有选举记录
        if not hasattr(self.this_round, 'elect') or not self.this_round.elect:
            # 如果没有选举记录，说明没有警长
            self.this_round.sheriff = None
            announcement = "No one is running for sheriff. This round will proceed without a sheriff."
        else:
            try:
                # 尝试获取最后一次选举结果
                last_election = self.this_round.elect[-1]
                if not last_election:  # 如果最后一次选举为空
                    self.this_round.sheriff = None
                    announcement = "No one is running for sheriff. This round will proceed without a sheriff."
                else:
                    most_elected, elect_count = Counter(last_election.values()).most_common(1)[0]
                    self.this_round.sheriff = most_elected
                    announcement = f"The majority elected {most_elected} as the sheriff."
                    sheriff = most_elected
            except (IndexError, AttributeError):
                # 如果出现任何错误，设置为没有警长
                self.this_round.sheriff = None
                announcement = "No one is running for sheriff. This round will proceed without a sheriff."
    else:
        if badge_flow:
            # 警长死亡，进行警徽流
            self.this_round.sheriff = sheriff
            announcement = f"The former sheriff named {sheriff} as the sheriff."
        else:
            # 警长继续连任
            self.this_round.sheriff = sheriff
            announcement = f"The {sheriff} is still the sheriff."
    
    # 每个玩家的私有信息
    for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
            player.gamestate.add_sheriff(self.this_round.sheriff)
            if name == sheriff:
                player.is_sheriff = True
            else:
                player.is_sheriff = False
        player.add_announcement(announcement)

    tqdm.tqdm.write(announcement)

  def determine_statement_order(self, order):
    """Determine the order in which statements are made."""
# 将字符串 order 转换为列表
    if isinstance(order, str):
        match = re.match(r'\[(.*?)\]', order)
        if match:
          order_list= [item.strip() for item in match.group(1).split(',')]
        else:
          order_list=[]
    self.this_round.statement_order = order_list
    announcement=(f'The order of statements is {order}')
    for name in self.this_round.players:
      player = self.state.players[name]
      player.add_announcement(announcement)
    tqdm.tqdm.write(announcement)
  
  def add_round_candidate(self,candidate):
    """Add a candidate to the list of candidates."""
    self.this_round.sheriff_candidates.append(candidate.name)
    announcement=(f'{candidate.name} is now a candidate for the  sheriff.')
    for name in self.this_round.players:
      player = self.state.players[name]
      player.add_announcement(announcement)
      player.gamestate.add_candidates(candidate.name)
    tqdm.tqdm.write(announcement)

  def resolve_night_phase(self):
    """Resolve elimination and protection during the night phase."""
    # 解决晚上杀人和保护的事情
    announcement = ''
    if self.this_round.eliminated != self.this_round.protected and self.this_round.eliminated != self.this_round.saved:
      # 杀的人没有被保护，且杀的人没有被救
      eliminated_player = self.this_round.eliminated  # 确实被杀了
      self.this_round.players.remove(eliminated_player)  # 把他移除
      announcement = f"{eliminated_player} was removed from the game during the night. "
      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.remove_player(self.this_round.eliminated)
        player.add_announcement(announcement)
      tqdm.tqdm.write(announcement)
    if self.this_round.poisoned and self.this_round.poisoned in self.this_round.players:
      poisoned_player = self.this_round.poisoned # 确实被毒死了
      self.this_round.players.remove(poisoned_player)  # 把他移除
      announcement = f"{poisoned_player} was removed from the game during the night."
      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.remove_player(self.this_round.poisoned)
        player.add_announcement(announcement)
      announcement = f"{poisoned_player} was poisoned from the game during the night."
      tqdm.tqdm.write(announcement)
    if announcement == '':
      announcement = "No one was removed from the game during the night."
      for name in self.this_round.players:
        player = self.state.players[name]
        player.add_announcement(announcement)
      tqdm.tqdm.write(announcement)

    

  def run_round(self):
    """Run a single round of the game."""
    self.state.rounds.append(Round())
    self.logs.append(RoundLog())

    self.this_round.players = (
        list(self.state.players.keys())
        if self.current_round_num == 0
        else self.state.rounds[self.current_round_num - 1].players.copy()
    )

    # 狼人投票杀人，守卫保护，预言家查验，结束夜晚阶段，检查输赢，走白天的流程，投票出局，检查获胜，总结
    # 狼人杀人，守卫保护，女巫毒/救，预言家查验，（夜晚猎人跳过），结束夜晚阶段，检查是否是猎人，检查输赢，走白天的流程，投票出局，检查获胜，总结
    for action, message in [
        (self.eliminate, "The Werewolves are picking someone to remove from the game."),
        (self.protect, "The Guard is protecting someone."),
        (self.save, "The Witch is deciding whether to save the killed player."),
        (self.poison, "The Witch is deciding whether to poison someone."),
        (self.unmask, "The Seer is investigating someone."),
        (self.resolve_night_phase, ""),
        (self.check_for_shoot, "Checking if the Hunter is active(can shoot) and shoot which player."),
        (self.check_for_winner, "Checking for a winner after Night Phase."),
        (self.run_day_phase, "The Players are debating and voting."),
        (self.exile, ""),
        (self.check_for_winner, "Checking for a winner after Day Phase."),
        (self.run_summaries, "The Players are summarizing the debate."),
    ]:
      tqdm.tqdm.write(message)  # 现在在干嘛
      action()  # 开始行动

      if self.state.winner:
        tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
        self.this_round.success = True
        return

    tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
    self.this_round.success = True
  def run_round_new(self):
    """Run a single round of the game."""
    self.state.rounds.append(Round())
    self.logs.append(RoundLog())

    self.this_round.players = (
        list(self.state.players.keys())
        if self.current_round_num == 0
        else self.state.rounds[self.current_round_num - 1].players.copy()
    )

    # for action, message in [
    #     (
    #         self.eliminate,
    #         "The Werewolves are picking someone to remove from the game.",
    #     ),
    #     (self.protect, "The Guard is protecting someone."),
    #     (self.unmask, "The Seer is investigating someone."),
    #     (self.resolve_night_phase, ""),
    #     (self.check_for_winner, "Checking for a winner after Night Phase."),
    #     (self.run_day_phase, "The Players are debating and voting."),
    #     (self.exile, ""),
    #     (self.check_for_winner, "Checking for a winner after Day Phase."),
    #     (self.run_summaries, "The Players are summarizing the debate."),
    # ]:
    for action, message in [
        (self.eliminate, "The Werewolves are picking someone to remove from the game."),
        (self.protect, "The Guard is protecting someone."),
        (self.save, "The Witch is deciding whether to save the killed player."),
        (self.poison, "The Witch is deciding whether to poison someone."),
        (self.unmask, "The Seer is investigating someone."),
        (self.resolve_night_phase, ""),
        (self.check_for_shoot, "Checking if the Hunter is active(can shoot) and shoot which player."),
        (self.check_for_winner, "Checking for a winner after Night Phase."),
        (self.run_day_phase, "The Players are debating and voting."),
        (self.exile, ""),
        (self.check_for_winner, "Checking for a winner after Day Phase."),
        (self.run_summaries, "The Players are summarizing the debate."),
    ]:
      tqdm.tqdm.write(message)
      action()

      if self.state.winner:
        tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
        self.this_round.success = True
        return

    tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
    self.this_round.success = True

  def get_winner(self) -> str:
    """Determine the winner of the game."""
    active_wolves = set(self.this_round.players) & set(
        w.name for w in self.state.werewolves
    )
    active_villagers = set(self.this_round.players) - active_wolves
    if len(active_wolves) >= len(active_villagers):
      return "Werewolves"
    return "Villagers" if not active_wolves else ""

  def check_for_winner(self):
    """Check if there is a winner and update the state accordingly."""
    self.state.winner = self.get_winner()
    if self.state.winner:
      tqdm.tqdm.write(f"The winner is {self.state.winner}!")

  def run_game(self) -> str:
    """Run the entire Werewolf game and return the winner."""
    while not self.state.winner:
      tqdm.tqdm.write(f"STARTING ROUND: {self.current_round_num}")
      self.run_round()  # 一天
      for name in self.this_round.players:
        if self.state.players[name].gamestate:
          self.state.players[name].gamestate.round_number = (
              self.current_round_num + 1
          )
          self.state.players[name].gamestate.clear_debate() # 清空此轮每个人的debate
      self.current_round_num += 1

    tqdm.tqdm.write("Game is complete!")
    return self.state.winner
