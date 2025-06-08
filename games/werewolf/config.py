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

import random

RETRIES = 3
<<<<<<< HEAD
NAMES = ["Derek", "Jackson", "Will", "Jacob", "Harold", "Sam", "Scott", "David",  "Isaac"]  # names of famous Werewolves according to Wikipedia
=======
NAMES = ["Derek", "Jackson", "Will", "Jacob", "Harold", "Sam", "Scott", "David",  "Isaac", "Hayley","Paul", "Leah"]
# names of famous Werewolves according to Wikipedia
>>>>>>> origin/addrole_db
# NAMES = [
#   "Derek", "Scott", "Jacob", "Isaac", "Hayley", "David", "Tyler",
#   "Ginger", "Jackson", "Mason", "Dan", "Bert", "Will", "Sam",
#   "Paul", "Leah", "Harold"
# ] 
RUN_SYNTHETIC_VOTES = True
MAX_DEBATE_TURNS = 12
NUM_PLAYERS = 12
<<<<<<< HEAD
NUM_VILLAGERS = 4
_THREADS=1
def get_player_names(num_players=NUM_PLAYERS,names=NAMES): 
    return random.sample(names, num_players)
=======
_THREADS = 1

# 添加调试模式标志
#! debug
DEBUG_MODE = False # 设置为 True 时启用调试模式

def get_player_names(): 
    if DEBUG_MODE:
        # 在调试模式下按顺序返回前 NUM_PLAYERS 个名称
        return NAMES[:NUM_PLAYERS]
    else:
        # 正常模式下随机选择
        return random.sample(NAMES, NUM_PLAYERS)
>>>>>>> origin/addrole_db
