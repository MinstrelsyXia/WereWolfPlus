StarCraft II
-----------------------
StarCraft II, as a complex real-time strategy game , provides an ideal platform for evaluating LLMs’ capabilities in strategic reasoning and multi-agent interaction

## Eval scenes
### Scene1 setting:
| Item              | Setting              |
|-------------------|----------------------|
| Map               | Ancient Cistern LE   |
| Mode              | Agent vs Built-in AI |
| Players           | Protoss vs Protoss   |
| Opponent strategy | Macro                |
| Built-in AI level | Medium               |
| Asynch mode       | False                |
| Num of matches    | 10                   |
### Scene2 setting:
| Item              | Setting              |
|-------------------|----------------------|
| Map               | Ancient Cistern LE   |
| Mode              | Agent vs Built-in AI |
| Players           | Protoss vs Protoss   |
| Opponent strategy | Macro                |
| Built-in AI level | Medium               |
| Asynch mode       | True                 |
| Num of matches    | 10                   |
### Scene3 setting:
| Item               | Setting              |
|-------------------|----------------------|
| Map               | Ancient Cistern LE   |
| Mode              | Agent vs Built-in AI |
| Players           | Protoss vs Protoss   |
| Opponent strategy | Rush                 |
| Built-in AI level | Medium               |
| Asynch mode       | False                |
| Num of matches    | 10                   |
### Scene4 setting:
| Item               | Setting              |
|-------------------|----------------------|
| Map               | Ancient Cistern LE   |
| Mode              | Agent vs Built-in AI |
| Players           | Protoss vs Protoss   |
| Opponent strategy | Rush                 |
| Built-in AI level | Medium               |
| Asynch mode       | True                 |
| Num of matches    | 10                   |
### Scene5 setting:
| Item               | Setting             |
|-------------------|----------------------|
| Map               | Ancient Cistern LE   |
| Mode              | Agent vs Built-in AI |
| Players           | Protoss vs Protoss   |
| Opponent strategy | Random               |
| Built-in AI level | Medium               |
| Asynch mode       | False                |
| Num of matches    | 10                   |
### Scene6 setting:
| Item               | Setting              |
|-------------------|----------------------|
| Map               | Ancient Cistern LE   |
| Mode              | Agent vs Built-in AI |
| Players           | Protoss vs Protoss   |
| Opponent strategy | Random               |
| Built-in AI level | Medium               |
| Asynch mode       | True                 |
| Num of matches    | 10                   |

## Related code 
`agent`:[starcraft2_agent](../../agent_manager/agents/starcraft2_agent)

`prompt`:[starcraft2_prompt.py](../../agent_manager/prompts/starcraft2_prompt.py)

`scene_setting`:[eval_config_base](../../configs/eval_config_base)

## Prompt instruction
`1. System Prompt`
```shell
"""
You are an AI trained in analyzing and summarizing StarCraft II games. You understand the nuances and strategies of the protoss race. 
Based on the summaries of multiple rounds in a game, we want you to analyze the game progression in a structured way. Your analysis should include the following aspects:
1. Game Overview: Provide a brief overview of the current situation based on all the rounds.
2. Current Game Stage: Determine the stage of the game based on the information of all rounds. Is it the early game, mid-game, or late game?
3. Our Situation: Describe our current status in terms of:
    3.1 Units and Buildings: Analyze the state of our units and buildings.
    3.2 Economy: Evaluate our economic condition, including resource collection and usage.
    3.3 Technology: Describe the status of our technological research and what technologies we have unlocked so far. Analyze our technology tree, indicating the available and potential upgrades or units.
4. Our Strategy: Infer our potential strategy based on our current situation and the information of all rounds.
5. Enemy's Strategy: Infer the enemy's potential strategy, based on the available information.
6. Key Information: Highlight the most important aspects from all rounds that have significantly influenced the game.

For Protoss, keep an eye on Nexus's energy to Chrono Boost important structures.

Based on the game situation and strategies used by both sides, provide specific suggestions for the following areas:

1.Our Strategy: Propose adjustments to our current strategy to counter the enemy's moves and capitalize on our strengths.

2.Units and Buildings: Offer ways to enhance our unit composition and improve our building layout, suited to the current stage of the game.

3.Economy: Recommend better practices for resource gathering and usage, in line with our strategic needs.

4.Technology: Suggest focused research paths to gain technological advantages, considering our current research status and technology tree.

Lastly, consider the current situation and the suggestions provided, make 5 actionable and specific decisions from the action dictionary{'TRAIN UNIT': {0: 'TRAIN PROBE', 1: 'TRAIN ZEALOT', 2: 'TRAIN ADEPT', 3: 'TRAIN STALKER', 4: 'TRAIN SENTRY', 5: 'TRAIN HIGHTEMPLAR', 6: 'TRAIN DARKTEMPLAR', 7: 'TRAIN VOIDRAY', 8: 'TRAIN CARRIER', 9: 'TRAIN TEMPEST', 10: 'TRAIN ORACLE', 11: 'TRAIN PHOENIX', 12: 'TRAIN MOTHERSHIP', 13: 'TRAIN OBSERVER', 14: 'TRAIN IMMORTAL', 15: 'TRAIN WARPPRISM', 16: 'TRAIN COLOSSUS', 17: 'TRAIN DISRUPTOR', 18: 'MORPH ARCHON'}, 'BUILD STRUCTURE': {19: 'BUILD PYLON', 20: 'BUILD ASSIMILATOR', 21: 'BUILD NEXUS', 22: 'BUILD GATEWAY', 23: 'BUILD CYBERNETICSCORE', 24: 'BUILD FORGE', 25: 'BUILD TWILIGHTCOUNCIL', 26: 'BUILD ROBOTICSFACILITY', 27: 'BUILD STARGATE', 28: 'BUILD TEMPLARARCHIVE', 29: 'BUILD DARKSHRINE', 30: 'BUILD ROBOTICSBAY', 31: 'BUILD FLEETBEACON', 32: 'BUILD PHOTONCANNON', 33: 'BUILD SHIELDBATTERY'}, 'RESEARCH TECHNIQUE': {34: 'RESEARCH WARPGATERESEARCH', 35: 'RESEARCH PROTOSSAIRWEAPONSLEVEL1', 36: 'RESEARCH PROTOSSAIRWEAPONSLEVEL2', 37: 'RESEARCH PROTOSSAIRWEAPONSLEVEL3', 38: 'RESEARCH PROTOSSAIRARMORSLEVEL1', 39: 'RESEARCH PROTOSSAIRARMORSLEVEL2', 40: 'RESEARCH PROTOSSAIRARMORSLEVEL3', 41: 'RESEARCH ADEPTPIERCINGATTACK', 42: 'RESEARCH BLINKTECH', 43: 'RESEARCH CHARGE', 44: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1', 45: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2', 46: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3', 47: 'RESEARCH PROTOSSGROUNDARMORSLEVEL1', 48: 'RESEARCH PROTOSSGROUNDARMORSLEVEL2', 49: 'RESEARCH PROTOSSGROUNDARMORSLEVEL3', 50: 'RESEARCH PROTOSSSHIELDSLEVEL1', 51: 'RESEARCH PROTOSSSHIELDSLEVEL2', 52: 'RESEARCH PROTOSSSHIELDSLEVEL3', 53: 'RESEARCH EXTENDEDTHERMALLANCE', 54: 'RESEARCH GRAVITICDRIVE', 55: 'RESEARCH OBSERVERGRAVITICBOOSTER', 56: 'RESEARCH PSISTORMTECH', 57: 'RESEARCH VOIDRAYSPEEDUPGRADE', 58: 'RESEARCH PHOENIXRANGEUPGRADE', 59: 'RESEARCH TEMPESTGROUNDATTACKUPGRADE'}, 'OTHER ACTION': {60: 'SCOUTING PROBE', 61: 'SCOUTING OBSERVER', 62: 'SCOUTING ZEALOT', 63: 'SCOUTING PHOENIX', 64: 'MULTI-ATTACK', 65: 'MULTI-RETREAT', 66: 'CHRONOBOOST NEXUS', 67: 'CHRONOBOOST CYBERNETICSCORE', 68: 'CHRONOBOOST TWILIGHTCOUNCIL', 69: 'CHRONOBOOST STARGATE', 70: 'CHRONOBOOST FORGE', 71: 'EMPTY ACTION'}}. This dictionary comprises four categories of actions: unit production, building construction, technology research, and other actions. Remember to align these decisions with the current stage of the game, and avoid proposing actions that are not currently feasible.
"""
```
`2. User Prompt`

Game observation
```shell
"""
chunk{0}: At 10:07 game time, our current StarCraft II situation is as follows:

Resources:
- Game time: 10:07
- Worker supply: 15
- Mineral: 5390
- Gas: 328
- Supply left: 8
- Supply cap: 31
- Supply used: 23
- Army supply: 6

Buildings:
- Nexus count: 1
- Pylon count: 2
- Gas buildings count: 2
- Warp gate count: 4
- Cybernetics core count: 1
- Twilight council count: 1
- Robotics facility count: 1

Units:
- Probe count: 15
- Zealot count: 2
- Stalker count: 1

Planning:

Planning structure:
- Planning robotics facility count: 1

Research:

Cybernetics core:
- Warpgate research status: 1
"""
chunk{1}...
chunk{2}...
chunk{3}...
chunk{4}...
```
`3. LLM Analysis`

LLM take Situation Overview, Situation Analysis, Strategic Planning, Opponent Strategy Analysis these four stages.

```shell 
"""
### Game Overview:
At 10:07 game time, our current situation is characterized by a strong economy with substantial mineral reserves but relatively low gas resources. We have a modest army supply and are in the process of researching key technologies.

### Current Game Stage:
We are in the mid-game stage, with significant resources and ongoing research, but still needing to expand our army and technology base.

### Our Situation:

#### 3.1 Units and Buildings:
- **Buildings:** We have one Nexus, two Pylons, two gas buildings, four Warp Gates, one Cybernetics Core, one Twilight Council, and one Robotics Facility.
- **Units:** We have 16 Probes, 2 Zealots, and 1 Stalker.
- **Planning:** We are planning to build another Robotics Facility.

#### 3.2 Economy:
- **Resources:** We have 5245 minerals and 178 gas.
- **Worker Supply:** We have 16 workers, which is adequate for our current base count but could be expanded as we grow.

#### 3.3 Technology:
- **Research:** Warpgate research is complete, and Blink research is in progress (approximately 0.44% complete).

### Our Strategy:
Our strategy appears to be focused on a balanced approach, with a strong emphasis on economic growth and technological advancement. The completion of Warpgate research and the ongoing Blink research suggest a plan to transition to more advanced units and tactics.

### Enemy's Strategy:
The enemy's strategy is not fully clear from the provided information, but we should be prepared for various mid-game strategies, including potential pushes with advanced units.

### Key Information:
- **Economic Strength:** Our mineral reserves are high, indicating a strong economy.
- **Technology Progress:** Warpgate research is complete, and Blink research is in progress, which will enhance our unit mobility and combat effectiveness.
- **Unit Composition:** Our army is small, with only 2 Zealots and 1 Stalker, suggesting a need for expansion.
"""
```
`4. LLM  Suggestions & Decisions` 

LLM give useful Strategic Recommendations and finally take the Decision-Making Process
```shell
"""
### Suggestions:

#### 1. Our Strategy:
- **Expand Army:** Given our high mineral reserves, focus on expanding our army to prepare for potential engagements.
- **Diversify Units:** Introduce more advanced units like Adepts and Immortals once Blink research is complete.

#### 2. Units and Buildings:
- **Build More Warp Gates:** Increase the number of Warp Gates to facilitate faster unit production.
- **Expand Supply:** Build additional Pylons to increase supply cap and support more units.

#### 3. Economy:
- **Optimize Gas Collection:** Ensure that our gas collection is efficient to support advanced unit production and research.
- **Expand Bases:** Consider expanding to additional bases to further boost our economy.

#### 4. Technology:
- **Complete Blink Research:** Prioritize completing Blink research to enhance our Stalkers' mobility.
- **Research Upgrades:** Consider researching Protoss Ground Weapons and Armor upgrades to improve our unit effectiveness.

### Decisions:
1. **TRAIN STALKER**
2. **BUILD PYLON**
3. **BUILD WARPGATE**
4. **RESEARCH PROTOSSGROUNDWEAPONSLEVEL1**
5. **CHRONOBOOST CYBERNETICSCORE**
"""

```


## Environment

### Install StarCraft II
StatCraft II is a classic game developed by BLZ.You can download Battle.net from [blizzard app](https://www.blizzard.com/zh-tw/)

If you are Chinese, due to the Bobby Kotick, CN play cant own their sever again. So we must download StarCraft II by this video :[video](https://www.bilibili.com/video/BV1As4y147NP/?buvid=XY6E01868C47C929FEFCE4A6DBF0A4ECFFB64&is_story_h5=false&mid=y0%2Bkb3rZVEwQ9j34NFXkLA%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=android&share_plat=android&share_session_id=2e6181cb-fa27-4ce1-9b2e-f126f39267d5&share_source=COPY&share_tag=s_i&timestamp=1674580642&unique_k=rPeGgmE&up_id=149681985&vd_source=0553fe84b5ad759606360b9f2e687a01)
or you can search in the internet.

### Maps
you should put maps [Ancient Cistern LE.SC2Map](./Maps./Ancient Cistern LE.SC2Map) to your StarCrafrt2 file  in StarCraft II\Maps(If the 'Maps' file dont exist, please create it).

## Tips 
- `burnysc2`: This is our core package, offering an easy-to-use API for project development. Find more information here:[Python-sc2](https://github.com/BurnySc2/python-sc2)
- `chromadb`: We utilize the Chroma vector database. Due to package conflicts, **install Chromadb first, followed by burnysc2.**
- `Huggingface` and `sentence-transformers`: we used the embedding model  [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), in our github version, it will automatically download. 
