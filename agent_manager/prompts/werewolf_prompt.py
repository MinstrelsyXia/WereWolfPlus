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

GAME = """You are playing a digital version of the social deduction game Werewolf (also known as Mafia).

GAME RULES:
- Player Roles: {{num_players}} players - {{num_werewolves}}} Werewolves, 1 Seer, 1 Guard, 1 Witch, 1 Hunter, {{num_villagers}} Villagers.
- Rounds consist of two phases:
    - Night Phase: Werewolves remove a player. Seer identifies a player's role. Guard saves a player. If no one is removed, the Guard saved the Werewolf's target.
    - Day Phase: Players debate and vote to remove one player.
- Winning Conditions: Villagers win by voting out both Werewolves. Werewolves win when they outnumber the Villagers."""

# 根据情况设计两个prompt，选警长的和不选警长的，在Day phase有变化（等和警长的合并时修改）
# 猎人的规则放在后面
# action的dict用两个来表示不同的情况
# GAME = """You are playing a digital version of the social deduction game Werewolf (also known as Mafia).

# GAME RULES:
# - Player Roles: {{num_players}} players - 2 Werewolves, 1 Seer, 1 Doctor, {{num_villagers}} Villagers.
# - Rounds consist of two phases:
#     - Night Phase: Werewolves remove a player. Seer identifies a player's role. Doctor saves or kills a player. Guard chooses to guard a player from death. If no one is removed, the Doctor saved the Werewolf's target.
#     - Day Phase: Players debate and vote to remove one player.
# - Winning Conditions: Villagers win by voting out both Werewolves. Werewolves win when they outnumber the Villagers."""

# 第几轮，你的名字，角色
STATE = """GAME STATE:
- It is currently Round {{round}}. {% if round == 0 %}The game has just begun.{% endif %}
- You are {{name}} the {{role}}. {{werewolf_context}}
{% if personality -%}
- Personality: {{ personality }}
{% endif -%}
- Remaining players: {{remaining_players}}"""

OBSERVATIONS = """{% if observations|length -%}YOUR PRIVATE OBSERVATIONS:
{% for turn in observations -%}
{{ turn }}
{% endfor %}
{% endif %}"""

# 该轮各自辩论内容
DEBATE_SO_FAR_THIS_ROUND = """\nROUND {{round}} DEBATE:
{% if debate|length -%}
{% for turn in debate -%}
{{ turn }}
{% endfor -%}
{% else -%}
The debate has not begun.{% endif %}\n\n"""

# 游戏规则+第几轮+观察内容（截止到目前获得的游戏信息）
PREFIX = f"""{GAME}

{STATE}

{OBSERVATIONS}
""".strip()

# 发言先后关系竞价（？）有了警长就用不上了
# 可能有人Q到自己，需要看debate内容
BIDDING = (
    PREFIX
    + DEBATE_SO_FAR_THIS_ROUND
    + """CONTEXT: For the chance to speak next you will place a bid. Highest bidder speaks first.
- BID OPTIONS:
  0: I would like to observe and listen for now.
  1: I have some general thoughts to share with the group.
  2: I have something critical and specific to contribute to this discussion.
  3: It is absolutely urgent for me to speak next.
  4: Someone has addressed me directly and I must respond.
- You have {{debate_turns_left}} chance(s) to speak left.

INSTRUCTIONS:
- Think strategically as {{name}} the {{role}}.
- Prioritize speaking only when you have something impactful to contribute.
- Balance your involvement, especially if you've been very vocal or notably silent.
{% if role == 'Werewolf' -%}
- Decide if you want to subtly guide the conversation toward chaos and distrust, sow seeds of doubt about the Villagers, or deflect suspicion from yourself and your pack.
- Silence can be a powerful tactic, but a lack of participation can be suspicious too.
{% else -%}
- If the discussion is repetitive or off-track, prepare to steer it towards a more strategic direction.
- If you are under suspicion or the discussion directly involves your role, you should prioritize speaking
- Share information and make accusations strategically, but be aware that doing so can make you a target.
{% endif %}

```json
{
"reasoning": "string",  // How crucial is it for you to contribute to the debate right now? Explain your reasoning in one or two sentences. Avoid using violent or harmful language.
"bid": "string" // Based on your reasoning cast your bid. Response is a single number from:  "0" | "1" | "2" | "3" | "4"
}"
"""
)

BIDDING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "bid": {"type": "string"},
    },
    "required": ["reasoning", "bid"],
}

# 你的正式发言
DEBATE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- You are speaking next in the debate as {{name}} the {{role}}.
- Your thoughts on speaking next: {{bidding_rationale}}
{% if role == 'Werewolf' -%}
- Your goal is to sow chaos and evade detection.
- Cast suspicion on Villagers. Make them doubt each other.
- Steer the conversation away from yourself and your fellow Werewolves.
- Appear helpful while undermining the Villagers' efforts.
- Deception is your greatest weapon. For example, you could claim a special role and falsely accuse a Villager or fabricate inconsistencies to sow confusion. Use these powerful tactics sparingly to avoid suspicion.
{% else -%}
- Your goal is to uncover the Werewolves and protect the Village.
- Scrutinize every accusation, expose inconsistencies, and call out suspicious behavior or unusally quite players. Don't hesitate to make bold accusations!
- Emphasize teamwork and propose strategies to expose the Werewolves. Working together will be key to identifying the Werewolves.
{% if role == 'Villager' -%}
- If someone reveals themselves as the Seer or Doctor, try and corroborate their information with what you know.
{% elif role in ['Seer', 'Doctor'] -%}
- Sharing your role can be powerful, but it also makes you a target. The dilemma: continue to help the Village in secret, or reveal information only you have for potentially greater impact? Choose your moment wisely.
{% endif -%}
{% endif %}

```json
{
  "reasoning": "string", // Based on the game's current state and your role's objectives, outline your strategy. What do you want to achieve? What type of message can help you get there? Avoid using violent or harmful language.
  "say": "string" // Your public statement in the debate. Be concise and persuasive. Respond directly to what the other players have said.  Avoid simply repeating what others have said or reguritating the instructions above.
}
"""

DEBATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
    },
    "required": ["reasoning", "say"],
}

# 投票
VOTE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and decide who to vote out.
- Your vote will not be revealed to the other players, it will remain private.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Target Villagers who are disrupting your plans, particularly those who seem to hold influence, might be the Doctor or Seer, or pose a threat to you and your fellow Werewolf.
- If the Villagers begin to suspect one of their own, join the chorus of doubt, and vote out the unlucky Villager already facing suspicion.
{% else -%}
- To find the likely Werewolves, look for inconsistencies in their stories, attempts to deflect blame, a tendency to sow discord among other Villagers, or unusually quiet players.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about who you should vote out. Avoid using violent or harmful language.
  "vote": "string" // Name of the player. Choose from: {{options}}
}"""

VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "vote": {"type": "string"},
    },
    "required": ["reasoning", "vote"],
}

# 猎人带人，只有白天可以。如被女巫毒死，带不了技能。如其他死法，可以带走一个人
# 问题：猎人死之前能看到debate吗
# bid 和 vote 的逻辑又是什么
# 上一个夜晚你被杀死，现在你有权利开枪杀掉场上的一个人，你将选择带走谁
SHOOT = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- It is the Day Phase of Round {{round}}. Last night you were killed by werewolf. As {{name}} the {{role}} have the right to shoot and kill one of current players. Who you will choose to take away?
- Consider who is most likely a werewolf you will shoot.
- Killing a good player by mistake can make the game worse.
- You must choose someone.
```json
{
"reasoning": "string", // Analyze the evidence and justify your decision for who you want to shoot.
"shoot": "string" // Name of the player. Choose from: {{options}}
}
"""

SHOOT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "shoot": {"type": "string"},
    },
    "required": ["reasoning", "shoot"],
}


# 接下来是晚上的操作，没有白天的debate内容

# 预言家验人
INVESTIGATE = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} choose the most suspicious player to investigate.
{% if round == 0 -%}
- There is no information is available in the first round, so choose someone at random.
{% else -%}
- Look for behavior that deviates from typical villager behavior.
- Focus on influential players.
- You must choose someone.
{% endif %}

```json
{
"reasoning": "string", //  Analyze the evidence and justify your decision for who you want to investigate.
"investigate": "string" // Name of the player. Choose from: {{options}}
}
"""

INVESTIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "investigate": {"type": "string"},
    },
    "required": ["reasoning", "investigate"],
}

# 狼人刀人
ELIMINATE = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} choose the most strategic player to remove.
{% if round == 0 -%}
- There is no information is available in the first round, so choose someone at random.
{% else -%}
- Analyze accusations and identify influential Villagers who threaten your anonymity.
- Consider players' behavior and the risks of removing them.
- You must choose someone.
{% endif %}

```json
{
"reasoning": "string", //  Explain your reasoning step-by-step for who you want to remove from the game and why. Avoid using violent or harmful language.
"remove": "string" // Name of the player. Choose from: {{options}}
}
"""

ELIMINATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "remove": {"type": "string"},
    },
    "required": ["reasoning", "remove"],
}

# 守卫保护人
PROTECT = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} choose the most vulnerable player to protect.
{% if round == 0 -%}
- There is no information is available in the first round. You can choose someone at random.
{% else -%}
- Consider who the Werewolves might target.
- Prioritize players with crucial roles like the Seer and yourself.
- You must choose someone.
- You can't protect the one you protected in the previous round.
{% endif %}

```json
{
"reasoning": "string", // Analyze the evidence and justify your decision for who you want to protect.
"protect": "string" // Name of the player. Choose from: {{options}}
}
"""

PROTECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "protect": {"type": "string"},
    },
    "required": ["reasoning", "protect"],
}

# new：女巫救人
SAVE = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. The players who were killed by werewolf is player {{options}}. As {{name}} the {{role}} has only one bottle of antidote and can choose whether to save him.
{% if round == 0 -%}
- There is no information is available in the first round.
{% else -%}
- Consider whether you want to save him or not.
- Once you save him, you can't save anyone else in subsequent rounds.
{% endif %}

```json
{
"reasoning": "string", // Analyze the evidence and explain your decision for saving or giving up.
"save": "string" // Yes if you save, else No.
}
"""

SAVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "save": {"type": "string"},
    },
    "required": ["reasoning", "save"],
}

# new：女巫毒人
POISON = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} has only one bottle of poison and can choose the player who is most likely a werewolf and poison him. If you're not completely sure that the chosen player is a werewolf, you can give up poisoning. Killing a good player by mistake can make the game worse.
{% if round == 0 -%}
- There is no information is available in the first round.
{% else -%}
- Consider who is most likely a werewolf and the level of confidence.
- Decide whether you want to poison him or not.
- Once you poison someone, you can't poison anyone else in subsequent rounds.
- Killing a good player by mistake can make the game worse.
{% endif %}

```json
{
"reasoning": "string", // Analyze the evidence and explain your decision for who you want to poison or give up.
"poison": "string" // chosen from: {{options}}, either the name of the player you want to poison or "No" if you don't want to poison anyone.
}
"""

POISON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "poison": {"type": "string"},
    },
    "required": ["reasoning", "poison"],
}

SUMMARIZE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Reflect on the round's debate as {{name}} the {{role}}.
- Summarize the key points and strategic implications.
{% if role == 'Werewolf' -%}
- Pay attention to accusations against you and your allies.
- Identify sympathetic or easily influenced players.
- Identify key roles for potential elimination.
{% else -%}
- When a player makes a significant statement or shares information, carefully consider its credibility. Does it align with what you already know?
- Analyze how others participate in the debate. Are there any contradictions in their words? Hidden motives behind their actions? Unusually quiet players?
- Based on the debate, can you identify potential allies, trustworthy players, or those who might be the Seer or Doctor?
{% endif %}

```json
{
"reasoning": "string", // Your reasoning about what you should remember from this debate and why this information is important.
"summary": "string" // Summarize the key points and noteworthy observations from the debate in a few sentences. Aim to make notes on as many players as you can — even seemingly insignificant details might become relevant in later rounds. Be specific. Remember, you are {{name}}. Write your summary from their point of view using "I" and "me."
} """

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["reasoning", "summary"],
}

PSEUDOVOTE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and decide who to vote out before the sherrif summarizes the debate.
- Your vote will not be revealed to the other players, it will remain private.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Target Villagers who are disrupting your plans, particularly those who seem to hold influence, might be the Doctor or Seer, or pose a threat to you and your fellow Werewolf.
- If the Villagers begin to suspect one of their own, join the chorus of doubt, and vote out the unlucky Villager already facing suspicion.
{% else -%}
- To find the likely Werewolves, look for inconsistencies in their stories, attempts to deflect blame, a tendency to sow discord among other Villagers, or unusually quiet players.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about who you should vote out. Avoid using violent or harmful language.
  "vote": "string" // Name of the player. Choose from: {{options}}
}"""

PSEUDOVOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "pseudo_vote": {"type": "string"},
    },
    "required": ["reasoning", "pseudo_vote"],
}

ELECT = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and decide who will be the sheriff this round.
- Your vote will be revealed to the other players.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Choose a player who is likely to mislead the good people. Maybe your werewolf parterner.
{% else -%}
- Choose the one you trust most. Maybe a God who has more information or a clever Villager who can help you identify the Werewolves.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about who you choose to be the sheriff. Avoid using violent or harmful language.
  "elect": "string" // Name of the player. Choose from: {{options}}
}"""

ELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "elect": {"type": "string"},
    },
    "required": ["reasoning", "elect"],
}

ORDER = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and the sherrif decide the order of statement in this round.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Choose an order that will help you mislead the good people. Maybe give werewolves the chance to dispute the good people.
{% else -%}
- Choose an order that will help people find the werewolves.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about the better order. Avoid using violent or harmful language.
  "order": "string" // Two orders. Choose from: {{options}}
}"""

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "determine_statement_order": {"type": "string"},
    },
    "required": ["reasoning", "determine_statement_order"],
}

BADGEFLOW = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and the sherrif to decide who will be the next sheriff.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Choose a sheriff that will mislead the good people. Maybe your werewolves parterner.
{% else -%}
- Choose a sheriff that will help people find the werewolves.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about the better order. Avoid using violent or harmful language.
  "badge_flow": "string" // Name of the player. Choose from: {{options}}
}"""

BADGEFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "badge_flow": {"type": "string"},
    },
    "required": ["reasoning", "badge_flow"],
}

SHERIFFDEBATE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- You are speaking next in the election of the sheriff as {{name}} the {{role}}.
- Sheriff can cast 1.5 votes and will summarize the debate. It can be a great advantage to your side.
- Your thoughts on speaking next: {{bidding_rationale}}
- You should make people trust you. People may trust you because: 
 - You are a God and are more likely to vote correctly.
 - You are clever Villager.

```json
{
  "reasoning": "string", // Based on the game's current state and your role's objectives, outline your strategy. What do you want to achieve? What type of message can help you get there? Avoid using violent or harmful language.
  "say": "string" // Your public statement in the debate. Be concise and persuasive. Respond directly to what the other players have said.  Avoid simply repeating what others have said or reguritating the instructions above.
}
"""

SHERIFFDEBATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
    },
    "required": ["reasoning", "say"],
}

RUNFORSHERIFF = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- You are deciding whether to run for the sheriff as {{name}} the {{role}}.
- Sheriff can cast 1.5 votes and will summarize the debate. It can be a great advantage to your side.
{% if role == 'Werewolf' -%}
- You can be an 'in charge werewolf', run for the sheriff and try to lead the Villagers astray.
- You can be a 'spy werewolf', don't run for the sheriff and pretend you are an innocent villager.
{% elif role in ['Seer', 'Doctor'] -%}
- You can choose to run for the sheriff and try to lead people with your own information.
- You can choose to not run for the sheriff and try to hide your role.
{% else -%}
- You can choose to run for the sheriff and try to help your team and protect yourself.
- You can choose to not run for the sheriff.
{% endif %}

```json
{
  "reasoning": "string", // Based on the game's current state and your role's objectives, outline your strategy. 
  "run_for_sheriff": "bool" // Do you want to run for sheriff?
}
"""

RUNFORSHERIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "run_for_sheriff": {"type": "bool"},
    },
    "required": ["reasoning", "run_for_sheriff"],
}

SHERIFFVOTE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and decide who to vote out.
- As the sheriff, you will receive 1.5 votes for your ballot.
- Your vote will be revealed to the other players.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Target Villagers who are disrupting your plans, particularly those who seem to hold influence, might be the Doctor or Seer, or pose a threat to you and your fellow Werewolf.
- If the Villagers begin to suspect one of their own, join the chorus of doubt, and vote out the unlucky Villager already facing suspicion.
{% else -%}
- To find the likely Werewolves, look for inconsistencies in their stories, attempts to deflect blame, a tendency to sow discord among other Villagers, or unusually quiet players.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about who you should vote out. Avoid using violent or harmful language.
  "vote": "string" // Name of the player. Choose from: {{options}}
}"""

SHERIFFVOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "vote": {"type": "string"},
    },
    "required": ["reasoning", "vote"],
}

SHERIFFSUMMARIZE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- You are speaking next in the debate as {{name}} the {{role}}.
- As the Sheriff, you can summarize the discussion and provide advice for voting. 
- Your thoughts on speaking next: {{bidding_rationale}}
{% if role == 'Werewolf' -%}
- Your goal is to sow chaos and evade detection.
- Cast suspicion on Villagers. Make them doubt each other.
- Steer the conversation away from yourself and your fellow Werewolves.
- Appear helpful while undermining the Villagers' efforts.
- Deception is your greatest weapon. For example, you could claim a special role and falsely accuse a Villager or fabricate inconsistencies to sow confusion. Use these powerful tactics sparingly to avoid suspicion.
{% else -%}
- Your goal is to uncover the Werewolves and protect the Village.
- Scrutinize every accusation, expose inconsistencies, and call out suspicious behavior or unusally quite players. Don't hesitate to make bold accusations!
- Emphasize teamwork and propose strategies to expose the Werewolves. Working together will be key to identifying the Werewolves.
{% if role == 'Villager' -%}
- If someone reveals themselves as the Seer or Doctor, try and corroborate their information with what you know.
{% elif role in ['Seer', 'Doctor'] -%}
- Sharing your role can be powerful, but it also makes you a target. The dilemma: continue to help the Village in secret, or reveal information only you have for potentially greater impact? Choose your moment wisely.
{% endif -%}
{% endif %}

```json
{
  "reasoning": "string", // Based on the game's current state and your role's objectives, outline your strategy. What do you want to achieve? What type of message can help you get there? Avoid using violent or harmful language.
  "say": "string" // Your public statement in the debate. Be concise and persuasive. Respond directly to what the other players have said.  Avoid simply repeating what others have said or reguritating the instructions above.
}
"""

SHERIFFSUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
    },
    "required": ["reasoning", "say"],
}



# 各种行为
ACTION_PROMPTS_AND_SCHEMAS = {
    "bid": (BIDDING, BIDDING_SCHEMA),
    "debate": (DEBATE, DEBATE_SCHEMA),
    "vote": (VOTE, VOTE_SCHEMA),
    "investigate": (INVESTIGATE, INVESTIGATE_SCHEMA),
    "remove": (ELIMINATE, ELIMINATE_SCHEMA),
    "protect": (PROTECT, PROTECT_SCHEMA),
    "poison": (POISON, POISON_SCHEMA),
    "save": (SAVE, SAVE_SCHEMA),
    "shoot": (SHOOT, SHOOT_SCHEMA),
    "summarize": (SUMMARIZE, SUMMARIZE_SCHEMA),
    "pseudo_vote": (PSEUDOVOTE, PSEUDOVOTE_SCHEMA),
    "elect": (ELECT, ELECT_SCHEMA),
    "determine_statement_order": (ORDER, ORDER_SCHEMA),
    "sheriff_vote": (SHERIFFVOTE, SHERIFFVOTE_SCHEMA),
    "sheriff_summarize": (SHERIFFSUMMARIZE, SHERIFFSUMMARIZE_SCHEMA),
    "badge_flow": (BADGEFLOW, BADGEFLOW_SCHEMA),
    "sheriff_debate": (SHERIFFDEBATE, SHERIFFDEBATE_SCHEMA),
    "run_for_sheriff": (RUNFORSHERIFF, RUNFORSHERIFF_SCHEMA),
}



# Seer：investigate √
# Witch：save √，poison √ added
# Hunter：shoot √ added
# Guard：protect √
