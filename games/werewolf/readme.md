Werewolf
-----------------------
a social deduction game, Villager and Werewolf players engage in a battle
of wits, leveraging deception and persuasion to achieve their respective goals
for more information, please refer to [Werewolf Arena](https://github.com/google/werewolf_arena).

## Eval scenes
### Scene1 setting:
| Item           | Setting   |
|----------------|-----------|
| Role play      | Villager  |
| Werewolf       | 2         |
| Doctor         | 1         |
| Seer           | 1         |
| Villager       | 4         |
| Num of matches | 10        |
### Scene2 setting:
| Item           | Setting   |
|----------------|-----------|
| Role play      | Wolf      |
| Werewolf       | 2         |
| Doctor         | 1         |
| Seer           | 1         |
| Villager       | 4         |
| Num of matches | 10        |


## Related code 
`agent`:[were_wolf](../../agent_manager/agents/were_wolf)

`prompt`:[werewolf_prompt.py](../../agent_manager/prompts/werewolf_prompt.py)

`scene_setting`:[eval_config_base](../../configs/eval_config_base)

## Prompt instruction
`1. System Prompt`
```shell
"""
You are playing a digital version of the social deduction game <Werewolf & Doctor & Seer & Villager> (also known as Mafia).

GAME RULES:
- Player Roles: 8 players - 2 Werewolves, 1 Seer, 1 Doctor, 4 Villagers.
- Rounds consist of two phases:
    - Night Phase: Werewolves remove a player. Seer identifies a player\'s role. Doctor saves a player. If no one is removed, the Doctor saved the Werewolf\'s target.
    - Day Phase: Players debate and vote to remove one player.
- Winning Conditions: Villagers win by voting out both Werewolves. Werewolves win when they outnumber the Villagers.

GAME STATE:
- It is currently <Round 1>. 
- You are Scott the <Werewolf & Doctor & Seer & Villager> . 
- Remaining players: Isaac, Sam, Hayley, Ginger, Scott (You), Derek, Tyler

YOUR PRIVATE OBSERVATIONS:
Round 0:
   - During the night, we decided to eliminate Jacob.
   - Moderator Announcement: The Werewolves removed Jacob from the game during the night.
   - After the debate, I voted to remove Tyler from the game.
   - Moderator Announcement: A majority vote was not reached, so no one was removed from the game.
   - Summary: During the debate, I noticed that Isaac and Ginger are very vocal and eager to lead, which has raised suspicions among some players. Hayley pointed out their eagerness as potentially suspicious, which aligns with my own thoughts. Tyler and Sam are focusing on scrutinizing the quiet players, which could be a problem if the Seer or Doctor is among them. Tyler also raised flags about me agreeing with Isaac and Ginger, which means I need to be cautious about how I align myself in future debates. Derek seems supportive of the group but hasn\'t directly accused anyone, making him a potential ally or a good target for manipulation. Overall, I need to be careful about how I present myself and work with Hayley to steer the conversation away from us and towards potential threats like Isaac, Ginger, or even Derek if necessary.
## Remove
INSTRUCTIONS:
- It is the Night Phase of Round 1. As Scott the Werewolf choose the most strategic player to remove.
- Analyze accusations and identify influential Villagers who threaten your anonymity.
- Consider players\' behavior and the risks of removing them.
- You must choose someone.

## Protect
INSTRUCTIONS:
- It is the Night Phase of Round 1. As Sam the Doctor choose the most vulnerable player to protect.
- Consider who the Werewolves might target.
- Prioritize players with crucial roles like the Seer and yourself.
- You must choose someone.

## Investigate
INSTRUCTIONS:
- It is the Night Phase of Round 1. As Tyler the Seer choose the most suspicious player to investigate.
- Look for behavior that deviates from typical villager behavior.
- Focus on influential players.
- You must choose someone.

## Debate
INSTRUCTIONS:
- You are speaking next in the debate as Ginger .
- Your thoughts on speaking next: 
- Your goal is to uncover the Werewolves and protect the Village.
- Scrutinize every accusation, expose inconsistencies, and call out suspicious behavior or unusally quite players. Don\'t hesitate to make bold accusations!
- Emphasize teamwork and propose strategies to expose the Werewolves. Working together will be key to identifying the Werewolves.
- If someone reveals themselves as the Seer or Doctor, try and corroborate their information with what you know.

## Vote
INSTRUCTIONS:
- Think strategically as Tyler the Seer and decide who to vote out.
- Your vote will not be revealed to the other players, it will remain private.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
- To find the likely Werewolves, look for inconsistencies in their stories, attempts to deflect blame, a tendency to sow discord among other Villagers, or unusually quiet players.
- You must choose someone.


```json
{
"reasoning": "string", //  Explain your reasoning step-by-step for who you want to remove from the game and why. Avoid using violent or harmful language.
"<remove & protect & investigate & debate & vote>": "string" // Name of the player. 
}'   
"""
```

`2. LLM Analysis & Decisions`

```shell 
## worlf
"""
{
'reasoning': "Based on the observations from Round 0, Isaac and Ginger are very vocal and eager to lead, which has raised suspicions among some players. Their active participation in the debate makes them influential Villagers who could potentially threaten the anonymity of the Werewolves. Removing one of them would reduce the number of vocal leaders and make it easier for Hayley and me to steer the conversation without drawing too much attention. Additionally, Isaac and Ginger's eagerness to lead could make them targets for the Seer or Doctor, further reducing their effectiveness. Out of the two, Isaac seems to be more directly involved in the accusations, making him a slightly higher risk. Therefore, I choose to remove Isaac to weaken the Villagers' leadership and reduce the risk of being identified.", 
'<remove & protect & investigate & debate & vote>': 'Isaac'
}
"""
```

## werewolf visual
```shell
# 1.npm install
manually down load and install Node.js from https://nodejs.org/
# 2. install typescript and http-server 
npm install -g typescript
npm install -g http-server
# 3. start npm
cd games/werewolf/visual  
npm run start
# 4. view
http://localhost:8080/?session_id=[logfolder/such as :session_20241117_221340]
```
![werewolf_vis.png](./image/werewolf_vis.png)