Street Fighter III
-----------------------
a classic fighting game series developed by CAPCOM, Japan. 
in this project, we adopt [DIAMBRA Arena](https://github.com/diambra/) as our environments.
 
## Installation
- Follow instructions in https://docs.diambra.ai/#installation
- Download the ROM and put it in `~/.diambra/roms` (no need to dezip the content)

## Eval scenes
### Scene1 setting:
| Item           | Setting        |
|----------------|----------------|
| Game mode      | Agent vs Agent |
| Players        | Ken vs Ken     |
| Asynch mode    | True           |
| Num of matches | 10             |
### Scene2 setting:
| Item           | Setting        |
|----------------|----------------|
| Game mode      | Agent vs Agent |
| Players        | Ken vs Ken     |
| Asynch mode    | False          |
| Num of matches | 10             |


## Related code 
`agent`:[streetfight3_agent](../../agent_manager/agents/streetfight3_agent)

`prompt`:[streetfight3_prompt.py](../../agent_manager/prompts/streetfight3_prompt.py)

`scene_setting`:[eval_config_base](../../configs/eval_config_base)

## Prompt instruction
`1. System Prompt`
```shell
"""
You are the best and most aggressive Street Fighter III 3rd strike player in the world.
    Your character is Ken. Your goal is to beat the other opponent. You respond with a bullet point list of moves.
    You are very far from the opponent. Move closer to the opponent.Your opponent is on the left.
Your current health  is 128, and ennemy current health is 93.
You can now use a powerfull move. The names of the powerful moves are: Megafireball, Super attack 2.
Your last action was No-Move. The opponent's last action was Right.
Your current score is 35.0. You are winning. Keep attacking the opponent.
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.

The moves you can use are:
 - Move Closer
 - Move Away
 - Fireball
 - Megapunch
 - Hurricane
 - Megafireball
 - Super attack 2
 - Super attack 3
 - Super attack 4
 - Low Punch
 - Medium Punch
 - High Punch
 - Low Kick
 - Medium Kick
 - High Kick
 - Low Punch+Low Kick
 - Medium Punch+Medium Kick
 - High Punch+High Kick
 - Jump Closer
 - Jump Away
    ----
    Reply with a bullet point list of moves. The format should be: `- <name of the move>` separated by a new line.
    Example if the opponent is close:
    - Move closer
    - Medium Punch
    
    Example if the opponent is far:
    - Fireball
    - Move closer
    
"""
```
`2. User Prompt`

Game observation
```shell
"""
Your next moves are:
"""
```
`3. LLM Analysis & Decisions`

```shell 
"""
- Move Closer  
- Medium Punch
- Low Kick
"""
```
