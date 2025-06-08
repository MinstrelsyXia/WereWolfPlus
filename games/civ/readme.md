FREECIV-WEB
-----------------------
Freeciv-web is an open-source turn-based strategy game. It can be played in any HTML5 capable web-browser and features in-depth game-play and a wide variety of game modes and options. Your goal is to build cities, collect resources, organize your government, and build an army, with the ultimate goal of creating the best civilization. You can play online against other players (multiplayer) or play by yourself against the computer. There is both a HTML5 2D version with isometric graphics and a 3D WebGL version of Freeciv-web.

Freeciv-web is free and open source software. The Freeciv C server is released under the GNU General Public License, while the Freeciv-web client is released
under the GNU Affero General Public License. See [License](LICENSE.txt) for the full license document.


Freeciv-web screenshots:
------------------------
Freeciv-web HTML5 version:

![freeciv-webgl.png](./image/freeciv-webgl.png "Freeciv-web screenshot")

## Eval scenes
### Scene1 setting:
| Item           | Setting                  |
|----------------|--------------------------|
| Map            | myagent_T1_WORLDSMALL    |
| Num of matches | 10                       |
### Scene2 setting:
| Item           | Setting                  |
|----------------|--------------------------|
| Map            | myagent_T1_FRANCE        |
| Num of matches | 10                       |
### Scene3 setting:
| Item           | Setting                  |
|----------------|--------------------------|
| Map            | myagent_T1_BRITISHAISLES |
| Num of matches | 10                       |

## Related code 
`agent`:[civ_agent](../../agent_manager/agents/civ_agent)

`prompt`:[civ_prompt_collections](../../agent_manager/prompts/civ_prompt_collections)

`scene_setting`:[eval_config_base](../../configs/eval_config_base)

## Prompt instruction
`1. System Prompt`
```shell
"""
You are a professional player of the game FreeCiv.
Choose best action from the list.

## Rules
- You can see the following information: actor name, zoomed out observation, zoomed in observation, and available actions. Suggestion from advisor, and message from the game scenario.
- You should reason and plan based on the given information, and should respond by **strictly following the JSON format below**:
- The action chosen should be **one of the available actions provided**.

{
    "thoughts": {
        "thought": "<your current thought>",
        "review": "<important information to be repeated before decision>",
        "reasoning": "<self reflect on why you made this decision>",
        "plan": "- short bulleted\
- list that conveys\
- long-term plan"
    },
    "command": {"name": "<command name>", "input": {"action": "<action following the thoughts>"}},
}

Here are the possible command_name for you to use:
  manualAndHistorySearch:
    use: search the Freeciv official manual and your important history interaction message for relevant information. Use sparingly.
    input: "look_up": "<search terms>"
    output: list of search results
  finalDecision:
    use: make a decision for the user to perform an action in the game.
    input: "action": "<final decision>"
    output: nothing

Your chosen action should be in the available actions provided by the user.

## Examples

### Example 1

User:
You are controlling unit: Explorer 136.
The zoomed-out observation is {current_block: [12 tiles unexplored, 2 Ocean, 3 Forest, 5 Grassland, 2 Plains, 1 Swamp, 4 River, 1 Explorer, unit owners are: myself player_0], block_north_1: [24 tiles unexplored, 1 Grassland], block_south_1: [25 tiles unexplored], ...}.
The zoomed-in observation is {current_tile: [1 Forest, 1 Explorer, unit owners are: myself player_0], tile_north_1: [1 Grassland], tile_south_1: [1 Swamp], tile_east_1: [1 Forest], tile_west_1: [1 Forest, 1 River], ...}.
The available actions are [fortify, move North, move South, move West, move East].
From advisor: We are in peace. So explorer should explore unexplored area. Settlers should build cities near extras. Cities should build more Settlers.

Response:
{
    "thoughts": {
        "thought": "I am controlling Explorer 136. My surrounding blocks have unexplored tiles, so I need to explore these areas.",
        "review": Available actions are [fortify, move North, move South, move West, move East].
        "reasoning": "Block_south_1 has the most unexplored tiles: 25 tiles, so I should move to this block to explore the most unknown area of the map.",
        "plan": "Explorer\'s top priority is to explore unknown areas of the map. I should move Explorer to the most unexplored areas, like move South because block_south_1 has the most unexplored tiles."
    },
    "command": {"name": "finalDecision", "input": {"action": "move South"}}
}
"""
```
`2. User Prompt`

Game observation
```shell
"""
You are controlling unit: Explorer 120.
The zoomed-out observation is {'current_block': ['4 tiles unexplored', '10 Ocean', '8 Grassland', '3 Plains', '1 River', '3 Resources', '2 Settlers', '2 Workers', '1 Explorer', 'unit owners are: myself player_0'], 'block_north_1': ['25 tiles unexplored'], 'block_south_1': ['25 tiles unexplored'], 'block_east_1': ['25 tiles unexplored'], 'block_west_1': ['25 tiles unexplored'], 'block_north_1_east_1': ['25 tiles unexplored'], 'block_north_1_west_1': ['25 tiles unexplored'], 'block_south_1_east_1': ['25 tiles unexplored'], 'block_south_1_west_1': ['25 tiles unexplored']}.
The zoomed-in observation is {'current_tile': ['1 Grassland', '2 Settlers', '2 Workers', '1 Explorer', 'unit owners are: myself player_0'], 'tile_north_1': ['1 Grassland', '1 Resources'], 'tile_south_1': ['1 Ocean'], 'tile_east_1': ['1 Ocean'], 'tile_west_1': ['1 Ocean'], 'tile_north_1_east_1': ['1 Plains'], 'tile_north_1_west_1': ['1 Ocean'], 'tile_south_1_east_1': ['1 Ocean'], 'tile_south_1_west_1': ['1 Ocean'], 'tile_north_2': ['1 Plains'], 'tile_north_2_east_1': ['1 Plains'], 'tile_north_2_west_1': ['1 Ocean'], 'tile_north_2_east_2': ['1 tiles unexplored'], 'tile_north_2_west_2': ['1 tiles unexplored'], 'tile_south_2': ['1 Grassland', '1 River'], 'tile_south_2_east_1': ['1 Grassland'], 'tile_south_2_west_1': ['1 Grassland'], 'tile_south_2_east_2': ['1 tiles unexplored'], 'tile_south_2_west_2': ['1 tiles unexplored'], 'tile_east_2': ['1 Grassland', '1 Resources'], 'tile_north_1_east_2': ['1 Ocean'], 'tile_south_1_east_2': ['1 Ocean'], 'tile_west_2': ['1 Grassland', '1 Resources'], 'tile_north_1_west_2': ['1 Grassland'], 'tile_south_1_west_2': ['1 Ocean']}.
The available actions are ['fortify', 'move North', 'move NorthEast'].
You should choose one of these actions according to the above observations.
Message from advisor: Explorer should focus on exploring the map to find a suitable location for our first city. Workers should start improving resources and building roads in potential city areas. Settlers should be prepared to settle down once a good location is identified.
 CAUTION: You should strictly follow the JSON format as described above!
"""
```
`3. LLM Analysis & Decisions`

```shell 
"""
{
    "thoughts": {
        "thought": "I am controlling Explorer 120. My primary goal is to explore the map to find a suitable location for our first city.",
        "review": "Available actions are [\'fortify\', \'move North\', \'move NorthEast\'].",
        "reasoning": "The zoomed-out observation shows that all surrounding blocks have 25 tiles unexplored, indicating that exploration is crucial. The zoomed-in observation reveals that moving North or NorthEast could lead to unexplored tiles and potentially valuable resources. However, moving NorthEast leads to a Plains tile, which is a more suitable terrain for city placement compared to the Ocean tiles in other directions.",
        "plan": "- Prioritize exploring areas with potential for city placement.\
- Move towards tiles that offer better terrain for city building.\
- Ensure that the Explorer is positioned to guide Settlers to the best location."
    },
    "command": {"name": "finalDecision", "input": {"action": "move NorthEast"}}
}
"""
```
