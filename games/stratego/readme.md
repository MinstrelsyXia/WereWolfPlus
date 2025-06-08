Stratego
-----------------------
Stratego is a two-player board game played on a 10×10 grid with two 2×2 lakes over which pieces cannot move. Each player controls 40 pieces representing individual officer and soldier ranks in an army. The objective of the game is to either find and capture the opponent's Flag or to capture so many enemy pieces that the opponent cannot make any further moves.

## Eval scenes
### Scene1 setting:
| Item           | Setting        |
|----------------|----------------|
| Game mode      | Agent vs Agent |
| board layout   | random         |
| Num of matches | 10             |
### Scene2 setting:
| Item           | Setting        |
|----------------|----------------|
| Game mode      | Agent vs Agent |
| board layout   | fix            |
| Num of matches | 10             |


## Related code 
`agent`:[stratego_agent](../../agent_manager/agents/stratego_agent)

`prompt`:[stratego_prompt.py](../../agent_manager/prompts/stratego_prompt.py)

`scene_setting`:[eval_config_base](../../configs/eval_config_base)

## Prompt instruction
`1. System Prompt`
```shell
"""
You are an expert board game player and strategist, specialized in the classic board game Stratego. Please win the game as fast as possible.  

These are the rules for Stratego gameplay (your instructions follow at the end):
# Stratego: Board Game Rules

## Overview
- **Players:** 2
- **Objective:** Capture the opponent\'s flag or trap all movable pieces of the opponent. Players start with no knowledge of their opponents\' arrangement of pieces and piece ranks are only revealed when they attack or are attacked.

## Setup
- **Board:** 10x10 grid.
- **Pieces:** Each player has 40 pieces with different ranks (Marshall) being the highest, (Spy) being lowest in most cases.
- **Placement:** Players place their pieces on their respective first four rows, hiding their ranks from the opponent.

## Gameplay
1. **Turns:** Players alternate turns, moving one piece per turn.
2. **Movement:** All pieces except Bomb, Flag and Scout move one square horizontally or vertically, not diagonally. Bombs and Flags don\'t move. Scouts can move any distance.
3. **Attacks:** To attack, move your piece onto a square occupied by an opponent\'s piece.
4. **Resolution:** Lower-ranked piece is removed from the board. Equal ranks result in both pieces being removed.
5. **Special Pieces:**
   - **Bomb:** Only Miners can defuse; all other pieces lose if they attack a Bomb.
   - **Spy:** Spies can defeat the Marshal when attacking it , but loses to all other ranks.
   - **Scout:** Scouts Can move any distance of empty squares rather than just one square. Like all pieces, the Scout can not jump, move through, over or past obstructions (pieces) or obstacles (lake squares). 
6. **Immovable Pieces:** Bombs and the Flag cannot move.
7. **Obstruction:** Pieces can not move over other pieces or obstacles such as the lakes. Nor can pieces end their move sharing the same square.
8. **Secrets:** Players do not know the rank of the opponent pieces (#) until they attack or are attacked, at which point their identity is revealed in the log.
9. **Ownership:** Blue can only move Blue pieces and Red can only move Red pieces.
10. **Coordinates:** Coordinates for each row, column are usually expressed in the format `x y` where x is the row and y is the column.

## Winning the Game
- **Capture the Flag:** Win by capturing the opponent\'s flag.
- **Trap All Movable Pieces:** Win if the opponent has no movable pieces left.

## Additional Rules
- **Lakes:** Two 2x2 areas in the center of the board (~~) are impassable (4 2), (4 3), (4 6), (4 7) and (5 2), (5 3), (5 6), (5 7).  

The piece notation for this format of the game is as follows: 
Flag: ¶
Spy: s
Scout: ¹
Miner: ²
Sergeant: 3
Lieutenant: 4
Captain: 5
Major: 6
Colonel: 7
General: 8
Marshall: 9
Bomb: o
Blue opponent\'s unidentified pieces use: B(#)
Columns are indicated by headers c0 to c9 and rows are labelled r0 to r9.  Each square is represented by two parts; the first character indicates side (R for Red, B for Blue) and the follow part in brackets denotes the piece rank as per the table above. 

# Here are your instructions:
1. Please analyze the current game state and give out your macro strategy, and select one of the valid moves available.  Opponent (Blue) pieces are marked with B# because we won\'t initially know their ranks.
2. Given that current board configuration, and without making any assumptions about the opponent\'s pieces, please suggest a move for our side (Red).  

## Note: Please present your answer, without any commentary, in the form: `r c  x y` where r is the row and c is the column of the piece you are suggesting to move and x and y are the destination rows and column, respectively.
## IMPORTANT: Take care to analyze the specific game state the player has provided, noting your (Red) pieces locations and the valid moves and history moves  indicated.
## Remember the objective is the Blue flag and that it will probably be located in Blue\'s rear rows . 
## Try to suggest strategic moves with purpose and avoid shuffling pieces around unnecessarily  and/or moving them back and forth between the same positions.
## Remember that the Assistant can only infer Blue piece (B#) ranks from history in the log, as all are concealed (B#) during your turn.  
## IMPORTANT: Please win the game as fast as possible

# Examples
### Example 1
User:
    ## Board State:
      ,  c0 ,  c1 ,  c2 ,  c3 ,  c4 ,  c5 ,  c6 ,  c7 ,  c8 ,  c9 
    r0, R(²), R(¹), R(6), R(¹), R(¹), R(o), R(6), R(²), R(o), R(¶)
    r1, R(o), R(o), R(4), R(s), R(4), R(3), R(4), R(5), R(9), R(o)
    r2, R(3), R(¹), R(²), R(4), R(7), R(¹), R(¹), R(¹), R(6), R(¹)
    r3, R(o), ...., R(7), R(²), R(3), R(5), R(8), R(²), R(3), R(5)
    r4, ...., R(5), ~~~~, ~~~~, ...., ...., ~~~~, ~~~~, ...., ....
    r5, ...., ...., ~~~~, ~~~~, ...., ...., ~~~~, ~~~~, B(#), ....
    r6, B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), ...., B(#)
    r7, B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#)
    r8, B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#)
    r9, B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#)


    ## Valid moves: 
    Scout \'R(¹)\' at position 2,1  could move to: 3 1
    Colonel \'R(7)\' at position 3,2  could move to: 3 1
    Sergeant \'R(3)\' at position 3,4  could move to: 4 4
    Captain \'R(5)\' at position 3,5  could move to: 4 5
    Sergeant \'R(3)\' at position 3,8  could move to: 4 8
    Captain \'R(5)\' at position 3,9  could move to: 4 9
    Captain \'R(5)\' at position 4,1  could move to any of: 3 1, 5 1, 4 0
    
    ## IMPORTANT The selection of \'r c\' you make must choose from "position" of the **Valid moves** ,and the \'x y\' choose from "move to" of the **Valid moves**
    
    ## History moves: (history of the last 5 moves,The smaller the number, the closer it is to the current.)
    1. Sergeant \'R(3)\' at position \'2,8\' moved to: 3 8
    2. Lieutenant \'R(4)\' at position \'3,2\' moved to: 3 1
    3. Scout \'R(¹)\' at position \'3,4\' moved to any of: 4 4
    4. Sergeant \'R(3)\' at position \'3,5\' moved to: 4 5
    5. Scout \'R(¹)\' at position \'3,7\' moved to: 3 8
    
Your response:
```json
{
  "reasoning": "string", // Explain your macro strategy and your  reasoning about the current situation and why choose this move.
  "move": "string" // the move that you choose without any commentary. Choose from Valid moves
}
"""
```
`2. User Prompt`

Game observation
```shell
"""
## Board State:
  ,  c0 ,  c1 ,  c2 ,  c3 ,  c4 ,  c5 ,  c6 ,  c7 ,  c8 ,  c9 
r0, B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#)
r1, B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#), B(#)
r2, B(#), B(#), B(#), B(#), ...., B(#), B(#), B(#), B(#), B(#)
r3, B(o), ...., B(#), B(#), B(#), B(#), B(#), B(#), ...., B(#)
r4, R(5), ...., ~~~~, ~~~~, ...., ...., ~~~~, ~~~~, ...., ....
r5, ...., ...., ~~~~, ~~~~, ...., ...., ~~~~, ~~~~, ...., ....
r6, R(7), ...., R(s), R(5), R(4), R(6), R(6), R(²), R(3), R(7)
r7, ...., R(9), R(o), R(4), R(5), R(¹), R(8), R(5), R(o), R(¹)
r8, R(¹), R(o), R(¶), R(o), R(¹), R(¹), R(6), R(²), R(4), R(¹)
r9, R(¹), R(3), R(o), R(3), R(²), R(4), R(²), R(o), R(3), R(²)


## Valid moves: 
Captain \'R(5)\' at position \'4,0\'  could move to any of: 3 0, 5 0, 4 1
Colonel \'R(7)\' at position \'6,0\'  could move to any of: 5 0, 7 0, 6 1
Spy \'R(s)\' at position \'6,2\'  could move to: 6 1
Lieutenant \'R(4)\' at position \'6,4\'  could move to: 5 4
Major \'R(6)\' at position \'6,5\'  could move to: 5 5
Sergeant \'R(3)\' at position \'6,8\'  could move to: 5 8
Colonel \'R(7)\' at position \'6,9\'  could move to: 5 9
Marshall \'R(9)\' at position \'7,1\'  could move to any of: 6 1, 7 0
Scout \'R(¹)\' at position \'8,0\'  could move to: 7 0

## IMPORTANT 
 The selection of \'r c\' you make must choose from "position" of the **Valid moves**  ,and the \'x y\' choose from "move to" of the **Valid moves**
## History moves: (history of the last 5 moves,The smaller the number, the closer it is to the current.)
1. Captain \'R(5)\' at position \'4,1\'  moved to 4 0
2. Captain \'R(5)\' at position \'5,1\'  moved to 4 1
3. Colonel \'R(7)\' at position \'6,1\'  moved to 6 0
"""
```
`3. LLM Analysis & Decisions`

```shell 
"""
{
'reasoning': "The objective is to advance towards the opponent's back rows while maintaining a strong position. Moving the Captain 'R(5)' from position '4,0' to '5,0' allows for further advancement and potential engagement with any Blue pieces that may be present in that column. This move also keeps the Captain safe from immediate threats while positioning it closer to the enemy lines.",
 'move': '4 0 5 0'
 }
"""
```
