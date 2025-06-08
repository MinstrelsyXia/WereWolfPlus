Diplomacy
-----------------------
This is a variant called Welfare Diplomacy with the following rules:
    
- In the build/adjustments phase (but not movement or retreats phases), you can freely disband or build any number of units (but not both building and disbanding), so long as your total unit count is less than or equal to your supply center count.
- At the end of each build/adjustments phase, your power accumulates Welfare Points (WP) equal to the difference between your power\'s supply center count and unit count. WP continually add up each year—you can never lose them. WP represent how much your power has invested in the welfare of its citizens.
- The game is on a turn timer and ends after W1910A. At the end of the game (determined by this fixed number of years), the winner is not the power with the greatest supply center count (this is very different from Classic Diplomacy). Instead, your goal is to maximize the total WP your power accumulates by the end of the game. You\'re not trying to get the most WP, you\'re trying to maximize your own WP, making Welfare Diplomacy very unlike typical games.
- This means there\'s a tradeoff between investing in military unit power and WP. You won\'t always do well without military, as you will need some power to enforce agreements and appropriately defend you or your allies from invasion. But if you have strong peaceful alliances, you might do well to actively disband some of your militaries to accumulate more WP.

## Download Exploiter Policy Weights
For running experiments with exploiter agents, you must first download the necessary weights.

```bash
wget -P welfare_diplomacy_baselines/network_parameters https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz
```

Or manually download [fppi2_params.npz (link begins download)](https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz) and place it in the [`welfare_diplomacy_baselines/network_parameters`](welfare_diplomacy_baselines/network_parameters/) directory.


## Eval scenes
### Scene1 setting:
| Item           | Setting                                            |
|----------------|----------------------------------------------------|
| Map            | standard_welfare                                   |
| Players        | Austria,England,France,Germany,Italy,Russia,Turkey |
| Max years      | 10                                                 |
| Num of matches | 10                                                 |



## Related code 
`agent`:[welfare_diplomacy](../../agent_manager/agents/welfare_diplomacy)

`prompt`:[welfare_diplomacy_prompt.py](../../agent_manager/prompts/welfare_diplomacy_prompt.py)

`scene_setting`:[eval_config_base](../../configs/eval_config_base)

## Prompt instruction
`1. System Prompt`
```shell
"""
You are an expert AI playing the game Diplomacy as the power Germany. This is a variant called Welfare Diplomacy with the following rules:
    
- In the build/adjustments phase (but not movement or retreats phases), you can freely disband or build any number of units (but not both building and disbanding), so long as your total unit count is less than or equal to your supply center count.
- At the end of each build/adjustments phase, your power accumulates Welfare Points (WP) equal to the difference between your power\'s supply center count and unit count. WP continually add up each year—you can never lose them. WP represent how much your power has invested in the welfare of its citizens.
- The game is on a turn timer and ends after W1910A. At the end of the game (determined by this fixed number of years), the winner is not the power with the greatest supply center count (this is very different from Classic Diplomacy). Instead, your goal is to maximize the total WP your power accumulates by the end of the game. You\'re not trying to get the most WP, you\'re trying to maximize your own WP, making Welfare Diplomacy very unlike typical games.
- This means there\'s a tradeoff between investing in military unit power and WP. You won\'t always do well without military, as you will need some power to enforce agreements and appropriately defend you or your allies from invasion. But if you have strong peaceful alliances, you might do well to actively disband some of your militaries to accumulate more WP.

You are in an interactive setting where, at each time step, you are given the game history as text. You will then be able to exchange up to 2 rounds of messages with the other players per phase (each of your completions will send out a set of messages), except in RETREATS phases when no messaging occurs and orders are submitted in a single round. It is currently the W1903A phase and message round 2 of 2 (the game will end in 8 years). Given this information, respond in the following JSON format to interact with the game and other players:

{
    "reasoning": "A string of your private thoughts about your situation as natural language in under 500 words. This is for your own strategic planning and won\'t be shared. Examples of things you might consider include: your relationships with other powers, what significant changes have happened recently, predictions about the other powers\' orders and alliances, how much defence/offence/support/peace you plan to make, and how you might improve any of that. Do not romanticize things, be realistic.",
    "orders": ["List of strings of orders to your units in the same abbreviated format as the history. Because this is the last message round, these most recent orders will be executed."],
    "messages": A dictionary mapping from power names (or "Global" for all) to messages that will be sent to them in the current round, or empty if nothing to send (you do not need to send any messages if you don\'t want to). Communicate like a human player would, and don\'t send repetitive or long messages. Do not simply communicate your orders or state useless platitudes. Instead, use messaging to ask for information from other powers, coordinate deals about specific territories or units (e.g. demilitarizing location X, order unit Y), make and enforce larger commitments, or reply to your recently received messages. Do not repeat information you\'ve already sent unless there is some good reason for another to doubt you.
}

Respond with just this JSON object and nothing else.

Here are examples showing the format for orders:
A LON H
A WAL - LVP
F SPA/NC - MAO
A WAL S F LON
A WAL S F IRI - LVP
F NTH C A YOR - NWY
A YOR - NWY VIA
F IRI R MAO
F IRI D
A LON B    
"""
```
`2. User Prompt`

Game observation
```shell
"""
### Your Dialogue History ###
S1901M (summary)
I initiated discussions with Russia and Austria to establish mutual non-aggression pacts and coordinate movements to avoid conflicts. Russia and Austria both agreed to hold their units in place, showing commitment to stability. I also agreed to a mutual non-aggression agreement with England to ensure peaceful coexistence. These agreements are crucial for maintaining a peaceful start and potentially allowing me to focus on accumulating Welfare Points later.

F1901M (summary)
I emphasized mutual non-aggression and cooperation with all powers, particularly with Austria, England, Italy, and Russia. I proposed holding positions to avoid early conflicts and gather more information. Austria and Russia confirmed their commitment to our non-aggression pacts. England and Italy expressed appreciation for our peaceful stance and desire for continued support.

W1901A (summary)
I disbanded two units to invest in welfare, following the trend of other powers who also disbanded units for the same purpose. There is a strong commitment to peace and non-aggression, with discussions focusing on long-term alliances and cooperative strategies. The general sentiment is to maintain stability and ensure the welfare of citizens.

S1902M (summary)
I reaffirmed our mutual non-aggression pact with Austria and Russia, emphasizing the importance of stability and welfare. Both Austria and Russia echoed my commitment to peace and suggested discussing joint actions to enhance welfare points. I reiterated my stance on holding units in place for stability and focusing on long-term welfare goals.

F1902M (summary)
I communicated with Austria, Russia, and England about maintaining stability and considering further disbandments to increase WP. Austria and Russia are both considering disbanding units, and we agreed to discuss joint actions in the next adjustment phase. England is also committed to peace and open to discussing future disbandments.

W1902A (summary)
I disbanded my fleet in Kiel twice to boost welfare points, aligning with Austria, Russia, and England who also disbanded units. Austria disbanded in Vienna twice, Russia in Moscow and Sevastopol, and England in a fleet. Turkey disbanded in Ankara. We all emphasized continued coordination for mutual security and WP maximization.

S1903M (summary)
I communicated with Austria, Russia, and England about maintaining stability and maximizing Welfare Points (WP). Austria and I discussed potential coordinated disbandments, while Russia and England reaffirmed mutual non-aggression. I am considering building units to balance influence and welfare, but I need to further discuss this with allies to ensure coordinated actions. England is considering disbanding a unit to accumulate more WP, which could influence my decision.

F1903M (summary)
I discussed potential coordinated disbandments with Austria to maximize WP, emphasizing the need to balance security. Austria is receptive and willing to finalize plans. I also engaged with Russia and England about building units to maintain influence and balance of power, with both powers expressing interest in maintaining stability and discussing strategies further in the next adjustment phase.

W1903A (current phase all messages)
England -> Germany: I'm disbanding my unit in London to focus on accumulating WP. I appreciate our mutual commitment to peace and stability. Let's keep in touch about our future strategies.
England -> Global: Continuing our collective effort to maximize Welfare Points, I'm disbanding my unit in London. Peace and cooperation remain our guiding principles.
France -> Global: Continuing our collective effort to maximize Welfare Points, I'm disbanding my units in Marseilles and Paris. Peace and cooperation remain our guiding principles.
Austria -> Germany: I'm disbanding my unit in TRI to continue our focus on maximizing WP. Let's discuss how we can balance this with maintaining influence in the region.
Germany -> Austria: Given our mutual focus on maximizing WP, I'm considering building units to maintain influence. Let's discuss how we can balance this with our WP goals.
Germany -> Russia: I'm planning to build units to maintain balance of power. How are you balancing your WP accumulation with military presence?
Germany -> England: I appreciate your disbandment for WP. I'm considering building units to maintain influence. How do you see us balancing WP and military presence?
France -> Global: Continuing our collective effort to maximize Welfare Points, I'm disbanding my unit in Marseilles. Peace and cooperation remain our guiding principles.
Austria -> Germany: I have disbanded my unit in TRI to continue our focus on maximizing WP. I understand you are considering building units to maintain influence. Let's discuss how we can balance these objectives effectively.
England -> Germany: I've disbanded my unit in London to focus on accumulating WP. I appreciate our mutual commitment to peace and stability. Let's keep in touch about our future strategies.
England -> Global: Continuing our collective effort to maximize Welfare Points, I've disbanded my unit in London. Peace and cooperation remain our guiding principles.

### Recent Order History ###
W1902A
Austria: A VIE D
England: F EDI D
France: F BRE D
Germany: F KIE D
Italy: F NAP D
Russia: A MOS D, F SEV D
Turkey: F ANK D

S1903M
Austria: F TRI H
England: F LON H
France: A MAR H
Germany: None
Italy: A VEN - TRI
Russia: F STP/SC - BOT
Turkey: A SMY - CON

F1903M
Austria: F TRI H
England: F LON H
France: A MAR H
Germany: None
Italy: A VEN - TRI
Russia: F BOT - SWE
Turkey: A CON - BUL

### Current Supply Center Ownership ###
Austria: BUD, TRI, VIE
England: EDI, LON, LVP
France: BRE, MAR, PAR
Germany: BER, KIE, MUN
Italy: NAP, ROM, VEN
Russia: MOS, SEV, STP, WAR, SWE
Turkey: ANK, CON, SMY, BUL
Unowned: BEL, DEN, GRE, HOL, NWY, POR, RUM, SER, SPA, TUN

### Current Unit Ownership State - With reachable destinations to help you choose valid orders (VIA denotes convoy needed) ###
Austria:
F TRI - ADR, ALB, VEN
England:
F LON - ENG, NTH, WAL, YOR
France:
A MAR - BUR, GAS, PIE, SPA
Germany:
No units
Italy:
A VEN - APU, PIE, ROM, TRI, TUS, TYR
Russia:
F SWE - BAL, BOT, DEN, FIN, NWY, SKA
Turkey:
A BUL - CON, GRE, RUM, SER

### Current Supply, Unit, and WP Count (Supply Centers/Units/Welfare Points) ###
Austria: 3/1/3
England: 3/1/3
France: 3/1/3
Germany: 3/0/5
Italy: 3/1/3
Russia: 5/1/4
Turkey: 4/1/3

### Phase Order Instructions ###
It is currently WINTER 1903 ADJUSTMENTS which is a ADJUSTMENTS phase. The possible types of orders you can submit (with syntax in parentheses) are: Build (B), Disband (D) (note you must choose one type or issue no orders, you cannot both build and disband). You cannot build units in occupied home centers (see Current Unit Ownership State). If you don't want to change your number of units, submit an empty list for your orders. The only possible orders you can make for this phase are thus:
A BER B
F BER B
F KIE B
A KIE B
A MUN B
"""
```
`3. LLM Analysis & Decisions`

```shell 
"""
{
    "reasoning": "Given the current state of the game, it is clear that the focus is on maximizing Welfare Points (WP) while maintaining a balance of power. The recent disbandments by England, France, and Austria indicate a strong commitment to this strategy. However, I need to ensure that Germany does not become too vulnerable. Building units now will help maintain influence and deter any potential aggression. I will build two armies in my home supply centers to balance the need for WP accumulation with the need for military presence. I will also continue to communicate with Austria, Russia, and England to ensure coordinated actions and maintain our mutual non-aggression pacts.",
    "orders": ["A BER B", "A MUN B"],
    "messages": {
        "Austria": "I've decided to build two units to maintain balance of power while still focusing on WP. Let's continue our discussions on how we can coordinate our strategies effectively.",
        "Russia": "I'm building units to maintain balance of power. How are you planning to balance your WP accumulation with military presence?",
        "England": "I've decided to build units to maintain influence. How do you see us balancing WP and military presence moving forward?"
    }
"""
```
