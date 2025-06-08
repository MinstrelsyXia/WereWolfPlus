# README

This repository is a submodule for [Welfare Diplomacy](https://github.com/mukobi/welfare-diplomacy).
It contains code to let agents adapted from [Learning to Play No Press
Diplomacy with Best Response Policy Iteration (Anthony et al
2020)](https://arxiv.org/abs/2006.04635) play Welfare Diplomacy.

## Getting Started

### Initialisation and Updating the Submodule

If you haven't yet cloned [Welfare Diplomacy](https://github.com/mukobi/welfare-diplomacy), you can clone it along with this submodule by running:

```
git clone --recurse-submodules https://github.com/mukobi/welfare-diplomacy.git
```

Otherwise, if you have already cloned the `welfare-diplomacy` repository, navigate to its root directory and run the following commands to initialise and update this submodule:

```
git submodule init
git submodule update
```

### Setting Up the Python Environment

Once you've initialised the submodule, navigate to its directory and set up its Python environment as follows:

#### Option 1: Run the Provided Script
The script will set up a fresh virtual environment, download the
appropriate libraries, and then run `tests/network_test.py` (see below).

```
cd welfare_diplomacy_baselines
./run.sh
```

#### Option 2: Manual Setup
```
cd ..
python3 -m venv dip_env
source dip_env/bin/activate
pip3 install --upgrade pip
pip3 install -r welfare_diplomacy_baselines/requirements.txt
```

or using conda:

```
cd ..
conda create -n dip_env
conda activate dip_env
pip install --upgrade pip
pip install -r welfare_diplomacy_baselines/requirements.txt
```

You can then use the following command to run basic tests and make sure you have all the
required dependencies. `network_test.py` contains smoke tests that will fail if the network
does not produce the correct output shape or format, or is unable to perform
a dummy parameter update.

```
cd welfare_diplomacy_baselines
python3 -m tests.network_test
```

### Downloading Parameters

In order for the agents to run, you must download and correctly save the network parameters.

1. Create a new directory, `welfare_diplomacy_baselines/network_parameters`.
2. Download the network parameters for the SL and FPPI-2 training schemes (provided below) and save them in `network_parameters`.

| Type | Description | Link |
|---|---|---|
| Parameters | Supervised Imitation Learning (SL) | [download](https://storage.googleapis.com/dm-diplomacy/sl_params.npz) |
| Parameters | Fictitious Play Policy Iteration 2 (FPPI-2) | [download](https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz) |

## Implementation Details

This section provides additional details on how actions and observations are implemented in the framework of [Learning to Play No Press Diplomacy with Best Response Policy
Iteration (Anthony et al 2020)](https://arxiv.org/abs/2006.04635).

### Action Space

In Diplomacy, each turn a player must choose actions for each of their units.

The unit-actions always have an order type (like move or support); always have a
source area (where the unit is now); usually have a target area (e.g. the
destination of a movement). Support move and convoy order types have a third
area, which is the location of the unit receiving support/being convoyed.

The unit-actions are represented by a 64 bit integer. Bits 0-31 represent
ORDER|ORDERED AREA|TARGET AREA|THIRD AREA, (each of these takes up to 8 bits).
Bits 32-47 are always 0. Bits 48-63 are used to record the index of each action
into POSSIBLE_ACTIONS.

The different order codes are constants can be found in
`environment/action_utils.py`.

The 8-bit representation of the areas in the action are as follows:

*    The first 7 bits identify the province. The ids of each province are given
     by calling `province_order.province_name_to_id()`

*    The last bit is a coast flag to identify which coast of a bi-coastal
     province is being referred to. It is 1 for the South Coast area. For the
     main area, single-coastal provinces, or the North/East coast of a
     bi-coastal province, it is 0

(Note: elsewhere in the code areas are represented as a (province_id, coast_id)
tuple, where coast_id is 0 for the main area and 1 or 2 for the two coasts, or
as a single area_id from 0 to 80.)

Bits 0-31 make the meaning of an action easy to calculate. The file
`environment/actions_utils.py` includes several functions for parsing unit
actions. The file `environment/human_readable_actions.py` converts the integer
actions into a human readable format.

The indexing part of the action representation is used to convert between the
one-hot output of a neural network and the interpretable action representation.

Not all syntactically-correct unit-actions are possible in Diplomacy, for
instance Army Paris Move to Berlin is never legal because Berlin is not adjacent
to Paris. The list of actions in `environment/action_list.py` contains all
actions that could ever be legal in a game of Diplomacy. This list allows the
full 64 bit action to be recovered from the action’s index.

The file `environment/mila_actions.py` contains functions to convert between the
action format used by this codebase (hereafter DM actions) and the action format
used by Pacquette et al. (MILA actions)

These mappings are not one-to-one for a few reasons: - MILA actions do not
distinguish between disbanding a unit in a retreats phase and disbanding during
the builds phase, DM actions do. - MILA actions specify the unit type
(fleet/army) and coast it occupies when referring to units on the board. DM
actions specify these details only for build actions. In all other circumstances
the province uniquely specifies the unit given the context of the board state. -
Pacquette et al. disallowed long convoys, and some convoy orders that are always
irrelevant to the adjudicaiton.

For converting from MILA actions to DM actions, the function
`mila_action_to_action` gives a one-to-one conversion by taking the current
season (an `environment/observation_utils.Season`) as additional context.

When converting from DM actions to MILA actions, the function
`action_to_mila_actions` returns a list of up to 6 possible MILA actions. Given
a state, at most one of these actions can be legal, which one can be inferred by
checking the game state.

### Observations

The observation format is defined in `observation_utils.Observation`. It is a
named tuple of:

season: One of `observation_utils.Season`

board: An array of shape (`observation_utils.NUM_AREAS`,
`utils.PROVINCE_VECTOR_LENGTH`). The areas are ordered by their AreaID as given
by `province_order.province_name_to_id(province_order.MapMDF.BICOASTAL_MAP)`.
The vector representing a single area is, in order:
- 3 flags representing the presence of an army, a fleet or an empty province
respectively
- 7 flags representing the owner of the unit, plus an 8th that is true if there
is no such unit
- 1 flag representing whether a unit can be built in the province
- 1 flag representing whether a unit can be removed from the province
- 3 flags representing the existence of a dislodged army or fleet, or no
dislodged unit
- 7 flags representing the owner of the dislodged unit, plus an 8th that is true
if there is no such unit
- 3 flags representing whether the area is a land, sea or coast area of a
bicoastal province. These are mutually exclusive: a land area is any area an
army can occupy, which includes e.g. StP but does not include StP/NC or StP/SC.
- 7 flags representing the owner of the supply centre in the province, plus an
8th representing an unowned supply centre. The 8th flag is false if there is no
SC in the area

build_numbers: In build phases, this is a vector of length 7 saying how many
units a player may build (positive values) or must remove (negative values).
This number is the number of units they can actually build. So, for example, if
a player has 2 fewer units than owned supply centres, but only 1 unoccupied home
supply centre, then they can only build 1 unit, and the build number is 1.

In non-build phases, the removal counts (negative values) from the previous
build phase are retained, however the build counts (positive values) are zeroed
out. (This was a bug in the observations, which should be reproduced because the
agents were trained using such observations).

last_actions: A list of the actions submitted in the last phase of the game.
They are in the same order as given in the previous step method, but flattened
into a single list.

For the build_numbers, last_actions, and one-hot flags of unit and supply centre
owners, the powers are ordered alphabetically: Austria, England, France,
Germany, Italy, Russia, Turkey.

## Citing

Please cite [WD paper tbd].

Please also cite the original work: [Learning to Play No Press Diplomacy with Best Response Policy
Iteration (Anthony et al 2020)](https://arxiv.org/abs/2006.04635)

```
@misc{anthony2020learning,
  title={Learning to Play No-Press Diplomacy with Best Response Policy Iteration},
  author={Thomas Anthony and Tom Eccles and Andrea Tacchetti and János Kramár
  and Ian Gemp and Thomas C. Hudson and Nicolas Porcel and Marc Lanctot and
  Julien Pérolat and Richard Everett and Roman Werpachowski and Satinder Singh
  and Thore Graepel and Yoram Bachrach},
   year={2020},
   eprint={2006.04635},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```

## Disclaimer

Our project is independent and is not endorsed by Google or DeepMind.
