import sys
import time

sys.path.append('../../')
from games.stratego import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions
from games.stratego.examples.util import softmax
import numpy as np


def nnet_choose_action_example(current_player, obs_from_env):
    # observation from the env is dict with multiple components
    board_observation = obs_from_env[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]
    valid_actions_mask = obs_from_env[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]

    # brief example as if we were choosing an action using a neural network.
    nnet_input = board_observation

    # neural network outputs logits in the same shape as the valid_actions_mask (board w x board h x ways_to_move).
    # since all logits here are the same value, this example will output a random valid action
    nnet_example_logits_output = np.ones_like(valid_actions_mask)

    # invalid action logits are changed to be -inf
    invalid_actions_are_neg_inf_valid_are_zero_mask = np.maximum(np.log(valid_actions_mask + 1e-8), np.finfo(np.float32).min)
    filtered_nnet_logits = nnet_example_logits_output + invalid_actions_are_neg_inf_valid_are_zero_mask

    # reshape logits from 3D to 1D since the Stratego env accepts 1D indexes in env.step()
    flattened_filtered_nnet_logits = np.reshape(filtered_nnet_logits, -1)

    # get action probabilities using a softmax over the filtered network logit outputs
    action_probabilities = softmax(flattened_filtered_nnet_logits)

    # choose an action from the output probabilities
    chosen_action_index = np.random.choice(range(len(flattened_filtered_nnet_logits)), p=action_probabilities)

    return chosen_action_index


def llm_gen(gameState,valid_move):
    from games.stratego.examples.stratepai_openai import get_openAI_move
    retimes=0
    while True:
        responce = get_openAI_move(gameState)
        if len(responce) == 7:
            action_key = responce[0] + ',' + responce[2]
            if action_key in valid_move.keys() and [int(responce[4]), int(responce[6])] in valid_move[action_key]:
                print("generate action is valid")
                break
            else:
                print("{} not in {} ".format(responce, valid_move))
                time.sleep(1)
                retimes+=1
                if retimes>3:
                    time.sleep(10)
                continue

    print(responce)

    action_cmd = [int(t) for t in responce.split(" ")]
    return action_cmd
def _player_index(player):
    # player 1 returns 0
    # player -1 returns 1
    return (player - 1) // -2
def get_gameState(env,current_player):
    piece_content_dict = {1: 'Spy', 2: 'Scout', 3: 'Miner', 4: 'Sergeant', 5: 'Lieutenant', 6: 'Captain', 7: 'Major',
                  8: 'Colonel', 9: 'General', 10: 'Marshall', 11: 'Flag', 12: 'Bomb'}
    piece_dict = {1: 's', 2: '¹', 3: '²', 4: '3', 5: '4', 6: '5', 7: '6',
                  8: '7', 9: '8', 10: '9', 11: '¶', 12: 'o'}
    # ['', '¶', 's', '¹', '²', '3', '4', '5', '6', '7', '8', '9', 'o']
    player_perspective_state = env.base_env.get_state_from_player_perspective(state=env.state, player=current_player)
    current_player_valid_move = env.base_env.get_dict_of_valid_moves_by_position(state=player_perspective_state, player=1)
    friend_board = player_perspective_state[_player_index(player=1)]
    enemy_board = player_perspective_state[_player_index(player=-1)]
    # friend_board = player_perspective_state[current_player, :, :]
    # enemy_board = player_perspective_state[(current_player - 1) // -2, :, :]
    enemy_part_view = player_perspective_state[_player_index(player=-1)+3]
    valid_moves_str = ''
    for k, v in current_player_valid_move.items():
        position = eval(k)
        piece_idx = friend_board[position[0], position[1]]

        valid_moves_str += piece_content_dict[piece_idx]+" 'R("+piece_dict[piece_idx] + ")' at position " + k + '  could move to'
        if len(v) > 1:
            valid_moves_str += " any of"
        valid_moves_str += ": " + ', '.join(' '.join(str(i) for i in item) for item in v)
        valid_moves_str += "\n"
        # print(valid_moves_str)
        # print("======")
        # print(friend_board)
        # print("======")
        # print(enemy_board)
    gameState = ''
    boardState = [['  ', ' c0 ', ' c1 ', ' c2 ', ' c3 ', ' c4 ', ' c5 ', ' c6 ', ' c7 ', ' c8 ', ' c9 ']]
    gameState += "## Board State:\n"
    gameState += '\n'.join([', '.join(map(str, sublist)) for sublist in boardState]) + "\n"
    # gameState += summary + "\n\n"
    # board_view=np.zeros_like(env.state[0,:,:])

    board_view = friend_board.copy()
    obstacle_view = env.state[2, :, :]
    for r in range(10):
        for c in range(10):
            if obstacle_view[r, c] == 1:
                board_view[r, c] = '-102'
            elif enemy_part_view[r, c] == 13:
                board_view[r, c] = '-101'
            elif enemy_part_view[r, c] > 0 and enemy_part_view[r, c] < 13:
                board_view[r, c] = -enemy_part_view[r, c]
    board_view_str = ''
    for r in range(10):
        board_view_str += "r{}".format(r)
        # print("r{}".format(r))
        for c in range(10):
            if board_view[r, c] == -101:
                show_str = 'B(#)'
                board_view_str += ", {}".format(show_str)
            elif board_view[r, c] == -102:
                show_str = "~~~~"
                board_view_str += ", {}".format(show_str)
            elif board_view[r, c] == 0:
                show_str = "...."
                board_view_str += ", {}".format(show_str)
            elif board_view[r, c] < 0 and board_view[r, c] > -13:
                show_str = -board_view[r, c]
                board_view_str += ", B({})".format(piece_dict[show_str])
            else:
                show_str = board_view[r, c]
                board_view_str += ", R({})".format(piece_dict[show_str])
            # print(", R{}".format(show_str), end='')
        board_view_str += "\n"
    gameState += board_view_str + "\n\n"
    gameState += "## Valid moves: \n" + valid_moves_str+"\n"
    gameState+='## IMPORTANT The selection you make must choose from the **Valid moves**'
    print(gameState)
    return gameState,current_player_valid_move

if __name__ == '__main__':
    config = {
        'version': GameVersions.STANDARD,
        'random_player_assignment': False,
        'human_inits': True,
        'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,

        'vs_human': True,  # one of the players is a human using a web gui
        'human_player_num': -1,  # 1 or -1
        'human_web_gui_port': 7000,
    }



    env = StrategoMultiAgentEnv(env_config=config)

    print(f"Visit \nhttp://localhost:{config['human_web_gui_port']}?player={config['human_player_num']} on a web browser")
    env_agent_player_num = config['human_player_num'] * -1

    number_of_games = 2
    for _ in range(number_of_games):
        print("New Game Started")
        obs = env.reset()
        while True:

            assert len(obs.keys()) == 1
            current_player = list(obs.keys())[0]
            assert current_player == env_agent_player_num

            current_player_action = nnet_choose_action_example(current_player=current_player, obs_from_env=obs)
            gameState, valid_move=get_gameState(env,current_player)
            action_cmd=llm_gen(gameState, valid_move)
            # star_r,star_c,end_r,end_c=[int(t) for t in responce.split(" ")]
            action = env.base_env.get_action_1d_index_from_positions(*action_cmd)
            obs, rew, done, info = env.step(action_dict={current_player: action}, is_spatial_index=False)
            # obs, rew, done, info = env.step(action_dict={current_player: current_player_action})
            # obs, rew, done, info = env.step(action_cmd=action_cmd)
            # print("0======",obs[0,:,:])
            # print("1======",obs[1,:,:])
            # env.gui_server.base_env.get_dict_of_valid_moves_by_position(state=env.state, player=1)
            print("======")
            # print(gameState)


            print(f"Player {current_player} made move {current_player_action}")



            if done["__all__"]:
                print(f"Game Finished, player {env_agent_player_num} rew: {rew[env_agent_player_num]}")
                break
            else:
                assert all(r == 0.0 for r in rew.values())
