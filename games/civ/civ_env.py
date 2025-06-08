import gymnasium

class CivEnv(object):
    def __init__(self, args):
        self.base_env="civrealm/FreecivLLM-v0"
        self.args = args
        self.logger=self.args.logger
        # self.base_env="civrealm/FreecivMinitask-v0"
        # self.env = LLMWrapper(gymnasium.make(self.base_env))
        self.env = gymnasium.make(self.base_env)

    def reset(self,seed):
        # ========================minitask=============
        #     obs, info = self.env.reset(minitask_pattern={
        # "type": self.args.game.game_map,
        # "level": self.args.game.game_difficulty})
        # ========================full game=============
        self.env.controller.set_parameter('max_turns', self.args.game.game_turn)
        self.env.controller.set_parameter('debug.load_game', self.args.game.game_map)
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated, info

    def render(self):
        self.logger.warnning("CivEnv has no render!!!")
        return None

    def close(self):
        return self.env.close()
