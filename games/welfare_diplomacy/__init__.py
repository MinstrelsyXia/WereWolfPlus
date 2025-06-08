from gym.envs.registration import register
register(
    id='WelfareDiplogues-v0',
    entry_point='games.welfare_diplomacy.welfare_diplomacy_env:WelfareDiplomacyEnv'
)