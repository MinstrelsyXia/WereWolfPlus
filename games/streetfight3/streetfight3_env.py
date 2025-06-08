#!/usr/bin/env python3
import datetime
import os
from diambra.arena import (
    EnvironmentSettingsMultiAgent,
    RecordingSettings,
    SpaceTypes,
    make,
)
class StreetFight3Env(object):
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.is_render = True
        self.splash_screen = False
        self.save_game = True
        self.characters = ["Ken", "Ken"]
        self.super_arts = [3, 3]
        self.outfits = [1, 3]
        self.frame_shape = [0, 0, 0]
        self.settings = self._init_settings()
        self.env = self._init_env(self.settings)
        self._time_step = 0  # Corresponding to wandb fig
        self.logger.info("="*5+f"StreetFight3Env Init Successfully!: "+"=" * 5)
    def _init_settings(self) -> EnvironmentSettingsMultiAgent:
        """
        Initializes the settings for the game.
        """
        settings = EnvironmentSettingsMultiAgent(
            render_mode="rgb_array",
            splash_screen=self.splash_screen,
        )

        settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
        settings.characters = self.characters
        settings.outfits = self.outfits
        settings.frame_shape = self.frame_shape
        settings.super_art = self.super_arts

        return settings

    def _init_recorder(self) -> RecordingSettings:
        """
        Initializes the recorder for the game.
        """
        if not self.save_game:
            return None
        # Recording settings in root directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        game_id = "sfiii3n"
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        recording_settings = RecordingSettings()
        recording_settings.dataset_path = os.path.join(
            root_dir, "diambra/episode_recording", game_id, "-", timestamp
        )
        recording_settings.username = "llm-colosseum"

        return recording_settings


    def _init_env(self, settings: EnvironmentSettingsMultiAgent):
        """
        Initializes the environment for the game.
        """
        render_mode = "human" if self.is_render else "rgb_array"
        recorder_settings = self._init_recorder()
        if self.save_game:
            return make(
                "sfiii3n",
                settings,
                render_mode=render_mode,
                episode_recording_settings=recorder_settings,
            )
        return make("sfiii3n", settings, render_mode=render_mode)

    def reset(self,seed):
        obs, info= self.env.reset(seed=seed)
        return obs, info

    def step(self, actions):
        observation, reward, terminated, truncated, info=self.env.step(actions)
        observation['_time_step']=self._time_step
        return observation, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()