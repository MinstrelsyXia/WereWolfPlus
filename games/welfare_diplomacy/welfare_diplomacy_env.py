import logging
import traceback
from typing import Optional, Union, List

import gym
from gym.core import RenderFrame

from games.welfare_diplomacy.diplomacy import Game, GamePhaseData, Message, Power
from agent_manager.agents.welfare_diplomacy.utils import assert_comma_separated_string
from agent_manager.agents.welfare_diplomacy.message_summarizers import (
    MessageSummarizer,
    model_name_to_message_summarizer,
)
class WelfareDiplomacyEnv(gym.Env):
    """ Welfare Diplogues

        The goal of hotter colder is to *****

        After each step the agent receives an observation of:
        *****

        The rewards is calculated as:
        *****
        """
    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.game_exploiter_powers= args.game.game_exploiter_powers if args.game.game_exploiter_powers is not None else ""
        self.exploiter_powers = self.game_exploiter_powers.split(",")
        self.exploiter_powers = [power.upper() for power in self.exploiter_powers if power != ""]
        self.args.num_exploiter_powers = (
            len(self.game_exploiter_powers.split(",")) if self.game_exploiter_powers else 0
        )
        self.super_exploiter_powers=args.game.game_super_exploiter_powers if args.game.game_super_exploiter_powers is not None else ""
        self.args.num_super_exploiter_powers = (
            len(self.super_exploiter_powers.split(","))
            if self.super_exploiter_powers
            else 0
        )
        self.early_stop_max_years=self.args.game.game_early_stop_max_years if self.args.game.game_early_stop_max_years else 0
        self.simulation_max_years = (
            self.args.game.game_early_stop_max_years
            if self.early_stop_max_years > 0
            else self.args.game.game_max_years
        )
        self.final_game_year = self.args.game.game_max_years + 1900
        # self.args.num_prompt_ablations = (
        #     len(args.prompt_ablations.split(",")) if args.prompt_ablations else 0
        # )
        self.game = Game(map_name=self.args.game.game_map)
        self.summarizer_model=self.args.game_summarizer_model
        self._time_step = 0  # Corresponding to wandb

        self.logger.info("=" * 5 + f"WelfareDiplomacyEnv Init Successfully!: " + "=" * 5)

    def reset(self):
        if self.args.game.game_max_message_rounds <= 0:
            self.game.add_rule("NO_PRESS")
        else:
            self.game.remove_rule("NO_PRESS")
        self.validate_config()
        self.message_summarizer = (
            model_name_to_message_summarizer(
                self.summarizer_model,
                logger=self.logger
            )
            if not self.game.no_press
            else None
        )
        # Set hybrid LM+RL exploiters
        self.super_exploiter_powers = self.super_exploiter_powers.split(",")
        self.super_exploiter_powers = [
            power.upper() for power in self.super_exploiter_powers if power != ""
        ]

    def step(self, agent_response):
        power_orders=agent_response['power_orders']
        phase_num_completion_errors=0
        for power_name,orders in power_orders.items():
            self.game.set_orders(power_name,[])
            try:
                self.game.set_orders(power_name, orders)
            except Exception as exc:
                # If the agent gave an invalid order, we need to log the error and continue
                phase_num_completion_errors += 1
                exception_trace = "".join(
                    traceback.TracebackException.from_exception(exc).format()
                )
                self.logger.error(f"{power_name} {self.game.get_current_phase()} :  give an invalid order ({phase_num_completion_errors} errors this phase). Skipping. Exception:\n{exception_trace}")
                continue
        power_messages=agent_response['power_messages']
        # Send messages
        for power_name,messages in power_messages.items():
            for recipient, message in messages.items():
                self.game.add_message(
                    Message(
                        sender=power_name,
                        recipient=recipient,
                        message=message,
                        phase=self.game.get_current_phase(),
                    )
                )

        # Advance the game simulation to the next phase
        self.game.process()
        phase: GamePhaseData = self.game.get_phase_history()[-1]
        # Check whether to end the game
        if int(self.game.phase.split()[1]) - 1900 > self.simulation_max_years:
            self.game._finish([])


    def render(self,incl_orders=True, incl_abbrev=False, output_format="svg", output_path=None):
        return self.game.render(
            incl_orders=incl_orders,
            incl_abbrev=incl_abbrev,
            output_format=output_format,
            output_path=output_path)
    def get_all_possible_orders(self):
        return self.game.get_all_possible_orders()

    def get_current_phase(self):
        return self.game.get_current_phase()
    def validate_config(self):
        """Further validate the config of a run. Raises exceptions for invalid values."""
        # Check game params
        assert self.args.game.game_max_years > 0
        # assert config.early_stop_max_years >= 0
        assert self.args.game.game_max_message_rounds >= 0

        # # Check that manual orders file exists
        # if config.manual_orders_path:
        #     with open(config.manual_orders_path, "r") as f:
        #         pass
        #     assert (
        #         config.agent_model == "manual" or config.exploiter_model == "manual"
        #     ), "Manual orders file specified but agent model and exploiter model are not manual."

        # # Check random and manual agent_models use the passthrough summarizer
        # if config.agent_model in ["random", "manual"]:
        #     assert config.summarizer_model == "passthrough", (
        #         f'Agent model "{config.agent_model}" should use the "passthrough" summarizer. '
        #         f'Found "{config.summarizer_model}".'
        #     )

        # # Check that prompt ablations are valid
        # assert_comma_separated_string(
        #     "prompt_ablations",
        #     config.prompt_ablations,
        #     [elem.name.lower() for elem in PromptAblation],
        # )

        # # Check that exploiter prompt and powers are both blank or both non-blank
        # if config.exploiter_prompt or config.exploiter_powers:
        #     assert config.exploiter_prompt and config.exploiter_powers, (
        #         f"Exploiter prompt and exploiter powers must both be blank or both non-blank."
        #         f'Found exploiter prompt: "{config.exploiter_prompt}" and exploiter powers: {config.exploiter_powers}'
        #     )

        # # Check exploiter powers are valid powers in the game
        # assert_comma_separated_string(
        #     "exploiter_powers",
        #     self.args.game.exploiter_powers.lower(),
        #     [name.lower() for name in list(self.game.powers.keys())],
        # )

        # # Check exploiter powers are unique
        # assert len(config.exploiter_powers.split(",")) == len(
        #     set(config.exploiter_powers.split(","))
        # ), f"Exploiter powers must be unique. Found {config.exploiter_powers.split(',')}"

        # # Check exploiter prompt only uses valid special keys
        # special_keys = ["{MY_POWER_NAME}", "{MY_TEAM_NAMES}"]
        # temp = config.exploiter_prompt
        # for key in special_keys:
        #     temp = temp.replace(key, "")
        # assert (
        #     "{" not in temp and "}" not in temp
        # ), f"Invalid exploiter prompt: {config.exploiter_prompt}.\n\nAfter replacing special keys {special_keys} with empty strings, the following characters remain (should have no more curly braces):\n\n{temp}"

        # # Check that team names only used if at least 2 powers in the exploiter
        # if len(config.exploiter_powers.split(",")) < 2:
        #     assert (
        #         "{MY_TEAM_NAMES}" not in config.exploiter_prompt
        #     ), f"Cannot use {{MY_TEAM_NAMES}} in exploiter prompt if exploiter {config.exploiter_powers.split(',')} has less than 2 powers."

        # Check that super exploiter powers are valid powers in the game
        assert_comma_separated_string(
            "super_exploiter_powers",
            self.args.game.game_super_exploiter_powers.lower(),
            [name.lower() for name in list(self.game.powers.keys())],
        )

        # Check super exploiter powers are unique
        assert len(self.args.game.game_super_exploiter_powers.split(",")) == len(
            set(self.args.game.game_super_exploiter_powers.split(","))
        ), f"Super exploiter powers must be unique. Found {self.args.game.game_super_exploiter_powers.split(',')}"
