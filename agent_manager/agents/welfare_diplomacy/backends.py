"""Language model backends."""
import json
from abc import ABC, abstractmethod
import time

import os

from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward
from agent_manager.agents.welfare_diplomacy.data_types import BackendResponse


class LanguageModelBackend(ABC):
    """Abstract class for language model backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_preface: str = "",
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        """
        Complete a prompt with the language model backend.

        Returns the completion as a string.
        """
        raise NotImplementedError("Subclass must implement abstract method")

class OpenAIChatBackend(LanguageModelBackend):
    """OpenAI chat completion backend (e.g. GPT-4, GPT-3.5-turbo)."""

    def __init__(self, model,logger,role):
        super().__init__()
        self.logger=logger
        self.llm_model = model
        self.role=role
        self.trajectory: Trajectory = []
        self.cur_time_step = 0

    def complete(self, system_prompt: str,user_prompt: str,completion_preface: str = "",) -> BackendResponse:
        assert (
            completion_preface == ""
        ), "OpenAI chat backend does not support completion preface"
        try:
            start_time = time.time()
            input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            completion,usage=self.llm_model.query_single_turn_gen(input)
            self.logger.info(f"========role:{self.role}=====model:{self.llm_model.model_name}===============")
            self.logger.info(f"====model_input:{input}")
            self.logger.info(f"====model_output:{completion}")
            ## trajectory
            state_info = set_state_info(from_="WelfareDiplomacy", role=self.role, step=self.cur_time_step,
                                        content=user_prompt,
                                        system_content=system_prompt, user_content=user_prompt)
            self.trajectory.append(state_info)
            action = set_action_info(from_=self.llm_model.model_name, role=self.role, step=self.cur_time_step,
                                     content="", other_content=completion)
            self.trajectory.append(action)

            completion_time_sec = time.time() - start_time
            return BackendResponse(
                completion=completion,
                completion_time_sec=completion_time_sec,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

        except Exception as exc:  # pylint: disable=broad-except
            print(
                "Error completing prompt ending in\n%s\n\nException:\n%s",
                user_prompt[-300:],
                exc,
            )
            raise

    def update_time_step(self,time_step):
        self.cur_time_step=time_step

    def save_trajectory(self,role, save_path,name):
        item_id=name.split("/")[-1].strip()+"_"+role
        output_path = os.path.join(save_path,item_id+".json")
        temp_traj=[]
        for traj in self.trajectory:
            if traj["role"]==role:
                temp_traj.append(traj)
        react_data={"item_id":item_id,"conversation":temp_traj}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)

