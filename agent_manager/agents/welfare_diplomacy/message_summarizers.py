"""Summarize message history to condense prompts."""

from abc import ABC, abstractmethod
from logging import Logger

from games.welfare_diplomacy.diplomacy import Game, Message, Power

from agent_manager.agents.welfare_diplomacy.backends import OpenAIChatBackend

from agent_manager.agents.welfare_diplomacy.data_types import AgentParams, PhaseMessageSummary
from agent_manager.agents.welfare_diplomacy import utils
from agent_manager.prompts import welfare_diplomacy_prompt as prompts


class MessageSummarizer(ABC):
    """Summarize message history to condense prompts."""

    @abstractmethod
    def summarize(self, params: AgentParams) -> PhaseMessageSummary:
        """
        Summarize the most recent phase's messages as visible to the power.

        Important: Must be called before game.process to get any messages!
        """


class PassthroughMessageSummarizer(MessageSummarizer):
    """Don't summarize, just copy over the messages."""

    def __init__(self, logger: Logger, **kwargs):
        self.logger = logger

    def __repr__(self) -> str:
        return f"PassthroughMessageSummarizer"

    def summarize(self, params: AgentParams) -> PhaseMessageSummary:
        """Generate a summary with an OpenAI model."""
        if len(params.game.messages) == 0:
            utils.log_warning(self.logger, "No messages to summarize!")

        system_prompt = prompts.get_summarizer_system_prompt(params)  # For debugging
        original_message_list = get_messages_list(params.game, params.power)
        messages_string = combine_messages(original_message_list)

        return PhaseMessageSummary(
            phase=params.game.get_current_phase(),
            original_messages=original_message_list,
            summary=messages_string,
            prompt_tokens=len(messages_string.split()),
            completion_tokens=100,
        )


class LLMMessageSummarizer:
    """Message summarizer using a language model backend."""

    def __init__(self, model, logger,role,**kwargs):
        self.backend = OpenAIChatBackend(model,logger, role)
        self.model_mame=model.model_name
        self.logger=logger


    def __repr__(self) -> str:
        return f"LLMMessageSummarizer(backend={self.backend})"

    def summarize(self, params: AgentParams,turn_orders={}) -> PhaseMessageSummary:
        """Generate a summary with an OpenAI model."""
        if len(params.game.messages) == 0:
            utils.log_warning(self.logger, "No messages to summarize!")
            return PhaseMessageSummary(
                phase=params.game.get_current_phase(),
                original_messages=[],
                summary="No messages sent or received.",
                prompt_tokens=0,
                completion_tokens=0,
            )

        original_message_list = get_messages_list(params.game, params.power)
        messages_string = combine_messages(original_message_list)

        turn_orders_string = "Turn orders:\""
        for k, v in turn_orders.items():
            temp_str = k + ":" + ",".join(v)
            turn_orders_string += temp_str+" \n "
        turn_orders_string+="\""
        # print(turn_orders_string)
        dialogue_info="\n\n Dialogue information:\""+messages_string+"\""
        messages_string=turn_orders_string+dialogue_info
        system_prompt = prompts.get_summarizer_system_prompt(params)
        response = self.backend.complete(system_prompt, messages_string)
        self.logger.info(f"=============model:{self.model_mame}===============")
        # self.logger.info(f"=============player:{game_state['role']}--{game_state['name']}=action:{action}=============")
        # self.logger.info(f"====model_input:{prompt.}")
        # self.logger.info(f"====model_output:{result}")
        completion = response.completion.strip()

        return PhaseMessageSummary(
            phase=params.game.get_current_phase(),
            original_messages=original_message_list,
            summary=completion,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )


def model_name_to_message_summarizer(model,logger, **kwargs) -> MessageSummarizer:
    """Given a model name, return an instantiated corresponding agent."""
    return LLMMessageSummarizer(model=model,logger=logger,role="LLMMessageSummarizer", **kwargs)


def get_messages_list(game: Game, power: Power) -> list[str]:
    """Get a list of messages to pass through to the summarizer."""
    message: Message
    original_message_list = []
    for message in game.messages.values():
        if (
            message.sender != power.name
            and message.recipient != power.name
            and message.recipient != "GLOBAL"
        ):
            # Limit messages seen by this power
            continue
        message_repr = f"{message.sender.title()} -> {message.recipient.title()}: {message.message}\n"
        original_message_list.append(message_repr)
    return original_message_list


def combine_messages(original_message_list: list[str]) -> str:
    """Combine the messages into a single string."""
    messages_string = ""
    for message_repr in original_message_list:
        messages_string += message_repr
    if not messages_string:
        messages_string = "None\n"
    messages_string = messages_string.strip()  # Remove trailing newline
    return messages_string
