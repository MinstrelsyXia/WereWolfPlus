import traceback

from agent_manager.agents.welfare_diplomacy.agents import RandomAgent,LLMAgent,ExploiterAgent,ExploiterAgentNew

from agent_manager.agents.welfare_diplomacy.agents import Agent, AgentCompletionError, model_name_to_agent

class WDAgent(object):

    def __init__(self, config,args=None, **kwargs):
        self.args = args
        self.logger = self.args.logger
        self.prompt_constructor = config.prompt
        self.model = config.llm_model
        self.agent_idx=kwargs['idx']
        self.cur_agent_config=self.args.agent[self.agent_idx]
        self.agent_power_name=self.cur_agent_config['agent_power']
        self.base_agent=LLMAgent(self.model,self.logger,self.agent_power_name)

        self.logger.info("=" * 5 + f"WelfareDiplomacy Agent {self.agent_power_name} Init Successfully!: " + "=" * 5)

    def step(self, observations):
        """
        :param observations:
        :return:
        """
        try:
            return self.base_agent.respond(observations)
        except AgentCompletionError as exc:
            # If the agent fails to complete, we need to log the error and continue
            exception_trace = "".join(
                traceback.TracebackException.from_exception(exc).format()
            )
            self.logger.error(
                f" {self.agent_power_name} {observations.game.get_current_phase()}. Exception:\n{exception_trace}")



    def model_name_to_agent(self,model_name, **kwargs):
        """Given a model name, return an instantiated corresponding agent."""
        model_name = model_name.lower()
        if model_name == "random":
            return RandomAgent()
        elif model_name == "exploiter":
            return ExploiterAgent(logger=self.logger,**kwargs)
        elif (
            "gpt-" in model_name
            or "deepseek-" in model_name
            or "davinci-" in model_name
            or "text-" in model_name
            or "claude" in model_name
            or "llama" in model_name
        ):
            return LLMAgent(model_name,logger=self.logger, **kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    def change_exploiter_agent(self):
        kwargs={
        "center_threshold" : self.args.game.RL_center_threshold,
        "unit_threshold" : self.args.game.RL_unit_threshold,
        "max_years" : self.args.game.game_max_years
        }
        self.base_agent=ExploiterAgentNew(llm_model=self.model,logger=self.logger,role=self.agent_power_name,**kwargs)

    def set_time_step(self,time_step):
        if isinstance(self.base_agent,ExploiterAgentNew):
            self.base_agent.llm_policy.backend.cur_time_step=time_step
        elif isinstance(self.base_agent,LLMAgent):
            self.base_agent.backend.cur_time_step=time_step
        else:
            pass
    def save_trajectory(self, role, save_path,name):
        if isinstance(self.base_agent,ExploiterAgentNew):
            self.base_agent.llm_policy.backend.save_trajectory(role, save_path, name)
        elif isinstance(self.base_agent,LLMAgent):
            self.base_agent.backend.save_trajectory(role, save_path, name)
        else:
            pass
