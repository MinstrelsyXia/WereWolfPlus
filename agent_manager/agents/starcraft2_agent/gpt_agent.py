

from agent_manager.agents.starcraft2_agent.gpt4_agent import ChatGPTAgent
class GPTAgent(object):

    def __init__(self, config,args=None, **kwargs):
        # super(GPTAgent, self).__init__(config)
        self.args = args
        self.logger = self.args.logger
        self.agent_name="GPTAgent"
        self.prompt_constructor = config.prompt
        self.model = config.llm_model
        self.gpt4_agent=ChatGPTAgent(self.model, self.prompt_constructor,self.logger)

    def step(self, observations,wandb_flag=True):
        """

        :param observations:
        :return:
        """
        # print("======================",observations.get('_time_step',-2))
        return self.gpt4_agent.action(observations,wandb_flag=wandb_flag)

    def set_trajectory_reward(self, env, role, score):
        self.gpt4_agent.L2.set_trajectory_reward(env, role, score)

    def save_trajectory(self, role, save_path,name):
        self.gpt4_agent.L2.save_trajectory(role, save_path, name)

