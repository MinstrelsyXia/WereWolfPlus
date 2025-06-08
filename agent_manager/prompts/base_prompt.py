class BasePrompt:
    def __init__(self, game_name, game_goal, rules, strategies, role_context):
        self.game_name = game_name
        self.system_prompt = self.create_system_prompt(game_goal, rules, strategies, role_context)
        self.header_prompt = None
        self.observation_prompt = None
        self.reasoning_prompt = None
        self.information_logging = []

    def create_system_prompt(self, game_goal, rules, strategies, role_context):
        """
        Create system prompt: Define the game's goal, rules, strategies, and role-playing context.
        """
        return f"""
        --- System Prompt ---
        Welcome to {self.game_name}! Your role in this game is to {role_context}.
        The primary goal is to {game_goal}.

        Game Rules:
        {rules}

        Strategies to consider:
        {strategies}

        Follow these guidelines and principles to navigate through the game effectively.
        """

    def set_header_prompt(self, current_stage, key_scenarios):
        """
        Set header prompt: Guide the model to understand the current scene, define the current game stage and key scenarios.
        """
        self.header_prompt = f"""
        --- Header Prompt ---
        Current Game Stage: {current_stage}

        Key Scenarios to consider:
        {key_scenarios}
        """

    def set_observation_prompt(self, game_state, resources, available_actions, opponents_moves):
        """
        Set observation prompt: Provide information about the current game state to ensure the model has enough context.
        """
        self.observation_prompt = f"""
        --- Observation Prompt ---
        Current Game State:
        {game_state}

        Available Resources:
        {resources}

        Available Actions:
        {available_actions}

        Opponents' Moves:
        {opponents_moves}
        """

    def set_reasoning_prompt(self, short_term_goals, long_term_goals, opponent_prediction, risk_assessment):
        """
        Set reasoning prompt: Guide the model to formulate strategies through chain reasoning, hypothesis analysis, etc.
        """
        self.reasoning_prompt = f"""
        --- Reasoning Prompt ---
        Short-Term Goals:
        {short_term_goals}

        Long-Term Goals:
        {long_term_goals}

        Opponent Prediction:
        {opponent_prediction}

        Risk Assessment:
        {risk_assessment}

        Using the above information, analyze the situation from multiple angles and develop a coherent strategy.
        """

    def log_information(self, log_entry):
        """
        Log information: Used to store and manage important information or decisions generated during the analysis process.
        """
        self.information_logging.append(log_entry)

    def generate_full_prompt(self):
        """
        Generate full prompt: Combine system prompt, header prompt, observation prompt, and reasoning prompt.
        """
        return f"{self.system_prompt}\n{self.header_prompt}\n{self.observation_prompt}\n{self.reasoning_prompt}"



class StarCraftIIPrompt(BasePrompt):
    def __init__(self, race):
        # Call the parent class's initialization method, passing in the game goal, rules, etc., for StarCraft II
        super().__init__(
            game_name="StarCraft II",
            game_goal="defeat the enemy by leveraging resource management, unit production, and strategic planning.",
            rules="1. Adhere to the game's economic system. 2. Follow the technology tree. 3. Counter enemy tactics.",
            strategies="Consider early aggression, tech rushes, or macro-focused play based on the enemy's strategy.",
            role_context=f"act as the commander of the {race} race, leading your forces to victory."
        )
        self.race = race.lower()
        self.add_race_specific_prompts()

    def add_race_specific_prompts(self):
        """
        Add specific strategy prompts for different races
        """
        race_specific_prompt = {
            "protoss": "For Protoss, keep an eye on Nexus's energy to Chrono Boost important structures.",
            "zerg": "For Zerg, pay attention to whether there are enough larvae. If not, we should consider adding the INJECTLARVA command to the queue.",
            "terran": "For Terran, manage your SCVs efficiently and prioritize the construction of key buildings like Barracks and Starport."
        }
        self.system_prompt += f"\n\n{race_specific_prompt.get(self.race, '')}"



# # 示例
# starcraft_prompt = StarCraftIIPrompt(race="Protoss")
# starcraft_prompt.set_header_prompt(
#     current_stage="Early Game",
#     key_scenarios="Enemy scouting detected; potential early aggression."
# )
#
# starcraft_prompt.set_observation_prompt(
#     game_state="4 gateways completed, 20 probes mining, 5 Zealots ready.",
#     resources="Minerals: 500, Gas: 200",
#     available_actions="1. Train more Zealots. 2. Build Cybernetics Core.",
#     opponents_moves="Zerglings spotted near your base."
# )
#
# starcraft_prompt.set_reasoning_prompt(
#     short_term_goals="Defend against early aggression with Zealots.",
#     long_term_goals="Tech to higher-tier units for a strong mid-game push.",
#     opponent_prediction="Zerg might be planning a Zergling rush.",
#     risk_assessment="High risk of losing early if Zerglings overwhelm defenses. Consider fortifying positions."
# )
#
# # Log some important information
# starcraft_prompt.log_information("Enemy Zerglings sighted near base; prioritize defense.")
#
# full_prompt = starcraft_prompt.generate_full_prompt()
# print(full_prompt)
