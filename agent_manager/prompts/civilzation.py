import base_prompt
class CivilizationPrompt(base_prompt):
    def __init__(self, civilization):
        # Call the parent class's initialization method, passing in the game's goals, rules, etc.
        super().__init__(
            game_name="Civilization",
            game_goal="achieve victory through either domination, culture, science, religion, or diplomacy.",
            rules="1. Manage your civilization's resources wisely. 2. Develop cities and expand your empire. 3. Engage in diplomacy, trade, and warfare as necessary. 4. Follow the technology and culture trees to advance.",
            strategies="Consider the strengths and weaknesses of your civilization. Balance military expansion with economic development and cultural influence.",
            role_context=f"lead the {civilization} civilization to achieve one of the possible victory conditions."
        )
        self.civilization = civilization.lower()
        self.add_civilization_specific_prompts()

    def add_civilization_specific_prompts(self):
        """
        Add specific strategy prompts for different civilizations
        """
        civilization_specific_prompt = {
            "rome": "As Rome, leverage your civilization's ability to build roads automatically and spread your empire rapidly. Focus on military and infrastructure development.",
            "greece": "As Greece, take advantage of your unique city-state interactions. Focus on diplomacy and culture to secure alliances and influence.",
            "egypt": "As Egypt, utilize your civilization's ability to build wonders faster. Focus on cultural development and wonder production to achieve a cultural or religious victory.",
            "china": "As China, use your civilization's abilities to advance quickly in technology and culture. Focus on early-game advantages to build a strong foundation.",
            "america": "As America, leverage your civilization's military and economic strengths. Focus on expansion and dominance through military and trade."
        }
        self.system_prompt += f"\n\n{civilization_specific_prompt.get(self.civilization, '')}"

    def set_header_prompt(self, current_stage, key_scenarios):
        """
        Set the header prompt: guide the model to understand the current scenario, define the current game stage and key scenarios.
        """
        self.header_prompt = f"""
        --- Header Prompt ---
        Current Game Stage: {current_stage}

        Key Scenarios to consider:
        {key_scenarios}
        """

    def set_observation_prompt(self, game_state, resources, available_actions, opponents_moves):
        """
        Set the observation prompt: provide information about the current game state to ensure the model has enough context.
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
        {opponents_moves}``````````````````
        """

    def set_reasoning_prompt(self, short_term_goals, long_term_goals, opponent_prediction, risk_assessment):
        """
        Set the reasoning prompt: guide the model to formulate strategies through chain reasoning, hypothesis analysis, etc.
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

        Analyze the situation from multiple angles and develop a strategy that balances expansion, diplomacy, and cultural influence.
        """

    def determine_output_actions(self, suggested_actions):
        """
        Determine output actions: select and output appropriate actions based on current prompts and model analysis.
        """
        self.output_actions = f"""
        --- Output Actions ---
        Based on the current game state and reasoning:
        {suggested_actions}

        Choose the most appropriate actions that align with your civilization's strengths and current situation.
        """

    def log_information(self, log_entry):
        """
        Log information: used to store and manage important information or decisions generated during the analysis process.
        """
        self.information_logging.append(log_entry)

    def generate_full_prompt(self):
        """
        Generate the full prompt: combine system prompts, header prompts, observation prompts, reasoning prompts, and output actions.
        """
        return f"{self.system_prompt}\n{self.header_prompt}\n{self.observation_prompt}\n{self.reasoning_prompt}\n{self.output_actions}"


civ_prompt = CivilizationPrompt(civilization="Rome")
civ_prompt.set_header_prompt(
    current_stage="Classical Era",
    key_scenarios="Your empire is expanding rapidly, but you are bordered by a militaristic civilization. Consider whether to focus on defense or further expansion."
)

civ_prompt.set_observation_prompt(
    game_state="You have four cities, with two more settlers ready to establish new cities.",
    resources="Gold: 500, Science: 45 per turn, Culture: 30 per turn",
    available_actions="1. Build military units to defend against potential attacks. 2. Establish new cities to expand your empire. 3. Focus on research to unlock new technologies.",
    opponents_moves="Neighboring civilization has moved troops near your borders, indicating possible aggression."
)

civ_prompt.set_reasoning_prompt(
    short_term_goals="Secure your borders while continuing to expand your empire.",
    long_term_goals="Achieve a military or scientific victory by leveraging your empire's size and resources.",
    opponent_prediction="The neighboring civilization may attack soon; consider preparing a defensive strategy.",
    risk_assessment="Moderate risk of being attacked if defenses are not prioritized."
)

civ_prompt.determine_output_actions(
    suggested_actions="1. Train additional military units to fortify your borders. 2. Establish one of the new cities in a strategic location to secure resources. 3. Begin researching a technology that provides a defensive advantage."
)

# Log some important information
civ_prompt.log_information("Bordering civilization is moving troops; defense should be prioritized.")

full_prompt = civ_prompt.generate_full_prompt()
print(full_prompt)
