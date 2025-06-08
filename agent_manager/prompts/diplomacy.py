import base_prompt
class DiplomacyPrompt(base_prompt):
    def __init__(self, country):
        # Call the parent class's initialization method, passing in the game's goals, rules, etc.
        super().__init__(
            game_name="Diplomacy",
            game_goal="achieve dominance through alliances, negotiations, and strategic decisions.",
            rules="1. Engage in diplomacy to form and maintain alliances. 2. Balance cooperation with competition. 3. Use negotiations to influence other players' decisions. 4. Adapt to changing geopolitical landscapes.",
            strategies="Leverage your country's strengths and geopolitical position. Manage relationships carefully to avoid isolation while pushing towards your goals.",
            role_context=f"act as the leader of {country}, guiding your nation to success through careful diplomacy."
        )
        self.country = country.lower()
        self.add_country_specific_prompts()

    def add_country_specific_prompts(self):
        """
        Add specific strategy prompts for different countries
        """
        country_specific_prompt = {
            "france": "As France, leverage your central position in Europe to form key alliances. Balance power between neighboring nations to prevent any single country from becoming too dominant.",
            "germany": "As Germany, focus on building strong military and economic alliances. Use your central location to influence both Eastern and Western Europe.",
            "russia": "As Russia, use your vast resources and size to your advantage. Form alliances with neighboring countries to secure your borders and expand your influence.",
            "britain": "As Britain, utilize your naval superiority to control sea routes and influence global trade. Form strategic alliances to maintain balance on the continent.",
            "america": "As America, focus on diplomacy across both hemispheres. Leverage your economic power to build influence and form key alliances in strategic regions."
        }
        self.system_prompt += f"\n\n{country_specific_prompt.get(self.country, '')}"

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

    def set_observation_prompt(self, geopolitical_state, alliances, tensions, available_actions, opponents_moves):
        """
        Set the observation prompt: provide information about the current game state to ensure the model has enough context.
        """
        self.observation_prompt = f"""
        --- Observation Prompt ---
        Current Geopolitical State:
        {geopolitical_state}

        Existing Alliances:
        {alliances}

        Rising Tensions:
        {tensions}

        Available Actions:
        {available_actions}

        Opponents' Moves:
        {opponents_moves}
        """

    def set_reasoning_prompt(self, short_term_goals, long_term_goals, alliance_strategy, risk_assessment):
        """
        Set the reasoning prompt: guide the model to formulate strategies through chain reasoning, hypothesis analysis, etc.
        """
        self.reasoning_prompt = f"""
        --- Reasoning Prompt ---
        Short-Term Goals:
        {short_term_goals}

        Long-Term Goals:
        {long_term_goals}

        Alliance Strategy:
        {alliance_strategy}

        Risk Assessment:
        {risk_assessment}

        Analyze the situation from multiple angles and develop a diplomatic strategy that balances alliance-building with strategic competition.
        """

    def determine_output_actions(self, suggested_actions):
        """
        Determine output actions: select and output appropriate actions based on current prompts and model analysis.
        """
        self.output_actions = f"""
        --- Output Actions ---
        Based on the current geopolitical state and reasoning:
        {suggested_actions}

        Choose the most appropriate diplomatic actions that align with your nation's strategic goals.
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


# Example usage
dip_prompt = DiplomacyPrompt(country="France")
dip_prompt.set_header_prompt(
    current_stage="Mid-Game",
    key_scenarios="You are at the center of a fragile alliance system; consider whether to strengthen your current alliances or form new ones to shift the balance of power."
)

dip_prompt.set_observation_prompt(
    geopolitical_state="Most of Europe is engaged in minor skirmishes, with tensions rising between Germany and Russia.",
    alliances="France is allied with Britain and Italy, but relations with Germany are deteriorating.",
    tensions="Germany is pressuring Russia, and there is growing concern about an Eastern bloc forming.",
    available_actions="1. Strengthen alliances with Britain and Italy. 2. Negotiate a non-aggression pact with Germany. 3. Seek new alliances in Eastern Europe.",
    opponents_moves="Germany has increased military presence near the Russian border, indicating possible aggressive intentions."
)

dip_prompt.set_reasoning_prompt(
    short_term_goals="Maintain stability in Western Europe while preventing any single power from dominating.",
    long_term_goals="Ensure France's security and influence in Europe by balancing power among neighboring countries.",
    alliance_strategy="Strengthen existing alliances while cautiously engaging with potential new allies in Eastern Europe.",
    risk_assessment="Moderate risk of Germany escalating tensions; consider diplomatic measures to de-escalate."
)

dip_prompt.determine_output_actions(
    suggested_actions="1. Reaffirm alliances with Britain and Italy to ensure their support. 2. Open diplomatic channels with Germany to negotiate a temporary ceasefire. 3. Begin discussions with potential Eastern European allies to counterbalance German influence."
)

# Log some important information
dip_prompt.log_information(
    "Germany's increased military presence is concerning; diplomatic efforts should focus on de-escalation.")

full_prompt = dip_prompt.generate_full_prompt()
print(full_prompt)
