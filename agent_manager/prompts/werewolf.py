import base_prompt

class WerewolfPrompt(base_prompt):
    def __init__(self, role):
        super().__init__(
            game_name="Werewolf",
            game_goal="survive and achieve your team's victory by identifying threats and forming alliances.",
            rules="1. Use deduction to identify the roles of other players. 2. Form alliances to protect yourself and eliminate threats. 3. Balance aggression with caution to avoid drawing suspicion.",
            strategies="Leverage your role's unique abilities to influence the game. Manage trust and deception carefully to achieve your goals.",
            role_context=f"act as a {role}, navigating the complexities of the game to secure victory for your side."
        )
        self.role = role.lower()
        self.add_role_specific_prompts()

    def add_role_specific_prompts(self):
        """
        Add specific strategy prompts for different roles
        """
        role_specific_prompt = {
            "werewolf": "As a Werewolf, work with your fellow wolves to eliminate villagers without drawing suspicion. Use deception to mislead others and survive the day phase.",
            "villager": "As a Villager, your goal is to identify and eliminate the Werewolves. Pay attention to others' behavior and form alliances to protect yourself and your fellow villagers.",
            "seer": "As the Seer, you can identify the true roles of other players. Use this ability wisely to guide the villagers without revealing your own identity too early.",
            "doctor": "As the Doctor, your role is to protect other players from being killed at night. Choose who to save carefully to maximize your impact on the game.",
            "hunter": "As the Hunter, if you are eliminated, you can take down one other player with you. Use this power strategically to remove key threats."
        }
        self.system_prompt += f"\n\n{role_specific_prompt.get(self.role, '')}"

    def set_header_prompt(self, current_phase, key_events):
        """
        Set the header prompt: guide the model to understand the current scenario, define the current game phase and key events.
        """
        self.header_prompt = f"""
        --- Header Prompt ---
        Current Game Phase: {current_phase}

        Key Events to consider:
        {key_events}
        """

    def set_observation_prompt(self, player_behaviors, recent_votes, known_roles, available_actions):
        """
        Set the observation prompt: provide information about the current game state to ensure the model has enough context.
        """
        self.observation_prompt = f"""
        --- Observation Prompt ---
        Player Behaviors:
        {player_behaviors}

        Recent Votes:
        {recent_votes}

        Known Roles:
        {known_roles}

        Available Actions:
        {available_actions}
        """

    def set_reasoning_prompt(self, suspicion_levels, alliance_options, risk_assessment):
        """
        Set the reasoning prompt: guide the model to formulate strategies through chain reasoning, hypothesis analysis, etc.
        """
        self.reasoning_prompt = f"""
        --- Reasoning Prompt ---
        Suspicion Levels:
        {suspicion_levels}

        Alliance Options:
        {alliance_options}

        Risk Assessment:
        {risk_assessment}

        Analyze the situation to determine the most likely threats and the best strategies to protect yourself and your team.
        """

    def determine_output_actions(self, suggested_actions):
        """
        Determine output actions: select and output appropriate actions based on current prompts and model analysis.
        """
        self.output_actions = f"""
        --- Output Actions ---
        Based on the current phase and reasoning:
        {suggested_actions}

        Choose the most appropriate actions to either deceive, protect, or eliminate other players.
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
werewolf_prompt = WerewolfPrompt(role="Seer")
werewolf_prompt.set_header_prompt(
    current_phase="Night Phase",
    key_events="You have identified one player's true role; consider how to use this information in the next day phase."
)

werewolf_prompt.set_observation_prompt(
    player_behaviors="Player A has been quiet, Player B is aggressively accusing others.",
    recent_votes="Player C was nearly voted out but was saved by a last-minute shift.",
    known_roles="You know that Player D is a Villager.",
    available_actions="1. Investigate another player. 2. Share your information subtly with trusted players. 3. Keep your knowledge secret to avoid becoming a target."
)

werewolf_prompt.set_reasoning_prompt(
    suspicion_levels="Player B's aggressive behavior could indicate they are a Werewolf trying to mislead others.",
    alliance_options="Player D seems trustworthy based on their voting pattern; consider forming an alliance.",
    risk_assessment="Revealing too much could make you a target for the Werewolves, but holding back could allow them to gain the upper hand."
)

werewolf_prompt.determine_output_actions(
    suggested_actions="1. Investigate Player B to confirm their role. 2. Consider sharing your findings with Player D in the next day phase."
)

# Log some important information
werewolf_prompt.log_information("Player B's behavior is suspicious; investigate them during the night.")

full_prompt = werewolf_prompt.generate_full_prompt()
print(full_prompt)
