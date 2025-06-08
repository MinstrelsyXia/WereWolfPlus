import base_prompt
class StrategoPrompt(base_prompt):
    def __init__(self, role):
        # Call the parent class's initialization method, passing in the goals, rules, etc. of the Stratego game
        super().__init__(
            game_name="Stratego",
            game_goal="capture the opponent's flag while defending your own.",
            rules="1. Use your pieces strategically, considering their ranks and abilities. 2. Plan your moves to outmaneuver the opponent. 3. Protect your flag while trying to discover the opponent's.",
            strategies="Leverage the strengths of your higher-ranked pieces while using lower-ranked ones to gather intelligence. Balance offense with defense to maintain a solid strategy.",
            role_context=f"act as the commander of your army, using strategic thinking to lead your forces to victory."
        )
        self.role = role.lower()
        self.add_role_specific_prompts()

    def add_role_specific_prompts(self):
        """
        Add specific strategy prompts for different roles or units
        """
        role_specific_prompt = {
            "scout": "As a Scout, use your speed to explore the battlefield and gather information about the opponent's pieces. Be cautious of potential traps.",
            "miner": "As a Miner, your primary role is to disarm bombs. Move strategically to locate and neutralize threats.",
            "marshal": "As the Marshal, you are the highest-ranking piece. Use your strength to eliminate key opponent pieces but be wary of hidden bombs.",
            "spy": "As a Spy, your ability to eliminate the Marshal is crucial. Use stealth and deception to approach the enemy's strongest piece.",
            "general": "As the General, you are second only to the Marshal. Lead the charge against the opponent's key pieces while protecting your own high-ranking units."
        }
        self.system_prompt += f"\n\n{role_specific_prompt.get(self.role, '')}"

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

    def set_observation_prompt(self, board_state, known_enemy_positions, recent_moves, available_actions):
        """
        Set the observation prompt: provide information about the current game state to ensure the model has enough context.
        """
        self.observation_prompt = f"""
        --- Observation Prompt ---
        Board State:
        {board_state}

        Known Enemy Positions:
        {known_enemy_positions}

        Recent Moves:
        {recent_moves}

        Available Actions:
        {available_actions}
        """

    def set_reasoning_prompt(self, potential_threats, strategic_goals, risk_assessment):
        """
        Set the reasoning prompt: guide the model to formulate strategies through chain reasoning, hypothesis analysis, etc.
        """
        self.reasoning_prompt = f"""
        --- Reasoning Prompt ---
        Potential Threats:
        {potential_threats}

        Strategic Goals:
        {strategic_goals}

        Risk Assessment:
        {risk_assessment}

        Develop a plan that considers both immediate and long-term objectives, balancing offensive maneuvers with defensive positioning.
        """

    def determine_output_actions(self, suggested_actions):
        """
        Determine output actions: select and output appropriate actions based on current prompts and model analysis.
        """
        self.output_actions = f"""
        --- Output Actions ---
        Based on the current board state and reasoning:
        {suggested_actions}

        Choose the most appropriate moves to achieve your strategic goals while minimizing risk.
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
stratego_prompt = StrategoPrompt(role="Marshal")
stratego_prompt.set_header_prompt(
    current_stage="Mid-Game",
    key_scenarios="The opponent has moved a high-ranking piece near your flag; consider whether to engage or reposition your defenses."
)

stratego_prompt.set_observation_prompt(
    board_state="Your Marshal is positioned near the center, with known bombs protecting the flag.",
    known_enemy_positions="A potential General is approaching your flag from the left flank.",
    recent_moves="The opponent has been cautious, likely testing your defenses.",
    available_actions="1. Move the Marshal to engage the approaching piece. 2. Reposition your defenses to protect the flag. 3. Send a lower-ranked piece to scout the opponent's intentions."
)

stratego_prompt.set_reasoning_prompt(
    potential_threats="The approaching piece could be the opponent's General, posing a significant threat to your defenses.",
    strategic_goals="Neutralize the opponent's high-ranking pieces while maintaining a solid defense around your flag.",
    risk_assessment="High risk if the approaching piece is indeed the General. Consider a cautious approach to confirm the threat before engaging."
)

stratego_prompt.determine_output_actions(
    suggested_actions="1. Move a lower-ranked piece to scout the approaching unit. 2. Prepare the Marshal for engagement if the threat is confirmed. 3. Reposition a nearby bomb as a deterrent."
)

# Log some important information
stratego_prompt.log_information(
    "The opponent's movements suggest a targeted approach towards your flag; prepare for a potential high-ranking engagement.")

full_prompt = stratego_prompt.generate_full_prompt()
print(full_prompt)
