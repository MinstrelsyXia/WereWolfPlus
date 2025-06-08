import base_prompt
class Dota2Prompt(base_prompt):
    def __init__(self, role):
        # 调用父类的初始化方法，传入Dota 2的游戏目标、规则等
        super().__init__(
            game_name="Dota 2",
            game_goal="destroy the enemy's Ancient while defending your own.",
            rules="1. Adhere to the lane dynamics and map control. 2. Manage hero abilities and item purchases strategically. 3. Counter enemy heroes and strategies.",
            strategies="Consider different lane strategies (e.g., safe lane, offlane, mid lane), hero power spikes, and item timing.",
            role_context=f"act as the {role} in your team, ensuring to fulfill your role's responsibilities (e.g., carry, support, mid, offlane, or roamer)."
        )
        self.role = role.lower()
        self.add_role_specific_prompts()

    def add_role_specific_prompts(self):
        """
        为不同的角色添加特定的策略提示
        """
        role_specific_prompt = {
            "carry": "As the carry, focus on farming efficiently in the early game to secure late-game dominance. Position carefully in fights to maximize your damage output.",
            "support": "As a support, prioritize warding key areas and protecting your carry. Provide healing and crowd control during fights.",
            "mid": "As the mid player, aim to dominate your lane and rotate to other lanes to create kill opportunities. Control the runes and apply pressure across the map.",
            "offlane": "As the offlaner, apply pressure to the enemy carry while staying alive. Initiate team fights and control the enemy's movements.",
            "roamer": "As a roamer, create early game pressure by ganking lanes. Secure map control and help your team establish dominance in the early game."
        }
        self.system_prompt += f"\n\n{role_specific_prompt.get(self.role, '')}"

    def set_header_prompt(self, current_stage, key_scenarios):
        """
        设置头部提示：引导模型理解当前场景，定义当前游戏阶段和关键情境。
        """
        self.header_prompt = f"""
        --- Header Prompt ---
        Current Game Stage: {current_stage}

        Key Scenarios to consider:
        {key_scenarios}
        """

    def set_observation_prompt(self, game_state, hero_status, resources, available_actions, opponents_moves):
        """
        设置观察提示：提供当前游戏状态的信息，确保模型有足够的上下文。
        """
        self.observation_prompt = f"""
        --- Observation Prompt ---
        Current Game State:
        {game_state}

        Hero Status:
        {hero_status}

        Available Resources:
        {resources}

        Available Actions:
        {available_actions}

        Opponents' Moves:
        {opponents_moves}
        """

    def set_reasoning_prompt(self, short_term_goals, long_term_goals, opponent_prediction, risk_assessment):
        """
        设置推理提示：引导模型通过链式推理、假设分析等方式制定策略。
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

        Analyze the situation from multiple angles and develop a coherent strategy balancing aggression with defense.
        """

    def log_information(self, log_entry):
        """
        记录信息：用于存储和管理在分析过程中生成的重要信息或决策。
        """
        self.information_logging.append(log_entry)

    def generate_full_prompt(self):
        """
        生成完整的Prompt：组合系统提示、头部提示、观察提示和推理提示。
        """
        return f"{self.system_prompt}\n{self.header_prompt}\n{self.observation_prompt}\n{self.reasoning_prompt}"


# 示例使用
dota2_prompt = Dota2Prompt(role="Carry")
dota2_prompt.set_header_prompt(
    current_stage="Early Game",
    key_scenarios="Your lane is being pressured by the enemy offlaner; consider retreating to jungle for safer farm."
)

dota2_prompt.set_observation_prompt(
    game_state="Your team has 2 towers down in the offlane; enemy is pushing mid.",
    hero_status="Your hero has completed Power Treads and is working on Battle Fury.",
    resources="Gold: 1200, XP: Level 9",
    available_actions="1. Continue farming in the jungle. 2. Rotate to mid to assist with defense. 3. Call for support assistance.",
    opponents_moves="Enemy midlaner missing; likely rotating to gank."
)

dota2_prompt.set_reasoning_prompt(
    short_term_goals="Avoid ganks while maximizing farm efficiency.",
    long_term_goals="Secure enough farm to become the primary damage dealer in the late game.",
    opponent_prediction="The enemy will likely focus on ganking you to delay your farm.",
    risk_assessment="High risk if caught alone; consider sticking with teammates or placing wards."
)

# Log some important information
dota2_prompt.log_information("Enemy midlaner missing; play cautiously and prioritize safe farm.")

full_prompt = dota2_prompt.generate_full_prompt()
print(full_prompt)
