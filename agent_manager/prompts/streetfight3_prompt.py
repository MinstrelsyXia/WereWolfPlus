class StreetFight3Prompt:
    def __init__(self):
        self.sys_prompt=self.get_sys_template()
        self.user_prompt=self.get_user_template()

    def get_user_template(self):
        return """Your next moves are:"""

    def get_sys_template(self):
        return """You are the best and most aggressive Street Fighter III 3rd strike player in the world.
    Your character is {character}. Your goal is to beat the other opponent. You respond with a bullet point list of moves.
    {context_prompt}
    The moves you can use are:
    {move_list}
    ----
    Reply with a bullet point list of moves. The format should be: `- <name of the move>` separated by a new line.
    Example if the opponent is close:
    - Move closer
    - Medium Punch
    
    Example if the opponent is far:
    - Fireball
    - Move closer
    """