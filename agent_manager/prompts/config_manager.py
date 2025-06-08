# config_manager.py
class ConfigManager:
    @staticmethod
    def load_config(game_name):
        # Here you can load the corresponding configuration from a configuration file or database based on the game name
        if game_name == "StarCraft II":
            return {
                "rules": "Standard 1v1 rules...",
                "roles": "You are the commander of the Protoss race...",
                "strategy_background": "This is a high-stakes match..."
            }
        elif game_name == "Dota 2":
            return {
                "rules": "Standard 5v5 rules...",
                "roles": "You are the captain...",
                "strategy_background": "Drafting phase is critical..."
            }
        else:
            raise ValueError("Unknown game name")

# Use ConfigManager to load configuration
config = ConfigManager.load_config("StarCraft II")
