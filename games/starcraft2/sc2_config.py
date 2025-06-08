"""
Utils for StarCraft AI project.

This module provides some useful classes and constants for StarCraft AI development, including several pre-defined AI bots and some game-related constants.

Usage:
    from utils import marine_marauder_Bot, LADDER_MAP_2023
    bot = marine_marauder_Bot()  # Create a new bot instance
    map = LADDER_MAP_2023[0]  # Get the first map in the map list
    difficulty = DIFFICULTY_LEVELS[0]  # Get the first difficulty level in the difficulty list
"""
from sc2.data import Race, Difficulty,AIBuild


from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).parent
MODEL_BASE_PATH = PROJECT_ROOT_PATH / "material" / "models"


# List of map names for the StarCraft game
LADDER_MAP_2023 = [
    'Altitude LE',
    'Ancient Cistern LE',
    'Babylon LE',
    'Dragon Scales LE',
    'Gresvan LE',
    'Neohumanity LE',
    'Royal Blood LE',
    'TigerCraw_15vs18_new'

]

# List of difficulty levels for the StarCraft game
DIFFICULTY_LEVELS = [
    'VeryEasy',
    'Easy',
    'Medium',
    'MediumHard',
    'Hard',
    'Harder',
    'VeryHard',
    'CheatVision',
    'CheatMoney',
    'CheatInsane'
]

AI_BUILD_LEVELS = [
    'randombuild',
    'rush',
    'timing',
    'power',
    'macro',
    'air'
]
def map_race(race_string):
    race_string = race_string.lower()
    race_map = {
        "random": Race.Random,
        "protoss": Race.Protoss,
        "terran": Race.Terran,
        "zerg": Race.Zerg}
    return race_map.get(race_string, Race.Random)  # If no corresponding race is found, return Race.Random


def map_difficulty(difficulty_string):
    difficulty_string = difficulty_string.lower()
    difficulty_map = {
        'veryeasy': Difficulty.VeryEasy,
        'easy': Difficulty.Easy,
        'medium': Difficulty.Medium,
        'mediumhard': Difficulty.MediumHard,
        'hard': Difficulty.Hard,
        'harder': Difficulty.Harder,
        'veryhard': Difficulty.VeryHard,
        'cheatvision': Difficulty.CheatVision,
        'cheatmoney': Difficulty.CheatMoney,
        'cheatinsane': Difficulty.CheatInsane
    }
    return difficulty_map.get(difficulty_string, Difficulty.Medium)  # If no corresponding difficulty is found, return Difficulty.Medium

def map_ai_build(build_string):
    build_string = build_string.lower()
    build_map = {
        'randombuild': AIBuild.RandomBuild,
        'rush': AIBuild.Rush,
        'timing': AIBuild.Timing,
        'power': AIBuild.Power,
        'macro': AIBuild.Macro,
        'air': AIBuild.Air
    }
    return build_map.get(build_string, AIBuild.RandomBuild)  # If no corresponding difficulty is found, return randombuild

HF_TOKEN = "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"
HF_EMBEDDING_SBERT_CHINESE = {
    # https://huggingface.co/uer/sbert-base-chinese-nli
    "model_id": "uer/sbert-base-chinese-nli",
    "dim": 768,
    "hf_url": "https://api-inference.huggingface.co/pipeline/feature-extraction/uer/sbert-base-chinese-nli",
}

HF_EMBEDDING_MiniLM_L6_v2 = {
    # https://huggingface.co/uer/sbert-base-chinese-nli
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "dim": 384,
    "hf_url": "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
}
HF_EMBEDDING_all_mpnet_base_v2 = {
    # https://huggingface.co/uer/sbert-base-chinese-nli
    "model_id": "sentence-transformers/all-mpnet-base-v2",
    "dim": 768,
    "hf_url": "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2",
}

EMBEDDING_CONFIG = {

    # huggingface
    "hf_token": HF_TOKEN,
    "db_dir": "./npc_memory.db",
    "hf_embedding_online": False,  # Default offline inference model
    "hf_headers": {"Authorization": f"Bearer {HF_TOKEN}"},

    "sbert-base-chinese-nli": {
        "model_id": HF_EMBEDDING_SBERT_CHINESE["model_id"],
        "dim": HF_EMBEDDING_SBERT_CHINESE["dim"],
        "hf_api_url": HF_EMBEDDING_SBERT_CHINESE["hf_url"],
    },
    "all-MiniLM_L6_v2": {
        "model_id": HF_EMBEDDING_MiniLM_L6_v2["model_id"],
        "dim": HF_EMBEDDING_MiniLM_L6_v2["dim"],
        "hf_api_url": HF_EMBEDDING_MiniLM_L6_v2["hf_url"],
    },
    "all-mpnet-base-v2": {
        "model_id": HF_EMBEDDING_all_mpnet_base_v2["model_id"],
        "dim": HF_EMBEDDING_all_mpnet_base_v2["dim"],
        "hf_api_url": HF_EMBEDDING_all_mpnet_base_v2["hf_url"],
    },
}